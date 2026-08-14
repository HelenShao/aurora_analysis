"""Ball-query halo matching (Hanyu Zhang).

A MAP is a candidate for a halo only when it lies inside that halo's sphere
(``d <= radius_factor * R``). When several spheres contain the same MAP, one
host is chosen by an overlap rule:

- ``"most_massive"`` (default) — largest mass / particle count; ties by
  distance, then halo id. The usual CompaSO host-assignment convention.
- ``"deepest"`` — smallest ``(d / R)^2``. Closest sibling of the legacy
  ACJ / ``method="knn"`` metric, but with a true enclosure query instead of
  a k-nearest shortlist.
- ``"nearest"`` — smallest centre distance.

Unmatched MAPs receive local index ``-1``. That differs from ``method="knn"``,
which always returns a host and leaves the caller to apply ``d < R``.

Periodic boxes use ``cKDTree(..., boxsize=...)`` on coordinates folded into
``[0, box)``, matching the reference implementation in
``hackAurora/AuroraVeldir/matching-hanyu.py``.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "BallMatchResult",
    "OverlapRule",
    "match_maps_ball",
    "match_maps_ball_local",
]

OverlapRule = Literal["nearest", "deepest", "most_massive"]


class BallMatchResult(NamedTuple):
    """Full ball-match output: per-MAP host and per-halo occupancy."""

    local_ih: np.ndarray
    """Local halo row per MAP; ``-1`` if outside every sphere."""
    n_enclosed: np.ndarray
    """Geometric count of MAPs inside each halo sphere (overlaps allowed)."""
    n_assigned: np.ndarray
    """Count of MAPs for which this halo is the unique chosen host."""


def match_maps_ball_local(
    map_pos,
    halo_pos,
    halo_R,
    box: float,
    *,
    overlap: OverlapRule = "most_massive",
    halo_mass=None,
    halo_id=None,
    radius_factor: float = 1.0,
    batch_size: int = 100_000,
    workers: int = -1,
    verbose: bool = False,
) -> np.ndarray:
    """Assign each MAP to a containing halo; return local indices (``-1`` = none)."""
    return match_maps_ball(
        map_pos,
        halo_pos,
        halo_R,
        box,
        overlap=overlap,
        halo_mass=halo_mass,
        halo_id=halo_id,
        radius_factor=radius_factor,
        batch_size=batch_size,
        workers=workers,
        verbose=verbose,
    ).local_ih


def match_maps_ball(
    map_pos,
    halo_pos,
    halo_R,
    box: float,
    *,
    overlap: OverlapRule = "most_massive",
    halo_mass=None,
    halo_id=None,
    radius_factor: float = 1.0,
    batch_size: int = 100_000,
    workers: int = -1,
    verbose: bool = False,
) -> BallMatchResult:
    """Containment-first MAP→halo match with configurable overlap resolution."""
    if overlap not in {"nearest", "deepest", "most_massive"}:
        raise ValueError(
            f"overlap must be 'nearest', 'deepest', or 'most_massive'; got {overlap!r}"
        )
    if not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(radius_factor) or float(radius_factor) <= 0.0:
        raise ValueError("radius_factor must be finite and positive")

    map_pos = np.ascontiguousarray(map_pos, dtype=np.float64)
    halo_pos = np.ascontiguousarray(halo_pos, dtype=np.float64)
    halo_R = np.ascontiguousarray(halo_R, dtype=np.float64)
    if map_pos.ndim != 2 or map_pos.shape[1] != 3:
        raise ValueError(f"map_pos must have shape (N, 3); got {map_pos.shape}")
    if halo_pos.ndim != 2 or halo_pos.shape[1] != 3:
        raise ValueError(f"halo_pos must have shape (M, 3); got {halo_pos.shape}")
    if halo_R.shape != (halo_pos.shape[0],):
        raise ValueError("halo_R must be one-dimensional with length len(halo_pos)")

    n_map = map_pos.shape[0]
    n_halo = halo_pos.shape[0]
    box = float(box)
    if not np.isfinite(box) or box <= 0.0:
        raise ValueError("box must be finite and positive")

    if halo_id is None:
        halo_ids = np.arange(n_halo, dtype=np.int64)
    else:
        halo_ids = np.ascontiguousarray(halo_id, dtype=np.int64)
        if halo_ids.shape != (n_halo,):
            raise ValueError("halo_id must have shape (n_halo,)")
        if halo_ids.size and np.min(halo_ids) < 0:
            raise ValueError("halo_id values must be non-negative (-1 is reserved)")

    masses: np.ndarray | None = None
    if overlap == "most_massive":
        if halo_mass is None:
            raise ValueError(
                "overlap='most_massive' requires halo_mass (e.g. CompaSO N)"
            )
        masses = np.ascontiguousarray(halo_mass, dtype=np.float64)
        if masses.shape != (n_halo,):
            raise ValueError("halo_mass must have shape (n_halo,)")

    with np.errstate(over="ignore", invalid="ignore"):
        effective_R = halo_R * float(radius_factor)

    eligible = np.all(np.isfinite(halo_pos), axis=1)
    eligible &= np.isfinite(effective_R) & (effective_R > 0.0)
    if masses is not None:
        eligible &= np.isfinite(masses) & (masses >= 0.0)
    finite_maps = np.all(np.isfinite(map_pos), axis=1)

    # scipy boxsize expects coordinates in [0, box).
    tree_halo_pos = np.mod(halo_pos, box)
    query_map_pos = np.mod(map_pos, box)
    periodic_box = np.full(3, box, dtype=np.float64)

    local_ih = np.full(n_map, -1, dtype=np.int64)
    n_enclosed = np.zeros(n_halo, dtype=np.int64)

    valid_halo = np.flatnonzero(eligible)
    valid_map = np.flatnonzero(finite_maps)
    if verbose:
        print(
            f"  ball match: n_MAP={n_map:,} eligible_halos={len(valid_halo):,} "
            f"overlap={overlap} R_factor={radius_factor}",
            flush=True,
        )

    if len(valid_halo) and len(valid_map):
        valid_tree_pos = tree_halo_pos[valid_halo]
        tree = cKDTree(valid_tree_pos, boxsize=periodic_box)
        maximum_radius = float(np.max(effective_R[valid_halo]))
        query_radius = np.nextafter(maximum_radius, np.inf)
        batch = int(batch_size)

        for start in range(0, len(valid_map), batch):
            map_rows = valid_map[start : start + batch]
            map_batch = query_map_pos[map_rows]
            candidate_lists = tree.query_ball_point(
                map_batch, r=query_radius, workers=workers
            )
            candidate_lengths = np.fromiter(
                (len(indices) for indices in candidate_lists),
                dtype=np.int64,
                count=len(map_rows),
            )
            if not np.any(candidate_lengths):
                continue

            map_in_batch = np.repeat(
                np.arange(len(map_rows), dtype=np.int64), candidate_lengths
            )
            tree_halo_index = np.concatenate(candidate_lists).astype(
                np.int64, copy=False
            )
            catalog_halo_index = valid_halo[tree_halo_index]

            displacement = map_batch[map_in_batch] - valid_tree_pos[tree_halo_index]
            displacement -= periodic_box * np.rint(displacement / periodic_box)
            distance_squared = np.einsum(
                "ij,ij->i", displacement, displacement, optimize=True
            )
            radius_squared = np.square(effective_R[catalog_halo_index])
            enclosed = distance_squared <= np.nextafter(radius_squared, np.inf)
            if not np.any(enclosed):
                continue

            map_in_batch = map_in_batch[enclosed]
            catalog_halo_index = catalog_halo_index[enclosed]
            distance_squared = distance_squared[enclosed]
            radius_squared = radius_squared[enclosed]

            enclosed_halos, enclosed_counts = np.unique(
                catalog_halo_index, return_counts=True
            )
            n_enclosed[enclosed_halos] += enclosed_counts

            if overlap == "nearest":
                order = np.lexsort(
                    (
                        halo_ids[catalog_halo_index],
                        distance_squared,
                        map_in_batch,
                    )
                )
            elif overlap == "deepest":
                order = np.lexsort(
                    (
                        halo_ids[catalog_halo_index],
                        distance_squared / radius_squared,
                        map_in_batch,
                    )
                )
            else:
                assert masses is not None
                order = np.lexsort(
                    (
                        halo_ids[catalog_halo_index],
                        distance_squared,
                        -masses[catalog_halo_index],
                        map_in_batch,
                    )
                )

            ordered_maps = map_in_batch[order]
            first_for_map = np.empty(len(order), dtype=bool)
            first_for_map[0] = True
            first_for_map[1:] = ordered_maps[1:] != ordered_maps[:-1]
            winners = order[first_for_map]

            winning_map_rows = map_rows[map_in_batch[winners]]
            winning_halo_rows = catalog_halo_index[winners]
            local_ih[winning_map_rows] = winning_halo_rows

    n_assigned = np.zeros(n_halo, dtype=np.int64)
    assigned = local_ih >= 0
    if np.any(assigned):
        assigned_halos, assigned_counts = np.unique(local_ih[assigned], return_counts=True)
        n_assigned[assigned_halos] = assigned_counts

    if verbose:
        print(
            f"  ball match: matched={int(assigned.sum()):,} / {n_map:,} "
            f"({100.0 * assigned.mean():.2f}%)",
            flush=True,
        )

    return BallMatchResult(local_ih=local_ih, n_enclosed=n_enclosed, n_assigned=n_assigned)
