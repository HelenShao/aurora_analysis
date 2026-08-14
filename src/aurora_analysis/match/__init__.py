"""Matching MAPs to halo catalogs.

Three methods share the same positional signature and return local halo row
indices into ``halo_pos`` / ``halo_R``::

    local_ih = match_maps(..., method="ball")   # default (Hanyu)
    local_ih = match_maps(..., method="knn")    # ACJ (d/R)^2 via kNN
    local_ih = match_maps(..., method="brute")  # exhaustive (d/R)^2

``method="ball"`` is containment-first: only halos whose sphere enclose the
MAP are candidates, and ``local_ih == -1`` means unmatched. Overlap among
enclosing spheres is resolved by ``overlap`` (default ``"most_massive"``).

``method="knn"`` / ``"brute"`` always return a host index; the caller applies
``d < R`` afterwards. Their metric is minimum periodic ``(d / R)^2``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .ball import (
    BallMatchResult,
    OverlapRule,
    match_maps_ball,
    match_maps_ball_local,
)
from .brute import match_maps_brute, match_maps_brute_local
from .catalog import (
    CURRENT_SCHEMA_VERSION,
    MatchCatalog,
    MatchProvenance,
    SchemaError,
    load_match_catalog,
    save_match_catalog,
)
from .censat import central_index_per_halo, label_cen_sat
from .geometry import (
    minimum_image_delta,
    periodic_distance,
    periodic_distance_sq,
    tile_periodic,
    wrap_centered,
    wrap_corner,
)
from .knn import match_maps_knn, match_maps_knn_local
from .naming import (
    METHOD_CATALOG_TAGS,
    format_match_stem,
    match_catalog_tag,
    parse_match_catalog_tag,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "BallMatchResult",
    "METHOD_CATALOG_TAGS",
    "MatchCatalog",
    "MatchProvenance",
    "OverlapRule",
    "SchemaError",
    "central_index_per_halo",
    "format_match_stem",
    "label_cen_sat",
    "load_match_catalog",
    "match_catalog_tag",
    "match_maps",
    "match_maps_ball",
    "match_maps_ball_local",
    "match_maps_brute",
    "match_maps_brute_local",
    "match_maps_knn",
    "match_maps_knn_local",
    "minimum_image_delta",
    "parse_match_catalog_tag",
    "periodic_distance",
    "periodic_distance_sq",
    "save_match_catalog",
    "tile_periodic",
    "wrap_centered",
    "wrap_corner",
]

Method = Literal["ball", "knn", "brute"]


def match_maps(
    map_pos,
    halo_pos,
    halo_R,
    box: float,
    *,
    method: Method = "ball",
    overlap: OverlapRule = "most_massive",
    halo_mass=None,
    halo_id=None,
    radius_factor: float = 1.0,
    k: int = 128,
    batch_size: int = 100_000,
    workers: int = -1,
    verbose: bool = True,
) -> np.ndarray:
    """Assign each MAP to a halo, returning row indices into the halo arrays.

    Parameters
    ----------
    method
        ``"ball"`` (default) — Hanyu containment query; unmatched → ``-1``.
        ``"knn"`` — tiled cKDTree + ``(d/R)^2`` re-rank (ACJ-compatible).
        ``"brute"`` — exhaustive ``(d/R)^2`` reference.
    overlap
        Only for ``method="ball"``: ``"most_massive"``, ``"deepest"``, or
        ``"nearest"``.
    halo_mass
        Required for ``method="ball"`` with ``overlap="most_massive"``
        (e.g. CompaSO ``N``).
    """
    if method == "ball":
        return match_maps_ball_local(
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
        )
    if method == "knn":
        return match_maps_knn_local(map_pos, halo_pos, halo_R, box, k=k, verbose=verbose)
    if method == "brute":
        return match_maps_brute_local(map_pos, halo_pos, halo_R, box)
    raise ValueError(
        f"unknown match method {method!r}; choose 'ball', 'knn', or 'brute'"
    )
