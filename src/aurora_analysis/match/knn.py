"""KD-tree halo matching by minimum ``(d / R)^2``.

Extracted verbatim (behaviour-wise) from ``parent_cen_sat_props.match_maps_knn``,
where it was buried in a 30 kB plotting script and reached by three other trees
through ``sys.path`` manipulation.

The matcher assigns each MAP to the halo minimising the periodic distance
scaled by that halo's radius, which is *not* the same as the nearest halo: a
massive halo slightly further away can win. That is why the tree query returns
``k`` candidates that are then re-ranked -- the tree orders by raw distance,
the metric orders by ``d/R``.

``k=128`` reproduces the brute-force result exactly on the ELG sample; ``k>=64``
was the observed threshold. See :func:`aurora_analysis.match.brute.match_maps_brute`
for the reference implementation used to validate this one.
"""

from __future__ import annotations

import numpy as np

from .._compat import HAVE_NUMBA, njit, prange
from .geometry import tile_periodic

__all__ = ["match_maps_knn", "match_maps_knn_local"]


@njit(nogil=True, parallel=True, cache=True)
def _rank_knn_metric(map_pos, halo_pos, R2, local_of_tile, j_idx, box):
    """Among the kNN tile candidates, pick the halo with min (periodic d/R)^2."""
    n, k = j_idx.shape
    half = 0.5 * box
    best_local = np.empty(n, dtype=np.int64)
    for im in prange(n):
        best = 1.0e300
        bi = -1
        for t in range(k):
            ih = local_of_tile[j_idx[im, t]]
            d2 = 0.0
            for ax in range(3):
                dx = abs(map_pos[im, ax] - halo_pos[ih, ax])
                if dx > half:
                    dx = box - dx
                d2 += dx * dx
            met = d2 / R2[ih]
            if met < best:
                best = met
                bi = ih
        best_local[im] = bi
    return best_local


def _rank_knn_metric_numpy(map_pos, halo_pos, R2, local_of_tile, j_idx, box):
    """Vectorised fallback with identical semantics, used when numba is absent.

    Processed in row blocks because the intermediate is ``(n_block, k, 3)``.
    """
    n, k = j_idx.shape
    best_local = np.empty(n, dtype=np.int64)
    block = max(1, int(4_000_000 / max(k, 1)))
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        cand = local_of_tile[j_idx[lo:hi]]  # (nb, k)
        d = map_pos[lo:hi, None, :] - halo_pos[cand]  # (nb, k, 3)
        d -= box * np.round(d / box)
        met = np.einsum("ijk,ijk->ij", d, d) / R2[cand]
        best_local[lo:hi] = np.take_along_axis(
            cand, np.argmin(met, axis=1)[:, None], axis=1
        )[:, 0]
    return best_local


def match_maps_knn_local(map_pos, halo_pos, halo_R, box: float, *, k: int = 128,
                         verbose: bool = True) -> np.ndarray:
    """Match each MAP to a halo; return the *local* halo row index.

    Parameters
    ----------
    map_pos : (N, 3) array
        MAP positions, same units and origin as ``halo_pos``.
    halo_pos, halo_R : (M, 3) and (M,) arrays
        Halo centres and match radii (e.g. ``r100_L2com``). ``halo_R`` must be
        strictly positive; callers should repair non-finite entries first.
    box : float
        Periodic box side length.
    k : int
        Candidates retrieved per MAP before re-ranking by ``(d/R)^2``.

    Returns
    -------
    (N,) int64 array of row indices into ``halo_pos`` / ``halo_R``.
    """
    from scipy.spatial import cKDTree

    map_pos = np.ascontiguousarray(map_pos, dtype=np.float64)
    halo_pos = np.ascontiguousarray(halo_pos, dtype=np.float64)
    halo_R = np.ascontiguousarray(halo_R, dtype=np.float64)

    if halo_pos.shape[0] == 0:
        raise ValueError("no halos supplied to match against")
    if not np.all(halo_R > 0):
        raise ValueError(
            "halo_R must be strictly positive; repair non-finite or zero radii first"
        )
    k = int(min(k, 27 * halo_pos.shape[0]))

    tiled, local_of_tile = tile_periodic(halo_pos, box)
    if verbose:
        print(f"  building tiled KDTree ({len(tiled):,} nodes) ...", flush=True)
    tree = cKDTree(tiled)
    if verbose:
        print(f"  querying k={k} ...", flush=True)
    _, j = tree.query(map_pos, k=k)
    j = np.ascontiguousarray(j, dtype=np.int64)
    if j.ndim == 1:
        j = j[:, None]

    R2 = np.ascontiguousarray(halo_R * halo_R, dtype=np.float64)
    local_of_tile = np.ascontiguousarray(local_of_tile, dtype=np.int64)
    rank = _rank_knn_metric if HAVE_NUMBA else _rank_knn_metric_numpy
    return rank(map_pos, halo_pos, R2, local_of_tile, j, float(box))


def match_maps_knn(map_pos, halo_pos, halo_R, halo_idx, box, k=128):
    """Legacy signature kept for drop-in compatibility with existing scripts.

    Returns ``(halo_idx[local], local)``, where ``local`` indexes the arrays
    passed in and ``halo_idx[local]`` maps back to global catalog ids.
    """
    local = match_maps_knn_local(map_pos, halo_pos, halo_R, box, k=k)
    return np.asarray(halo_idx)[local], local
