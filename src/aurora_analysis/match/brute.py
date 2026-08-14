"""Brute-force all-pairs halo matching -- the reference implementation.

This is ACJ's ``compare_pos``: every MAP compared against every halo, keeping
the minimum ``(periodic d / R)^2``. It is O(N_map x N_halo) and impractical at
full-box scale (roughly 93k x 240k was already the practical ceiling), but it
is unambiguous, which makes it the ground truth that
:mod:`aurora_analysis.match.knn` is validated against.

Use it for small samples and for the agreement test; use the KD-tree matcher
for production.
"""

from __future__ import annotations

import numpy as np

from .._compat import HAVE_NUMBA, njit, prange

__all__ = ["match_maps_brute", "match_maps_brute_local"]


@njit(nogil=True, parallel=True, cache=True)
def _compare_pos(map_pos, halo_pos, halo_R, box):
    n_map = map_pos.shape[0]
    n_halo = halo_pos.shape[0]
    out = np.empty(n_map, dtype=np.int64)
    half = 0.5 * box
    for im in prange(n_map):
        best = 1.0e300
        best_ih = -1
        for ih in range(n_halo):
            d2 = 0.0
            for ax in range(3):
                dx = abs(map_pos[im, ax] - halo_pos[ih, ax])
                if dx > half:
                    dx = box - dx
                d2 += dx * dx
            r = halo_R[ih]
            metric = d2 / (r * r)
            if metric < best:
                best = metric
                best_ih = ih
        out[im] = best_ih
    return out


def _compare_pos_numpy(map_pos, halo_pos, halo_R, box):
    """Blocked numpy fallback with identical semantics."""
    n_map = map_pos.shape[0]
    out = np.empty(n_map, dtype=np.int64)
    r2 = halo_R * halo_R
    block = max(1, int(2_000_000 / max(halo_pos.shape[0], 1)))
    for lo in range(0, n_map, block):
        hi = min(lo + block, n_map)
        d = map_pos[lo:hi, None, :] - halo_pos[None, :, :]
        d -= box * np.round(d / box)
        met = np.einsum("ijk,ijk->ij", d, d) / r2[None, :]
        out[lo:hi] = np.argmin(met, axis=1)
    return out


def match_maps_brute_local(map_pos, halo_pos, halo_R, box: float) -> np.ndarray:
    """Match each MAP to a halo by exhaustive search; return local row indices.

    Same contract as :func:`aurora_analysis.match.knn.match_maps_knn_local`, so
    the two are interchangeable in tests and in
    :func:`aurora_analysis.match.match_maps`.
    """
    map_pos = np.ascontiguousarray(map_pos, dtype=np.float64)
    halo_pos = np.ascontiguousarray(halo_pos, dtype=np.float64)
    halo_R = np.ascontiguousarray(halo_R, dtype=np.float64)
    if halo_pos.shape[0] == 0:
        raise ValueError("no halos supplied to match against")
    if not np.all(halo_R > 0):
        raise ValueError("halo_R must be strictly positive")
    impl = _compare_pos if HAVE_NUMBA else _compare_pos_numpy
    return impl(map_pos, halo_pos, halo_R, float(box))


def match_maps_brute(map_pos, halo_pos, halo_R, halo_idx, box):
    """Brute-force analogue of the legacy ``match_maps_knn`` signature."""
    local = match_maps_brute_local(map_pos, halo_pos, halo_R, box)
    return np.asarray(halo_idx)[local], local
