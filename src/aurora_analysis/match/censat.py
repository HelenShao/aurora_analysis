"""Central / satellite labelling.

Rule: within each matched halo, the member with the highest multiplicity is
the central (halo primary) and the rest are satellites.

**Host-primary / HOD (correct production path).** Call this on the *full*
matched MAP catalog at ``z_cat`` (Stage A) *before* ELG/LRG/ram cuts. Join
``is_halo_primary`` onto later selections by ``maps`` id. See the package
README section "Matching MAPs to halos (correct order)".

**Sample-internal (diagnostics only).** If you pass only MAPs that already
passed a tracer cut, the same MAP can flip cen/sat depending on who else is in
the sample. That is *not* the host-primary definition used for HOD overlays.

The implementation here is vectorised. The original looped over halo groups in
Python, which cost minutes on the 83 M matched MAPs of a full-box highbase
match.
"""

from __future__ import annotations

import numpy as np

__all__ = ["central_index_per_halo", "label_cen_sat", "label_cen_sat_loop"]


def label_cen_sat(mult, local_halo_ih) -> np.ndarray:
    """Boolean ``is_central`` per entry: highest ``mult`` in each halo wins.

    Parameters
    ----------
    mult : (N,) array
        Multiplicity (or any "which one is the main object" ranking quantity).
    local_halo_ih : (N,) int array
        Host halo index per entry. Pass only matched entries; there is no
        unmatched sentinel.

    Ties in ``mult`` are broken by the lowest entry index, which makes the
    result deterministic and independent of input ordering. The original
    implementation broke ties via an unstable ``argsort`` and could therefore
    return different centrals for the same input across runs.
    """
    mult = np.asarray(mult, dtype=np.float64)
    ih = np.asarray(local_halo_ih)
    n = len(ih)
    is_cen = np.zeros(n, dtype=bool)
    if n == 0:
        return is_cen
    if len(mult) != n:
        raise ValueError(f"mult has length {len(mult)} but local_halo_ih has {n}")
    is_cen[central_index_per_halo(mult, ih)] = True
    return is_cen


def central_index_per_halo(mult, local_halo_ih) -> np.ndarray:
    """Indices of the central entry of each occupied halo.

    Returns one index per distinct value in ``local_halo_ih``, sorted by halo
    index. Useful when you want the centrals themselves rather than a mask.
    """
    mult = np.asarray(mult, dtype=np.float64)
    ih = np.asarray(local_halo_ih)
    n = len(ih)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    # Primary key is the halo, secondary is descending multiplicity, tertiary
    # is the original index so ties resolve deterministically. lexsort applies
    # the last key first.
    order = np.lexsort((np.arange(n), -mult, ih))
    ih_sorted = ih[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ih_sorted)) + 1))
    return order[starts].astype(np.int64)


def label_cen_sat_loop(elg_mult, local_halo_ih) -> np.ndarray:
    """The original group-loop implementation, kept for regression testing.

    Do not use in production: it is O(N log N) in the sort plus a Python loop
    over halos, and its tie-breaking is not reproducible.
    """
    n = len(elg_mult)
    is_cen = np.zeros(n, dtype=bool)
    order = np.argsort(local_halo_ih)
    ih_sorted = np.asarray(local_halo_ih)[order]
    breaks = np.flatnonzero(np.diff(ih_sorted)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))
    for s, e in zip(starts, ends):
        idx = order[s:e]
        if len(idx) == 0:
            continue
        j = idx[np.argmax(np.asarray(elg_mult)[idx])]
        is_cen[j] = True
    return is_cen
