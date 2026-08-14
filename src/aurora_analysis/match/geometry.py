"""Periodic-box geometry.

This module replaces fourteen separate copies of the minimum-image convention
that had accumulated across the analysis trees (``_periodic_delta`` x8,
``_periodic_dist`` x4, ``periodic_distance``, and a nested ``wrap``). They all
computed the same thing; having one implementation means a fix lands once.

Convention: ``minimum_image_delta(a, b, box)`` returns ``a - b`` folded into
``[-box/2, box/2)`` on every axis, matching the sign convention used
throughout the Abacus scripts. Abacus stores positions in centered coordinates
``[-box/2, box/2)``, but the minimum-image result is independent of whether the
inputs use centered or corner-origin coordinates.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TILE_SHIFTS",
    "minimum_image_delta",
    "periodic_distance",
    "periodic_distance_sq",
    "tile_periodic",
    "wrap_centered",
    "wrap_corner",
]


def minimum_image_delta(a, b, box: float) -> np.ndarray:
    """Separation ``a - b`` under the minimum-image convention.

    Broadcasts, so ``a`` may be ``(N, 3)`` against a single ``(3,)`` centre.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    box = float(box)
    return d - box * np.round(d / box)


def periodic_distance(a, b, box: float) -> np.ndarray:
    """Euclidean distance between ``a`` and ``b`` across periodic boundaries."""
    return np.linalg.norm(minimum_image_delta(a, b, box), axis=-1)


def periodic_distance_sq(a, b, box: float) -> np.ndarray:
    """Squared periodic distance. Avoids the sqrt when only ranking matters."""
    d = minimum_image_delta(a, b, box)
    return np.einsum("...i,...i->...", d, d)


def wrap_centered(pos, box: float) -> np.ndarray:
    """Fold positions into ``[-box/2, box/2)`` (the Abacus convention)."""
    box = float(box)
    p = np.asarray(pos, dtype=np.float64)
    return (p + 0.5 * box) % box - 0.5 * box


def wrap_corner(pos, box: float) -> np.ndarray:
    """Fold positions into ``[0, box)`` (the convention pycorr/Corrfunc expect)."""
    return np.asarray(pos, dtype=np.float64) % float(box)


#: The 27 integer offsets of a periodic 3x3x3 replication, ordered so that the
#: identity shift is not special-cased anywhere downstream.
TILE_SHIFTS = np.array(
    [(ix, iy, iz) for ix in (-1, 0, 1) for iy in (-1, 0, 1) for iz in (-1, 0, 1)],
    dtype=np.float64,
)


def tile_periodic(pos, box: float) -> tuple[np.ndarray, np.ndarray]:
    """Replicate points into the 27 neighbouring periodic images.

    Returns ``(tiled, local_of_tile)`` where ``tiled`` has ``27 * len(pos)``
    rows and ``local_of_tile`` maps each tiled row back to its index in the
    original array. Feeding ``tiled`` to a non-periodic KD-tree is what makes
    the tree honour the box wrap.
    """
    p = np.ascontiguousarray(pos, dtype=np.float64)
    n = len(p)
    tiled = np.vstack([p + s * float(box) for s in TILE_SHIFTS])
    local_of_tile = np.tile(np.arange(n, dtype=np.int64), len(TILE_SHIFTS))
    return tiled, local_of_tile
