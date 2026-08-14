"""KDE kernel shapes for MAP environments.

Two shapes are in use, and the choice matters scientifically rather than just
numerically:

``hollow`` (``w = s^2 (1 - s^2)``)
    The production MAP kernel. Downweighting the core maximises *environmental
    contrast*, which is what you want for infall and satellite diagnostics.

``filled`` (``w = 1 - s^2``)
    The DM-like kernel. Keeping the core makes it a better estimator of the
    halo's mean velocity, which is what you want for a central-galaxy velocity
    proxy.

There is no single kernel that is best for both, and running dual kernels in
production is not practical -- hence the compromise of a hollow kernel plus a
stored relative-velocity direction. Both are defined on ``s = r / b < 1`` and
vanish outside.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "KERNEL_ALIASES",
    "inv_kde_for_radius",
    "kernel_volume_integral",
    "kernel_weights",
    "normalized_kernel_weights",
]

#: Accepted spellings. "map"/"dm" are the names used in the original scripts;
#: "hollow"/"filled" are the names used in the notes and the readiness deck.
KERNEL_ALIASES: dict[str, str] = {
    "map": "hollow",
    "hollow": "hollow",
    "dm": "filled",
    "filled": "filled",
}

#: Analytic value of the dimensionless radial integral
#: ``int_0^1 w(s) s^2 ds``, so that ``int w d^3r = 4 pi b^3 * this``.
_RADIAL_INTEGRAL: dict[str, float] = {
    "hollow": 2.0 / 35.0,  # int s^2(1-s^2) s^2 ds = 1/5 - 1/7
    "filled": 2.0 / 15.0,  # int (1-s^2) s^2 ds   = 1/3 - 1/5
}


def _canonical(kind: str) -> str:
    try:
        return KERNEL_ALIASES[str(kind).lower()]
    except KeyError:
        raise ValueError(
            f"unknown kernel {kind!r}; choose from {', '.join(sorted(KERNEL_ALIASES))}"
        ) from None


def kernel_weights(distances, b: float, *, kind: str = "map") -> np.ndarray:
    """Unnormalised KDE weights for neighbours at ``distances`` given radius ``b``."""
    shape = _canonical(kind)
    b = float(b)
    if not b > 0:
        raise ValueError(f"kernel radius b must be positive, got {b}")
    s2 = (np.asarray(distances, dtype=np.float64) / b) ** 2
    w = np.zeros_like(s2)
    inside = s2 < 1.0
    if shape == "hollow":
        w[inside] = s2[inside] * (1.0 - s2[inside])
    else:
        w[inside] = 1.0 - s2[inside]
    return w


def kernel_volume_integral(b: float, *, kind: str = "map") -> float:
    """``int w(r) d^3r`` over the sphere of radius ``b``.

    ``8 pi b^3 / 35`` for the hollow kernel and ``8 pi b^3 / 15`` for the
    filled one. This is the normalisation that turns a weight sum into a
    density, and the reason a filled-kernel run needs a different
    ``inv_kde`` than a hollow one at the same ``b``.
    """
    return 4.0 * np.pi * float(b) ** 3 * _RADIAL_INTEGRAL[_canonical(kind)]


def normalized_kernel_weights(distances, b: float, *, kind: str = "map") -> np.ndarray:
    """Kernel weights divided by ``kernel_volume_integral``.

    Summing these over neighbours gives a number density directly, without
    needing the simulation's stored ``invMap_KDEunitdensity``.
    """
    return kernel_weights(distances, b, kind=kind) / kernel_volume_integral(b, kind=kind)


def inv_kde_for_radius(inv_kde_ref: float, b_ref: float, b: float) -> float:
    """Rescale a stored ``invMap_KDEunitdensity`` from ``b_ref`` to ``b``.

    The kernel integral scales as ``b^3`` for both shapes, so the inverse
    normalisation scales as ``(b_ref / b)^3``. This only converts between radii
    for the *same* kernel shape; switching shape changes the ``2/35`` vs
    ``2/15`` factor as well.
    """
    return float(inv_kde_ref) * (float(b_ref) / float(b)) ** 3
