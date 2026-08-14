"""KDE environment estimates around a MAP from its DM neighbours.

Recomputes, in post-processing, the density / velocity-dispersion /
relative-velocity triple that Abacus computes on the fly, so that the kernel
radius and shape can be varied without rerunning the simulation.

This consolidates three previous implementations: two near-verbatim copies of
``compute_map_env_from_neighbors`` (in ``merge_validation_compare.py`` and
``map_kde_radius_scan.py``, differing only in whether the kernel shape was
selectable) and a third variant inside ``test5_alt_vel._kde_map_velocity``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..match.geometry import minimum_image_delta
from .kernel import kernel_weights

__all__ = ["MapEnv", "kde_mean_velocity", "map_env_from_neighbors"]


class MapEnv(NamedTuple):
    """Environment of one MAP. Unpacks as the legacy 4-tuple."""

    density: float
    velocity_dispersion: float
    relative_velocity: float
    weight_sum: float


def _weighted_moments(pos_map, vel_map, pos_dm, vel_dm, *, b, box, kernel):
    """Shared core: kernel weights, weighted mean velocity, weighted sigma."""
    dr = minimum_image_delta(pos_dm, pos_map, box)
    r = np.linalg.norm(dr, axis=-1)
    w = kernel_weights(r, b, kind=kernel)
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        nan3 = np.full(3, np.nan)
        return 0.0, nan3, float("nan")

    vel = np.asarray(vel_dm, dtype=np.float64)
    v_mean = (w[:, None] * vel).sum(axis=0) / w_sum
    mean_speed_sq = float((w * np.einsum("ij,ij->i", vel, vel)).sum() / w_sum)
    sigma = float(np.sqrt(max(mean_speed_sq - float(v_mean @ v_mean), 0.0)))
    return w_sum, v_mean, sigma


def map_env_from_neighbors(
    pos_map,
    vel_map,
    pos_dm,
    vel_dm,
    *,
    b: float,
    lbox: float,
    inv_map_kde: float,
    kernel: str = "map",
) -> MapEnv:
    """Density, velocity dispersion and relative velocity for one MAP.

    Parameters
    ----------
    pos_map, vel_map : (3,) arrays
        The MAP's own position and velocity.
    pos_dm, vel_dm : (M, 3) arrays
        Neighbouring DM particles. The caller is responsible for pre-selecting
        neighbours within ``b`` (typically via a KD-tree ``query_ball_point``);
        particles beyond ``b`` receive zero weight anyway but cost time.
    b : float
        Kernel radius, same length units as the positions.
    lbox : float
        Periodic box side.
    inv_map_kde : float
        Inverse kernel unit density. Must correspond to *this* ``b`` and
        ``kernel``; use :func:`aurora_analysis.env.kernel.inv_kde_for_radius`
        to rescale a stored value to a new radius, and note that the hollow and
        filled kernels have different normalisations at the same radius.
    kernel : str
        ``"map"``/``"hollow"`` or ``"dm"``/``"filled"``.

    Notes
    -----
    ``velocity_dispersion`` and ``relative_velocity`` do not depend on
    ``inv_map_kde``; only ``density`` does. If you are scanning kernel radii and
    only care about the velocity quantities, the normalisation is irrelevant.
    """
    w_sum, v_mean, sigma = _weighted_moments(
        pos_map, vel_map, pos_dm, vel_dm, b=b, box=lbox, kernel=kernel
    )
    if w_sum <= 0.0:
        return MapEnv(float("nan"), float("nan"), float("nan"), 0.0)
    v_rel = float(np.linalg.norm(np.asarray(vel_map, dtype=np.float64) - v_mean))
    return MapEnv(w_sum * float(inv_map_kde), sigma, v_rel, w_sum)


def kde_mean_velocity(
    pos_map,
    pos_dm,
    vel_dm,
    *,
    b: float,
    lbox: float,
    kernel: str = "dm",
) -> np.ndarray:
    """Kernel-weighted mean DM velocity around a MAP, as a ``(3,)`` vector.

    This is the central-galaxy velocity proxy: placing a MAP at the local mean
    DM velocity instead of its own velocity drops the median offset from the
    CompaSO L2 velocity dramatically for well-resolved MAPs. It defaults to the
    filled kernel because a hollow kernel deliberately downweights the core and
    therefore picks up infalling substructure in the mean.

    Returns ``[nan, nan, nan]`` when no neighbour carries weight.
    """
    _w_sum, v_mean, _sigma = _weighted_moments(
        pos_map, None, pos_dm, vel_dm, b=b, box=lbox, kernel=kernel
    )
    return np.asarray(v_mean, dtype=np.float64)
