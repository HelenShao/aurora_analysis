"""Ball-query matcher (Hanyu) and agreement with deepest ≈ knn on simple cases."""

from __future__ import annotations

import numpy as np
import pytest

from aurora_analysis.match import match_maps
from aurora_analysis.match.ball import match_maps_ball, match_maps_ball_local
from aurora_analysis.match.geometry import periodic_distance
from aurora_analysis.match.knn import match_maps_knn_local

BOX = 100.0


def _toy(*, seed=0, n_map=800, n_halo=120):
    rng = np.random.default_rng(seed)
    halo_pos = rng.uniform(-BOX / 2, BOX / 2, size=(n_halo, 3))
    halo_R = 10 ** rng.uniform(-0.3, 0.7, size=n_halo)
    halo_N = 10 ** rng.uniform(1.5, 4.0, size=n_halo)
    # Place many MAPs inside random host spheres so enclosure is common.
    host = rng.integers(0, n_halo, size=n_map)
    offset = rng.normal(scale=0.3, size=(n_map, 3))
    offset *= (0.5 * halo_R[host] / np.linalg.norm(offset, axis=1).clip(1e-9))[:, None]
    map_pos = halo_pos[host] + offset
    map_pos = map_pos - BOX * np.round(map_pos / BOX)  # keep roughly centered
    return map_pos, halo_pos, halo_R, halo_N


def test_unmatched_are_minus_one_not_a_fake_host():
    map_pos = np.array([[0.0, 0.0, 0.0]])
    halo_pos = np.array([[40.0, 0.0, 0.0]])
    halo_R = np.array([1.0])
    local = match_maps_ball_local(
        map_pos, halo_pos, halo_R, BOX, overlap="nearest", verbose=False
    )
    assert local[0] == -1


def test_single_enclosing_halo_is_chosen():
    map_pos = np.array([[0.1, 0.0, 0.0]])
    halo_pos = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    halo_R = np.array([1.0, 1.0])
    local = match_maps_ball_local(
        map_pos, halo_pos, halo_R, BOX, overlap="nearest", verbose=False
    )
    assert local[0] == 0


def test_most_massive_prefers_the_larger_host_in_an_overlap():
    # MAP at origin; small nearby halo and large farther halo both enclose it.
    map_pos = np.array([[0.0, 0.0, 0.0]])
    halo_pos = np.array([[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]])
    halo_R = np.array([1.0, 5.0])
    halo_N = np.array([50.0, 5000.0])
    local_mass = match_maps_ball_local(
        map_pos,
        halo_pos,
        halo_R,
        BOX,
        overlap="most_massive",
        halo_mass=halo_N,
        verbose=False,
    )
    local_deep = match_maps_ball_local(
        map_pos, halo_pos, halo_R, BOX, overlap="deepest", verbose=False
    )
    assert local_mass[0] == 1  # massive
    assert local_deep[0] == 0  # deeper in the small halo


def test_deepest_disagrees_with_most_massive_often_enough():
    map_pos, halo_pos, halo_R, halo_N = _toy(seed=1)
    deep = match_maps_ball_local(
        map_pos, halo_pos, halo_R, BOX, overlap="deepest", verbose=False
    )
    mass = match_maps_ball_local(
        map_pos,
        halo_pos,
        halo_R,
        BOX,
        overlap="most_massive",
        halo_mass=halo_N,
        verbose=False,
    )
    both = (deep >= 0) & (mass >= 0)
    assert both.any()
    # Not asserting a minimum fraction — just that the rules are not identical.
    assert np.any(deep[both] != mass[both])


def test_deepest_agrees_with_knn_on_matched_when_k_is_large():
    map_pos, halo_pos, halo_R, _ = _toy(seed=2, n_map=500, n_halo=80)
    ball = match_maps_ball_local(
        map_pos, halo_pos, halo_R, BOX, overlap="deepest", verbose=False
    )
    knn = match_maps_knn_local(map_pos, halo_pos, halo_R, BOX, k=128, verbose=False)
    d = periodic_distance(map_pos, halo_pos[knn], BOX)
    knn_matched = d < halo_R[knn]
    ball_matched = ball >= 0
    # Where both claim a match, deepest and knn should usually agree.
    both = ball_matched & knn_matched
    assert both.mean() > 0.5
    agree = np.mean(ball[both] == knn[both])
    assert agree > 0.95


def test_enclosed_counts_can_exceed_assigned_under_overlap():
    map_pos = np.array([[0.0, 0.0, 0.0]])
    halo_pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    halo_R = np.array([2.0, 2.0])
    halo_N = np.array([10.0, 100.0])
    result = match_maps_ball(
        map_pos,
        halo_pos,
        halo_R,
        BOX,
        overlap="most_massive",
        halo_mass=halo_N,
        verbose=False,
    )
    assert result.n_enclosed.sum() == 2  # in both spheres
    assert result.n_assigned.sum() == 1  # one unique host
    assert result.local_ih[0] == 1


def test_periodic_wrap_enclosure():
    map_pos = np.array([[49.0, 0.0, 0.0]])
    halo_pos = np.array([[-49.0, 0.0, 0.0]])
    halo_R = np.array([5.0])
    local = match_maps_ball_local(
        map_pos, halo_pos, halo_R, BOX, overlap="nearest", verbose=False
    )
    assert local[0] == 0
    assert periodic_distance(map_pos[0], halo_pos[0], BOX) <= halo_R[0]


def test_dispatcher_default_is_ball_most_massive():
    map_pos, halo_pos, halo_R, halo_N = _toy(seed=3, n_map=100, n_halo=40)
    a = match_maps(
        map_pos, halo_pos, halo_R, BOX, halo_mass=halo_N, verbose=False
    )
    b = match_maps(
        map_pos,
        halo_pos,
        halo_R,
        BOX,
        method="ball",
        overlap="most_massive",
        halo_mass=halo_N,
        verbose=False,
    )
    np.testing.assert_array_equal(a, b)
    assert (a >= -1).all()


def test_most_massive_requires_halo_mass():
    map_pos, halo_pos, halo_R, _ = _toy(seed=4, n_map=10, n_halo=5)
    with pytest.raises(ValueError, match="halo_mass"):
        match_maps(map_pos, halo_pos, halo_R, BOX, method="ball", verbose=False)


def test_radius_factor_expands_enclosure():
    map_pos = np.array([[1.5, 0.0, 0.0]])
    halo_pos = np.array([[0.0, 0.0, 0.0]])
    halo_R = np.array([1.0])
    assert match_maps_ball_local(
        map_pos, halo_pos, halo_R, BOX, overlap="nearest", radius_factor=1.0, verbose=False
    )[0] == -1
    assert match_maps_ball_local(
        map_pos, halo_pos, halo_R, BOX, overlap="nearest", radius_factor=2.0, verbose=False
    )[0] == 0
