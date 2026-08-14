"""The KD-tree matcher must reproduce the brute-force reference."""

import numpy as np
import pytest

from aurora_analysis.match import match_maps
from aurora_analysis.match.brute import match_maps_brute_local
from aurora_analysis.match.geometry import periodic_distance
from aurora_analysis.match.knn import match_maps_knn, match_maps_knn_local

BOX = 100.0


def _sample(seed=0, n_map=2000, n_halo=300):
    rng = np.random.default_rng(seed)
    map_pos = rng.uniform(-BOX / 2, BOX / 2, size=(n_map, 3))
    halo_pos = rng.uniform(-BOX / 2, BOX / 2, size=(n_halo, 3))
    # Wide radius range so that d/R genuinely disagrees with nearest-neighbour.
    halo_R = 10 ** rng.uniform(-0.5, 0.9, size=n_halo)
    return map_pos, halo_pos, halo_R


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_knn_reproduces_brute_force(seed):
    map_pos, halo_pos, halo_R = _sample(seed)
    knn = match_maps_knn_local(map_pos, halo_pos, halo_R, BOX, k=128, verbose=False)
    brute = match_maps_brute_local(map_pos, halo_pos, halo_R, BOX)
    np.testing.assert_array_equal(knn, brute)


def test_the_dR_metric_is_not_just_nearest_neighbour():
    """Guard against a matcher that silently degrades to plain nearest-halo."""
    map_pos, halo_pos, halo_R = _sample(0)
    chosen = match_maps_brute_local(map_pos, halo_pos, halo_R, BOX)
    nearest = np.argmin(
        np.stack([periodic_distance(map_pos, h, BOX) for h in halo_pos], axis=1), axis=1
    )
    assert np.mean(chosen != nearest) > 0.1


def test_small_k_can_disagree_which_is_why_the_default_is_128():
    map_pos, halo_pos, halo_R = _sample(3)
    brute = match_maps_brute_local(map_pos, halo_pos, halo_R, BOX)
    k1 = match_maps_knn_local(map_pos, halo_pos, halo_R, BOX, k=1, verbose=False)
    k128 = match_maps_knn_local(map_pos, halo_pos, halo_R, BOX, k=128, verbose=False)
    assert np.any(k1 != brute)
    np.testing.assert_array_equal(k128, brute)


def test_matching_respects_the_periodic_boundary():
    # A MAP just inside one face; the only halo sits just inside the opposite
    # face. Without the wrap they are a box apart and would not be matched.
    map_pos = np.array([[49.0, 0.0, 0.0]])
    halo_pos = np.array([[-49.0, 0.0, 0.0]])
    halo_R = np.array([5.0])
    local = match_maps_knn_local(map_pos, halo_pos, halo_R, BOX, k=1, verbose=False)
    assert local[0] == 0
    assert periodic_distance(map_pos[0], halo_pos[0], BOX) < halo_R[0]


def test_dispatcher_agrees_with_the_direct_calls():
    map_pos, halo_pos, halo_R = _sample(4, n_map=300, n_halo=80)
    a = match_maps(map_pos, halo_pos, halo_R, BOX, method="knn", verbose=False)
    b = match_maps(map_pos, halo_pos, halo_R, BOX, method="brute")
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="unknown match method"):
        match_maps(map_pos, halo_pos, halo_R, BOX, method="nope")  # type: ignore[arg-type]


def test_knn_and_brute_still_available_alongside_ball_default():
    map_pos, halo_pos, halo_R = _sample(7, n_map=50, n_halo=20)
    halo_N = np.full(len(halo_R), 100.0)
    ball = match_maps(
        map_pos, halo_pos, halo_R, BOX, halo_mass=halo_N, verbose=False
    )
    knn = match_maps(map_pos, halo_pos, halo_R, BOX, method="knn", verbose=False)
    assert ball.shape == knn.shape
    assert (ball >= -1).all()
    assert (knn >= 0).all()


def test_legacy_signature_returns_global_ids_and_local_rows():
    map_pos, halo_pos, halo_R = _sample(5, n_map=200, n_halo=40)
    halo_idx = np.arange(1000, 1040)
    host, local = match_maps_knn(map_pos, halo_pos, halo_R, halo_idx, BOX, k=64)
    np.testing.assert_array_equal(host, halo_idx[local])
    np.testing.assert_array_equal(
        local, match_maps_brute_local(map_pos, halo_pos, halo_R, BOX)
    )


def test_non_positive_radii_are_rejected_rather_than_silently_matched():
    map_pos, halo_pos, halo_R = _sample(6, n_map=10, n_halo=5)
    halo_R[2] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        match_maps_knn_local(map_pos, halo_pos, halo_R, BOX, verbose=False)
