import numpy as np

from aurora_analysis.match.censat import (
    central_index_per_halo,
    label_cen_sat,
    label_cen_sat_loop,
)


def test_matches_the_original_loop_when_multiplicities_are_unique():
    """Pin the vectorised labeller against the implementation it replaces.

    Ties are excluded here because the original broke them with an unstable
    argsort and is therefore not a well-defined reference in that case.
    """
    rng = np.random.default_rng(0)
    n = 5000
    ih = rng.integers(0, 400, size=n)
    mult = rng.permutation(n).astype(float)  # all distinct
    np.testing.assert_array_equal(label_cen_sat(mult, ih), label_cen_sat_loop(mult, ih))


def test_exactly_one_central_per_occupied_halo():
    rng = np.random.default_rng(1)
    ih = rng.integers(0, 50, size=1000)
    mult = rng.random(1000)
    is_cen = label_cen_sat(mult, ih)
    counts = np.bincount(ih[is_cen], minlength=50)
    occupied = np.bincount(ih, minlength=50) > 0
    assert np.array_equal(counts[occupied], np.ones(occupied.sum(), dtype=int))
    assert counts[~occupied].sum() == 0


def test_the_central_is_the_highest_multiplicity_member():
    ih = np.array([0, 0, 0, 1, 1])
    mult = np.array([1.0, 9.0, 3.0, 4.0, 2.0])
    np.testing.assert_array_equal(
        label_cen_sat(mult, ih), np.array([False, True, False, True, False])
    )


def test_ties_break_on_the_lowest_index_deterministically():
    ih = np.array([0, 0, 0])
    mult = np.array([5.0, 5.0, 5.0])
    np.testing.assert_array_equal(
        label_cen_sat(mult, ih), np.array([True, False, False])
    )


def test_result_does_not_depend_on_input_ordering():
    rng = np.random.default_rng(2)
    n = 2000
    ih = rng.integers(0, 100, size=n)
    mult = rng.permutation(n).astype(float)
    base = label_cen_sat(mult, ih)
    perm = rng.permutation(n)
    shuffled = label_cen_sat(mult[perm], ih[perm])
    np.testing.assert_array_equal(base[perm], shuffled)


def test_handles_empty_input():
    assert label_cen_sat(np.array([]), np.array([], dtype=int)).shape == (0,)
    assert central_index_per_halo(np.array([]), np.array([], dtype=int)).shape == (0,)


def test_central_indices_agree_with_the_mask():
    rng = np.random.default_rng(3)
    ih = rng.integers(0, 30, size=500)
    mult = rng.permutation(500).astype(float)
    idx = central_index_per_halo(mult, ih)
    mask = label_cen_sat(mult, ih)
    np.testing.assert_array_equal(np.sort(idx), np.flatnonzero(mask))
