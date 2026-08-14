import numpy as np
import pytest

from aurora_analysis.match.geometry import (
    minimum_image_delta,
    periodic_distance,
    periodic_distance_sq,
    tile_periodic,
    wrap_centered,
    wrap_corner,
)

BOX = 100.0


def test_delta_is_antisymmetric_and_bounded():
    rng = np.random.default_rng(0)
    a = rng.uniform(-BOX / 2, BOX / 2, size=(500, 3))
    b = rng.uniform(-BOX / 2, BOX / 2, size=(500, 3))
    d = minimum_image_delta(a, b, BOX)
    assert np.all(np.abs(d) <= BOX / 2 + 1e-12)
    np.testing.assert_allclose(d, -minimum_image_delta(b, a, BOX))


def test_delta_wraps_across_the_boundary():
    a = np.array([49.0, 0.0, 0.0])
    b = np.array([-49.0, 0.0, 0.0])
    # Naively 98 apart; across the periodic edge they are 2 apart.
    assert periodic_distance(a, b, BOX) == pytest.approx(2.0)


def test_distance_matches_sqrt_of_squared():
    rng = np.random.default_rng(1)
    a = rng.uniform(-BOX / 2, BOX / 2, size=(200, 3))
    b = rng.uniform(-BOX / 2, BOX / 2, size=(200, 3))
    np.testing.assert_allclose(
        periodic_distance(a, b, BOX), np.sqrt(periodic_distance_sq(a, b, BOX))
    )


def test_distance_is_invariant_to_a_global_shift():
    rng = np.random.default_rng(2)
    a = rng.uniform(-BOX / 2, BOX / 2, size=(200, 3))
    b = rng.uniform(-BOX / 2, BOX / 2, size=(200, 3))
    shift = np.array([37.5, -11.25, 62.0])
    np.testing.assert_allclose(
        periodic_distance(a, b, BOX), periodic_distance(a + shift, b + shift, BOX)
    )


def test_delta_broadcasts_against_a_single_centre():
    rng = np.random.default_rng(3)
    pos = rng.uniform(-BOX / 2, BOX / 2, size=(50, 3))
    centre = np.array([10.0, -20.0, 30.0])
    assert minimum_image_delta(pos, centre, BOX).shape == (50, 3)
    assert periodic_distance(pos, centre, BOX).shape == (50,)


def test_wrap_conventions_land_in_their_intervals():
    rng = np.random.default_rng(4)
    p = rng.uniform(-3 * BOX, 3 * BOX, size=(1000, 3))
    c = wrap_centered(p, BOX)
    assert np.all(c >= -BOX / 2) and np.all(c < BOX / 2)
    k = wrap_corner(p, BOX)
    assert np.all(k >= 0.0) and np.all(k < BOX)
    # The two conventions differ by whole boxes only.
    np.testing.assert_allclose(np.round((k - c) / BOX), (k - c) / BOX, atol=1e-9)


def test_tiling_produces_27_images_mapped_back_to_their_source():
    rng = np.random.default_rng(5)
    pos = rng.uniform(-BOX / 2, BOX / 2, size=(7, 3))
    tiled, local = tile_periodic(pos, BOX)
    assert tiled.shape == (27 * 7, 3)
    assert np.array_equal(np.bincount(local), np.full(7, 27))
    # Every image sits on the original modulo a whole box in each axis.
    offsets = (tiled - pos[local]) / BOX
    np.testing.assert_allclose(offsets, np.round(offsets), atol=1e-9)
