import numpy as np
import pytest

from aurora_analysis.env import (
    inv_kde_for_radius,
    kde_mean_velocity,
    kernel_volume_integral,
    kernel_weights,
    map_env_from_neighbors,
    normalized_kernel_weights,
)

BOX = 100.0


@pytest.mark.parametrize("kind", ["map", "hollow", "dm", "filled"])
def test_weights_vanish_outside_the_kernel_radius(kind):
    r = np.array([0.0, 0.5, 0.999, 1.0, 1.5, 10.0])
    w = kernel_weights(r, 1.0, kind=kind)
    assert np.all(w[r >= 1.0] == 0.0)
    assert np.all(w[r < 1.0] >= 0.0)


def test_the_hollow_kernel_is_hollow_and_the_filled_one_is_not():
    assert kernel_weights([0.0], 1.0, kind="hollow")[0] == 0.0
    assert kernel_weights([0.0], 1.0, kind="filled")[0] == 1.0
    # The hollow kernel peaks at s = 1/sqrt(2).
    s = np.linspace(0, 1, 2001)
    assert s[np.argmax(kernel_weights(s, 1.0, kind="hollow"))] == pytest.approx(
        1 / np.sqrt(2), abs=1e-3
    )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("hollow", 8 * np.pi / 35), ("filled", 8 * np.pi / 15)],
)
def test_analytic_volume_integral_matches_numerical_quadrature(kind, expected):
    b = 1.0
    r = np.linspace(0, b, 200_001)
    numeric = np.trapezoid(kernel_weights(r, b, kind=kind) * 4 * np.pi * r**2, r)
    assert kernel_volume_integral(b, kind=kind) == pytest.approx(expected)
    assert numeric == pytest.approx(expected, rel=1e-5)


def test_volume_integral_scales_as_b_cubed():
    a = kernel_volume_integral(1.0, kind="map")
    assert kernel_volume_integral(2.0, kind="map") == pytest.approx(8 * a)


def test_inv_kde_rescaling_is_consistent_with_the_integral():
    b_ref, b = 0.25, 0.18
    inv_ref = 1.0 / kernel_volume_integral(b_ref, kind="map")
    assert inv_kde_for_radius(inv_ref, b_ref, b) == pytest.approx(
        1.0 / kernel_volume_integral(b, kind="map")
    )


def test_normalized_weights_integrate_to_one():
    b = 0.3
    r = np.linspace(0, b, 100_001)
    w = normalized_kernel_weights(r, b, kind="map")
    assert np.trapezoid(w * 4 * np.pi * r**2, r) == pytest.approx(1.0, rel=1e-5)


def test_uniform_field_recovers_its_number_density():
    """A uniform Poisson field must come back at the density it was drawn at."""
    rng = np.random.default_rng(0)
    n_dens = 500.0
    b = 0.5
    half = 2.0
    n_part = int(n_dens * (2 * half) ** 3)
    pos = rng.uniform(-half, half, size=(n_part, 3))
    vel = np.zeros_like(pos)
    env = map_env_from_neighbors(
        np.zeros(3),
        np.zeros(3),
        pos,
        vel,
        b=b,
        lbox=BOX,
        inv_map_kde=1.0 / kernel_volume_integral(b, kind="map"),
        kernel="map",
    )
    assert env.density == pytest.approx(n_dens, rel=0.15)


def test_dispersion_recovers_an_isotropic_gaussian():
    rng = np.random.default_rng(1)
    sigma_1d = 3.0
    pos = rng.uniform(-0.4, 0.4, size=(40_000, 3))
    vel = rng.normal(0.0, sigma_1d, size=(40_000, 3))
    env = map_env_from_neighbors(
        np.zeros(3), np.zeros(3), pos, vel, b=1.0, lbox=BOX, inv_map_kde=1.0
    )
    # The reported dispersion is the 3-D quantity, sqrt(3) * sigma_1d.
    assert env.velocity_dispersion == pytest.approx(np.sqrt(3) * sigma_1d, rel=0.05)


def test_relative_velocity_is_measured_against_the_weighted_mean():
    pos = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    vel = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    v_map = np.array([13.0, 4.0, 0.0])
    env = map_env_from_neighbors(
        np.zeros(3), v_map, pos, vel, b=1.0, lbox=BOX, inv_map_kde=1.0
    )
    assert env.relative_velocity == pytest.approx(5.0)  # (3, 4, 0)
    assert env.velocity_dispersion == pytest.approx(0.0, abs=1e-9)


def test_empty_neighbourhood_reports_nan_not_zero():
    env = map_env_from_neighbors(
        np.zeros(3),
        np.zeros(3),
        np.array([[50.0, 50.0, 50.0]]),
        np.zeros((1, 3)),
        b=0.1,
        lbox=BOX,
        inv_map_kde=1.0,
    )
    assert np.isnan(env.density)
    assert env.weight_sum == 0.0


def test_env_unpacks_as_the_legacy_four_tuple():
    pos = np.array([[0.2, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0]])
    rho, sig, vrel, d = map_env_from_neighbors(
        np.zeros(3), np.zeros(3), pos, vel, b=1.0, lbox=BOX, inv_map_kde=2.0
    )
    assert np.isfinite([rho, sig, vrel, d]).all()


def test_mean_velocity_is_the_weighted_average():
    pos = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    vel = np.array([[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    v = kde_mean_velocity(np.zeros(3), pos, vel, b=1.0, lbox=BOX, kernel="filled")
    np.testing.assert_allclose(v, [4.0, 0.0, 0.0])


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError, match="unknown kernel"):
        kernel_weights([0.1], 1.0, kind="gaussian")
