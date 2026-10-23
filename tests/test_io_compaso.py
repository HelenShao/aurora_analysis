"""CompaSO slab overlap. No halo files required."""

from __future__ import annotations

import numpy as np
import pytest

from aurora_analysis.io.compaso import (
    compaso_slab_edges,
    default_compaso_root,
    nominal_rank_x_domain,
    overlapping_compaso_slab_indices,
    rank_x_domain_from_particles,
)


def test_import_has_no_default_sim_root():
    import aurora_analysis.io.compaso as c

    assert "DEFAULT_SIM_ROOT" not in c.__all__
    with pytest.raises(AttributeError, match="DEFAULT_SIM_ROOT"):
        getattr(c, "DEFAULT_SIM_ROOT")
    assert default_compaso_root() == c.DEFAULT_COMPASO_ROOT


def test_slab_edges_span_the_centered_box():
    edges = compaso_slab_edges(1000.0, 16)
    assert len(edges) == 17
    assert edges[0] == pytest.approx(-500.0)
    assert edges[-1] == pytest.approx(500.0)
    np.testing.assert_allclose(np.diff(edges), 62.5)


def test_full_box_overlaps_every_slab():
    idxs = overlapping_compaso_slab_indices(-500.0, 500.0, box=1000.0, delta=0.0)
    assert idxs == list(range(16))


def test_rank_domain_picks_neighbouring_slabs_with_delta():
    x0, x1 = nominal_rank_x_domain(1000.0, node_rank_x=16, node_size_x=32)
    assert x0 == pytest.approx(0.0)
    assert x1 == pytest.approx(31.25)
    idxs = overlapping_compaso_slab_indices(x0, x1, box=1000.0, delta=5.0)
    # x in [0, 31.25] ± 5 → [-5, 36.25]; slabs of width 62.5 starting at -500.
    assert idxs  # at least the slab covering x=0
    edges = compaso_slab_edges(1000.0, 16)
    for i in idxs:
        assert edges[i] < 36.25 and edges[i + 1] > -5.0


def test_empty_particle_domain_falls_back_to_header():
    with pytest.raises(ValueError, match="need particles or header"):
        rank_x_domain_from_particles(np.empty((0, 3)))
    x0, x1 = rank_x_domain_from_particles(
        np.empty((0, 3)),
        box=1000.0,
        header={"NodeRankX": 0, "NodeSizeX": 32},
    )
    assert (x0, x1) == nominal_rank_x_domain(1000.0, 0, 32)


def test_particle_bbox_wins_over_header():
    pos = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert rank_x_domain_from_particles(pos, box=1000.0, header={"NodeRankX": 0}) == (
        -10.0,
        10.0,
    )
