"""MapNode control packing and step ↔ z. No files required."""

from __future__ import annotations

import numpy as np
import pytest

from aurora_analysis.epochs import (
    NODE_FORMATION,
    NODE_MERGER,
    NODE_TIMESLICE,
    build_maplog_epochs,
    encode_control,
    epoch_step,
    map_epoch_for_redshift,
    node_type,
    parse_abacus_kv_file,
    pick_z_cat_near_zeff,
    resolve_timeslice_epoch,
    step_to_z_from_header_array,
    timeslice_control,
)


def test_control_round_trips_step_and_type():
    for ntype in (NODE_FORMATION, NODE_MERGER, NODE_TIMESLICE):
        ctrl = encode_control(892, ntype)
        assert epoch_step(ctrl) == 892
        assert int(node_type(ctrl)) == ntype
    assert timeslice_control(892) == 4 * 892 + NODE_TIMESLICE
    assert timeslice_control(892) == encode_control(892, NODE_TIMESLICE)


def test_encode_control_broadcasts_over_an_array_of_steps():
    steps = np.array([10, 20, 30])
    ctrl = encode_control(steps, NODE_TIMESLICE)
    np.testing.assert_array_equal(epoch_step(ctrl), steps)
    np.testing.assert_array_equal(node_type(ctrl), NODE_TIMESLICE)


def test_unknown_node_type_is_rejected():
    with pytest.raises(ValueError, match="unknown node type"):
        encode_control(1, 7)


def test_step_to_z_header_index_is_the_write_state_step():
    # No -1 offset: index i is FullStepNumber i.
    table = step_to_z_from_header_array([3.0, 1.1, 0.3])
    assert table[0] == 3.0
    assert table[2] == pytest.approx(0.3)
    assert 892 not in table


def test_resolve_timeslice_epoch_picks_the_nearest_redshift():
    step_to_z = {100: 1.0, 200: 0.3, 300: 0.0}
    assert resolve_timeslice_epoch(step_to_z, 0.25) == (200, 0.3)
    with pytest.raises(ValueError, match="empty"):
        resolve_timeslice_epoch({}, 0.3)


def test_pick_z_cat_prefers_the_epoch_above_on_an_exact_tie():
    control = np.array(
        [encode_control(10, NODE_TIMESLICE), encode_control(20, NODE_TIMESLICE)]
    )
    maplog = {"control": control, "timestep": np.array([10, 20])}
    step_to_z = {10: 1.0, 20: 0.5}
    z_cat, step, zs = pick_z_cat_near_zeff(maplog, 0.75, step_to_z)
    assert z_cat == pytest.approx(1.0)
    assert step == 10
    np.testing.assert_allclose(sorted(zs), [0.5, 1.0])


def test_build_maplog_epochs_requires_a_matching_step_to_z_entry():
    maplog = {
        "control": np.array([timeslice_control(5)]),
        "timestep": np.array([5]),
    }
    steps, zs = build_maplog_epochs(maplog, {5: 0.8})
    np.testing.assert_array_equal(steps, [5])
    np.testing.assert_allclose(zs, [0.8])
    with pytest.raises(ValueError, match="no Step_to_Z"):
        build_maplog_epochs(maplog, {4: 0.8})


def test_map_epoch_for_redshift_uses_only_timeslice_rows():
    control = np.array(
        [
            encode_control(1, NODE_FORMATION),
            encode_control(10, NODE_TIMESLICE),
            encode_control(20, NODE_TIMESLICE),
        ]
    )
    maplog = {"control": control, "timestep": epoch_step(control)}
    step_to_z = {1: 9.0, 10: 1.0, 20: 0.1}
    assert map_epoch_for_redshift(maplog, 0.2, step_to_z) == (20, pytest.approx(0.1))


def test_parse_abacus_kv_file_reads_state_scalars(tmp_path):
    p = tmp_path / "step892.state"
    p.write_text("# comment\nFullStepNumber = 892\nRedshift = 0.3\nBoxSize = 1000.\n")
    st = parse_abacus_kv_file(p)
    assert st["FullStepNumber"] == 892
    assert st["Redshift"] == pytest.approx(0.3)
    assert st["BoxSize"] == pytest.approx(1000.0)
