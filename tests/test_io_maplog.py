"""Maplog segment scan and TimeSlice control select. No ASDF required."""

from __future__ import annotations

import numpy as np
import pytest

from aurora_analysis.epochs import encode_control, timeslice_control
from aurora_analysis.io.maplog import (
    MAPLOG_LAYOUT,
    _combine_pid_columns,
    maplog_as_u4_rows,
    segment_ranges_from_starts,
    select_timeslice_indices_parallel,
    slice_maplog,
)


def test_layout_tag_matches_abacusnbody():
    assert MAPLOG_LAYOUT == "mapnode_24b"


def test_combine_pid_columns_inverts_auxstruct_set_pid():
    """C++ ``auxstruct::pid()`` is ``x | y<<16 | z<<32`` (particlestruct.cpp)."""
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0xFFFF, 3, 7]])
    expected = (
        xyz[:, 0].astype(np.int64)
        | (xyz[:, 1].astype(np.int64) << 16)
        | (xyz[:, 2].astype(np.int64) << 32)
    )
    np.testing.assert_array_equal(_combine_pid_columns(xyz), expected)


def test_combine_pid_columns_is_injective_over_the_lattice():
    """Distinct (x, y, z) must give distinct keys — a MAP PID is an identity.

    The regression this guards: with ``_PID_SHIFT1 = 2**31`` the j and k
    contributions overlapped (``2**32 == 2 * 2**31``), so e.g. (0, 2, 0) and
    (0, 0, 1) collided and PID-keyed groupings fused unrelated particles.
    """
    n = 24
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1)
    xyz = g.reshape(-1, 3)
    keys = _combine_pid_columns(xyz)
    assert len(np.unique(keys)) == len(xyz)

    collided = np.array([[0, 2, 0], [0, 0, 1]])
    assert _combine_pid_columns(collided)[0] != _combine_pid_columns(collided)[1]


def test_maplog_as_u4_rows_views_flat_and_shaped_input():
    flat = np.arange(18, dtype=np.uint32)
    rows = maplog_as_u4_rows(flat)
    assert rows.shape == (3, 6)
    np.testing.assert_array_equal(rows[1], np.arange(6, 12, dtype=np.uint32))

    already = rows.copy()
    np.testing.assert_array_equal(maplog_as_u4_rows(already), already)

    with pytest.raises(ValueError, match="multiple of 6"):
        maplog_as_u4_rows(np.arange(7, dtype=np.uint32))


def test_segment_ranges_cover_the_file_and_skip_empty():
    assert segment_ranges_from_starts(None, 10) == [(0, 10)]
    assert segment_ranges_from_starts(np.array([], dtype=np.int64), 10) == [(0, 10)]
    assert segment_ranges_from_starts(np.array([0, 4, 8]), 10) == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    # First start not 0 → prepend 0.
    assert segment_ranges_from_starts(np.array([3, 7]), 10) == [
        (0, 3),
        (3, 7),
        (7, 10),
    ]
    assert segment_ranges_from_starts(np.array([0]), 0) == []


def test_timeslice_select_is_control_equals_4_step_plus_3():
    raw = np.zeros((12, 6), dtype=np.uint32)
    raw[1, 4] = timeslice_control(100)
    raw[4, 4] = timeslice_control(100)
    raw[5, 4] = encode_control(100, ntype=0)  # formation at same step
    raw[8, 4] = timeslice_control(99)
    raw[11, 4] = timeslice_control(100)

    idx = select_timeslice_indices_parallel(
        raw, timeslice_control(100), n_workers=1
    )
    np.testing.assert_array_equal(idx, [1, 4, 11])

    starts = np.array([0, 6], dtype=np.int64)
    idx_seg = select_timeslice_indices_parallel(
        raw, timeslice_control(100), segment_starts=starts, n_workers=2
    )
    np.testing.assert_array_equal(idx_seg, [1, 4, 11])


def test_empty_raw_select_returns_empty():
    raw = np.zeros((0, 6), dtype=np.uint32)
    idx = select_timeslice_indices_parallel(raw, timeslice_control(1), n_workers=1)
    assert idx.dtype == np.int64
    assert len(idx) == 0


def test_slice_maplog_preserves_keys_and_row_range():
    ml = {
        "pos": np.arange(12, dtype=np.float32).reshape(4, 3),
        "control": np.arange(4, dtype=np.int32),
    }
    seg = slice_maplog(ml, 1, 3)
    np.testing.assert_array_equal(seg["pos"], ml["pos"][1:3])
    np.testing.assert_array_equal(seg["control"], [1, 2])
