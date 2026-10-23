"""Subsample path helpers and coordinate masks. No ASDF required."""

from __future__ import annotations

import numpy as np
import pytest

from aurora_analysis.io.subsample import (
    center_pos_to_box,
    column_dir_names,
    normalize_redshift,
    parse_column_index,
    pid_keys,
    rank_x_label_from_asdf,
    subbox_mask,
    subsample_asdf_stem,
    z_slice_mask,
)


def test_normalize_redshift_is_three_decimals():
    assert normalize_redshift(0.8) == "0.800"
    assert normalize_redshift("3") == "3.000"
    assert normalize_redshift(0.25) == "0.250"


def test_parse_column_index_accepts_col_and_column_spellings():
    assert parse_column_index(3) == 3
    assert parse_column_index("001") == 1
    assert parse_column_index("col007") == 7
    assert parse_column_index("column0007") == 7


def test_column_dir_names_try_four_then_three_digits():
    assert column_dir_names(7) == ("column0007", "col007")


def test_subsample_asdf_stem_embeds_sampler_and_z():
    assert subsample_asdf_stem(0.8, sampler="subP8") == "particles_subP8_z0.800_"


def test_rank_label_parses_legacy_and_jun19_basenames():
    assert (
        rank_x_label_from_asdf("particles_subP8_z0.800_0007.asdf", "0.800") == "0007"
    )
    assert (
        rank_x_label_from_asdf("particles_subP8_z0.800_000_003.asdf", "0.800")
        == "0003"
    )
    with pytest.raises(ValueError, match="Cannot parse"):
        rank_x_label_from_asdf("particles_subP8.asdf", "0.800")


def test_pid_keys_are_unique_and_stable():
    pid = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 3]], dtype=np.int16)
    keys = pid_keys(pid)
    assert keys[0] == keys[2]
    assert keys[0] != keys[1]


def test_z_slice_and_subbox_are_half_open_on_the_hi_edge():
    pos = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    m = z_slice_mask(pos, -1.0, 1.0)
    np.testing.assert_array_equal(m, [True, True, False])
    m_inc = z_slice_mask(pos, -1.0, 1.0, inclusive_max=True)
    np.testing.assert_array_equal(m_inc, [True, True, True])

    box_pos = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-0.1, 0.0, 0.0]]
    )
    sm = subbox_mask(box_pos, 0.0, 5.0, -1.0, 1.0, -1.0, 1.0)
    np.testing.assert_array_equal(sm, [True, False, False])


def test_center_pos_to_box_lands_in_zero_l():
    lbox = 100.0
    pos = np.array([[-50.0, 0.0, 49.9], [49.9, -49.9, 0.0]])
    out = center_pos_to_box(pos, lbox)
    assert np.all(out >= 0.0) and np.all(out < lbox)
    np.testing.assert_allclose(out[0, 0], 0.0, atol=1e-12)
