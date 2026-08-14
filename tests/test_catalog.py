"""Match-catalog schema: write, read, and refuse to lie about the aperture."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from aurora_analysis.match.catalog import (
    CURRENT_SCHEMA_VERSION,
    MatchProvenance,
    SchemaError,
    load_match_catalog,
    save_match_catalog,
    validate_arrays,
)


def _arrays(n=8, *, r_alias: str | None = "host_R"):
    rng = np.random.default_rng(0)
    matched = np.zeros(n, dtype=bool)
    matched[:5] = True
    is_cen = np.zeros(n, dtype=bool)
    is_cen[:2] = True
    is_sat = np.zeros(n, dtype=bool)
    is_sat[2:5] = True
    arrays = {
        "map_pos": rng.normal(size=(n, 3)),
        "map_vel": rng.normal(size=(n, 3)),
        "map_mult": rng.uniform(20, 200, size=n),
        "matched": matched,
        "is_central": is_cen,
        "is_satellite": is_sat,
        "d": rng.uniform(0, 1, size=n),
        "d_over_R": rng.uniform(0, 1.2, size=n),
        "host_x_L2com": rng.normal(size=(n, 3)),
        "host_v_L2com": rng.normal(size=(n, 3)),
        "host_sigmav3d_L2com": rng.uniform(100, 400, size=n),
        "host_N": rng.uniform(50, 5000, size=n),
    }
    R = rng.uniform(0.2, 1.5, size=n)
    if r_alias == "host_R":
        arrays["host_R"] = R
    elif r_alias == "host_r100_L2com":
        arrays["host_r100_L2com"] = R
    elif r_alias == "both_same":
        arrays["host_R"] = R
        arrays["host_r100_L2com"] = R.copy()
    elif r_alias == "both_differ":
        arrays["host_R"] = R
        arrays["host_r100_L2com"] = R * 0.98  # r98 vs r100
    return arrays


def _prov(**kwargs) -> MatchProvenance:
    defaults = dict(
        source="subsample",
        halo_R_field="r100_L2com",
        match_cut="d < r100_L2com",
        matcher="knn",
        knn_k=128,
        box=1000.0,
        z="0.800",
        sim_root="/sim",
        halo_root="/compaso",
        maplog_known_duplicates=False,
    )
    defaults.update(kwargs)
    return MatchProvenance(**defaults)


def test_roundtrip_preserves_arrays_and_provenance(tmp_path):
    arrays = _arrays()
    path, meta = save_match_catalog(arrays, tmp_path / "match_z0.800.npz", _prov())
    cat = load_match_catalog(path, strict=True)
    np.testing.assert_array_equal(cat["matched"], arrays["matched"])
    np.testing.assert_allclose(cat["host_R"], arrays["host_R"])
    assert cat.provenance.halo_R_field == "r100_L2com"
    assert cat.provenance.schema_version == CURRENT_SCHEMA_VERSION
    assert cat.n_matched == 5
    assert meta.is_file()


def test_writer_adds_r100_alias_only_when_aperture_is_r100(tmp_path):
    arrays = _arrays()
    path, _ = save_match_catalog(arrays, tmp_path / "a.npz", _prov())
    with np.load(path) as raw:
        assert "host_r100_L2com" in raw.files
        np.testing.assert_allclose(raw["host_r100_L2com"], raw["host_R"])

    path98, _ = save_match_catalog(
        arrays, tmp_path / "b.npz", _prov(halo_R_field="r98_L2com", match_cut="d < r98_L2com")
    )
    with np.load(path98) as raw:
        assert "host_r100_L2com" not in raw.files
        assert "host_R" in raw.files


def test_legacy_dict_does_not_invent_r100_for_an_r98_catalog(tmp_path):
    path, _ = save_match_catalog(
        _arrays(),
        tmp_path / "match_maplog_z0.800_fullbox.npz",
        _prov(source="maplog_timeslice", halo_R_field="r98_L2com", maplog_known_duplicates=True),
    )
    cat = load_match_catalog(path, strict=True)
    legacy = cat.as_legacy_dict()
    assert "host_r100_L2com" not in legacy
    assert "host_R" in legacy


def test_legacy_dict_does_alias_when_aperture_really_is_r100(tmp_path):
    path, _ = save_match_catalog(_arrays(), tmp_path / "m.npz", _prov())
    legacy = load_match_catalog(path, strict=True).as_legacy_dict()
    np.testing.assert_allclose(legacy["host_r100_L2com"], legacy["host_R"])


def test_reader_copies_host_r100_into_host_R_and_warns(tmp_path):
    arrays = _arrays(r_alias="host_r100_L2com")
    np.savez(tmp_path / "old.npz", **arrays)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cat = load_match_catalog(tmp_path / "old.npz")
    assert "host_R" in cat
    np.testing.assert_allclose(cat["host_R"], arrays["host_r100_L2com"])
    assert any("host_r100_L2com" in str(w.message) for w in caught)


def test_differing_host_R_and_host_r100_keeps_host_R_and_warns(tmp_path):
    arrays = _arrays(r_alias="both_differ")
    np.savez(tmp_path / "mixed.npz", **arrays)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cat = load_match_catalog(tmp_path / "mixed.npz")
    np.testing.assert_allclose(cat["host_R"], arrays["host_R"])
    assert not np.allclose(cat["host_R"], arrays["host_r100_L2com"])
    assert any("differ" in str(w.message) for w in caught)


def test_filename_infers_maplog_duplicates_when_sidecar_is_missing(tmp_path):
    arrays = _arrays()
    np.savez(tmp_path / "match_maplog_z0.800_fullbox.npz", **arrays)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cat = load_match_catalog(tmp_path / "match_maplog_z0.800_fullbox.npz")
    assert cat.provenance.source == "maplog_timeslice"
    assert cat.provenance.maplog_known_duplicates is True
    assert cat.provenance.z == "0.800"
    assert cat.provenance.inferred is True


def test_sidecar_without_schema_version_is_accepted_and_marked_inferred(tmp_path):
    arrays = _arrays()
    np.savez(tmp_path / "match_z0.300_fullbox.npz", **arrays)
    (tmp_path / "match_z0.300_fullbox.json").write_text(
        json.dumps(
            {
                "box": 1000.0,
                "z": "0.300",
                "halo_R_field": "r100_L2com",
                "sim_root": "/sim",
                "compaso_root": "/compaso",
                "source": "subsample",
                "n_matched": 5,
            }
        )
    )
    cat = load_match_catalog(tmp_path / "match_z0.300_fullbox.npz", strict=True)
    assert cat.provenance.schema_version == 0
    assert cat.provenance.halo_root == "/compaso"
    assert cat.provenance.inferred is True


def test_meta_directory_convention_is_discovered(tmp_path):
    catdir = tmp_path / "catalogs"
    metadir = tmp_path / "meta"
    catdir.mkdir()
    metadir.mkdir()
    arrays = _arrays()
    np.savez(catdir / "match_z0.800_fullbox.npz", **arrays)
    (metadir / "match_z0.800_fullbox.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "subsample",
                "halo_R_field": "r100_L2com",
                "box": 1000.0,
                "z": "0.800",
                "sim_root": "/sim",
            }
        )
    )
    cat = load_match_catalog(catdir / "match_z0.800_fullbox.npz", strict=True)
    assert cat.meta_path == metadir / "match_z0.800_fullbox.json"
    assert cat.provenance.halo_R_field == "r100_L2com"


def test_strict_rejects_missing_arrays(tmp_path):
    arrays = _arrays()
    del arrays["d_over_R"]
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(SchemaError, match="missing required"):
        load_match_catalog(tmp_path / "bad.npz", strict=True)


def test_strict_rejects_incomplete_provenance(tmp_path):
    np.savez(tmp_path / "bare.npz", **_arrays())
    with pytest.raises(SchemaError, match="missing provenance"):
        load_match_catalog(tmp_path / "bare.npz", strict=True)


def test_writer_refuses_invalid_cen_sat_masks(tmp_path):
    arrays = _arrays()
    arrays["is_central"][0] = True
    arrays["is_satellite"][0] = True
    with pytest.raises(SchemaError, match="overlap"):
        save_match_catalog(arrays, tmp_path / "x.npz", _prov())


def test_validate_flags_unmatched_centrals():
    arrays = _arrays()
    arrays["is_central"][-1] = True  # last MAP is unmatched
    problems = validate_arrays(arrays)
    assert any("unmatched" in p for p in problems)


def test_writer_refuses_incomplete_provenance(tmp_path):
    with pytest.raises(SchemaError, match="provenance incomplete"):
        save_match_catalog(_arrays(), tmp_path / "x.npz", MatchProvenance())
