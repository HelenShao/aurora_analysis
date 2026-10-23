"""ASDF helpers: UAN detection and YAML header sniff. No Flare files required."""

from __future__ import annotations

import warnings

import pytest

from aurora_analysis.io.asdf import (
    DEFAULT_LAZY_LOAD,
    is_likely_login_node,
    read_header,
    reset_login_warning,
    warn_if_login_node,
)


def test_eager_open_is_the_default():
    assert DEFAULT_LAZY_LOAD is False


def test_login_node_detection_honours_overrides(monkeypatch):
    monkeypatch.setenv("AURORA_ANALYSIS_SKIP_UAN_WARN", "1")
    monkeypatch.delenv("AURORA_ANALYSIS_FORCE_UAN_WARN", raising=False)
    assert is_likely_login_node() is False

    monkeypatch.delenv("AURORA_ANALYSIS_SKIP_UAN_WARN", raising=False)
    monkeypatch.setenv("AURORA_ANALYSIS_FORCE_UAN_WARN", "1")
    assert is_likely_login_node() is True


def test_pbs_allocation_is_treated_as_compute(monkeypatch):
    monkeypatch.delenv("AURORA_ANALYSIS_SKIP_UAN_WARN", raising=False)
    monkeypatch.delenv("AURORA_ANALYSIS_FORCE_UAN_WARN", raising=False)
    monkeypatch.setenv("PBS_JOBID", "12345.debug")
    assert is_likely_login_node() is False


def test_uan_warning_fires_once(monkeypatch):
    monkeypatch.setenv("AURORA_ANALYSIS_FORCE_UAN_WARN", "1")
    reset_login_warning()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_login_node("shard.asdf", decompressing=True)
        warn_if_login_node("shard2.asdf", decompressing=True)
    assert len(caught) == 1
    assert "login/UAN" in str(caught[0].message)

    reset_login_warning()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_login_node("shard.asdf", decompressing=False)
    assert caught == []


def test_read_header_sniffs_yaml_without_asdf(tmp_path):
    p = tmp_path / "particles_subP8_z0.800_0000.asdf"
    p.write_bytes(
        b"#ASDF 1.0.0\n"
        b"%YAML 1.1\n"
        b"---\n"
        b"asdf_library: {author: test}\n"
        b"header:\n"
        b"  BoxSize: 1000.0\n"
        b"  NodeRankX: 3\n"
        b"  NodeSizeX: 32\n"
        b"  SimName: HighBase\n"
        b"...\n"
        b"this is not yaml and would be a binary block\n"
    )
    hdr = read_header(p)
    assert hdr["BoxSize"] == pytest.approx(1000.0)
    assert hdr["NodeRankX"] == 3
    assert hdr["NodeSizeX"] == 32
    assert hdr["SimName"] == "HighBase"
