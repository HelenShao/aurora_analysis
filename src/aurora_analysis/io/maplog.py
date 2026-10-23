"""Maplog readers with two documented modes.

``mode="timeslice"`` (the matching / catalog path)
    Resolve the TimeSlice epoch ``ts`` once from ``Step_to_Z``, then scan
    ``control == 4*ts + NODE_TIMESLICE`` over ``MapLogSegmentStarts`` ranges
    and unpack only those rows. This is :func:`load_timeslice_maps`.

``mode="full"`` (the tree-walk path)
    Eager-open the shard and unpack every field of every node. This is
    :func:`decode_maplog_shard`. Use it when ``dominant_branch_walk`` /
    ``tag_maps`` need Formation and Merger rows around a TimeSlice.

Both modes open with :func:`aurora_analysis.io.asdf.open_asdf`
(``lazy_load=False``). The previous ``decode_maplog_shard`` went through
``read_asdf``, which hardcodes ``lazy_load=True`` and is the slow path on
Aurora single-block files.

2D MPI (``NumZRanks > 1``) writes one ``col%03d/`` folder per Z-rank.
:func:`discover_maplog_asdf_files` walks every subdirectory of ``maplogs/``.
Restricting to ``col000`` would miss the rest of the box.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Literal

import numpy as np

from ..epochs import (
    build_maplog_epochs,
    build_step_to_z_from_maplog,
    build_step_to_z_from_state_files,
    encode_control,
    epoch_step,
    map_epoch_for_redshift,
    node_type,
    pick_z_cat_near_zeff,
    resolve_timeslice_epoch,
    step_to_z_from_header_array,
    timeslice_control,
)
from .asdf import open_asdf

__all__ = [
    "BASE_KEYS",
    "BASE_KEYS_WITH_ENV",
    "ENV_KEYS",
    "MAPLOG_LAYOUT",
    "MAPLOG_LOAD_FIELDS",
    "MaplogFileMeta",
    "MaplogSegment",
    "SegmentTask",
    "build_maplog_epochs",
    "build_step_to_z_from_maplog",
    "build_step_to_z_from_state_files",
    "catalog_at_redshift_segmented",
    "catalog_from_segment",
    "decode_maplog_shard",
    "decode_maplog_timeslice_fast",
    "discover_maplog_asdf_files",
    "encode_control",
    "epoch_step",
    "inject_timeslice_pids",
    "iter_maplog_segments",
    "load_maplog_run_metadata",
    "load_maplogs",
    "load_timeslice_maps",
    "load_timeslice_maps_streaming",
    "map_epoch_for_redshift",
    "maplog_as_u4_rows",
    "maplog_col_counts",
    "merge_segment_catalogs",
    "multiprocessing_get_context",
    "node_type",
    "pick_z_cat_near_zeff",
    "process_shard_segments",
    "read_maplog_asdf_header",
    "read_maplog_dir_chunked",
    "reduce_over_segments",
    "resolve_step_to_z",
    "resolve_timeslice_epoch",
    "segment_bounds",
    "segment_ranges_from_starts",
    "select_timeslice_indices_parallel",
    "slice_maplog",
    "step_to_z_from_header_array",
    "timeslice_control",
]

BASE_KEYS = ("pos", "vel", "pid", "mult", "control", "timestep", "length")
ENV_KEYS = ("density", "vel_disp", "rel_vel", "rel_vel_healpix", "mult_sec")
BASE_KEYS_WITH_ENV = BASE_KEYS + ENV_KEYS

#: Fields the matching / Ye path unpacks from TimeSlice rows.
MAPLOG_LOAD_FIELDS = (
    "pos",
    "vel",
    "mult",
    "control",
    "timestep",
    "length",
    "density",
    "vel_disp",
    "rel_vel",
    "rel_vel_healpix",
)

#: On-disk MapNode layout tag. Absent on files written before the multiplicity
#: word was split; a *different* present value is a hard error.
MAPLOG_LAYOUT = "mapnode_24b"

# A Formation node stores the PID as the integer3 lattice index (pid_i, pid_j,
# pid_k) = auxstruct::xyz(), one uint16 each (maplog.cpp CreateFormationNode).
# The canonical scalar PID is auxstruct::pid() = x | y<<16 | z<<32
# (particlestruct.cpp), so these are the shifts that invert set_pid(x, y, z).
# 2**31 here was wrong: 2**32 == 2 * 2**31, so the j and k contributions
# overlapped and the key collapsed to (x, j + 2k) — collisions, not identities.
_PID_SHIFT1 = np.int64(2**16)
_PID_SHIFT2 = np.int64(2**32)
_CONTROL_MASK = 0xFFFF
_MP_SHARD: dict = {}

LoadMode = Literal["timeslice", "full"]


@dataclass
class MaplogFileMeta:
    """Header-only metadata for one maplog ASDF shard (no node decode)."""

    path: Path
    n_nodes: int
    segment_starts: np.ndarray  # int64, monotone, segment_starts[0] == 0
    step_to_z: dict[int, float]
    box_size: float | None = None


@dataclass
class MaplogSegment:
    """One in-memory row-range slice of one loaded maplog shard."""

    maplog: dict
    file_path: Path
    global_offset: int
    segment_index: int
    n_segments_in_file: int = field(default=1)


@dataclass
class SegmentTask:
    """Work item for one segment slice within a loaded shard."""

    i0: int
    i1: int
    segment_index: int
    global_offset: int
    file_path: Path


# MAP env dens is accumulated on the GPU with the *filled* kernel W=1−s²
# (volume weight ∝ 8π/15; Direct/DirectGPUKernels.sycl.hpp), but Abacus still
# divides by the *hollow*-kernel unit density Map_KDEunitdensity ∝ 8π/35
# (Timestep/proepi.cpp).  Stored dens is therefore high by 35/15.  Undo that
# at load time so callers see filled-kernel mean-relative dens without
# touching the on-disk bytes.
#
# Only AuroraHighBase runs have this filled/hollow mismatch in the written
# dens; do **not** rescale any other sim path.
MAP_DENS_FILLED_KERNEL_CORR = 15.0 / 35.0
AURORA_HIGHBASE_DENS_CORR_ROOT = Path(
    "/lus/flare/projects/Abacus/ajohnson/AuroraHighBase"
)


def _path_needs_highbase_dens_corr(path: str | Path | None) -> bool:
    """True iff ``path`` lives under the AuroraHighBase tree (dens rescale)."""
    if path is None:
        return False
    try:
        resolved = Path(path).resolve()
        root = AURORA_HIGHBASE_DENS_CORR_ROOT.resolve()
    except OSError:
        resolved = Path(path)
        root = AURORA_HIGHBASE_DENS_CORR_ROOT
    return root == resolved or root in resolved.parents


def _unpack_maplog(raw: np.ndarray, fields, *, path: str | Path | None = None):
    try:
        from abacusnbody.data.maplog import unpack_maplog
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ImportError(
            "maplog decode needs abacusutils (pip install 'aurora_analysis[abacus]'). "
            "Use the @aurora branch for maplog rel_vel unpack."
        ) from e
    cols = unpack_maplog(raw, fields=list(fields))
    if "density" in cols and _path_needs_highbase_dens_corr(path):
        # Same unpack path as before; only rescale dens for AuroraHighBase.
        cols["density"] = np.asarray(cols["density"]) * MAP_DENS_FILLED_KERNEL_CORR
    return cols


def _tree_tags():
    """``map_tree_tags`` from the aurora_maplogs tree (not yet extracted)."""
    import sys

    cached = sys.modules.get("map_tree_tags")
    if cached is not None:
        return cached
    try:
        import map_tree_tags as mod  # type: ignore

        return mod
    except ImportError:
        from ..config import path
        from ..external._locate import load_module, require_tree

        root = require_tree(
            path("maplogs_root"),
            what="aurora_maplogs",
            env_var="AURORA_MAPLOGS_ROOT",
            expect="map_tree_tags.py",
        )
        return load_module(
            root / "map_tree_tags.py", name="map_tree_tags", deps=root
        )


def discover_maplog_asdf_files(mdir, z=0.2):
    """All ``maplog_z{z:.2f}_*.asdf`` shards under every ``maplogs/*/`` directory.

    2D MPI (``NumZRanks`` > 1) writes one ``col%03d/`` folder per Z-rank, each
    holding the X-rank shards ``maplog_z{z}_{col}_{rank}.asdf``. Walks every
    subdirectory of ``maplogs/`` (not only ``col000``), then falls back to a
    recursive glob.
    """
    mdir = Path(mdir)
    ztag = f"{float(z):.2f}"
    pat = f"maplog_z{ztag}_*.asdf"
    maplogs = mdir / "maplogs"
    files = []
    if maplogs.is_dir():
        for d in sorted(p for p in maplogs.iterdir() if p.is_dir()):
            files.extend(sorted(d.glob(pat)))
    if not files:
        files = [
            Path(f) for f in glob(str(mdir / "maplogs" / "**" / pat), recursive=True)
        ]
    return sorted(set(files))


def maplog_col_counts(files):
    """``{col000: n, …}`` from shard parent directory names."""
    counts: dict[str, int] = {}
    for p in files:
        counts[p.parent.name] = counts.get(p.parent.name, 0) + 1
    return dict(sorted(counts.items()))


def read_maplog_asdf_header(path) -> MaplogFileMeta:
    """Open one maplog ASDF shard and return header metadata only.

    Uses ``lazy_load=True`` so the data block is not decompressed. This is
    the one maplog open that *should* be lazy.
    """
    with open_asdf(path, lazy_load=True, warn=False) as af:
        header = af.tree.get("header", {}) or {}
        n_nodes = int(af.tree["data"]["maplogs"].shape[0])

        segment_starts = header.get("MapLogSegmentStarts")
        if segment_starts is None:
            segment_starts = np.array([0], dtype=np.int64)
        else:
            segment_starts = np.asarray(segment_starts, dtype=np.int64)

        total_nodes = int(header.get("MapLogTotalNodes", n_nodes))
        if total_nodes != n_nodes:
            print(
                f"WARNING: {path} MapLogTotalNodes={total_nodes} != decoded "
                f"n_nodes={n_nodes}; using n_nodes for segment bounds."
            )

        step_to_z_header = header.get("Step_to_Z")
        step_to_z = (
            step_to_z_from_header_array(step_to_z_header)
            if step_to_z_header is not None
            else {}
        )

        box_size = header.get("BoxSize")
        box_size = float(box_size) if box_size is not None else None

        layout = header.get("MapLogLayout")
        if layout is not None and layout != MAPLOG_LAYOUT:
            raise ValueError(
                f"MapLogLayout {layout!r} in {path} does not match "
                f"the only layout this reader supports ({MAPLOG_LAYOUT!r})"
            )

    return MaplogFileMeta(
        path=Path(path),
        n_nodes=n_nodes,
        segment_starts=segment_starts,
        step_to_z=step_to_z,
        box_size=box_size,
    )


def resolve_step_to_z(rdir=None, header=None, maplog=None, mdir=None, z_maplog=0.2):
    """FullStepNumber → redshift conversion table.

    Priority:

    1. ASDF header ``Step_to_Z`` (``header`` arg, or first maplog shard at ``mdir``)
    2. ``log/step*/step*.state`` under ``rdir``
    3. Legacy acj ``maplog['step']`` / ``maplog['redshift']`` arrays

    When ``mdir`` is omitted but ``rdir`` is set, ``mdir`` defaults to ``rdir``
    so maplog ASDF headers are consulted before state files.
    """
    if mdir is None and rdir is not None:
        mdir = rdir

    if header is not None:
        step_to_z = (
            header.step_to_z
            if isinstance(header, MaplogFileMeta)
            else header.get("step_to_z")
        )
        if step_to_z:
            return step_to_z

    if mdir is not None:
        files = discover_maplog_asdf_files(mdir, z=z_maplog)
        if files:
            meta = read_maplog_asdf_header(files[0])
            if meta.step_to_z:
                return meta.step_to_z

    if rdir is not None:
        step_to_z = build_step_to_z_from_state_files(rdir)
        if step_to_z:
            return step_to_z

    if maplog is not None and "step" in maplog and "redshift" in maplog:
        return build_step_to_z_from_maplog(maplog)

    raise ValueError(
        "resolve_step_to_z: no usable source (no header Step_to_Z, no state files, "
        "and no maplog step/redshift arrays provided)"
    )


def load_maplog_run_metadata(mdir, z=0.2) -> list[MaplogFileMeta]:
    """Per-shard header metadata (no node decode) for every ASDF file at ``z``."""
    files = discover_maplog_asdf_files(mdir, z=z)
    if not files:
        raise FileNotFoundError(
            f"no maplog ASDF files under {mdir}/maplogs/ at z={z:.2f}"
        )
    return [read_maplog_asdf_header(f) for f in files]


def segment_ranges_from_starts(
    segment_starts: np.ndarray | None, n_nodes: int
) -> list[tuple[int, int]]:
    """``(i0, i1)`` pairs from ``MapLogSegmentStarts``.

    Missing / empty ``MapLogSegmentStarts`` → one range ``[0, n_nodes)``.
    If the first start is not 0, ``0`` is prepended.
    """
    n_nodes = int(n_nodes)
    if n_nodes <= 0:
        return []
    if segment_starts is None or len(segment_starts) == 0:
        starts = np.array([0], dtype=np.int64)
    else:
        starts = np.asarray(segment_starts, dtype=np.int64)
        if int(starts[0]) != 0:
            starts = np.concatenate((np.array([0], dtype=np.int64), starts))
    ends = np.append(starts[1:], n_nodes)
    return [
        (int(i0), int(i1))
        for i0, i1 in zip(starts, ends, strict=True)
        if int(i1) > int(i0)
    ]


def segment_bounds(meta: MaplogFileMeta):
    """Yield ``(i0, i1)`` pairs from ``MapLogSegmentStarts`` (nominal, as written)."""
    yield from segment_ranges_from_starts(meta.segment_starts, meta.n_nodes)


def slice_maplog(maplog, i0, i1, keys=None):
    """Return a segment-local dict ``maplog[key][i0:i1]``."""
    if keys is None:
        keys = maplog.keys()
    return {key: maplog[key][i0:i1] for key in keys}


def maplog_as_u4_rows(data: np.ndarray) -> np.ndarray:
    """``(N, 6) uint32`` MapNode rows — identical to ``unpack_maplog``'s preamble.

    Copied from ``abacusnbody.data.maplog.unpack_maplog`` (``.view(np.uint32)``,
    then reshape). Do not replace ``.view`` with ``astype`` / ``dtype=`` casts.
    """
    data = np.ascontiguousarray(data).view(np.uint32)
    if data.ndim == 1:
        if data.size % 6 != 0:
            raise ValueError(
                f"flat input size {data.size} is not a multiple of 6 uint32 "
                "(MapNode = 24 bytes = 6 uint32)"
            )
        data = data.reshape(-1, 6)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(
            f"expected (N, 6) uint32 after reinterpretation; got shape {data.shape}"
        )
    return data


def _combine_pid_columns(pid_cols) -> np.ndarray:
    pid_cols = np.asarray(pid_cols)
    return (
        pid_cols[:, 0].astype(np.int64)
        + pid_cols[:, 1].astype(np.int64) * _PID_SHIFT1
        + pid_cols[:, 2].astype(np.int64) * _PID_SHIFT2
    )


def inject_timeslice_pids(maplog):
    """Set TimeSlice MAP PIDs from the dominant-branch Formation node.

    Skips trees whose ``length`` span has no Formation (truncated / corrupt
    logs); prints a one-line count if any were skipped.
    """
    tags = _tree_tags()
    control = maplog["control"]
    pid = maplog["pid"]
    n_skip = 0
    for ctrl in np.unique(control):
        if int(ctrl) % 4 != tags.NODE_TIMESLICE:
            continue
        maps = np.flatnonzero(control == ctrl)
        for imap in maps:
            try:
                form_row = tags.formation_row_dominant(int(imap), maplog)
            except ValueError:
                n_skip += 1
                continue
            pid[imap] = pid[form_row]
    if n_skip:
        print(
            f"WARNING: inject_timeslice_pids skipped {n_skip} TimeSlice "
            f"nodes with no Formation in length span"
        )


_inject_timeslice_pids = inject_timeslice_pids


def decode_maplog_shard(path, keys=BASE_KEYS, inject_pids=True, *, lazy_load=False):
    """Load and unpack one maplog ASDF shard (full decode for that file).

    This is ``mode="full"``: every requested field of every node. Tree tagging
    needs Formation / Merger rows, so matching-time TimeSlice selection is the
    wrong tool here.

    Default ``lazy_load=False`` is the Aurora fast path. The previous
    implementation went through ``read_asdf`` (``lazy_load=True``).

    Set ``inject_pids=False`` if you will slice to a segment first and inject
    (or tag) only on that slice.

    Accepts legacy load key ``vel_rel`` (mapped to ``rel_vel``). When
    ``rel_vel`` is present, also sets ``maplog['vel_rel']`` as an alias.
    """
    keys = list(dict.fromkeys(keys))
    load_keys = [("rel_vel" if k == "vel_rel" else k) for k in keys]
    load_keys = list(dict.fromkeys(load_keys))

    with open_asdf(path, lazy_load=lazy_load) as af:
        header = af.tree.get("header", {}) or {}
        layout = header.get("MapLogLayout")
        if layout is not None and layout != MAPLOG_LAYOUT:
            raise ValueError(
                f"MapLogLayout {layout!r} in {path} does not match "
                f"the only layout this reader supports ({MAPLOG_LAYOUT!r})"
            )
        raw = np.array(maplog_as_u4_rows(af.tree["data"]["maplogs"]), copy=True)

    cols = _unpack_maplog(raw, load_keys, path=path)
    maplog = {key: np.asarray(cols[key]) for key in load_keys}
    if "pid" in maplog and getattr(maplog["pid"], "ndim", 1) == 2:
        maplog["pid"] = _combine_pid_columns(maplog["pid"])
    if "rel_vel" in maplog and "vel_rel" not in maplog:
        maplog["vel_rel"] = maplog["rel_vel"]
    if (
        inject_pids
        and "pid" in maplog
        and "control" in maplog
        and "length" in maplog
    ):
        inject_timeslice_pids(maplog)
    return maplog


def select_timeslice_indices_parallel(
    raw: np.ndarray,
    target_control: int,
    *,
    segment_starts: np.ndarray | None = None,
    n_workers: int | None = None,
) -> np.ndarray:
    """Parallel ``control == target_control`` over ``MapLogSegmentStarts`` ranges.

    Uses threads so chunks share ``raw`` with no pickle/fork overhead; NumPy
    releases the GIL on the bitwise/compare/`nonzero` work. Falls back to a
    single ``[0, N)`` range when segment starts are absent.
    """
    n = int(raw.shape[0])
    if n == 0:
        return np.array([], dtype=np.int64)

    target = int(target_control)
    ranges = segment_ranges_from_starts(segment_starts, n)
    if not ranges:
        return np.array([], dtype=np.int64)

    if n_workers is None:
        n_workers = min(8, max(1, os.cpu_count() or 1))
    n_workers = max(1, min(int(n_workers), len(ranges)))

    def _chunk(i0: int, i1: int) -> np.ndarray:
        ctrl = raw[i0:i1, 4] & _CONTROL_MASK
        return (i0 + np.flatnonzero(ctrl == target)).astype(np.int64)

    if n_workers == 1 or len(ranges) == 1:
        parts = [_chunk(i0, i1) for i0, i1 in ranges]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            parts = list(ex.map(lambda r: _chunk(r[0], r[1]), ranges))
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts, axis=0)


def decode_maplog_timeslice_fast(
    path: str | Path,
    *,
    epoch_step: int,
    epoch_z: float,
    fields: tuple[str, ...] | list[str] = MAPLOG_LOAD_FIELDS,
    n_workers: int | None = None,
    return_row_indices: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Decode one shard; return TimeSlice rows at pre-resolved ``epoch_step``.

    If ``return_row_indices``, also include ``imap`` = absolute row indices in
    the shard node array (needed for ``dominant_branch_walk`` / tree tags).
    """
    path = Path(path)
    fields = list(dict.fromkeys(fields))
    target_control = timeslice_control(epoch_step)

    t0 = time.perf_counter()
    with open_asdf(path, lazy_load=False) as af:
        header = af.tree.get("header", {}) or {}
        ss = header.get("MapLogSegmentStarts")
        segment_starts = None if ss is None else np.asarray(ss, dtype=np.int64)
        raw = np.array(maplog_as_u4_rows(af.tree["data"]["maplogs"]), copy=True)
    t_open = time.perf_counter() - t0

    n_nodes = int(raw.shape[0])
    ranges = segment_ranges_from_starts(segment_starts, n_nodes)

    t1 = time.perf_counter()
    idx = select_timeslice_indices_parallel(
        raw,
        target_control,
        segment_starts=segment_starts,
        n_workers=n_workers,
    )
    t_sel = time.perf_counter() - t1

    n_ts = int(len(idx))

    t2 = time.perf_counter()
    if n_ts:
        cols = _unpack_maplog(raw[idx], fields, path=path)
        out = {k: np.asarray(cols[k]) for k in fields}
    else:
        probe = _unpack_maplog(raw[:0], fields, path=path)
        out = {k: np.asarray(probe[k]) for k in fields}
    if return_row_indices:
        out["imap"] = np.asarray(idx, dtype=np.int64)
    t_unpack = time.perf_counter() - t2

    stats = {
        "n_nodes": n_nodes,
        "n_timeslice": n_ts,
        "n_segments": len(ranges),
        "epoch_step": int(epoch_step),
        "epoch_z": float(epoch_z),
        "target_control": int(target_control),
        "n_workers_sel": int(n_workers or 1),
        "wall_open_s": t_open,
        "wall_select_s": t_sel,
        "wall_unpack_ts_s": t_unpack,
        "wall_total_s": time.perf_counter() - t0,
        "lazy_load": False,
        "sel_mode": "MapLogSegmentStarts",
        "return_row_indices": bool(return_row_indices),
    }
    return out, stats


def load_timeslice_maps(
    sim_root: str | Path,
    *,
    z_maplog: float = 0.25,
    z_cat: float = 0.8,
    fields: tuple[str, ...] | list[str] = MAPLOG_LOAD_FIELDS,
    n_workers: int | None = None,
    return_row_indices: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load TimeSlice rows at ``z_cat`` from all ``maplog_z{z_maplog}_*.asdf`` shards.

    Resolves the TimeSlice epoch ``ts`` once from ``step_to_z``, then selects
    ``control == 4*ts + 3`` in parallel over ``MapLogSegmentStarts`` on each
    shard. The filename tag ``z_maplog`` is *not* the catalog epoch.

    When ``return_row_indices`` (default True for Ye tree tagging), also returns
    ``imap`` (row in shard) and ``shard_id`` (index into ``info['asdf_files']``).
    """
    sim_root = Path(sim_root)
    files = discover_maplog_asdf_files(sim_root, z=float(z_maplog))
    if not files:
        raise FileNotFoundError(
            f"no maplogs under {sim_root}/maplogs/*/maplog_z{float(z_maplog):.2f}_*.asdf"
        )
    col_counts = maplog_col_counts(files)
    n_cols = len(col_counts)

    step_to_z = resolve_step_to_z(rdir=sim_root, mdir=sim_root, z_maplog=float(z_maplog))
    epoch_step_, epoch_z = resolve_timeslice_epoch(step_to_z, float(z_cat))
    target_control = timeslice_control(epoch_step_)
    dz = abs(epoch_z - float(z_cat))
    if n_workers is None:
        n_workers = min(8, max(1, os.cpu_count() or 1))

    col_summary = ", ".join(f"{k}:{v}" for k, v in col_counts.items())
    print(
        f"  maplog shards={len(files)}  cols={n_cols} ({col_summary})  "
        f"z_maplog={float(z_maplog):.2f}  "
        f"z_cat={z_cat}  step_to_z entries={len(step_to_z)}",
        flush=True,
    )
    if n_cols > 1:
        print(
            f"  2D maplog layout: loading all {n_cols} col*/ folders "
            f"(not col000 only)",
            flush=True,
        )
    print(
        f"  resolved epoch once: step={epoch_step_}  z_epoch={epoch_z:.6f}  "
        f"|z-z_cat|={dz:.6g}  "
        f"target_control={target_control}  n_workers_sel={n_workers}  "
        f"sel=MapLogSegmentStarts  lazy_load=False",
        flush=True,
    )
    if dz > 0.05:
        print(
            f"  WARNING: |epoch_z - z_cat|={dz:.4f} is large; check Step_to_Z / z_cat",
            flush=True,
        )

    chunks: dict[str, list[np.ndarray]] = {k: [] for k in fields}
    imap_chunks: list[np.ndarray] = []
    shard_chunks: list[np.ndarray] = []
    n_nodes_total = 0
    n_ts_total = 0
    n_segments_total = 0

    for i, path in enumerate(files):
        print(f"  [{i + 1}/{len(files)}] decode {path.name} …", flush=True)
        shard, stats = decode_maplog_timeslice_fast(
            path,
            epoch_step=epoch_step_,
            epoch_z=epoch_z,
            fields=fields,
            n_workers=n_workers,
            return_row_indices=return_row_indices,
        )
        n_nodes_total += stats["n_nodes"]
        n_ts_total += stats["n_timeslice"]
        n_segments_total += int(stats["n_segments"])
        for k in fields:
            chunks[k].append(shard[k])
        if return_row_indices:
            n_here = int(stats["n_timeslice"])
            imap_chunks.append(np.asarray(shard["imap"], dtype=np.int64))
            shard_chunks.append(np.full(n_here, i, dtype=np.int32))
        del shard
        print(
            f"    nodes={stats['n_nodes']:,}  segs={stats['n_segments']:,}  "
            f"TimeSlice@{z_cat}={stats['n_timeslice']:,}  "
            f"open={stats['wall_open_s']:.1f}s sel={stats['wall_select_s']:.1f}s "
            f"unpack_ts={stats['wall_unpack_ts_s']:.1f}s "
            f"total={stats['wall_total_s']:.1f}s",
            flush=True,
        )

    arrays = {k: np.concatenate(chunks[k], axis=0) for k in fields}
    if return_row_indices:
        arrays["imap"] = (
            np.concatenate(imap_chunks, axis=0)
            if imap_chunks
            else np.array([], dtype=np.int64)
        )
        arrays["shard_id"] = (
            np.concatenate(shard_chunks, axis=0)
            if shard_chunks
            else np.array([], dtype=np.int32)
        )
    info = {
        "asdf_files": [str(p) for p in files],
        "n_asdf_files": len(files),
        "n_maplog_cols": int(n_cols),
        "maplog_col_counts": col_counts,
        "n_nodes_all_epochs": int(n_nodes_total),
        "n_map_timeslice": int(n_ts_total),
        "n_segments_total": int(n_segments_total),
        "z_maplog": float(z_maplog),
        "z_cat": float(z_cat),
        "epoch_step": int(epoch_step_),
        "epoch_z": float(epoch_z),
        "epoch_z_minus_z_cat": float(dz),
        "target_control": int(target_control),
        "n_workers_sel": int(n_workers),
        "fields": list(fields),
        "return_row_indices": bool(return_row_indices),
        "decode_mode": "resolve_ts_once_MapLogSegmentStarts_control_select",
        "lazy_load": False,
    }
    print(
        f"  epoch step={epoch_step_}  z_epoch={epoch_z:.6f}  "
        f"|z-z_cat|={dz:.6g}  n_MAP={n_ts_total:,}",
        flush=True,
    )
    return arrays, info


load_timeslice_maps_streaming = load_timeslice_maps


def load_maplogs(
    source: str | Path,
    *,
    mode: LoadMode = "timeslice",
    z_maplog: float = 0.25,
    z_cat: float | None = None,
    keys=BASE_KEYS,
    fields: tuple[str, ...] | list[str] = MAPLOG_LOAD_FIELDS,
    inject_pids: bool = True,
    return_row_indices: bool = True,
    n_workers: int | None = None,
):
    """Unified entry: ``mode='timeslice'`` or ``mode='full'``.

    ``source`` is a sim root (directory with ``maplogs/``) for ``timeslice``,
    or a single ASDF shard for ``full``.
    """
    if mode == "timeslice":
        if z_cat is None:
            z_cat = float(z_maplog)
        return load_timeslice_maps(
            source,
            z_maplog=float(z_maplog),
            z_cat=float(z_cat),
            fields=fields,
            n_workers=n_workers,
            return_row_indices=return_row_indices,
        )
    if mode == "full":
        return decode_maplog_shard(source, keys=keys, inject_pids=inject_pids)
    raise ValueError(f"unknown maplog mode {mode!r}; choose 'timeslice' or 'full'")


def catalog_from_segment(seg_maplog, z_cat, step_to_z, global_offset=0):
    """Build a compact per-MAP catalog for one in-memory segment slice."""
    tags = _tree_tags()
    maps = tags.maps_at_redshift(seg_maplog, z_cat, step_to_z=step_to_z)
    if len(maps) == 0:
        return _empty_catalog()

    tagged = tags.tag_maps(seg_maplog, maps, step_to_z)
    return {
        "global_map": global_offset + maps.astype(np.int64),
        "local_map": maps.astype(np.int64),
        "pos": np.asarray(seg_maplog["pos"][maps]),
        "mult": np.asarray(seg_maplog["mult"][maps]),
        "pid": np.asarray(seg_maplog["pid"][maps]),
        "form_z": tagged["form_z"],
        "merger_max_ratio": tagged["merger_max_ratio"],
        "merger_count": tagged["merger_count"],
        "accr_rate": tagged["accr_rate"],
    }


def _empty_catalog():
    return {
        "global_map": np.array([], dtype=np.int64),
        "local_map": np.array([], dtype=np.int64),
        "pos": np.empty((0, 3), dtype=np.float32),
        "mult": np.array([], dtype=np.float32),
        "pid": np.array([], dtype=np.int64),
        "form_z": np.array([], dtype=np.float64),
        "merger_max_ratio": np.array([], dtype=np.float64),
        "merger_count": np.array([], dtype=np.int32),
        "accr_rate": np.array([], dtype=np.float64),
    }


def merge_segment_catalogs(catalogs):
    """Concatenate catalog dicts produced per segment / per shard."""
    catalogs = [c for c in catalogs if len(c.get("global_map", ())) > 0]
    if not catalogs:
        return _empty_catalog()
    out = {}
    for key in catalogs[0]:
        out[key] = np.concatenate([c[key] for c in catalogs], axis=0)
    return out


def _mp_init_worker(shard_maplog, step_to_z, z_cat):
    global _MP_SHARD
    _MP_SHARD = {
        "maplog": shard_maplog,
        "step_to_z": step_to_z,
        "z_cat": z_cat,
    }


def _mp_process_segment_task(task: SegmentTask):
    full = _MP_SHARD["maplog"]
    seg = slice_maplog(full, task.i0, task.i1)
    cat = catalog_from_segment(
        seg,
        _MP_SHARD["z_cat"],
        _MP_SHARD["step_to_z"],
        global_offset=task.global_offset,
    )
    cat["file_path"] = np.array(
        [str(task.file_path)] * len(cat["global_map"]), dtype=object
    )
    cat["segment_index"] = np.full(
        len(cat["global_map"]), task.segment_index, dtype=np.int32
    )
    return cat


def multiprocessing_get_context():
    """Use fork on Linux so workers inherit the loaded shard via copy-on-write."""
    import multiprocessing as mp

    if os.name == "posix":
        return mp.get_context("fork")
    return mp.get_context("spawn")


def process_shard_segments(
    meta,
    shard_maplog,
    z_cat,
    step_to_z,
    *,
    n_workers=1,
    global_offset=0,
    process_fn=None,
):
    """Process every nominal segment of one already-loaded shard."""
    if process_fn is None:

        def process_fn(task, seg_maplog):
            return catalog_from_segment(
                seg_maplog, z_cat, step_to_z, global_offset=task.global_offset
            )

    tasks = [
        SegmentTask(
            i0=i0,
            i1=i1,
            segment_index=idx,
            global_offset=global_offset + i0,
            file_path=meta.path,
        )
        for idx, (i0, i1) in enumerate(segment_bounds(meta))
    ]

    if n_workers <= 1:
        out = []
        for task in tasks:
            seg = slice_maplog(shard_maplog, task.i0, task.i1)
            out.append(process_fn(task, seg))
        return out

    ctx = multiprocessing_get_context()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_mp_init_worker,
        initargs=(shard_maplog, step_to_z, z_cat),
    ) as pool:
        return list(pool.map(_mp_process_segment_task, tasks))


def iter_maplog_segments(mdir, z=0.2, keys=BASE_KEYS):
    """Yield :class:`MaplogSegment` slices using nominal ``MapLogSegmentStarts``.

    Loads each shard file once, yields every segment slice, then drops the
    shard before opening the next file.
    """
    load_keys = list(dict.fromkeys(list(keys) + ["control", "length"]))
    metas = load_maplog_run_metadata(mdir, z=z)
    global_offset = 0
    for meta in metas:
        full = decode_maplog_shard(meta.path, keys=load_keys)
        bounds = list(segment_bounds(meta))
        for seg_idx, (i0, i1) in enumerate(bounds):
            yield MaplogSegment(
                maplog=slice_maplog(full, i0, i1, keys=full),
                file_path=meta.path,
                global_offset=global_offset + i0,
                segment_index=seg_idx,
                n_segments_in_file=len(bounds),
            )
        global_offset += meta.n_nodes
        del full


def catalog_at_redshift_segmented(
    mdir,
    z_cat=0.2,
    *,
    z_maplog=None,
    rdir=None,
    n_workers=1,
    keys=BASE_KEYS,
):
    """Full-run MAP catalog at ``z_cat`` via load-once / segment-parallel I/O."""
    if z_maplog is None:
        z_maplog = z_cat
    if rdir is None:
        rdir = mdir

    step_to_z = resolve_step_to_z(mdir=mdir, rdir=rdir, z_maplog=z_maplog)
    metas = load_maplog_run_metadata(mdir, z=z_maplog)
    load_keys = list(dict.fromkeys(list(keys) + ["control", "length"]))

    catalogs = []
    global_offset = 0
    for meta in metas:
        shard = decode_maplog_shard(meta.path, keys=load_keys)
        parts = process_shard_segments(
            meta,
            shard,
            z_cat,
            step_to_z,
            n_workers=n_workers,
            global_offset=global_offset,
        )
        catalogs.extend(parts)
        global_offset += meta.n_nodes
        del shard

    merged = merge_segment_catalogs(catalogs)
    merged["step_to_z"] = step_to_z
    return merged


def reduce_over_segments(mdir, z, init, fold, keys=BASE_KEYS):
    """Generic tree-reduce: ``acc = fold(acc, segment)`` over all segments."""
    acc = init
    for seg in iter_maplog_segments(mdir, z=z, keys=keys):
        acc = fold(acc, seg)
    return acc


def read_maplog_dir_chunked(mdir, z=0.2, max_nodes=None, keys=BASE_KEYS):
    """Concatenate segments (one shard at a time) up to ``max_nodes``."""
    parts = {key: [] for key in keys}
    n = 0
    for seg in iter_maplog_segments(mdir, z=z, keys=keys):
        for key in keys:
            parts[key].append(seg.maplog[key])
        n += len(seg.maplog[keys[0]])
        if max_nodes is not None and n >= max_nodes:
            break
    return {key: np.concatenate(parts[key], axis=0) for key in keys}
