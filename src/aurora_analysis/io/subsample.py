"""Timeslice subsample ASDF helpers.

Column discovery (``colNNN`` vs ``columnNNNN``), sampler detection
(``subP8`` / ``subP64`` / ``subA`` / ``all``), and merging rank files into
one particle table. Particle unpack still goes through abacusutils
``read_asdf`` (the timeslice_subsample handler); this module adds the
compute-node warning that used to live only in READMEs.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np

from .asdf import warn_if_login_node

__all__ = [
    "LOAD_TIMING",
    "SUBSAMPLE_SAMPLER",
    "center_pos_to_box",
    "column_dir_names",
    "detect_subsample_sampler",
    "discover_column_indices",
    "list_column_asdf_files",
    "load_merged_1d",
    "load_merged_2d_full_box",
    "load_merged_column_particles",
    "load_merged_columns",
    "load_merged_full_box",
    "masked_dm_positions",
    "masked_map_positions",
    "match_positions_by_pid",
    "normalize_redshift",
    "parse_column_index",
    "pid_keys",
    "rank_asdf_filename_candidates",
    "rank_x_label_from_asdf",
    "resolve_column_subsample_dir",
    "resolve_matched_subsample_sampler",
    "resolve_rank_asdf_path",
    "subbox_mask",
    "subsample_asdf_stem",
    "subsample_sampler",
    "z_slice_mask",
]

SUBSAMPLE_SAMPLER = os.environ.get("SUBSAMPLE_SAMPLER", "subP8")
_SAMPLER_ASDF_RE = re.compile(r"^particles_(subP\d+|subA|all)_z", re.IGNORECASE)
LOAD_TIMING = os.environ.get("LOAD_TIMING", "0") == "1"


def subsample_sampler() -> str:
    """Explicit SUBSAMPLE_SAMPLER env override (optional; auto-detect preferred)."""
    return os.environ.get("SUBSAMPLE_SAMPLER", SUBSAMPLE_SAMPLER)


def normalize_redshift(z: str | float) -> str:
    """Abacus subsample dirs and ASDF names use three decimals (``z0.500`` not ``z0.5``)."""
    return f"{float(z):.3f}"


def detect_subsample_sampler(column_path: str | Path, redshift: str) -> str:
    """Detect sampler (``subP*``, ``subA``, or ``all``) from ASDF basenames."""
    redshift = normalize_redshift(redshift)
    column_path = Path(column_path)
    ztag = f"_z{redshift}_"
    counts: dict[str, int] = {}
    for path in column_path.iterdir():
        if not path.is_file() or path.suffix != ".asdf":
            continue
        m = _SAMPLER_ASDF_RE.match(path.name)
        if m is None or ztag not in path.name:
            continue
        sampler = m.group(1)
        if sampler.lower() == "all":
            sampler = "all"
        elif sampler.lower() == "suba":
            sampler = "subA"
        counts[sampler] = counts.get(sampler, 0) + 1
    if not counts:
        raise FileNotFoundError(
            f"No particles_{{subP*,subA,all}}_z{redshift}_*.asdf under {column_path}"
        )
    if len(counts) > 1:
        detail = ", ".join(f"{k} ({n} files)" for k, n in sorted(counts.items()))
        raise ValueError(f"Ambiguous subsample samplers under {column_path}: {detail}")
    return next(iter(counts))


def subsample_asdf_stem(redshift: str, sampler: str | None = None) -> str:
    redshift = normalize_redshift(redshift)
    samp = subsample_sampler() if sampler is None else sampler
    return f"particles_{samp}_z{redshift}_"


def parse_column_index(column_id: str | int) -> int:
    """Parse COLUMN_ID / COMPARE_COLUMN_2D (1, 001, 0001, col001, column0001)."""
    if isinstance(column_id, int):
        return column_id
    s = str(column_id).strip().lower()
    if s.startswith("column"):
        s = s[6:]
    elif s.startswith("col"):
        s = s[3:]
    return int(s, 10)


def column_dir_names(column_index: int) -> tuple[str, str]:
    """Candidate directory basenames: column{04d} then col{03d}."""
    return (f"column{column_index:04d}", f"col{column_index:03d}")


def resolve_column_subsample_dir(
    sim_dir: str | Path, redshift: str, column_index: int
) -> str:
    """Return subsamples/z{z}/column… or col… path, auto-detecting naming convention."""
    redshift = normalize_redshift(redshift)
    zroot = Path(sim_dir) / f"subsamples/z{redshift}"
    for name in column_dir_names(column_index):
        if (zroot / name).is_dir():
            return f"subsamples/z{redshift}/{name}"
    tried = ", ".join(column_dir_names(column_index))
    raise FileNotFoundError(
        f"No column directory for index {column_index} under {zroot} (tried {tried})"
    )


def list_column_asdf_files(
    column_path: Path,
    redshift: str,
    sampler: str | None = None,
) -> list[Path]:
    """All subsample ASDF files in a column directory (auto-detects subP* if needed)."""
    redshift = normalize_redshift(redshift)
    column_path = Path(column_path)
    if sampler is None:
        sampler = detect_subsample_sampler(column_path, redshift)
    stem = f"particles_{sampler}_z{redshift}_"
    return sorted(column_path.glob(stem + "*.asdf"))


def rank_asdf_filename_candidates(
    redshift: str,
    rank_x: str,
    column_index: int | None = None,
    sampler: str | None = None,
) -> list[str]:
    """Candidate ASDF basenames for a logical rank (RANK_X env).

    Supports legacy ``…_0000.asdf`` and JUN19-style ``…_000_001.asdf``.
    """
    rx = str(rank_x).strip()
    midparts: list[str] = []

    if "_" in rx:
        midparts.append(rx)
    else:
        midparts.append(rx)
        if rx.isdigit():
            i = int(rx)
            midparts.append(f"{i:04d}")
            midparts.append(f"{i:03d}")
            if column_index is not None:
                midparts.append(f"{column_index:03d}_{i:03d}")

    stem = subsample_asdf_stem(redshift, sampler=sampler)
    seen: list[str] = []
    for mid in midparts:
        name = f"{stem}{mid}.asdf"
        if name not in seen:
            seen.append(name)
    return seen


def resolve_rank_asdf_path(
    column_path: str | Path,
    redshift: str,
    rank_x: str,
    column_index: int | None = None,
    sampler: str | None = None,
) -> Path:
    """Resolve a single-rank subsample ASDF under a column directory."""
    column_path = Path(column_path)
    if sampler is None:
        sampler = detect_subsample_sampler(column_path, redshift)
    tried: list[str] = []
    for name in rank_asdf_filename_candidates(
        redshift, rank_x, column_index, sampler=sampler
    ):
        path = column_path / name
        tried.append(name)
        if path.is_file():
            return path

    if str(rank_x).strip().isdigit():
        files = list_column_asdf_files(column_path, redshift, sampler=sampler)
        idx = int(str(rank_x).strip(), 10)
        if 0 <= idx < len(files):
            return files[idx]

    listing = [
        p.name for p in list_column_asdf_files(column_path, redshift, sampler=sampler)
    ]
    msg = f"No rank file for RANK_X={rank_x!r} in {column_path}"
    if column_index is not None:
        msg += f" (column index {column_index})"
    msg += f"; tried {tried}"
    if listing:
        msg += f"; found {listing}"
    raise FileNotFoundError(msg)


def rank_x_label_from_asdf(
    path: str | Path, redshift: str, column_index: int | None = None
) -> str:
    """4-digit x-rank label parsed from an ASDF basename (e.g. …_0007 or …_000_003)."""
    redshift = normalize_redshift(redshift)
    stem = Path(path).stem
    tag = f"_z{redshift}_"
    if tag not in stem:
        raise ValueError(f"Cannot parse rank from {path!r}")
    rank_part = stem.split(tag, 1)[1]
    if "_" in rank_part:
        xrank = int(rank_part.rsplit("_", 1)[-1], 10)
    else:
        xrank = int(rank_part, 10)
    return f"{xrank:04d}"


def z_slice_mask(pos, z_min, z_max, *, inclusive_max: bool = False):
    pos = np.asarray(pos)
    z_ok_lo = pos[:, 2] >= z_min
    if inclusive_max:
        z_ok_hi = pos[:, 2] <= z_max
    else:
        z_ok_hi = pos[:, 2] < z_max
    return z_ok_lo & z_ok_hi


def subbox_mask(
    pos,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
):
    """Axis-aligned subbox in centered Abacus coords [Mpc/h], half-open on each axis."""
    pos = np.asarray(pos)
    return (
        (pos[:, 0] >= x_min)
        & (pos[:, 0] < x_max)
        & (pos[:, 1] >= y_min)
        & (pos[:, 1] < y_max)
        & (pos[:, 2] >= z_min)
        & (pos[:, 2] < z_max)
    )


def center_pos_to_box(pos, lbox):
    """Abacus centered coords [-L/2, L/2] → abacusutils [0, Lbox)."""
    return np.mod(np.asarray(pos, dtype=np.float64) + 0.5 * lbox, lbox)


def pid_keys(pid):
    return (
        pid[:, 0].astype(np.uint64)
        | (pid[:, 1].astype(np.uint64) << np.uint64(16))
        | (pid[:, 2].astype(np.uint64) << np.uint64(32))
    )


def masked_dm_positions(pos, pid, is_map):
    dm = ~np.asarray(is_map, dtype=bool)
    return pos[dm], pid[dm], is_map[dm]


def masked_map_positions(pos, pid, is_map):
    mp = np.asarray(is_map, dtype=bool)
    return pos[mp], pid[mp], is_map[mp]


def match_positions_by_pid(pos_a, keys_a, pos_b, keys_b):
    order_a = np.argsort(keys_a, kind="mergesort")
    order_b = np.argsort(keys_b, kind="mergesort")
    keys_a_s = keys_a[order_a]
    keys_b_s = keys_b[order_b]
    _, idx_a, idx_b = np.intersect1d(keys_a_s, keys_b_s, return_indices=True)
    return pos_a[order_a][idx_a], pos_b[order_b][idx_b]


def load_merged_column_particles(
    sim_dir, subsample_dir, redshift, load=None, sampler: str | None = None
):
    from abacusnbody.data import read_asdf

    if load is None:
        load = ["pos"]
    column_path = Path(sim_dir) / subsample_dir
    if sampler is None:
        sampler = detect_subsample_sampler(column_path, redshift)
        print(f"  subsample sampler: {sampler} (auto-detected)", flush=True)
    else:
        print(f"  subsample sampler: {sampler} (matched)", flush=True)
    files = list_column_asdf_files(column_path, redshift, sampler=sampler)
    if not files:
        pattern = f"particles_{sampler}_z{redshift}_*.asdf"
        raise FileNotFoundError(f"No files matching {pattern} in {column_path}")

    print(f"Loading {len(files)} files from {column_path}/", flush=True)
    warn_if_login_node(files[0], decompressing=True)
    chunks = {col: [] for col in load}
    read_cols = list(dict.fromkeys([*load, "pid"]))
    t_column = time.perf_counter()
    for fn in files:
        t_file = time.perf_counter()
        print(f"  {fn.name}", flush=True)
        particles = read_asdf(str(fn), load=read_cols)
        dt_file = time.perf_counter() - t_file
        print(f"    N = {len(particles):,}", flush=True)
        if LOAD_TIMING:
            print(f"    read time = {dt_file:.1f}s", flush=True)
        for col in load:
            chunks[col].append(np.asarray(particles[col]))

    t_cat = time.perf_counter()
    merged = {col: np.concatenate(parts, axis=0) for col, parts in chunks.items()}
    n_merged = len(next(iter(merged.values())))
    print(f"Merged total: N = {n_merged:,}", flush=True)
    if LOAD_TIMING:
        print(
            f"  column concat = {time.perf_counter() - t_cat:.1f}s, "
            f"column wall = {time.perf_counter() - t_column:.1f}s",
            flush=True,
        )
    return merged, files


def discover_column_indices(sim_dir: str | Path, redshift: str) -> list[int]:
    """Sorted column indices under ``subsamples/z{redshift}/`` (colNNN or columnNNNN)."""
    redshift = normalize_redshift(redshift)
    zroot = Path(sim_dir) / f"subsamples/z{redshift}"
    if not zroot.is_dir():
        raise FileNotFoundError(f"Missing subsample redshift directory {zroot}")
    indices: list[int] = []
    for path in sorted(zroot.iterdir()):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if name.startswith("column"):
            indices.append(int(name[6:], 10))
        elif name.startswith("col"):
            indices.append(int(name[3:], 10))
    if not indices:
        raise FileNotFoundError(f"No column directories under {zroot}")
    return sorted(indices)


def resolve_matched_subsample_sampler(
    sim_dir_1d: str | Path,
    redshift: str,
    *,
    sim_dir_2d: str | Path | None = None,
    sim_dir_zd: str | Path | None = None,
    column_index_1d: int = 0,
    sim_dir_2d_columns: tuple[int, ...] = (0, 1),
    reference_sampler: str | None = None,
) -> str:
    """Pick one subP* sampler from 1D and verify 2D/ZD have matching ASDF files."""
    if reference_sampler is not None:
        sampler = reference_sampler
        source = f"override {reference_sampler}"
    else:
        subsample_dir = resolve_column_subsample_dir(
            sim_dir_1d, redshift, column_index_1d
        )
        sampler = detect_subsample_sampler(Path(sim_dir_1d) / subsample_dir, redshift)
        source = f"1D {Path(subsample_dir).name}"

    def _require(
        sim_dir: str | Path, column_indices: tuple[int, ...], label: str
    ) -> None:
        for cid in column_indices:
            subsample_dir = resolve_column_subsample_dir(sim_dir, redshift, cid)
            col_path = Path(sim_dir) / subsample_dir
            files = list_column_asdf_files(col_path, redshift, sampler=sampler)
            if not files:
                pattern = f"particles_{sampler}_z{redshift}_*.asdf"
                raise FileNotFoundError(
                    f"{label}: no {sampler} files at z={redshift} in {col_path} "
                    f"(expected {pattern})"
                )

    if sim_dir_2d is not None:
        _require(sim_dir_2d, sim_dir_2d_columns, "2D")
    if sim_dir_zd is not None:
        zd_cols = tuple(discover_column_indices(sim_dir_zd, redshift))
        _require(sim_dir_zd, zd_cols, "ZD")

    print(
        f"z={redshift}: matched subsample sampler {sampler} ({source})",
        flush=True,
    )
    return sampler


def load_merged_columns(
    sim_dir,
    redshift,
    column_indices: tuple[int, ...] | list[int],
    load=None,
    *,
    label: str = "merged",
    sampler: str | None = None,
):
    """Merge all rank ASDF files across the given z-column indices."""
    if load is None:
        load = ["pos"]
    parts = []
    files_all = []
    col_names: list[str] = []
    for cid in column_indices:
        subsample_dir = resolve_column_subsample_dir(sim_dir, redshift, cid)
        col_names.append(Path(subsample_dir).name)
        merged, files = load_merged_column_particles(
            sim_dir, subsample_dir, redshift, load=load, sampler=sampler
        )
        parts.append(merged)
        files_all.extend(files)
    out = {col: np.concatenate([part[col] for part in parts], axis=0) for col in parts[0]}
    n_merged = len(next(iter(out.values())))
    cols_join = " + ".join(col_names)
    print(
        f"{label}: N = {n_merged:,} from {len(files_all)} files ({cols_join})",
        flush=True,
    )
    return out, files_all


def load_merged_1d(
    sim_dir, redshift, load=None, column_index: int = 0, sampler: str | None = None
):
    """Merge all rank ASDF files in one 1D z-column (col{idx:03d} / column{idx:04d})."""
    subsample_dir = resolve_column_subsample_dir(sim_dir, redshift, column_index)
    merged, files = load_merged_column_particles(
        sim_dir, subsample_dir, redshift, load=load, sampler=sampler
    )
    col_name = Path(subsample_dir).name
    n_merged = len(next(iter(merged.values())))
    print(
        f"1D column {col_name}: N = {n_merged:,} from {len(files)} ASDF files",
        flush=True,
    )
    return merged, files


def load_merged_2d_full_box(sim_dir, redshift, load=None, sampler: str | None = None):
    """Merge both z-columns and all rank_x files (full 2D box)."""
    return load_merged_columns(
        sim_dir, redshift, (0, 1), load=load, label="2D full box", sampler=sampler
    )


def load_merged_full_box(sim_dir, redshift, load=None, sampler: str | None = None):
    """Merge every z-column under ``subsamples/z{redshift}/`` (auto-discovered)."""
    column_indices = discover_column_indices(sim_dir, redshift)
    return load_merged_columns(
        sim_dir, redshift, column_indices, load=load, label="full box", sampler=sampler
    )
