"""Cleaned CompaSO halo catalogs and timeslice-subsample one-rank loads.

Slab overlap: Abacus writes 16 ``halo_info_XXX.asdf`` slabs along x. A rank's
particles occupy a subset of that x range, so the matcher loads the slabs
whose edges overlap ``[x_lo - delta, x_hi + delta]`` and then filters by
``x_L2com``. ``cleaned=True`` is required; uncleaned ``N==0`` rows are dropped
by default.

Machine-specific roots resolve through :mod:`aurora_analysis.config`.
There is no package-wide default simulation: callers pass ``sim_root``.
Test-5 scripts that still want JUL28 should use ``path("jul28_root")``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import path as config_path
from .asdf import read_header, warn_if_login_node
from .subsample import (
    detect_subsample_sampler,
    normalize_redshift,
    resolve_column_subsample_dir,
    resolve_rank_asdf_path,
)

__all__ = [
    "COMPASO_FIELDS",
    "DEFAULT_LOAD",
    "asdf_header",
    "compaso_slab_edges",
    "default_compaso_root",
    "load_compaso_overlap",
    "load_timeslice_one_rank",
    "nominal_rank_x_domain",
    "normalize_env_fields",
    "overlapping_compaso_slab_indices",
    "rank_x_domain_from_particles",
    "smoke_load",
    "DEFAULT_COMPASO_ROOT",
]

DEFAULT_LOAD = ("pos", "vel", "is_map", "density", "vel_disp", "rel_vel", "mult")
COMPASO_FIELDS = (
    "N",
    "x_L2com",
    "v_L2com",
    "sigmav3d_L2com",
    "r100_L2com",
)


def default_compaso_root() -> Path:
    return config_path("compaso_root")


def __getattr__(name: str):
    """Lazy ``DEFAULT_COMPASO_ROOT`` so importing this module does not touch unused paths."""
    if name == "DEFAULT_COMPASO_ROOT":
        return default_compaso_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def normalize_env_fields(merged: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(merged)
    if "velocity_dispersion" not in out and "vel_disp" in out:
        out["velocity_dispersion"] = out["vel_disp"]
    if "relative_velocity" not in out and "rel_vel" in out:
        out["relative_velocity"] = out["rel_vel"]
    return out


def load_timeslice_one_rank(
    sim_root: str | Path,
    z: str | float,
    *,
    col: int = 0,
    rank: str | int = 0,
    load: tuple[str, ...] | list[str] = DEFAULT_LOAD,
) -> tuple[dict[str, np.ndarray], Path, str]:
    """Load one timeslice subsample ASDF.

    Returns (arrays, asdf_path, sampler).
    """
    from abacusnbody.data import read_asdf

    sim_root = Path(sim_root)
    z = normalize_redshift(z)
    subsample_dir = resolve_column_subsample_dir(sim_root, z, int(col))
    column_path = sim_root / subsample_dir
    sampler = detect_subsample_sampler(column_path, z)
    path = resolve_rank_asdf_path(
        column_path, z, str(rank), column_index=int(col), sampler=sampler
    )
    print(f"  load_timeslice_one_rank: {path}", flush=True)
    print(f"  sampler={sampler}  load={list(load)}", flush=True)
    warn_if_login_node(path, decompressing=True)
    tab = read_asdf(str(path), load=list(load))
    merged = {col: np.asarray(tab[col]) for col in load}
    return normalize_env_fields(merged), path, sampler


def nominal_rank_x_domain(
    box: float, node_rank_x: int, node_size_x: int
) -> tuple[float, float]:
    """Box-centered x interval for an Abacus NodeRankX slab."""
    width = box / float(node_size_x)
    x0 = -0.5 * box + node_rank_x * width
    x1 = x0 + width
    return float(x0), float(x1)


def rank_x_domain_from_particles(
    pos: np.ndarray,
    *,
    box: float | None = None,
    header: dict | None = None,
) -> tuple[float, float]:
    """Prefer particle bbox; fall back to NodeRankX/NodeSizeX from header."""
    pos = np.asarray(pos)
    if pos.size:
        return float(pos[:, 0].min()), float(pos[:, 0].max())
    if header is not None and box is not None:
        return nominal_rank_x_domain(
            float(box),
            int(header.get("NodeRankX", 0)),
            int(header.get("NodeSizeX", 32)),
        )
    raise ValueError("need particles or header to get x domain")


def compaso_slab_edges(box: float = 1000.0, n_slab: int = 16) -> np.ndarray:
    return np.linspace(-0.5 * box, 0.5 * box, n_slab + 1)


def overlapping_compaso_slab_indices(
    x_lo: float,
    x_hi: float,
    *,
    box: float = 1000.0,
    n_slab: int = 16,
    delta: float = 5.0,
) -> list[int]:
    edges = compaso_slab_edges(box, n_slab)
    lo, hi = x_lo - delta, x_hi + delta
    return [i for i in range(n_slab) if edges[i] < hi and edges[i + 1] > lo]


def load_compaso_overlap(
    z: str | float,
    x_lo: float,
    x_hi: float,
    *,
    delta: float = 5.0,
    compaso_root: str | Path | None = None,
    fields: tuple[str, ...] | list[str] = COMPASO_FIELDS,
    box: float = 1000.0,
    n_slab: int = 16,
    drop_n_eq_0: bool = True,
) -> tuple[dict[str, np.ndarray], list[Path]]:
    """Load cleaned CompaSO halos whose slabs overlap [x_lo, x_hi] (+ delta).

    Still filters by ``x_L2com`` after load. Drops cleaned ``N==0`` rows by default.
    ``compaso_root`` defaults to :func:`aurora_analysis.config.path` ``compaso_root``.
    """
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    z = normalize_redshift(z)
    compaso_root = Path(compaso_root) if compaso_root is not None else default_compaso_root()
    zdir = compaso_root / "halos" / f"z{z}"
    idxs = overlapping_compaso_slab_indices(
        x_lo, x_hi, box=box, n_slab=n_slab, delta=delta
    )
    if not idxs:
        raise ValueError(f"no CompaSO slabs overlap x=[{x_lo}, {x_hi}] delta={delta}")

    paths = [zdir / "halo_info" / f"halo_info_{i:03d}.asdf" for i in idxs]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    print(
        f"  load_compaso_overlap: z={z} x=[{x_lo:.3f},{x_hi:.3f}] "
        f"delta={delta} → slabs {idxs}",
        flush=True,
    )
    warn_if_login_node(paths[0], decompressing=True)

    chunks: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    header: dict | None = None
    for p in paths:
        cat = CompaSOHaloCatalog(
            str(p),
            cleaned=True,
            fields=list(fields),
            subsamples=False,
            convert_units=True,
        )
        if header is None:
            header = dict(cat.header)
        h = cat.halos
        for f in fields:
            chunks[f].append(np.asarray(h[f]))

    out = {f: np.concatenate(chunks[f], axis=0) for f in fields}
    x = np.asarray(out["x_L2com"])[:, 0]
    keep = (x >= x_lo - delta) & (x <= x_hi + delta)
    if drop_n_eq_0 and "N" in out:
        keep &= np.asarray(out["N"]) > 0
    n_before = len(x)
    out = {f: v[keep] for f, v in out.items()}
    print(
        f"  halos loaded={n_before}  after x-cut"
        f"{' + N>0' if drop_n_eq_0 else ''}={len(out['N'])}",
        flush=True,
    )
    out["_header"] = header or {}
    return out, paths


asdf_header = read_header


def smoke_load(
    *,
    sim_root: str | Path,
    z: float = 3.0,
    col: int = 0,
    rank: int = 0,
) -> None:
    """Lightweight smoke: header domain + CompaSO overlap (no full particle decompress)."""
    sim_root = Path(sim_root)
    z = normalize_redshift(z)
    subsample_dir = resolve_column_subsample_dir(sim_root, z, col)
    column_path = sim_root / subsample_dir
    sampler = detect_subsample_sampler(column_path, z)
    path = resolve_rank_asdf_path(
        column_path, z, str(rank), column_index=col, sampler=sampler
    )
    hdr = read_header(path)
    box = float(hdr.get("BoxSize", 1000.0))
    x0, x1 = nominal_rank_x_domain(
        box, int(hdr.get("NodeRankX", rank)), int(hdr.get("NodeSizeX", 32))
    )
    print(f"smoke path={path}")
    print(
        f"  BoxSize={box}  NodeRankX={hdr.get('NodeRankX')}  "
        f"NodeSizeX={hdr.get('NodeSizeX')}"
    )
    print(f"  nominal x domain=[{x0:.4f}, {x1:.4f})")
    halos, slabs = load_compaso_overlap(z, x0, x1, delta=5.0, box=box)
    print(f"  slab files={[p.name for p in slabs]}")
    print(f"  n_halos={len(halos['N'])}  r100 median={np.median(halos['r100_L2com']):.4g}")
