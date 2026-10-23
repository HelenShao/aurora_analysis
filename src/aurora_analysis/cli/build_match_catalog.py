#!/usr/bin/env python3
"""Build a versioned MAP–halo match catalog with ``aurora_analysis.match``.

Loads TimeSlice MAPs (maplog) and cleaned CompaSO hosts via
``aurora_analysis.io``, then:

- matches with ``match_maps`` (default ``method=ball``, ``overlap=most_massive``)
- labels cen/sat with ``label_cen_sat``
- writes ``.npz`` + sidecar JSON via ``save_match_catalog``

Example (compute node + tee)::

  python3 -m aurora_analysis.cli.build_match_catalog \\
    --sim aurora_highbase --z 0.25 --z-maplog 0.25 \\
    --outdir /lus/flare/projects/Abacus/helenshao/load/ye_repro_AHB_z0.25 \\
    2>&1 | tee logs/build_match_sph_AHB_z0.250.log
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from aurora_analysis.config import path, sim
from aurora_analysis.io.compaso import COMPASO_FIELDS, load_compaso_overlap
from aurora_analysis.io.maplog import (
    MAPLOG_LOAD_FIELDS,
    load_timeslice_maps,
    read_maplog_asdf_header,
)
from aurora_analysis.io.subsample import normalize_redshift
from aurora_analysis.match import (
    MatchProvenance,
    format_match_stem,
    label_cen_sat,
    match_catalog_tag,
    match_maps,
    periodic_distance,
    save_match_catalog,
)


def _ztag(z: float) -> str:
    return normalize_redshift(z)


def _load_maps(
    *,
    sim_root: Path,
    z_maplog: float,
    z_cat: float,
    maps_cache: Path | None,
    cache_maps: Path | None,
) -> tuple[dict[str, np.ndarray], dict]:
    if maps_cache is not None and Path(maps_cache).is_file():
        t0 = time.perf_counter()
        cache = Path(maps_cache)
        print(f"  loading MAP cache {cache} …", flush=True)
        raw = np.load(cache, allow_pickle=True)
        maps = {k: raw[k] for k in raw.files}
        meta_path = cache.with_suffix(".json")
        load_info: dict = {}
        if meta_path.is_file():
            import json

            load_info = json.loads(meta_path.read_text())
        print(
            f"  MAP cache wall={time.perf_counter() - t0:.1f}s  "
            f"n_MAP={len(maps['pos']):,}",
            flush=True,
        )
        return maps, load_info

    t0 = time.perf_counter()
    maps, load_info = load_timeslice_maps(
        sim_root,
        z_maplog=float(z_maplog),
        z_cat=float(z_cat),
        fields=MAPLOG_LOAD_FIELDS,
        return_row_indices=True,
    )
    print(f"  maplog TimeSlice load wall={time.perf_counter() - t0:.1f}s", flush=True)
    if cache_maps is not None:
        cache = Path(cache_maps)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **maps)
        import json

        cache.with_suffix(".json").write_text(
            json.dumps(load_info, indent=2, default=str) + "\n"
        )
        print(f"  wrote MAP cache {cache}", flush=True)
    return maps, load_info


def _load_halos(
    *,
    ztag: str,
    compaso_root: Path,
    box: float,
    halo_R_field: str,
    delta: float,
) -> tuple[dict[str, np.ndarray], list[Path]]:
    fields = list(dict.fromkeys([*COMPASO_FIELDS, halo_R_field, "r100_L2com"]))
    half = 0.5 * float(box)
    t0 = time.perf_counter()
    halos, slab_paths = load_compaso_overlap(
        ztag,
        -half,
        half,
        delta=delta,
        compaso_root=compaso_root,
        box=box,
        fields=fields,
    )
    print(f"  CompaSO load wall={time.perf_counter() - t0:.1f}s", flush=True)
    return halos, slab_paths


def _host_ids(host_x: np.ndarray) -> np.ndarray:
    hx = np.ascontiguousarray(host_x, dtype=np.float64)
    void = hx.view(np.dtype((np.void, hx.dtype.itemsize * 3))).reshape(-1)
    _, inv = np.unique(void, return_inverse=True)
    return inv.astype(np.int64)


def build_catalog(
    *,
    sim_name: str,
    z: float,
    z_maplog: float,
    outdir: Path,
    method: str = "ball",
    overlap: str = "most_massive",
    halo_R_field: str | None = None,
    knn_k: int = 128,
    delta: float = 5.0,
    maps_cache: Path | None = None,
    cache_maps: Path | None = None,
    box: float | None = None,
) -> tuple[Path, Path]:
    spec = sim(sim_name)
    halo_R_field = halo_R_field or spec.halo_R_field or "r100_L2com"
    box = float(box if box is not None else spec.box or 1000.0)
    ztag = _ztag(z)
    outdir = Path(outdir)
    catdir = outdir / "catalogs"
    metadir = outdir / "meta"
    catdir.mkdir(parents=True, exist_ok=True)
    metadir.mkdir(parents=True, exist_ok=True)

    print(
        f"aurora_analysis.build_match_catalog  sim={sim_name}  root={spec.root}  "
        f"z_maplog={z_maplog} z_cat={ztag}  R={halo_R_field}  method={method}",
        flush=True,
    )

    maps, load_info = _load_maps(
        sim_root=spec.root,
        z_maplog=z_maplog,
        z_cat=z,
        maps_cache=maps_cache,
        cache_maps=cache_maps,
    )
    map_pos = np.asarray(maps["pos"], dtype=np.float64)
    map_vel = np.asarray(maps["vel"], dtype=np.float64)
    map_mult = np.asarray(maps["mult"], dtype=np.float64)
    n_map = len(map_pos)

    try:
        files = load_info.get("asdf_files") or []
        if files:
            meta0 = read_maplog_asdf_header(files[0])
            if meta0.box_size is not None:
                box = float(meta0.box_size)
    except Exception as e:  # noqa: BLE001
        print(f"  warning: BoxSize from header failed ({e}); using {box}", flush=True)

    halo_root = path("compaso_root")
    halos, slab_paths = _load_halos(
        ztag=ztag,
        compaso_root=halo_root,
        box=box,
        halo_R_field=halo_R_field,
        delta=delta,
    )
    halo_pos = np.asarray(halos["x_L2com"], dtype=np.float64)
    halo_R = np.asarray(halos[halo_R_field], dtype=np.float64)
    halo_N = np.asarray(halos["N"], dtype=np.float64)
    bad_R = ~(np.isfinite(halo_R) & (halo_R > 0))
    if bad_R.any():
        halo_R = halo_R.copy()
        halo_R[bad_R] = np.nanmedian(halo_R[~bad_R]) if (~bad_R).any() else 1.0

    t2 = time.perf_counter()
    local_ih = match_maps(
        map_pos,
        halo_pos,
        halo_R,
        box,
        method=method,  # type: ignore[arg-type]
        overlap=overlap,  # type: ignore[arg-type]
        halo_mass=halo_N,
        k=knn_k,
        verbose=True,
    )
    print(f"  match_maps({method}) wall={time.perf_counter() - t2:.1f}s", flush=True)

    if method == "ball":
        matched = local_ih >= 0
        match_cut = f"enclosed in {halo_R_field}, overlap={overlap}"
    else:
        d_all = periodic_distance(map_pos, halo_pos[local_ih], box)
        matched = d_all < halo_R[local_ih]
        match_cut = f"d < {halo_R_field}"

    n_matched = int(matched.sum())
    print(
        f"  matched ({method}): {n_matched:,} / {n_map:,} "
        f"({100.0 * n_matched / max(n_map, 1):.2f}%)",
        flush=True,
    )

    d = np.full(n_map, np.nan, dtype=np.float64)
    d_over_R = np.full(n_map, np.nan, dtype=np.float64)
    R_host = np.full(n_map, np.nan, dtype=np.float64)
    host_x = np.full((n_map, 3), np.nan, dtype=np.float64)
    host_v = np.full((n_map, 3), np.nan, dtype=np.float64)
    host_sig = np.full(n_map, np.nan, dtype=np.float64)
    host_N = np.full(n_map, np.nan, dtype=np.float64)
    local_out = np.full(n_map, -1, dtype=np.int64)

    if matched.any():
        ih = local_ih[matched]
        local_out[matched] = ih
        R_host[matched] = halo_R[ih]
        d[matched] = periodic_distance(map_pos[matched], halo_pos[ih], box)
        d_over_R[matched] = d[matched] / np.maximum(R_host[matched], 1e-30)
        host_x[matched] = halo_pos[ih]
        host_v[matched] = np.asarray(halos["v_L2com"], dtype=np.float64)[ih]
        host_sig[matched] = np.asarray(halos["sigmav3d_L2com"], dtype=np.float64)[ih]
        host_N[matched] = halo_N[ih]

    is_cen = np.zeros(n_map, dtype=bool)
    is_sat = np.zeros(n_map, dtype=bool)
    if n_matched:
        is_cen_m = label_cen_sat(map_mult[matched], local_out[matched])
        is_cen[matched] = is_cen_m
        is_sat[matched] = ~is_cen_m

    arrays: dict[str, np.ndarray] = {
        "map_pos": map_pos,
        "map_vel": map_vel,
        "map_mult": map_mult,
        "matched": matched,
        "is_central": is_cen,
        "is_satellite": is_sat,
        "d": d,
        "d_over_R": d_over_R,
        "local_halo_ih": local_out,
        "host_x_L2com": host_x,
        "host_v_L2com": host_v,
        "host_sigmav3d_L2com": host_sig,
        "host_N": host_N,
        "host_R": R_host,
        "host_id": _host_ids(host_x),
    }
    for src, dst in (
        ("density", "map_density"),
        ("vel_disp", "map_vel_disp"),
        ("rel_vel", "map_rel_vel"),
        ("rel_vel_healpix", "map_rel_vel_healpix"),
        ("imap", "map_imap"),
        ("shard_id", "map_shard_id"),
        ("control", "map_control"),
        ("length", "map_length"),
    ):
        if src in maps:
            arrays[dst] = np.asarray(maps[src])

    stem = format_match_stem(
        ztag=ztag, method=method, source="maplog", suffix="fullbox"
    )
    extra = {
        "sim_name": sim_name,
        "compaso_root": str(halo_root),
        "asdf_files": list(load_info.get("asdf_files", [])),
        "load_info": {k: v for k, v in load_info.items() if k != "asdf_files"},
        "epoch_step": load_info.get("epoch_step"),
        "epoch_z": load_info.get("epoch_z"),
        "slab_paths": [str(p) for p in slab_paths],
        "n_halo": int(len(halo_R)),
        "n_unique_hosts_matched": int(np.unique(local_out[matched]).size)
        if n_matched
        else 0,
    }
    provenance = MatchProvenance(
        source="maplog_timeslice",
        halo_R_field=halo_R_field,
        match_cut=match_cut,
        matcher=method,
        catalog_tag=match_catalog_tag(method),
        knn_k=int(knn_k) if method == "knn" else None,
        overlap=overlap if method == "ball" else None,
        box=box,
        z=ztag,
        z_maplog=float(z_maplog),
        sim_root=str(spec.root),
        halo_root=str(halo_root),
        maplog_known_duplicates=bool(spec.maplog_known_duplicates),
        extra=extra,
    )
    npz_path = catdir / f"{stem}.npz"
    meta_path = metadir / f"{stem}.json"
    npz_path, meta_path = save_match_catalog(
        arrays, npz_path, provenance, meta_path=meta_path
    )
    print(f"  wrote {npz_path}", flush=True)
    print(f"  wrote {meta_path}", flush=True)
    print(
        f"  n_MAP={n_map:,} matched={n_matched:,} "
        f"cen={int(is_cen.sum()):,} sat={int(is_sat.sum()):,}",
        flush=True,
    )
    return npz_path, meta_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim", default="aurora_highbase", help="sim() registry name")
    p.add_argument("--z", type=float, required=True, help="catalog / TimeSlice redshift")
    p.add_argument("--z-maplog", type=float, default=0.25, help="maplog file tag")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--method", choices=("ball", "knn", "brute"), default="ball")
    p.add_argument(
        "--overlap",
        choices=("most_massive", "deepest", "nearest"),
        default="most_massive",
    )
    p.add_argument("--halo-R-field", type=str, default=None)
    p.add_argument("--knn-k", type=int, default=128)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--maps-cache", type=Path, default=None)
    p.add_argument("--cache-maps", type=Path, default=None)
    args = p.parse_args(argv)
    build_catalog(
        sim_name=args.sim,
        z=args.z,
        z_maplog=args.z_maplog,
        outdir=args.outdir,
        method=args.method,
        overlap=args.overlap,
        halo_R_field=args.halo_R_field,
        knn_k=args.knn_k,
        delta=args.delta,
        maps_cache=args.maps_cache,
        cache_maps=args.cache_maps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
