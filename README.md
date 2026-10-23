# aurora_analysis

Analysis toolkit for Abacus **MAPs** (Massless Aggregate Particles — the galaxy
proxies Abacus tracks on the fly), covering maplog and subsample I/O, matching
MAPs to halo catalogs, KDE environments, and clustering statistics.

This package consolidates code that had grown across three separate trees
(`InitialConditions/aurora_maplogs`, `InitialConditions/hackAurora`, and
`abacus/acj`) into one versioned, testable library. Campaign-specific figure
scripts stay where they are and import from here.

## Install

```bash
git clone <this repo>
cd aurora_analysis
pip install -e '.[all]'
```

On Aurora, `module load frameworks` first. `numba` is optional — every
accelerated kernel has a vectorized numpy fallback — but it is strongly
recommended for full-box work.

## Quickstart

```python
from aurora_analysis.config import sim, path
from aurora_analysis.match import match_maps, label_cen_sat, periodic_distance
from aurora_analysis.env import map_env_from_neighbors, kernel_weights
from aurora_analysis.io.maplog import load_timeslice_maps
from aurora_analysis.epochs import timeslice_control

spec = sim("highbase_veldir_v2")          # locked paths, box, seed, aperture
box  = spec.box

# maps, info = load_timeslice_maps(spec.root, z_maplog=0.25, z_cat=0.3)
# TimeSlice rows are control == timeslice_control(info["epoch_step"])

# Default: Hanyu ball query, most-massive overlap (CompaSO). Unmatched → -1.
# Pass *all* MAPs at z_cat here — before ELG/LRG/ram cuts (Stage A).
local_ih = match_maps(map_pos, halo_pos, halo_R, box, halo_mass=halo_N)
matched  = local_ih >= 0
is_cen   = label_cen_sat(map_mult[matched], local_ih[matched])

# Legacy ACJ path (always assigns a host; apply d < R yourself):
# local_ih = match_maps(map_pos, halo_pos, halo_R, box, method="knn", k=128)
```

See **Matching MAPs to halos (correct order)** below for Stage A → join
onto selected tracers (do not rematch ELGs alone for HOD).

Build a versioned match catalog (maplog TimeSlice → CompaSO) with the library
matcher + ``save_match_catalog``:

```bash
python3 -m aurora_analysis.cli.build_match_catalog \
  --sim aurora_highbase --z 0.25 --z-maplog 0.25 \
  --outdir /lus/flare/projects/Abacus/helenshao/load/ye_repro_AHB_z0.25 \
  2>&1 | tee logs/build_match.log
```

## Layout

| Module | Contents |
|---|---|
| `config` | Named paths and simulations. **The only module that knows about `/lus/flare`.** |
| `match.geometry` | Minimum-image delta, periodic distance, box wrapping, 27-image tiling |
| `match.knn` | Production KD-tree matcher, minimum `(d/R)²` over `k` candidates (ACJ-compatible; not the default) |
| `match.ball` | Hanyu containment query; **default** via `match_maps(..., method="ball")` with `overlap="most_massive"`. Catalogs labeled **`sph`** in the filename. |
| `match.brute` | Exhaustive reference matcher (ACJ's `compare_pos`) |
| `match.naming` | `format_match_stem` / `match_catalog_tag` — `sph` for ball, `knn` for kNN |
| `match.censat` | Central/satellite labelling |
| `env.kernel` | Hollow and filled KDE kernels, normalization, radius rescaling |
| `env.kde` | Density, velocity dispersion, relative velocity, mean DM velocity |
| `plotting.style` | FRESCO style entry points |
| `external.acj` | Adapter for ACJ's separately maintained modules |
| `external.hackaurora` | Adapter for the FRESCO rcParams modules |
| `io.asdf` | Eager `asdf.open(lazy_load=False)` + UAN warning; YAML header sniff |
| `io.maplog` | TimeSlice control-scan and full-shard unpack (two modes of one reader) |
| `io.subsample` | Timeslice subsample column / sampler / rank discovery |
| `io.compaso` | Cleaned CompaSO + x-slab overlap |
| `epochs` | `control = 4*step + node_type` and step ↔ z |
| `stats` | Placeholder for the next extraction milestone |

## Configuration

Every machine-specific path resolves in this order: environment variable, then
user config file, then the built-in Aurora/Flare defaults.

```bash
export AURORA_COMPASO_ROOT=/path/to/AbacusSummit_highbase_c000_ph100
export AURORA_ACJ_ROOT=/path/to/abacus/acj
```

Or `~/.config/aurora_analysis/paths.toml`:

```toml
[paths]
compaso_root = "/path/to/AbacusSummit_highbase_c000_ph100"
tng_base     = "/path/to/TNG300-2/output"
```

Inspect what resolved with `python -c "from aurora_analysis.config import paths; print(paths())"`.

## Matching MAPs to halos (correct order)

**Match the full MAP catalog at \(z_{\rm cat}\) to cleaned AbacusSummit
CompaSO *before* any target-specific cuts** (ELG mass window, accretion,
ram-pressure, LRG cuts, …). Then join host tags onto whatever subsample you
select later. Rematching only the ELG (or other tracer) sample is deprecated
for host-primary cen/sat and HOD.

### Why full catalog first

Halo primary = highest-`mult` MAP **among all matched MAPs in that host**, not
among ELGs only. If Stage A skips MAPs outside the ELG mass window, a
lower-mass ELG can be wrongly tagged central and LRG HOD centrals
(high `mult`) disappear. Selection cuts come **after** the match table exists.

```text
maplog TimeSlice @ z_cat
        │
        ▼
 cleaned CompaSO @ z_halo   ──Stage A──►  all-MAP match table
   (ball, r100_L2com)                     (maps, matched, local_halo_ih,
                                           is_halo_primary, M_host, R_host, …)
        │
        │  join by maps id  (Stage D)
        ▼
 ELG / LRG / ram / … selection  ──►  enriched inclusive + host-primary HOD
```

### Stage A — match aperture and rule

| Setting | Production default |
|---|---|
| Halos | Cleaned CompaSO at `z_halo` (same epoch as `z_cat` when dumps align) |
| Center | `x_L2com` |
| Aperture | `r100_L2com` — MAP matched if periodic \(d < R\) |
| Overlap | `most_massive` (when several hosts claim a MAP) |
| Matcher | `match_maps(..., method="ball", overlap="most_massive")` |
| Primary | Highest `mult` per host among **matched full-catalog** rows; ties → lower row index |

`r100_L2com` is only the **host membership radius**. It is not \(R_{\rm gal}\)
and not an HOD bin edge. Stored on matched rows as `R_host` /
`host_R`; `r_over_R = d / R_host`.

Library entry points:

```python
from aurora_analysis.match import match_maps, label_cen_sat

# Stage A inputs: ALL MAPs at z_cat (no ELG cut yet)
local_ih = match_maps(
    map_pos, halo_pos, halo_R, box,
    halo_mass=halo_N, method="ball", overlap="most_massive",
)
matched = local_ih >= 0
is_halo_primary = np.zeros(len(map_pos), dtype=bool)
is_halo_primary[matched] = label_cen_sat(
    map_mult[matched], local_ih[matched],
)
```

CLI (versioned match catalog):

```bash
python3 -m aurora_analysis.cli.build_match_catalog \
  --sim aurora_highbase --z <z_cat> --z-maplog <z_maplog> \
  --outdir /path/to/match_out \
  2>&1 | tee logs/build_match.log
```

Campaign wrapper used for HighBase ELG HOD (writes
`allmaps_compaso_cen_sat.npz`):
`InitialConditions/aurora_maplogs/elg_highbase_compaso_cen_sat.py`
with Stage A / `--full-catalog`. Reference product:
`catalog_runs/elg_dmult_AuroraHighBase_z1p10_v1/catalogs/allmaps_compaso_cen_sat.npz`.

### Stage D — tag selected samples (no rematch)

Join Stage A onto parent / inclusive / LRG catalogs **by `maps` id**:

```text
is_central_elg   = matched & is_halo_primary
is_satellite_elg = matched & ~is_halo_primary
```

Unmatched selected MAPs stay unmatched; do not invent cen/sat via an
ELG-only rematch. Sample-internal `label_cen_sat` on the tracer alone is
diagnostics only.

### Do / do not

| Do | Do not |
|---|---|
| Stage A once per `(box, z_cat, CompaSO epoch)` | Rematch ELGs ball-by-ball for HOD |
| Join Stage A into every later selection | Define halo primary inside the ELG mass window only |
| Use Stage A primaries for host-primary HOD | Treat parent-only `allmaps_*.npz` as sufficient for LRG HOD |

## Two things the library encodes that used to live only in READMEs

**Matching is not nearest-neighbour.** Both matchers minimize periodic
`(d/R)²`, so a massive halo slightly further away can win over a closer small
one. That is why the KD-tree query returns `k=128` candidates and re-ranks
them: the tree orders by raw distance, the metric does not. `test_matchers.py`
asserts the two matchers agree exactly and that the result genuinely differs
from plain nearest-halo.

**The maplog filename tag is not the catalog epoch.** TimeSlice selection is
`control == 4*step + 3`, with `step` from `Step_to_Z` nearest to `z_cat`.
`--z-maplog 0.25` only names `maplog_z0.25_*.asdf`; the epoch inside may be
`z=0.3`. `test_epochs.py` pins the packing; `test_io_maplog.py` pins the
segment scan.

**Kernel shape is a scientific choice, not a default.** The hollow kernel
`w = s²(1-s²)` maximizes environmental contrast and is what production uses;
the filled kernel `w = 1-s²` is the better estimator of a halo's mean
velocity. They have different normalizations at the same radius
(`8πb³/35` vs `8πb³/15`), so an `inv_kde` value is only valid for the shape and
radius it was derived at. `env.kernel.inv_kde_for_radius` handles the radius
part; switching shape needs a fresh value.

## Running on Aurora

Multi-GB ASDF opens belong on a **compute node**, not a login/UAN node — one
maplog shard opens in roughly 18 s on compute versus 150 s on a UAN. Meson and
ninja builds also need a compute node, for the unrelated reason that login
nodes lack `libxpmem`.

```bash
qsub -I -l select=1 -l walltime=2:00:00 -q debug-scaling -A Abacus
module load frameworks
export NUMEXPR_MAX_THREADS=256 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
```

Pipe long runs through `tee` so epoch resolution, match rates and wall times
become permanent records:

```bash
python3 -u your_script.py 2>&1 | tee logs/run_z0.300.log
```

## Tests

```bash
pytest
```

The suite is deliberately property-based rather than golden-file: periodic
distance is checked for antisymmetry, boundary wrapping and translation
invariance; kernel normalizations are checked against numerical quadrature; the
KDE recovers the density and dispersion of a synthetic field it was handed; and
the fast matcher and labeller are pinned against the slow reference
implementations they replace.

## Status

See [MIGRATION.md](MIGRATION.md) for what has moved, what still has duplicates,
and what is next.
