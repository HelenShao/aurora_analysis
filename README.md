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

spec = sim("highbase_veldir_v2")          # locked paths, box, seed, aperture
box  = spec.box

# Default: Hanyu ball query, most-massive overlap (CompaSO). Unmatched → -1.
local_ih = match_maps(map_pos, halo_pos, halo_R, box, halo_mass=halo_N)
matched  = local_ih >= 0
is_cen   = label_cen_sat(map_mult[matched], local_ih[matched])

# Legacy ACJ path (always assigns a host; apply d < R yourself):
# local_ih = match_maps(map_pos, halo_pos, halo_R, box, method="knn", k=128)
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
| `io`, `stats` | Placeholders documenting the next extraction milestones |

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

## Two things the library encodes that used to live only in READMEs

**Matching is not nearest-neighbour.** Both matchers minimize periodic
`(d/R)²`, so a massive halo slightly further away can win over a closer small
one. That is why the KD-tree query returns `k=128` candidates and re-ranks
them: the tree orders by raw distance, the metric does not. `test_matchers.py`
asserts the two matchers agree exactly and that the result genuinely differs
from plain nearest-halo.

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
