# Migration status

Strategy: **library-first with shims.** Shared code moves into
`aurora_analysis`; the original module keeps the old public name as a
re-export, so existing scripts and running campaigns keep working unchanged.

## Baseline

Measured across `InitialConditions/aurora_maplogs` and
`InitialConditions/hackAurora` before the migration:

| | |
|---|---|
| Python files | 126 |
| Lines of code | ~49,300 |
| Files mutating `sys.path` | 72 (one had five separate insertions) |
| Files with hardcoded `/lus/flare` or `/home/helenshao` | 54 |
| Separate `main()` functions | 95 |

## Round 1 — done

### Extracted

| Now in | Came from | Notes |
|---|---|---|
| `match.geometry.minimum_image_delta` / `periodic_distance` | 14 copies: `_periodic_delta` ×8, `_periodic_dist` ×4, `periodic_distance`, a nested `wrap` | All computed the same thing |
| `match.knn.match_maps_knn` | `aurora_maplogs/parent_cen_sat_props.py` | Was buried in a 30 kB plotting script and reached by three trees via `sys.path` |
| `match.brute.match_maps_brute` | `acj/match_maps_to_tng.compare_pos`, `elg_tng_cen_sat.compare_pos` | Now the tested reference for the KD-tree matcher |
| `match.censat.label_cen_sat` | `aurora_maplogs/elg_tng_cen_sat.py` | Vectorized; see behaviour change below |
| `env.kernel.kernel_weights` | `map_kde_radius_scan.py` + `merge_validation_compare.py` | Two near-verbatim copies |
| `env.kde.map_env_from_neighbors` | same two files, plus a third variant in `test5_alt_vel._kde_map_velocity` | |
| `plotting.style` | `plot_style.py`, `plot_params_fresco.py`, `plot_params.py`, `paths.setup_fresco()` | Four parallel entry points |
| `external.acj` | `sys.path` hacks in ~10 files | ACJ stays independently maintained |

### Shimmed (old names still work)

| File | Change |
|---|---|
| `aurora_maplogs/parent_cen_sat_props.py` | `match_maps_knn` re-exported; dropped the now-unused `numba` and `cKDTree` imports |
| `aurora_maplogs/elg_tng_cen_sat.py` | `compare_pos`, `label_cen_sat`, `_periodic_dist` re-exported |
| `compaso_preflight/match_maps_to_compaso.py` | Imports the matcher, labeller and distance from the package. **Side benefit:** running a match no longer imports matplotlib or `illustris_python`, which it previously pulled in transitively through two plotting scripts |
| `hackAurora/map_kde_radius_scan.py` | Kernel and env estimator re-exported |
| `hackAurora/merge_validation_compare.py` | Same, plus `minimum_image_delta` kept as a thin wrapper (see below) |

All five still import cleanly and `match_maps_knn` / `label_cen_sat` /
`compare_pos` resolve to the package implementations.

### Behaviour changes — read these

**`label_cen_sat` tie-breaking.** The original sorted with an unstable
`np.argsort` and then took `np.argmax` within each halo group, so when two
members of a halo had *identical* multiplicity, which one was tagged central
could vary between runs on the same input. The new implementation uses
`np.lexsort` with the entry index as the final key, so ties break on the lowest
index, deterministically. Results are identical whenever multiplicities within
a halo are distinct, which `test_censat.py` pins against the original loop. It
is also vectorized — the old Python loop over halo groups cost minutes on the
83 M matched MAPs of a full-box highbase match.

**`minimum_image_delta` argument order.** `merge_validation_compare` defined it
as `(pos_from, pos_to) -> pos_to - pos_from`, the reverse of the package's
`(a, b) -> a - b`. The old wrapper is preserved with its original order rather
than aliased, so existing callers are unaffected. Every current caller uses
only the norm, so the sign never actually mattered — but the wrapper avoids a
silent sign flip if that changes.

**Non-positive halo radii are now rejected.** `match_maps_knn` raises rather
than dividing by zero. `match_map_arrays_to_compaso` already repaired bad radii
before calling, so it is unaffected; other callers that were silently relying
on the old behaviour will now fail loudly.

## Round 2 — done (`match.catalog`)

The match ``.npz`` is the seam between matching and ~29 plot scripts. Its
keys used to be defined by README prose, and four writers each invented a
slightly different alias for the host aperture.

| Now in | Came from | Notes |
|---|---|---|
| `match.catalog.REQUIRED_ARRAY_KEYS` | `ye_repro/README.md` "Required match npz keys" + `ye_halo_props.REQUIRED_MATCH_KEYS` | One list, tested |
| `match.catalog.MatchProvenance` | sidecar JSON written ad hoc by each matcher | Includes `halo_R_field` and `maplog_known_duplicates` |
| `match.catalog.save_match_catalog` | `np.savez_compressed` + `json.dumps` in four writers | Writes `host_r100_L2com` **only** when the aperture really is `r100_L2com` |
| `match.catalog.load_match_catalog` | `np.load` + sibling-meta guess in `ye_halo_props` | Warn-and-infer by default; `strict=True` for new production |

### Behaviour change — the r100 lie

`ye_halo_props` used to do::

    if "host_r100_L2com" in raw: use it
    elif "host_R" in raw: call it host_r100_L2com anyway

That is how a catalog matched on ``r98_L2com`` (the maplog summary.pdf path)
could be plotted as if the aperture were ``r100``. The new reader keeps
``host_R`` as the match aperture and only aliases ``host_r100_L2com`` when
provenance says the field is ``r100_L2com``. Existing Flare catalogs without
a sidecar get a warning, not a rejection.

`ye_halo_props.load_match_catalog` is now a thin wrapper: schema load, then
the Ye-specific derived columns (``m_host``, ``shmr``, ``alpha_L2``).

### Still to do in this area

The four writers (`match_maps_to_compaso.py`, `match_veldir_fullbox.py`,
`match_maplog_fullbox.py`, `match_maplog_slabs.py`) still call
``np.savez_compressed`` themselves. Switching them to
``save_match_catalog`` is a one-function change per file and can wait until
the current ye_repro campaign is not mid-run.

## Round 2.5 — Hanyu ball matcher as default

| Now in | Came from | Notes |
|---|---|---|
| `match.ball` | `hackAurora/AuroraVeldir/matching-hanyu.py` | Containment-first; `overlap` in `{most_massive, deepest, nearest}` |
| `match_maps(..., method="ball")` | — | **New default.** Unmatched → ``-1``. Requires ``halo_mass`` when ``overlap="most_massive"`` |
| `match_maps(..., method="knn")` | previous default | Kept for ACJ / TNG ``(d/R)^2`` continuity |

`compaso_preflight/match_maps_to_compaso.py` now defaults to ball /
most_massive and accepts ``--method knn`` to restore the old path. Meta JSON
records ``method``, ``overlap``, and ``catalog_tag``. Ball catalogs also store
``halo_n_enclosed`` (geometric, overlaps allowed) alongside
``halo_n_assigned``.

**Catalog filename tag:** sphere matches embed ``sph`` in the stem so they
cannot be confused with legacy kNN files::

    match_sph_z0.300_rank000.npz
    match_maplog_sph_z0.300_fullbox.npz
    match_maplog_sph_z0.300_slab003.npz
    match_knn_z0.300_fullbox.npz          # explicit when --method knn

Existing Flare match npz files were built with knn and usually have **no**
method tag; re-run to pick up ``sph``. Do not mix knn and sph catalogs in one
Ye comparison without labeling them.

## Round 3 — next

`io/` is the highest-value remaining target, because it is where the operational
knowledge currently lives only in prose.

1. **`io.asdf`** — eager `asdf.open(lazy_load=False)` helpers with a
   compute-node warning. The fast path and the slow path are currently chosen
   ad hoc per script; `maplog_io.decode_maplog_shard` still hardcodes
   `lazy_load=True`, which is the slow path on Aurora.
2. **`io.maplog`** — merge `maplog_io.py` and `maplog_timeslice_load.py` into
   one reader with two documented modes: a control scan over
   `MapLogSegmentStarts` unpacking only TimeSlice rows, versus a full-field
   unpack for tree work.
3. **`io.subsample`** and **`io.compaso`** — from `subsample_io.py` and
   `compaso_io.py`.
4. **`epochs`** — `step ↔ z` and the `control = 4*step + node_type` convention,
   currently re-derived in several places.
5. **`match.catalog`** — a versioned schema for the match `.npz`, with a writer
   that stamps provenance (sim root, redshift, aperture field, matcher,
   `maplog_known_duplicates`) and a reader that validates required keys.

Then `tree/` (from `map_tree_tags.py`), `stats/` (abundance, ξ, P(k), running
medians), and finally the per-campaign scripts.

## Known duplicates not yet addressed

- `hackAurora/aurora_maplogs/utilities.py` is **byte-identical** to
  `aurora_maplogs/utilities_legacy.py` (38,827 bytes each).
- `summit_pk_monopole.py` and `summit_subsample_io.py` are duplicated verbatim
  between `hackAurora/` and `hackAurora/summit_pk_bundle/`.
- `_running_median` ×3, `maybe_downsample` ×6, `_finite` ×6, `redshift_label` /
  `_z_tag` / `z_tag` ×6 across trees.
- 95 `main()` functions re-declaring the same `--z`, `--sim-root`, `--outdir`,
  `--compaso-root` flags; these want one shared argparse parent.

Deleting the identical copies needs your say-so, so they are listed rather than
removed.
