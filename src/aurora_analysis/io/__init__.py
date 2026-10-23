"""Readers for Abacus products.

``asdf``
    Eager-open helpers. On Aurora, ``asdf.open(..., lazy_load=False)`` is far
    faster than the tip ``read_asdf`` path for single-block files, and multi-GB
    opens belong on a compute node (roughly 18 s there versus 150 s on a UAN
    for one maplog shard).
``maplog``
    Two documented modes of one reader: a control scan over
    ``MapLogSegmentStarts`` unpacking only TimeSlice rows
    (``load_timeslice_maps``), versus a full-field unpack for tree work
    (``decode_maplog_shard``). Both use the eager asdf path.
``subsample``
    Timeslice subsample column / sampler / rank discovery and merged loads.
``compaso``
    ``CompaSOHaloCatalog(cleaned=True)`` plus the slab-overlap logic.
"""

from __future__ import annotations

from .asdf import (
    DEFAULT_LAZY_LOAD,
    is_likely_login_node,
    open_asdf,
    read_header,
    warn_if_login_node,
)
from .compaso import (
    COMPASO_FIELDS,
    asdf_header,
    compaso_slab_edges,
    default_compaso_root,
    load_compaso_overlap,
    load_timeslice_one_rank,
    nominal_rank_x_domain,
    overlapping_compaso_slab_indices,
)
from .maplog import (
    BASE_KEYS,
    BASE_KEYS_WITH_ENV,
    ENV_KEYS,
    MAPLOG_LOAD_FIELDS,
    MaplogFileMeta,
    decode_maplog_shard,
    discover_maplog_asdf_files,
    load_maplogs,
    load_timeslice_maps,
    load_timeslice_maps_streaming,
    read_maplog_asdf_header,
    resolve_step_to_z,
)
from .subsample import (
    load_merged_full_box,
    normalize_redshift,
)

__all__ = [
    "BASE_KEYS",
    "BASE_KEYS_WITH_ENV",
    "COMPASO_FIELDS",
    "DEFAULT_LAZY_LOAD",
    "ENV_KEYS",
    "MAPLOG_LOAD_FIELDS",
    "MaplogFileMeta",
    "asdf_header",
    "compaso_slab_edges",
    "decode_maplog_shard",
    "default_compaso_root",
    "discover_maplog_asdf_files",
    "is_likely_login_node",
    "load_compaso_overlap",
    "load_maplogs",
    "load_merged_full_box",
    "load_timeslice_maps",
    "load_timeslice_maps_streaming",
    "load_timeslice_one_rank",
    "nominal_rank_x_domain",
    "normalize_redshift",
    "open_asdf",
    "overlapping_compaso_slab_indices",
    "read_header",
    "read_maplog_asdf_header",
    "resolve_step_to_z",
    "warn_if_login_node",
]
