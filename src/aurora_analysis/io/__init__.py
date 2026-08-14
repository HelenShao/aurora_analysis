"""Readers for Abacus products. Not yet populated -- next extraction milestone.

Planned modules, and the code they consolidate:

``asdf``
    Eager-open helpers. On Aurora, ``asdf.open(..., lazy_load=False)`` is far
    faster than the tip ``read_asdf`` path for single-block files, and multi-GB
    opens belong on a compute node (roughly 18 s there versus 150 s on a UAN
    for one maplog shard). That guidance currently lives only in READMEs; it
    belongs in the loader.
``maplog``
    ``maplog_io.py`` plus ``maplog_timeslice_load.py``. Two loading strategies
    exist for good reason -- a control scan over ``MapLogSegmentStarts``
    followed by unpacking only TimeSlice rows, versus a full-field unpack for
    tree work -- and they should be two documented modes of one reader rather
    than two modules that drifted apart.
``subsample``
    ``subsample_io.py`` and the reader half of ``compaso_io.py``.
``compaso``
    ``CompaSOHaloCatalog(cleaned=True)`` plus the slab-overlap logic.
``tng``
    TNG subhalo and FoF loaders (currently split between ``elg_tng_cen_sat`` and
    ``acj.illustris_utilities``).
``state``
    ``abacus.par`` / ``log/step*.state`` parsing and step-to-redshift mapping.
"""
