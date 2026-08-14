"""Adapter for ACJ's independently maintained analysis modules.

``abacus/acj/`` is a directory of loose scripts with no ``__init__.py`` and no
packaging metadata, and it is maintained outside this repo. It was previously
imported two incompatible ways depending on which ``sys.path`` hack ran first::

    from acj.utilities import get_xi          # needs abacus/ on the path
    from utilities import get_xi              # needs abacus/acj/ on the path

Both now route through here. Attributes are resolved lazily, because
``acj/utilities.py`` is ~122 kB and pulls in matplotlib, pycorr and the
Illustris helpers at import time -- too expensive to pay for on every
``import aurora_analysis``.

Usage
-----
>>> from aurora_analysis.external import acj
>>> xi = acj.get_xi(pos, edges, box)            # doctest: +SKIP

Point at a different checkout with ``$AURORA_ACJ_ROOT``.
"""

from __future__ import annotations

from functools import lru_cache
from types import ModuleType
from typing import Any

from ..config import path
from ._locate import load_module, require_tree

__all__ = ["match_maps_to_tng", "root", "utilities"]

#: Names re-exported from ``acj/utilities.py``. Anything not listed is still
#: reachable via ``acj.utilities.<name>``; this tuple documents the surface the
#: rest of aurora_analysis actually depends on, so an upstream rename shows up
#: as one failing test instead of a scattered breakage.
UTILITIES_SURFACE = (
    "get_xi",
    "get_threshold",
    "apply_zspace_poles",
    "read_maplog_dir",
    "read_subsample_dir",
    "H",
    "get_colors",
    "get_sci_not",
    "test_maplog",
)

#: Names re-exported from ``acj/match_maps_to_tng.py``.
MATCH_SURFACE = (
    "compare_pos",
    "make_HOD_plot",
    "load_map_data",
    "halowise_map_galaxies",
)


def root():
    """Validated location of the ACJ source tree."""
    return require_tree(
        path("acj_root"),
        what="acj",
        env_var="AURORA_ACJ_ROOT",
        expect="utilities.py",
    )


@lru_cache(maxsize=None)
def _module(stem: str) -> ModuleType:
    tree = root()
    return load_module(tree / f"{stem}.py", name=f"_acj_{stem}", deps=tree)


def utilities() -> ModuleType:
    """The ``acj.utilities`` module object."""
    return _module("utilities")


def match_maps_to_tng() -> ModuleType:
    """The ``acj.match_maps_to_tng`` module object."""
    return _module("match_maps_to_tng")


def illustris_utilities() -> ModuleType:
    """The ``acj.illustris_utilities`` module object."""
    return _module("illustris_utilities")


_SOURCES = {
    **{name: "utilities" for name in UTILITIES_SURFACE},
    **{name: "match_maps_to_tng" for name in MATCH_SURFACE},
}


def __getattr__(name: str) -> Any:
    """Lazily forward ``acj.<name>`` to the module that defines it (PEP 562)."""
    stem = _SOURCES.get(name)
    if stem is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. "
            f"Reach anything else through acj.utilities().{name} "
            f"or acj.match_maps_to_tng().{name}"
        )
    return getattr(_module(stem), name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_SOURCES))
