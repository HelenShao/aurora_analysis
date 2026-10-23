"""Analysis toolkit for Abacus MAP galaxy proxies.

Subpackages
-----------
:mod:`aurora_analysis.config`
    Named paths and simulations. The only module that knows about ``/lus/flare``.
:mod:`aurora_analysis.match`
    Periodic geometry, KD-tree and brute-force halo matching, central/satellite
    labelling.
:mod:`aurora_analysis.env`
    KDE kernels and MAP environment estimates.
:mod:`aurora_analysis.plotting`
    FRESCO style entry points.
:mod:`aurora_analysis.external`
    Adapters for source trees maintained elsewhere (``acj``, hackAurora
    plot params).
:mod:`aurora_analysis.io`
    ASDF, maplog, subsample and CompaSO readers.
:mod:`aurora_analysis.epochs`
    MapNode ``control`` packing and step ↔ redshift.
:mod:`aurora_analysis.cli`
    Catalog builders (``python -m aurora_analysis.cli.build_match_catalog``).

Submodules are imported lazily so that ``import aurora_analysis`` stays cheap
and does not drag in matplotlib or scipy until something needs them.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_SUBMODULES = ("cli", "config", "env", "epochs", "external", "io", "match", "plotting")

__all__ = ["__version__", *_SUBMODULES]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
