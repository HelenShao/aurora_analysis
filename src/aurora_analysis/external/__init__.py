"""Adapters for source trees maintained outside this repository.

Currently:

- :mod:`aurora_analysis.external.acj` -- ACJ's ``abacus/acj`` scripts
- :mod:`aurora_analysis.external.hackaurora` -- the FRESCO plot-params modules
"""

from __future__ import annotations

from . import acj, hackaurora
from ._locate import load_module, require_tree, tree_on_syspath

__all__ = ["acj", "hackaurora", "load_module", "require_tree", "tree_on_syspath"]
