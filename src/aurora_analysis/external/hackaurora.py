"""Adapter for the FRESCO plot-params modules in ``InitialConditions/hackAurora``.

``plot_params_fresco.py`` and its siblings are the canonical definition of the
FRESCO paper style, and they live in the hackAurora tree rather than here. This
adapter loads them on demand so :mod:`aurora_analysis.plotting.style` has a
single import site.

Vendoring these four small rcParams modules into the package is the natural
next step once this repo stands alone; at that point only this file changes.

Point at a different checkout with ``$AURORA_HACKAURORA_ROOT``.
"""

from __future__ import annotations

from functools import lru_cache
from types import ModuleType

from ..config import path
from ._locate import load_module, require_tree

__all__ = ["STYLE_MODULES", "params_module", "root"]

#: Style name -> module stem in the hackAurora tree.
STYLE_MODULES: dict[str, str] = {
    "fresco": "plot_params_fresco",
    "sm": "plot_params_sm",
    "legacy": "plot_params",
    "lucia": "plot_params_lucia",
}


def root():
    """Validated location of the hackAurora tree."""
    return require_tree(
        path("hackaurora_root"),
        what="hackAurora",
        env_var="AURORA_HACKAURORA_ROOT",
        expect="plot_params_fresco.py",
    )


@lru_cache(maxsize=None)
def params_module(style: str = "fresco") -> ModuleType:
    """Load the rcParams module for a named style."""
    key = style.lower()
    try:
        stem = STYLE_MODULES[key]
    except KeyError:
        raise ValueError(
            f"unknown style {style!r}; choose from {', '.join(sorted(STYLE_MODULES))}"
        ) from None
    tree = root()
    return load_module(tree / f"{stem}.py", name=f"_hackaurora_{stem}", deps=tree)
