"""FRESCO plot style.

The rcParams themselves live in the hackAurora tree and are loaded through
:mod:`aurora_analysis.external.hackaurora`, so there is exactly one definition
of the style rather than the four parallel setups that had accumulated
(``plot_style.py``, ``plot_params_fresco.py``, ``plot_params.py``, and a
``setup_fresco()`` helper in a per-campaign ``paths.py``).

Typical use at the top of a figure script::

    from aurora_analysis.plotting import use_style, palette
    use_style()               # FRESCO unless $PLOT_STYLE says otherwise
    C = palette()
    ax.plot(x, y, color=C["blue"])
"""

from __future__ import annotations

import os
from types import ModuleType

from ..external import hackaurora

__all__ = ["grid", "line_effects", "palette", "style_module", "use_style"]

DEFAULT_STYLE = "fresco"


def _resolve(name: str | None) -> str:
    return (name or os.environ.get("PLOT_STYLE") or DEFAULT_STYLE).lower()


def style_module(name: str | None = None) -> ModuleType:
    """The rcParams module backing a named style."""
    return hackaurora.params_module(_resolve(name))


def use_style(name: str | None = None) -> ModuleType:
    """Apply a style's rcParams to the active matplotlib session.

    Resets to matplotlib defaults first, so switching styles within one process
    does not leave the previous style's settings partially in place.
    """
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    mod = style_module(name)
    mod.apply_plot_params()
    return mod


def palette(name: str | None = None) -> dict:
    """Named colours for the style, e.g. ``palette()["orange"]``."""
    return getattr(style_module(name), "PALETTE", {})


def line_effects(lw: float | None = None):
    """FRESCO line path effects (the subtle outline on overlapping curves)."""
    mod = style_module("fresco")
    fn = getattr(mod, "line_path_effects", None)
    if fn is None:
        return None
    return fn() if lw is None else fn(lw=lw)


def grid(ax, *, faint: bool = True) -> None:
    """Diagnostic gridlines.

    FRESCO figures are gridless by default; this is for working plots. ``faint``
    uses the style's own low-alpha major/minor grid, otherwise a plain grey
    major grid at alpha 0.5 (the ``_gray_grid`` look used across the older
    scripts).
    """
    if faint:
        fn = getattr(style_module("fresco"), "fresco_grid", None)
        if fn is not None:
            fn(ax)
            return
    ax.grid(True, which="major", color="gray", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
