"""Optional-dependency shims.

Numba is a hard requirement for comfortable full-box runs but not for importing
the package, so every numba kernel here is paired with a vectorised numpy
fallback. ``HAVE_NUMBA`` lets callers report which path they took.
"""

from __future__ import annotations

__all__ = ["HAVE_NUMBA", "njit", "prange"]

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    prange = range

    def njit(*args, **kwargs):
        """No-op stand-in for ``numba.njit`` when numba is unavailable."""

        def decorate(func):
            return func

        # Support both @njit and @njit(...) spellings.
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return decorate
