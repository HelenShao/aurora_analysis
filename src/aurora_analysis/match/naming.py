"""Catalog filename tags for match method.

Sphere (ball) matches are labeled ``sph`` in npz / meta stems so they are not
confused with legacy kNN catalogs that omit a method tag::

    match_sph_z0.300_fullbox.npz
    match_maplog_sph_z0.300_slab000.npz
    match_knn_z0.300_rank000.npz   # explicit when method=knn
"""

from __future__ import annotations

__all__ = [
    "METHOD_CATALOG_TAGS",
    "format_match_stem",
    "match_catalog_tag",
    "parse_match_catalog_tag",
]

#: method name → short label embedded in catalog filenames / meta.
METHOD_CATALOG_TAGS: dict[str, str] = {
    "ball": "sph",
    "knn": "knn",
    "brute": "brute",
}


def match_catalog_tag(method: str = "ball") -> str:
    """Short label for a match method (``sph`` for the default ball matcher)."""
    key = str(method).lower()
    try:
        return METHOD_CATALOG_TAGS[key]
    except KeyError:
        known = ", ".join(sorted(METHOD_CATALOG_TAGS))
        raise ValueError(f"unknown match method {method!r}; choose from {known}") from None


def format_match_stem(
    *,
    ztag: str,
    method: str = "ball",
    source: str = "subsample",
    suffix: str = "",
    tag: str | None = None,
) -> str:
    """Build a catalog stem with an explicit method tag.

    Parameters
    ----------
    ztag
        Redshift tag as already used on disk, e.g. ``"0.300"``.
    method
        ``"ball"`` → ``sph``, ``"knn"`` → ``knn``, …
    source
        ``"subsample"`` → stem starts with ``match_``;
        ``"maplog"`` / ``"maplog_timeslice"`` → ``match_maplog_``.
    suffix
        Trailing piece without a leading underscore, e.g. ``"fullbox"``,
        ``"rank000"``, ``"slab003"``, ``"col000_rank003"``.
    """
    t = tag if tag is not None else match_catalog_tag(method)
    src = str(source).lower()
    if src in ("maplog", "maplog_timeslice"):
        head = "match_maplog"
    else:
        head = "match"
    stem = f"{head}_{t}_z{ztag}"
    if suffix:
        stem = f"{stem}_{suffix.lstrip('_')}"
    return stem


def parse_match_catalog_tag(name: str) -> str | None:
    """Extract ``sph`` / ``knn`` / ``brute`` from a catalog filename, if present."""
    base = name.rsplit("/", 1)[-1]
    for tag in METHOD_CATALOG_TAGS.values():
        if f"_{tag}_z" in base or base.startswith(f"match_{tag}_z") or base.startswith(
            f"match_maplog_{tag}_z"
        ):
            return tag
    return None
