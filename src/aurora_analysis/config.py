"""Path and simulation registry.

This is the only module in the package that knows about machine-specific
locations such as ``/lus/flare/projects/Abacus``. Everything else asks for a
path by name, so the package runs off Aurora without edits.

Resolution order for every entry, first hit wins:

1. environment variable (e.g. ``AURORA_COMPASO_ROOT``)
2. user config file -- ``$AURORA_ANALYSIS_CONFIG``, else
   ``~/.config/aurora_analysis/paths.toml`` (or ``.json``)
3. the built-in Aurora/Flare defaults below

Example
-------
>>> from aurora_analysis.config import path, sim
>>> path("compaso_root")                      # doctest: +SKIP
PosixPath('/lus/flare/projects/Abacus/helenshao/AbacusSummit/...')
>>> sim("highbase_veldir_v2").root            # doctest: +SKIP
PosixPath('/lus/flare/projects/Abacus/eisenste/Keepers/HighBase_veldir_eps015_v2')
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

try:  # Python >= 3.11
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover
    _tomllib = None


__all__ = [
    "DEFAULT_PATHS",
    "SIMS",
    "SimSpec",
    "config_file",
    "path",
    "paths",
    "register_sim",
    "sim",
    "sim_names",
]


# --------------------------------------------------------------------------
# Named locations
# --------------------------------------------------------------------------

_FLARE = "/lus/flare/projects/Abacus"

#: Built-in defaults for the Aurora/Flare deployment. Override with the
#: ``AURORA_<NAME>`` environment variable or a user config file.
DEFAULT_PATHS: dict[str, str] = {
    # Filesystem roots
    "flare_root": _FLARE,
    "keepers_root": f"{_FLARE}/eisenste/Keepers",
    "work_root": f"{_FLARE}/helenshao/load",
    # Halo catalogs
    "compaso_root": f"{_FLARE}/helenshao/AbacusSummit/AbacusSummit_highbase_c000_ph100",
    "cleaning_root": f"{_FLARE}/helenshao/AbacusSummit/cleaning",
    # IllustrisTNG
    "tng_base": f"{_FLARE}/ajohnson/TNG300/TNG300-2/output",
    # External source trees wrapped by aurora_analysis.external
    "acj_root": str(Path.home() / "InitialConditions" / "abacus" / "acj"),
    "hackaurora_root": str(Path.home() / "InitialConditions" / "hackAurora"),
}


def config_file() -> Path | None:
    """Return the user config file that will be read, or None if there is none."""
    explicit = os.environ.get("AURORA_ANALYSIS_CONFIG")
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    base = Path.home() / ".config" / "aurora_analysis"
    for name in ("paths.toml", "paths.json"):
        p = base / name
        if p.is_file():
            return p
    return None


def _read_config_file() -> dict[str, str]:
    p = config_file()
    if p is None:
        return {}
    if p.suffix == ".json":
        data = json.loads(p.read_text())
    else:
        if _tomllib is None:  # pragma: no cover
            raise RuntimeError(
                f"cannot read {p}: tomllib needs Python >= 3.11; use paths.json instead"
            )
        data = _tomllib.loads(p.read_text())
    entries = data.get("paths", data)
    return {str(k): str(v) for k, v in entries.items()}


def _env_var(name: str) -> str:
    return "AURORA_" + name.upper()


@lru_cache(maxsize=1)
def paths() -> dict[str, Path]:
    """Resolved path registry (cached). Call :func:`paths.cache_clear` to reload."""
    resolved = dict(DEFAULT_PATHS)
    resolved.update(_read_config_file())
    for name in list(resolved):
        env = os.environ.get(_env_var(name))
        if env:
            resolved[name] = env
    # Environment variables may introduce names absent from the defaults.
    for key, value in os.environ.items():
        if key.startswith("AURORA_") and key.endswith("_ROOT"):
            resolved.setdefault(key[len("AURORA_") :].lower(), value)
    return {k: Path(v).expanduser() for k, v in resolved.items()}


def path(name: str, *, must_exist: bool = False) -> Path:
    """Look up a named path.

    Raises KeyError with the override instructions rather than a bare miss, so
    a wrong deployment is self-explanatory in a traceback.
    """
    table = paths()
    try:
        p = table[name]
    except KeyError:
        known = ", ".join(sorted(table))
        raise KeyError(
            f"unknown path {name!r}. Known: {known}. "
            f"Add it via ${_env_var(name)} or {config_file() or '~/.config/aurora_analysis/paths.toml'}"
        ) from None
    if must_exist and not p.exists():
        raise FileNotFoundError(
            f"path {name!r} resolves to {p}, which does not exist. "
            f"Override with ${_env_var(name)}."
        )
    return p


# --------------------------------------------------------------------------
# Simulation registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SimSpec:
    """A named simulation and the analysis defaults that travel with it.

    ``box`` is in Mpc/h and ``np_side`` is the cube root of the particle count.
    Values here are conveniences; readers still prefer the on-disk ASDF header
    or ``abacus.par`` when one is available.
    """

    name: str
    root: Path
    box: float | None = None
    np_side: int | None = None
    zd_seed: int | None = None
    kind: str = "abacus"  # abacus | compaso | tng
    #: Default match aperture for this sim, e.g. "r100_L2com".
    halo_R_field: str | None = None
    #: True when maplog output is known to contain duplicate rows
    #: (Z-boundary write bug); match metadata records this.
    maplog_known_duplicates: bool = False
    notes: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def with_root(self, root: str | Path) -> SimSpec:
        return replace(self, root=Path(root))


def _keepers(name: str) -> Path:
    return path("keepers_root") / name


SIMS: dict[str, SimSpec] = {}


def register_sim(spec: SimSpec) -> SimSpec:
    """Add or replace an entry in the simulation registry."""
    SIMS[spec.name] = spec
    return spec


def _register_defaults() -> None:
    if SIMS:
        return
    register_sim(
        SimSpec(
            name="highbase_veldir_v2",
            root=_keepers("HighBase_veldir_eps015_v2"),
            box=1000.0,
            np_side=3456,
            zd_seed=22321,
            halo_R_field="r100_L2com",
            notes="Subsample-based matching. Preferred over the AbS maplog dump.",
        )
    )
    register_sim(
        SimSpec(
            name="highbase_veldir_abs",
            root=_keepers("HighBase_veldir_eps015_AbS"),
            box=1000.0,
            np_side=3456,
            zd_seed=22321,
            halo_R_field="r98_L2com",
            maplog_known_duplicates=True,
            notes=(
                "Maplog TimeSlice nodes carry Z-boundary duplicates "
                "(~10% redundant pairs). Subsample files are unaffected."
            ),
        )
    )
    register_sim(
        SimSpec(
            name="highbase_compaso",
            root=path("compaso_root"),
            box=1000.0,
            np_side=3456,
            zd_seed=22321,
            kind="compaso",
            halo_R_field="r100_L2com",
            notes="Cleaned AbacusSummit highbase c000 ph100. Load with cleaned=True.",
        )
    )
    register_sim(
        SimSpec(
            name="tng300_2",
            root=path("tng_base"),
            box=205.0,
            kind="tng",
            notes="IllustrisTNG300-2 group catalog; snapshot 84 is z~0.2.",
        )
    )
    register_sim(
        SimSpec(
            name="tng300_3_dark_ne6",
            root=path("flare_root") / "ajohnson" / "TNG300-3-Dark-Abacus_ne6",
            box=205.0,
            notes="MAP reference run for the MAP paper HOD figure.",
        )
    )


def sim(name: str) -> SimSpec:
    """Look up a registered simulation by short name."""
    _register_defaults()
    try:
        return SIMS[name]
    except KeyError:
        raise KeyError(
            f"unknown sim {name!r}. Known: {', '.join(sorted(SIMS))}. "
            "Use register_sim(SimSpec(...)) to add one."
        ) from None


def sim_names() -> list[str]:
    """Sorted names in the simulation registry."""
    _register_defaults()
    return sorted(SIMS)
