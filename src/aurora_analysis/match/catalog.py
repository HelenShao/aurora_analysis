"""Versioned match-catalog schema.

The match ``.npz`` is the seam between matching and every downstream plot.
Its keys used to be defined by README prose and four independent writers that
each invented a slightly different alias for the host aperture
(``host_R`` vs ``host_r100_L2com``). That is how a catalog matched on
``r98_L2com`` can be read as if it were ``r100`` without anyone noticing.

This module is the single place that:

- lists the required and optional keys
- writes a sidecar JSON with provenance (sim, redshift, aperture, matcher,
  ``maplog_known_duplicates``)
- reads an existing catalog, infers what it can from the sidecar or the
  filename, and warns rather than rejecting files written before this schema
  existed

Existing 6–7 GB catalogs on Flare stay usable. Pass ``strict=True`` to require
a complete provenance record (useful for new production runs).
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "OPTIONAL_ARRAY_KEYS",
    "REQUIRED_ARRAY_KEYS",
    "TREE_ARRAY_KEYS",
    "MatchCatalog",
    "MatchProvenance",
    "SchemaError",
    "discover_meta_path",
    "load_match_catalog",
    "save_match_catalog",
    "validate_arrays",
]

CURRENT_SCHEMA_VERSION = 1

#: Arrays every match catalog must have. Length is the number of MAPs, not
#: the number of matched MAPs — ``matched`` is a mask over that length.
REQUIRED_ARRAY_KEYS: tuple[str, ...] = (
    "map_pos",
    "map_vel",
    "map_mult",
    "matched",
    "is_central",
    "is_satellite",
    "d",
    "d_over_R",
    "host_x_L2com",
    "host_v_L2com",
    "host_sigmav3d_L2com",
    "host_N",
    "host_R",
)

#: Present on maplog-sourced catalogs so a later tree walk can find the MAP
#: again. Subsample matches do not have these.
TREE_ARRAY_KEYS: tuple[str, ...] = (
    "map_imap",
    "map_shard_id",
    "map_control",
    "map_length",
)

OPTIONAL_ARRAY_KEYS: tuple[str, ...] = TREE_ARRAY_KEYS + (
    "local_halo_ih",
    "host_id",
    "map_density",
    "map_vel_disp",
    "map_rel_vel",
    "map_rel_vel_healpix",
    "halo_x_L2com",
    "halo_v_L2com",
    "halo_sigmav3d_L2com",
    "halo_N",
    "halo_R",
    "halo_n_assigned",
    "halo_sum_assigned_mult",
)

_Z_IN_NAME = re.compile(r"z(\d+\.\d+)")


class SchemaError(ValueError):
    """The arrays on disk do not satisfy the match-catalog contract."""


@dataclass
class MatchProvenance:
    """Everything about a catalog that is not a per-MAP array.

    Stored as a sidecar ``.json`` next to the ``.npz`` (or under a sibling
    ``meta/`` directory, which is the existing convention). Keeping it out of
    the npz means a 6 GB catalog can be inspected without loading it.
    """

    schema_version: int = CURRENT_SCHEMA_VERSION
    source: str | None = None  # "subsample" | "maplog_timeslice"
    halo_R_field: str | None = None
    match_cut: str | None = None
    matcher: str | None = None  # "ball" | "knn" | "brute"
    catalog_tag: str | None = None  # "sph" | "knn" | "brute" — embedded in filenames
    knn_k: int | None = None
    overlap: str | None = None  # ball only: most_massive | deepest | nearest
    box: float | None = None
    z: str | None = None
    z_maplog: float | None = None
    sim_root: str | None = None
    halo_root: str | None = None
    maplog_known_duplicates: bool | None = None
    inferred: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def missing_required(self) -> list[str]:
        """Fields a newly written catalog is expected to set."""
        needed = ("source", "halo_R_field", "box", "z", "sim_root")
        return [name for name in needed if getattr(self, name) in (None, "")]

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra") or {}
        # Sidecar keeps the extra keys at the top level so existing plot
        # scripts that read the JSON directly still see ``n_matched`` etc.
        return {**extra, **d}

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> MatchProvenance:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in data.items():
            if key in known and key != "extra":
                kwargs[key] = value
            else:
                extra[key] = value
        extra.update(data.get("extra") or {})
        extra.pop("extra", None)
        kwargs["extra"] = extra
        if "halo_root" not in kwargs and extra.get("compaso_root"):
            kwargs["halo_root"] = extra["compaso_root"]
        # Sidecars written before this schema have no schema_version.
        had_version = "schema_version" in data
        kwargs.setdefault("schema_version", 0)
        kwargs.setdefault("inferred", not had_version)
        return cls(**kwargs)


@dataclass
class MatchCatalog:
    """In-memory match catalog: arrays plus provenance.

    Array access is dict-like (``cat["matched"]``) so existing plot code can
    switch from ``np.load`` with a one-line change. Provenance lives on
    ``cat.provenance``.
    """

    arrays: dict[str, np.ndarray]
    provenance: MatchProvenance
    path: Path | None = None
    meta_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def __contains__(self, key: object) -> bool:
        return key in self.arrays

    def get(self, key: str, default: Any = None) -> Any:
        return self.arrays.get(key, default)

    def keys(self):
        return self.arrays.keys()

    @property
    def n_map(self) -> int:
        return int(len(self.arrays["matched"]))

    @property
    def n_matched(self) -> int:
        return int(np.asarray(self.arrays["matched"]).sum())

    @property
    def halo_R_field(self) -> str | None:
        return self.provenance.halo_R_field

    @property
    def host_R(self) -> np.ndarray:
        """Aperture used for the match, whatever field it was."""
        return self.arrays["host_R"]

    def as_legacy_dict(self) -> dict[str, Any]:
        """Flat dict matching what ``ye_halo_props.load_match_catalog`` returns.

        Includes the ``host_r100_L2com`` alias *only* when the aperture really
        is ``r100_L2com``. That is the whole point of this module: do not lie
        about the radius field.
        """
        out: dict[str, Any] = dict(self.arrays)
        out["path"] = str(self.path) if self.path is not None else None
        out["meta"] = self.provenance.to_json_dict()
        out["box"] = self.provenance.box
        out["ztag"] = self.provenance.z
        if self.provenance.halo_R_field == "r100_L2com":
            out["host_r100_L2com"] = self.arrays["host_R"]
        return out


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> list[str]:
    """Return human-readable problems; empty list means the arrays are fine."""
    problems: list[str] = []
    missing = [k for k in REQUIRED_ARRAY_KEYS if k not in arrays]
    if missing:
        problems.append(f"missing required keys: {missing}")
        return problems

    n = len(np.asarray(arrays["matched"]))
    length_keys = [
        k
        for k in REQUIRED_ARRAY_KEYS
        if k in arrays and np.asarray(arrays[k]).shape[0] != n
    ]
    if length_keys:
        problems.append(f"arrays of length != n_MAP={n}: {length_keys}")

    matched = np.asarray(arrays["matched"], dtype=bool)
    is_cen = np.asarray(arrays["is_central"], dtype=bool)
    is_sat = np.asarray(arrays["is_satellite"], dtype=bool)
    if np.any(is_cen & ~matched):
        problems.append("is_central is True for unmatched MAPs")
    if np.any(is_sat & ~matched):
        problems.append("is_satellite is True for unmatched MAPs")
    if np.any(is_cen & is_sat):
        problems.append("is_central and is_satellite overlap")
    if np.any(matched & ~(is_cen | is_sat)):
        problems.append("matched MAPs that are neither central nor satellite")
    return problems


def _normalize_arrays(raw: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Copy arrays and resolve the host-aperture alias mess.

    Writers have used ``host_R``, ``host_r100_L2com``, or both (sometimes
    pointing at different physical radii). The canonical in-memory name is
    always ``host_R``.
    """
    notes: list[str] = []
    arrays = {k: np.asarray(v) for k, v in raw.items()}

    has_R = "host_R" in arrays
    has_r100 = "host_r100_L2com" in arrays
    if has_R and has_r100:
        a = np.asarray(arrays["host_R"], dtype=np.float64)
        b = np.asarray(arrays["host_r100_L2com"], dtype=np.float64)
        if a.shape == b.shape and not np.allclose(a, b, equal_nan=True, rtol=0, atol=0):
            notes.append(
                "host_R and host_r100_L2com both present and differ; "
                "keeping host_R as the match aperture. Do not treat "
                "host_r100_L2com as the match radius."
            )
    elif (not has_R) and has_r100:
        arrays["host_R"] = np.asarray(arrays["host_r100_L2com"], dtype=np.float64)
        notes.append(
            "no host_R; copied host_r100_L2com → host_R. "
            "Confirm the match aperture really was r100_L2com."
        )
    return arrays, notes


def discover_meta_path(npz: Path) -> Path | None:
    """Find the sidecar JSON using the conventions already on disk."""
    candidates = (
        npz.with_suffix(".json"),
        npz.parent / "meta" / (npz.stem + ".json"),
        npz.parent.parent / "meta" / (npz.stem + ".json"),
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def _infer_from_filename(npz: Path) -> dict[str, Any]:
    from .naming import parse_match_catalog_tag

    inferred: dict[str, Any] = {}
    name = npz.name.lower()
    m = _Z_IN_NAME.search(name)
    if m:
        inferred["z"] = m.group(1)
    if "maplog" in name:
        inferred["source"] = "maplog_timeslice"
        inferred["maplog_known_duplicates"] = True
    elif "match" in name:
        inferred["source"] = "subsample"
    tag = parse_match_catalog_tag(name)
    if tag is not None:
        inferred["catalog_tag"] = tag
        # Reverse map tag → method name used in match_maps.
        if tag == "sph":
            inferred["matcher"] = "ball"
        else:
            inferred["matcher"] = tag
    return inferred


def _merge_provenance(
    sidecar: Mapping[str, Any] | None,
    filename_guess: Mapping[str, Any],
) -> tuple[MatchProvenance, list[str]]:
    notes: list[str] = []
    if sidecar:
        prov = MatchProvenance.from_json_dict(sidecar)
    else:
        prov = MatchProvenance(inferred=True)
        notes.append("no sidecar JSON found; inferring provenance")

    # Fill blanks from the filename, but never overwrite an explicit value.
    for key, value in filename_guess.items():
        if getattr(prov, key, None) in (None, "") and value is not None:
            setattr(prov, key, value)
            prov.inferred = True
            notes.append(f"inferred {key}={value!r} from filename")

    if sidecar and "halo_R_field" not in sidecar and "halo_R_field" not in filename_guess:
        # The dangerous default: older scripts treated host_r100_L2com as
        # gospel. We refuse to guess r100.
        notes.append(
            "halo_R_field unknown. host_R is the match aperture; "
            "do not assume it is r100_L2com."
        )
    return prov, notes


def load_match_catalog(
    npz: str | Path,
    *,
    meta: str | Path | Mapping[str, Any] | None = None,
    strict: bool = False,
    mmap: bool = False,
) -> MatchCatalog:
    """Load a match ``.npz`` and attach provenance.

    Parameters
    ----------
    npz
        Path to the catalog.
    meta
        Sidecar JSON path, or an already-parsed dict. ``None`` walks the
        usual sibling / ``meta/`` locations.
    strict
        If True, missing required arrays *or* missing provenance fields
        raise :class:`SchemaError`. Default is warn-and-infer so existing
        Flare catalogs stay usable.
    mmap
        Open the npz with ``mmap_mode='r'`` (useful for the 6.8 GB full-box
        catalogs). Returned arrays are views; do not close the file.
    """
    npz_path = Path(npz)
    raw = np.load(npz_path, allow_pickle=True, mmap_mode="r" if mmap else None)
    try:
        files = {k: raw[k] for k in raw.files}
    finally:
        if not mmap:
            raw.close()

    arrays, alias_notes = _normalize_arrays(files)
    problems = validate_arrays(arrays)

    if isinstance(meta, Mapping):
        sidecar: dict[str, Any] | None = dict(meta)
        meta_path = None
    elif meta is not None:
        meta_path = Path(meta)
        sidecar = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    else:
        meta_path = discover_meta_path(npz_path)
        sidecar = json.loads(meta_path.read_text()) if meta_path else None

    prov, infer_notes = _merge_provenance(sidecar, _infer_from_filename(npz_path))
    notes = alias_notes + infer_notes
    if problems:
        notes.extend(problems)

    if strict:
        missing_prov = prov.missing_required()
        if problems or missing_prov:
            raise SchemaError(
                f"{npz_path}: "
                + "; ".join(problems + [f"missing provenance: {missing_prov}"])
            )
    else:
        for note in notes:
            warnings.warn(f"{npz_path.name}: {note}", UserWarning, stacklevel=2)

    return MatchCatalog(
        arrays=arrays,
        provenance=prov,
        path=npz_path,
        meta_path=meta_path,
        warnings=notes,
    )


def save_match_catalog(
    arrays: Mapping[str, np.ndarray],
    path: str | Path,
    provenance: MatchProvenance,
    *,
    meta_path: str | Path | None = None,
    compress: bool = True,
) -> tuple[Path, Path]:
    """Write a catalog and its sidecar. Returns ``(npz_path, meta_path)``.

    The writer always stores ``host_R`` (the match aperture) and, when the
    aperture really is ``r100_L2com``, also stores ``host_r100_L2com`` so
    older plot scripts keep working. It will not invent that alias for any
    other field.
    """
    arrays = dict(arrays)
    problems = validate_arrays(arrays)
    if problems:
        raise SchemaError("refusing to write an invalid catalog: " + "; ".join(problems))
    missing = provenance.missing_required()
    if missing:
        raise SchemaError(f"provenance incomplete: {missing}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path is None:
        meta_dir = path.parent.parent / "meta"
        if meta_dir.is_dir() or path.parent.name == "catalogs":
            meta_path = meta_dir / (path.stem + ".json")
        else:
            meta_path = path.with_suffix(".json")
    else:
        meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    save = dict(arrays)
    if provenance.halo_R_field == "r100_L2com" and "host_r100_L2com" not in save:
        save["host_r100_L2com"] = save["host_R"]

    saver = np.savez_compressed if compress else np.savez
    saver(path, **save)

    payload = provenance.to_json_dict()
    payload.setdefault("n_map", int(len(arrays["matched"])))
    payload.setdefault("n_matched", int(np.asarray(arrays["matched"]).sum()))
    payload.setdefault("n_central", int(np.asarray(arrays["is_central"]).sum()))
    payload.setdefault("n_satellite", int(np.asarray(arrays["is_satellite"]).sum()))
    meta_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path, meta_path
