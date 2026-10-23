"""MAP epoch bookkeeping: node type, ``control`` packing, and step ↔ z.

The 16-bit MapNode ``control`` word is how every maplog reader decides *what*
a row is and *when* it happened. That convention used to be re-derived in
comments, ``map_tree_tags.py``, ``maplog_io.py``, and
``maplog_timeslice_load.py`` independently::

    node_type = control % 4          # low 2 bits
    timestep  = control // 4         # WriteState.FullStepNumber
    control   = 4 * step + node_type

TimeSlice catalog selection is therefore ``control == 4 * step + 3``, never
the maplog filename tag (``maplog_z0.25_*.asdf`` can hold many epochs).

``Step_to_Z[i]`` is ``ReadState.Redshift`` for WriteState step ``i``. Lookup
with ``step_to_z[control // 4]``; there is no ``- 1`` offset.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

__all__ = [
    "NODE_CONTROL_MASK",
    "NODE_FORMATION",
    "NODE_LIGHTCONE",
    "NODE_MERGER",
    "NODE_NAMES",
    "NODE_TIMESLICE",
    "build_maplog_epochs",
    "build_step_to_z_from_maplog",
    "build_step_to_z_from_state_files",
    "encode_control",
    "epoch_step",
    "map_epoch_for_redshift",
    "node_type",
    "parse_abacus_kv_file",
    "pick_z_cat_near_zeff",
    "resolve_timeslice_epoch",
    "step_to_z_from_header_array",
    "timeslice_control",
]

#: Low 2 bits of ``control``. Matches ``maplog.cc`` / abacusnbody.
NODE_FORMATION = 0
NODE_MERGER = 1
NODE_LIGHTCONE = 2
NODE_TIMESLICE = 3
NODE_CONTROL_MASK = 0x3

NODE_NAMES: dict[int, str] = {
    NODE_FORMATION: "formation",
    NODE_MERGER: "merger",
    NODE_LIGHTCONE: "lightcone",
    NODE_TIMESLICE: "timeslice",
}


def node_type(control) -> np.ndarray:
    """Low 2 bits of ``control``: formation / merger / lightcone / timeslice."""
    return np.asarray(control) & NODE_CONTROL_MASK


def epoch_step(control) -> np.ndarray:
    """WriteState ``FullStepNumber`` packed into the upper bits of ``control``."""
    return np.asarray(control) // 4


def encode_control(step, ntype: int = NODE_TIMESLICE):
    """Pack ``control = 4 * step + node_type``.

    ``step`` may be a scalar or an array. The result has the same shape.
    """
    ntype = int(ntype)
    if ntype not in NODE_NAMES:
        raise ValueError(
            f"unknown node type {ntype}; expected one of {sorted(NODE_NAMES)}"
        )
    step_arr = np.asarray(step)
    out = step_arr * 4 + ntype
    if out.ndim == 0:
        return int(out)
    return out


def timeslice_control(step) -> int:
    """``control`` value of a TimeSlice node at ``step``: ``4 * step + 3``."""
    return int(encode_control(int(step), NODE_TIMESLICE))


def step_to_z_from_header_array(step_to_z_header) -> dict[int, float]:
    """Parse ASDF header ``Step_to_Z`` ndarray → ``{FullStepNumber: Redshift}``.

    Index ``i`` is the WriteState step stamped into MapNode ``control // 4``;
    the value is ``ReadState.Redshift`` for that same step. Lookup with
    ``step_to_z[map_epoch]`` (no ``- 1`` offset).
    """
    zs = np.asarray(step_to_z_header, dtype=np.float64)
    return {i: float(z) for i, z in enumerate(zs)}


def parse_abacus_kv_file(path: str | Path) -> dict:
    """Parse an Abacus ``key = value`` file (``abacus.par``, ``step*.state``).

    This is a restricted stand-in for ``abacus.param.InputFile`` that does not
    ``eval`` the file as Python. Scalars, quoted strings, and simple lists
    (``[1, 2]`` or space-separated ``1 2 3``) are understood; anything else is
    kept as a string.
    """
    out: dict = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        out[key] = _parse_kv_value(val)
    return out


def _parse_kv_value(val: str):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        parts = val.split()
        if len(parts) > 1:
            try:
                return [ast.literal_eval(p) for p in parts]
            except (ValueError, SyntaxError):
                return val
        return val


def build_step_to_z_from_state_files(rdir) -> dict[int, float]:
    """``{FullStepNumber: Redshift}`` from ``log/step*/step*.state``."""
    from glob import glob

    step_to_z: dict[int, float] = {}
    for f in glob(str(Path(rdir) / "log/step*/step*.state")):
        st = parse_abacus_kv_file(f)
        step_to_z[int(st["FullStepNumber"])] = float(st["Redshift"])
    return step_to_z


def build_step_to_z_from_maplog(maplog) -> dict[int, float]:
    """Legacy fallback from acj-injected ``maplog['step']`` / ``maplog['redshift']``."""
    return {int(s): float(z) for s, z in zip(maplog["step"], maplog["redshift"], strict=True)}


def resolve_timeslice_epoch(step_to_z: dict, z_cat: float) -> tuple[int, float]:
    """Nearest MAP output step to ``z_cat`` and its redshift."""
    if not step_to_z:
        raise ValueError("empty step_to_z")
    steps = np.fromiter(step_to_z.keys(), dtype=np.int64, count=len(step_to_z))
    zs = np.empty(len(steps), dtype=np.float64)
    for i, s in enumerate(steps):
        zs[i] = float(step_to_z[int(s)])
    j = int(np.argmin(np.abs(zs - float(z_cat))))
    return int(steps[j]), float(zs[j])


def build_maplog_epochs(
    maplog, step_to_z: dict[int, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse TimeSlice epoch table from node timesteps + ``Step_to_Z``.

    Returns ``(map_epoch, redshift)`` for the TimeSlice epochs present in
    ``maplog`` — the same information acj stores as ``maplog['step']`` /
    ``maplog['redshift']`` after ``read_maplog_dir``.
    """
    state_steps = np.array(sorted(int(s) for s in step_to_z), dtype=np.int32)
    state_z = np.array([step_to_z[int(s)] for s in state_steps], dtype=np.float32)
    if "timestep" in maplog:
        timesteps = np.unique(
            maplog["timestep"][node_type(maplog["control"]) == NODE_TIMESLICE]
        )
    else:
        timesteps = np.unique(
            epoch_step(maplog["control"])[node_type(maplog["control"]) == NODE_TIMESLICE]
        )
    maplog_steps = np.empty(len(timesteps), dtype=np.int32)
    maplog_redshifts = np.empty(len(timesteps), dtype=np.float32)
    for i, ts in enumerate(timesteps):
        ts = int(ts)
        maplog_steps[i] = ts
        matched = state_z[state_steps == ts]
        if len(matched) == 0:
            raise ValueError(
                f"no Step_to_Z entry for maplog epoch {ts} "
                f"(expected key {ts} = WriteState.FullStepNumber)"
            )
        maplog_redshifts[i] = matched[0]
    return maplog_steps, maplog_redshifts


def map_epoch_for_redshift(maplog, z, step_to_z: dict[int, float]) -> tuple[int, float]:
    """Nearest MapLog TimeSlice epoch to ``z`` using ``Step_to_Z``."""
    steps, redshifts = build_maplog_epochs(maplog, step_to_z)
    if len(redshifts) == 0:
        raise ValueError("map_epoch_for_redshift: no TimeSlice epochs")
    j = int(np.argmin(np.abs(redshifts - z)))
    return int(steps[j]), float(redshifts[j])


def pick_z_cat_near_zeff(
    maplog, z_eff, step_to_z: dict[int, float], *, tie_prefer_above: bool = True
) -> tuple[float, int, np.ndarray]:
    """Choose catalog redshift as nearest MapLog epoch to ``z_eff``.

    On an exact tie in |Δz|, prefer the epoch **above** ``z_eff`` when
    ``tie_prefer_above`` (plan default for DESI ELG).
    Returns ``(z_cat, step, all_redshifts)``.
    """
    steps, redshifts = build_maplog_epochs(maplog, step_to_z)
    if len(redshifts) == 0:
        raise ValueError("pick_z_cat_near_zeff: no TimeSlice epochs")
    z_eff = float(z_eff)
    d = np.abs(redshifts - z_eff)
    dmin = float(np.min(d))
    cand = np.flatnonzero(np.isclose(d, dmin, rtol=0.0, atol=1e-9))
    if len(cand) == 1:
        j = int(cand[0])
    else:
        above = [int(i) for i in cand if redshifts[i] >= z_eff]
        below = [int(i) for i in cand if redshifts[i] < z_eff]
        if tie_prefer_above and above:
            j = above[int(np.argmin(np.abs(redshifts[above] - z_eff)))]
        elif below:
            j = below[int(np.argmin(np.abs(redshifts[below] - z_eff)))]
        else:
            j = int(cand[0])
    return float(redshifts[j]), int(steps[j]), redshifts
