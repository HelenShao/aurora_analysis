"""ASDF open helpers for Aurora.

On Aurora, ``asdf.open(..., lazy_load=False)`` is far faster than the
abacusutils ``read_asdf`` path for single-block files (maplog shards,
timeslice subsample). ``read_asdf`` hardcodes ``lazy_load=True``, which is
the slow path: a multi-GB block is pulled in on demand through a less
efficient reader.

Multi-GB opens belong on a **compute node**, not a login/UAN node — one
AbS maplog shard opens in roughly 18 s on compute versus 150 s on a UAN.
This module is where that guidance lives, rather than only in READMEs.

Typical use::

    from aurora_analysis.io.asdf import open_asdf, read_header

    with open_asdf(path) as af:          # eager; warns on a UAN
        raw = af.tree["data"]["maplogs"]
    hdr = read_header(path)              # YAML sniff; does not decompress
"""

from __future__ import annotations

import ast
import os
import re
import socket
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "DEFAULT_LAZY_LOAD",
    "is_likely_login_node",
    "open_asdf",
    "read_header",
    "reset_login_warning",
    "warn_if_login_node",
]

#: Aurora fast path for single-block files. Pass ``lazy_load=True`` only for
#: header-only access that must not decompress the data block.
DEFAULT_LAZY_LOAD = False

_WARNED = False
_UAN_HOST_RE = re.compile(r"(^uan\b)|(^login\b)|(\.uan\.)", re.IGNORECASE)


def is_likely_login_node() -> bool:
    """True on an Aurora UAN / login node, False on a PBS compute allocation.

    Override with ``AURORA_ANALYSIS_SKIP_UAN_WARN=1`` (treat as compute) or
    ``AURORA_ANALYSIS_FORCE_UAN_WARN=1`` (always warn). A set ``PBS_JOBID``
    or ``PBS_NODEFILE`` counts as compute.
    """
    if os.environ.get("AURORA_ANALYSIS_SKIP_UAN_WARN"):
        return False
    if os.environ.get("AURORA_ANALYSIS_FORCE_UAN_WARN"):
        return True
    if os.environ.get("PBS_JOBID") or os.environ.get("PBS_NODEFILE"):
        return False
    host = socket.gethostname().lower()
    return bool(_UAN_HOST_RE.search(host))


def reset_login_warning() -> None:
    """Allow :func:`warn_if_login_node` to fire again (tests)."""
    global _WARNED
    _WARNED = False


def warn_if_login_node(
    path: str | Path | None = None, *, decompressing: bool = True
) -> None:
    """Warn once per process if this looks like a UAN and we are decompressing."""
    global _WARNED
    if _WARNED or not decompressing or not is_likely_login_node():
        return
    _WARNED = True
    where = f" ({path})" if path is not None else ""
    warnings.warn(
        "ASDF eager-open on an Aurora login/UAN node is much slower than on "
        "a compute node (one maplog shard is ~150 s here vs ~18 s there). "
        "Use `qsub -I -l select=1 -l walltime=2:00:00 -q debug-scaling "
        "-A Abacus`, then `module load frameworks`. "
        f"Opening{where}. Set AURORA_ANALYSIS_SKIP_UAN_WARN=1 to silence.",
        stacklevel=3,
    )


@contextmanager
def open_asdf(
    path: str | Path,
    *,
    lazy_load: bool = DEFAULT_LAZY_LOAD,
    memmap: bool = False,
    warn: bool = True,
) -> Iterator:
    """Open an ASDF file. Default is the Aurora fast path (``lazy_load=False``).

    Pass ``lazy_load=True`` only when you will not touch the binary blocks
    (header metadata). ``read_asdf`` is the slow path on Aurora and is not
    used here.
    """
    import asdf

    path = Path(path)
    if warn:
        warn_if_login_node(path, decompressing=not lazy_load)
    with asdf.open(path, lazy_load=lazy_load, memmap=memmap) as af:
        yield af


def read_header(path: str | Path, *, nbytes: int = 262_144) -> dict:
    """Read the ASDF YAML ``header`` mapping without decompressing blocks.

    Aurora subsample / maplog files are one large blosc block; a full
    ``asdf.open`` would re-read that block just for ``BoxSize`` / ``NodeRankX``.
    Falls back to an eager open if the YAML sniff cannot find ``header:``.
    """
    path = Path(path)
    raw = path.open("rb").read(int(nbytes))
    end = raw.find(b"\n...\n")
    text = raw[: end if end >= 0 else len(raw)].decode("utf-8", "replace")
    m = re.search(r"(?m)^header:\s*\n", text)
    if not m:
        with open_asdf(path, lazy_load=False, warn=True) as af:
            return dict(af.tree.get("header", {}) or {})

    hdr: dict = {}
    for line in text[m.end() :].splitlines():
        if not line.startswith("  "):
            break
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            hdr[key] = val[1:-1]
            continue
        try:
            hdr[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            hdr[key] = val
    return hdr
