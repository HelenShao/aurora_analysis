"""Load modules from external source trees that are not installable packages.

This is the **only** place in ``aurora_analysis`` that manipulates ``sys.path``
or loads a module by file path. Confining it here means a broken deployment
produces one clear error from one file instead of a different ``ImportError``
in each of the 70-odd scripts that used to do their own ``sys.path.insert``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

__all__ = ["load_module", "require_tree", "tree_on_syspath"]


def require_tree(root: Path, *, what: str, env_var: str, expect: str | None = None) -> Path:
    """Validate that an external source tree is present.

    ``expect`` is a filename that must exist inside ``root``; it distinguishes
    "the directory happens to exist" from "this is really the tree we want".
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"{what} tree not found at {root}. Set ${env_var} to its location."
        )
    if expect is not None and not (root / expect).is_file():
        raise FileNotFoundError(
            f"{root} does not look like the {what} tree ({expect} is missing). "
            f"Set ${env_var} to the correct location."
        )
    return root


def tree_on_syspath(root: Path) -> None:
    """Prepend a directory to ``sys.path`` once, idempotently."""
    entry = str(Path(root).expanduser())
    if entry not in sys.path:
        sys.path.insert(0, entry)


def load_module(file_path: Path, *, name: str, deps: Path | None = None) -> ModuleType:
    """Import a standalone .py file as a module under a private name.

    ``deps`` is a directory to put on ``sys.path`` first, for trees whose
    modules import each other by bare name (as both ``acj`` and the hackAurora
    plot-params modules do).
    """
    file_path = Path(file_path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"cannot load {name}: {file_path} does not exist")
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    if deps is not None:
        tree_on_syspath(deps)
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot build an import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so modules that import themselves do not recurse.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module
