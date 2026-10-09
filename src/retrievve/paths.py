"""Repository paths. The root is found by walking up to pyproject.toml, so modules can move freely."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def find_root(start: Path | None = None) -> Path:
    """Repo root: $RETRIEVVE_ROOT if set, else the first parent of `start` (or this file) holding pyproject.toml."""
    env = os.environ.get("RETRIEVVE_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path(__file__)).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file():
            return p
    return Path.cwd().resolve()


ROOT = find_root()
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
