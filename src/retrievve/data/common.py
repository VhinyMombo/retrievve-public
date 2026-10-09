"""Shared helpers for the v1 data pipeline (DATA_SPEC.md)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

from ..paths import CONFIGS, ROOT  # noqa: F401  (ROOT re-exported)

CONFIG_PATH = CONFIGS / "v1_pipeline.yaml"


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_hash(*parts: str) -> int:
    """Deterministic across runs/machines (unlike builtin hash())."""
    h = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_placeholder(path: Path) -> bool:
    """Dropbox online-only file: 0 bytes on disk (content not downloaded)."""
    try:
        return path.stat().st_size == 0
    except OSError:
        return True


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


WRITTEN: set[Path] = set()            # every file written by this process (stale-file cleanup in build)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    WRITTEN.add(path.resolve())
    return n


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)  # deterministic bytes
        f.write("\n")
    WRITTEN.add(path.resolve())
