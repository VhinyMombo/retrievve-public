"""Project paths and per-language config loading."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "languages.yaml"

RAW = ROOT / "data" / "raw"
OCR = ROOT / "data" / "ocr"
INTERIM = ROOT / "data" / "interim"
CORPUS = ROOT / "data" / "corpus"


@lru_cache
def all_langs() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def lang_cfg(lang: str) -> dict:
    cfgs = all_langs()
    if lang not in cfgs:
        known = [k for k in cfgs if k != "defaults"]
        raise KeyError(f"Unknown language '{lang}'. Known: {', '.join(known)}")
    return cfgs[lang]


def defaults() -> dict:
    return all_langs().get("defaults", {})


def ollama_models(lang: str) -> tuple[str, str]:
    """Return (chat_model, embed_model) for a language, falling back to defaults."""
    cfg = lang_cfg(lang)
    d = defaults()
    chat = cfg.get("ollama_chat_model", d.get("ollama_chat_model", "qwen3.5:9b"))
    embed = cfg.get("ollama_embed_model", d.get("ollama_embed_model", "embeddinggemma"))
    return chat, embed
