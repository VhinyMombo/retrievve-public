"""Source registry (DATA_SPEC §1, §10): data/v1/sources.yaml.

`discover()` scans data/raw and proposes an entry per file; `sync_registry()` merges proposals into the
existing sources.yaml without overwriting manual edits (release, license, domain, ... are human-owned).
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .common import ROOT, is_placeholder

# filename token -> (kind, domain). First match wins, so order matters.
KIND_RULES = [
    (r"bible|genesis|genese|exode|levitique|nombres_1|deuteronome|matthieu|_nt\b", "verse", "religious"),
    (r"grammaire|grammar", "grammar", "grammar"),
    (r"conv_|conversation|dialogue|interview|repliques", "dialogue_turn", "conversational"),
    (r"chanson|chant|berceuse|song", "song_line", "oral"),
    (r"poeme|poesie|poem", "poem_line", "literary"),
    (r"conte|recit|discours|texte|monolingual", "sentence", "literary"),
    (r"proverbe", "sentence", "oral"),
    (r"lexique|vocab|dict|nombres|famille|verbes|professions|anatomie|animaux|singulier|pluriel"
     r"|adjectifs|adverbes|temps|indefinis|traits|salutations|sensations|possessifs", "lexicon", "lexical"),
    (r"expression|phrase|cours|question|imperatif|negation|amour|affection|je_te|conjugaison|livre|"
     r"apprendre|glace|poisson", "sentence", "conversational"),
]

# patterns that mark a source as non-redistributable by default (§10)
NO_RELEASE = [r"tiktok", r"quizlet"]

PARSER_BY_EXT = {".csv": "csv", ".txt": "txt"}
SKIP_EXT = {".traineddata", ".gitkeep", ".ds_store"}
# directories that are one multi-file source (consumed via a v0 adapter) or assets, not individual sources
SKIP_DIR = re.compile(r"(_html|^fonts)$", re.I)


def guess_kind_domain(name: str) -> tuple[str, str]:
    low = name.lower()
    for pat, kind, domain in KIND_RULES:
        if re.search(pat, low):
            return kind, domain
    return "sentence", "conversational"


def guess_origin(name: str) -> str:
    low = name.lower()
    for tag, origin in [("tiktok", "TikTok @nkobefang (scrape)"), ("quizlet", "Quizlet (scrape)"),
                        ("dungandzi", "dungandzi"),
                        ("image01", "image transcription"), ("livre", "book (Parlons yipunu)")]:
        if tag in low:
            return origin
    return ""


def source_id_for(path: Path) -> str:
    stem = re.sub(r"\s*\(.*?\)\s*", " ", path.stem)          # drop parenthesised filename noise
    stem = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_").lower()
    return stem[:80]


def discover(raw_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.name.startswith(".") or p.suffix.lower() in SKIP_EXT:
            continue
        if any(SKIP_DIR.search(part) for part in p.relative_to(raw_dir).parts[:-1]):
            continue
        rel = p.relative_to(ROOT).as_posix()
        sid = source_id_for(p)
        if sid in out:                                        # same stem in two folders
            sid = f"{sid}_{p.parent.name.lower()}"
        kind, domain = guess_kind_domain(p.name + " " + p.parent.name)
        parser = PARSER_BY_EXT.get(p.suffix.lower())
        if parser == "txt" and kind == "lexicon":
            parser = None                                     # dictionary text dumps need a dedicated parser
        if p.name == "ls1910-nt.json":
            parser, kind, domain = None, "verse", "religious"   # French reference only, used via v0 alignments
        out[sid] = {
            "path": rel,
            "lang_hint": p.relative_to(raw_dir).parts[0],
            "parser": parser or "unsupported",
            "kind": kind,
            "domain": domain,
            "title": p.stem,
            "author": "",
            "origin": guess_origin(p.name),
            "acquired_on": "",
            "license": "unknown",
            "release": not any(re.search(x, p.name.lower()) for x in NO_RELEASE),
            "enabled": parser is not None,
            "notes": "",
        }
    return out


# v0 aligned corpora that we cannot rebuild from raw yet (PDF NT alignment, Bible HTML scrape).
V0_ADAPTED = {
    "fang_nt_web_v0": {
        "path": "data/corpus/fang-nt-parallel.scored.jsonl", "lang_hint": "fang", "parser": "v0_jsonl",
        "kind": "verse", "domain": "religious", "title": "Fang NT (WNTfan PDF) aligned to LS1910 — v0 alignment",
        "origin": "data/raw/fang/00-WNTfan-web-2.pdf + data/raw/ref/ls1910-nt.json", "orthography": "nt_ipa",
    },
    # Galley 1964 Fang-French dictionary: OCR (data/ocr/fang/fang-dictionary.txt) parsed by v0 into headwords +
    # example pairs. ocr=true -> ocr_review (blocks SFT/LM/eval; RAG allowed with a marker) until lifted (§4.6).
    "fang_galley_dict_v0": {
        "release": False, "license": "© 1964 S. Galley, Dictionnaire fang-français (printed by Henri Messeiller, Neuchâtel); internal use only",
        "path": "data/corpus/fang.jsonl", "lang_hint": "fang", "parser": "v0_galley",
        "kind": "lexicon", "domain": "lexical", "title": "Galley 1964, Dictionnaire fang-français (OCR) — headwords",
        "origin": "data/raw/fang/fang-dictionary.pdf -> data/ocr/fang/fang-dictionary.txt",
        "author": "S. Galley", "orthography": "galley1964", "headword_case": "upper", "ocr": True,
        "ocr_review_lifted": False, "rag_marker": "[dictionnaire Galley 1964, OCR non vérifié]",
    },
    # v1 parse straight from the OCR text (dehyphenation, tone groups, POS fixes, headers); replaces the v0 one.
    "fang_galley_dict": {
        "release": False, "license": "© 1964 S. Galley, Dictionnaire fang-français (printed by Henri Messeiller, Neuchâtel); internal use only",
        "path": "data/ocr/fang/fang-dictionary.txt", "lang_hint": "fang", "parser": "galley_ocr",
        "kind": "lexicon", "domain": "lexical", "title": "Galley 1964, Dictionnaire fang-français (OCR) — v1 parse",
        "origin": "data/raw/fang/fang-dictionary.pdf -> data/ocr/fang/fang-dictionary.txt",
        "author": "S. Galley", "orthography": "galley1964", "headword_case": "upper", "ocr": True,
        "ocr_review_lifted": False, "rag_marker": "[dictionnaire Galley 1964, OCR non vérifié]",
    },
    "fang_galley_examples_v0": {
        "release": False, "license": "© 1964 S. Galley, Dictionnaire fang-français (printed by Henri Messeiller, Neuchâtel); internal use only",
        "path": "data/corpus/fang-examples.jsonl", "lang_hint": "fang", "parser": "v0_galley",
        "kind": "example", "domain": "lexical", "title": "Galley 1964 dictionary — example pairs (OCR)",
        "origin": "data/raw/fang/fang-dictionary.pdf", "author": "S. Galley", "orthography": "galley1964",
        "ocr": True, "ocr_review_lifted": False, "rag_marker": "[dictionnaire Galley 1964, OCR non vérifié]",
    },
    # Realigned Fang NT (typography parser + bead DP vs LS1910); only books in config nt_books_accepted enter.
    "fang_nt_v1": {
        "path": "data/interim/fang_nt_aligned.jsonl", "lang_hint": "fang", "parser": "nt_aligned",
        "kind": "verse", "domain": "religious", "title": "Fang NT (WNTfan PDF) realigned to LS1910 — v1",
        "origin": "data/raw/fang/00-WNTfan-web-2.pdf + data/raw/ref/ls1910-nt.json (nt_align.py)",
        "orthography": "nt_ipa", "release": False, "accept_books_key": "nt_books_accepted",
        "license": "Fang NT (ScriptureEarth / Asociación Cristiana); licence to be checked; internal use only",
    },
    "punu_bible_html_v0": {
        "path": "data/corpus/punu-bible.jsonl", "lang_hint": "punu", "parser": "v0_jsonl",
        "kind": "verse", "domain": "religious", "title": "Punu Bible (puu_html scrape) — v0 parse",
        "origin": "data/raw/punu/puu_html",
    },
}


# Punu 1992 Bible: Hebrew numbering in these chapters only (88, align_check --scan-offsets); mapped with the
# standard Hebrew->English table in versification.py. Proposed once, then human-owned in sources.yaml.
VERSIFICATION = {
    "punu_bible_deuteronome_1_34": {"versification": "hebrew", "versification_scope": ["DEU.13", "DEU.23", "DEU.29"]},
    "punu_bible_nombres_1_36": {"versification": "hebrew", "versification_scope": ["NUM.17"]},
}

def propose_orthography(sid: str, lang_hint: str) -> str:
    """Spec v0.6/v0.8 orthography tags (mandatory per source; 'unknown' = to be documented, reported)."""
    if lang_hint == "fang":
        if "nkobefang" in sid or "tiktok" in sid:
            return "schwa_untoned"            # ə + ɔ, no tone marks (v0.8)
        if "grammaire" in sid or "galley" in sid or "dictionary" in sid:
            return "galley1964"
        if "nt" in sid.split("_"):
            return "nt_ipa"
        if "quizlet" in sid:
            return "mixed"
    return "unknown"


# per-source French parenthetical rule (spec v0.4); proposed once, then human-owned in sources.yaml
FR_PARENS = {"myene_proverbes_mandji": "literal", "punu_proverbes_dungandzi": "note"}

LANG_TOKENS = r"^(fang|punu|nzebi|myene|lembaama)_"


def _propose_shared_groups(proposed: dict) -> None:
    """Same deck in several languages (e.g. fang_/myene_/nzebi_expressions_quizlet share their French side):
    give them one language-independent group so they are split together (spec v0.1)."""
    by_stem = {}
    for sid in proposed:
        stem = re.sub(LANG_TOKENS, "", sid)
        if stem != sid:
            by_stem.setdefault(stem, []).append(sid)
    for stem, sids in by_stem.items():
        if len(sids) > 1:
            for sid in sids:
                proposed[sid]["group"] = f"shared/{stem}"


def sync_registry(path: Path, raw_dir: Path) -> dict:
    existing = {}
    if path.exists() and path.stat().st_size:
        existing = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("sources", {})
    proposed = discover(raw_dir)
    _propose_shared_groups(proposed)
    for sid, e in proposed.items():
        e.setdefault("orthography", propose_orthography(sid, e.get("lang_hint", "")))
        e.setdefault("variety", "")                      # empty unless confirmed (v0.6)
        e.setdefault("record_origin", "human")           # v0.5: human | synthetic | mt
    for sid, v in VERSIFICATION.items():
        if sid in proposed:
            proposed[sid].update(v)
    for sid, mode in FR_PARENS.items():
        if sid in proposed:
            proposed[sid]["fr_parens"] = mode
    for sid, e in V0_ADAPTED.items():
        proposed[sid] = {"author": "", "acquired_on": "", "license": "unknown", "release": False,
                         "variety": "", "record_origin": "human", "orthography": "unknown",
                         "enabled": True, "notes": "v0 adapter: reuses v0 alignment output; replace with a v1 parser.",
                         **e}
    merged = {}
    for sid, e in proposed.items():
        cur = existing.get(sid)
        merged[sid] = {**e, **cur} if cur else e             # human edits win
    for sid, cur in existing.items():                          # keep manual-only entries
        merged.setdefault(sid, cur)
    doc = {"note": "Auto-proposed by retrievve.data.sources; edit freely — manual values are preserved on rebuild. "
                   "release:false = internal experiments only (DATA_SPEC §10).",
           "sources": dict(sorted(merged.items()))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    return doc["sources"]


def availability(sources: dict) -> dict[str, bool]:
    return {sid: not is_placeholder(ROOT / e["path"]) for sid, e in sources.items()}
