"""Raw source -> canonical records (DATA_SPEC §1).

group is language-independent (spec v0.1): one group per source document, and `bible/<BOOK>` for
Bible books so a held-out book is held out in every language.

Each parser yields dicts with: lang, text_raw, fr_raw, kind, locator, group, order, align, extra, (lex).
`build.py` turns them into canonical records (id, normalization, release, ...).
"""
from __future__ import annotations

import csv
import io
import re

from .common import ROOT, read_jsonl

# French book names / codes -> USFM book codes (for Bible locators and book-level splits)
BOOKS = {
    "genese": "GEN", "genesis": "GEN", "gen": "GEN", "exode": "EXO", "exo": "EXO", "levitique": "LEV",
    "lev": "LEV", "nombres": "NUM", "num": "NUM", "deuteronome": "DEU", "deu": "DEU", "josue": "JOS",
    "juges": "JDG", "ruth": "RUT", "rut": "RUT", "jonas": "JON", "jon": "JON", "matthieu": "MAT",
    "mat": "MAT", "marc": "MRK", "mrk": "MRK", "luc": "LUK", "luk": "LUK", "jean": "JHN", "jhn": "JHN",
    "actes": "ACT", "act": "ACT", "romains": "ROM", "rom": "ROM", "1corinthiens": "1CO", "2corinthiens": "2CO",
    "galates": "GAL", "ephesiens": "EPH", "philippiens": "PHP", "php": "PHP", "colossiens": "COL", "col": "COL",
    "1thessaloniciens": "1TH", "2thessaloniciens": "2TH", "1timothee": "1TI", "2timothee": "2TI", "tite": "TIT",
    "philemon": "PHM", "hebreux": "HEB", "jacques": "JAS", "1pierre": "1PE", "2pierre": "2PE", "1jean": "1JN",
    "2jean": "2JN", "3jean": "3JN", "jude": "JUD", "apocalypse": "REV",
}
USFM = set(BOOKS.values()) | {
    "1CO", "2CO", "GAL", "EPH", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS", "1PE", "2PE",
    "1JN", "2JN", "3JN", "JUD", "REV", "JOS", "JDG", "1SA", "2SA", "1KI", "2KI", "PSA", "PRO", "ISA"}

COL_BOOK = {"livre", "book", "bk"}
COL_CHAP = {"chapitre", "chapter", "chap", "ch", "c"}
COL_VERSE = {"verset", "verse", "v", "vs"}
COL_REF = {"ref", "reference", "référence", "locator", "verse_id"}
COL_TURN = {"tour", "turn", "ligne", "line", "n", "num", "index", "id"}
COL_SPEAKER = {"locuteur", "speaker", "personne"}
COL_PLURAL = {"pluriel", "plural"}
COL_SINGULAR = {"singulier", "singular"}
COL_POS = {"pos", "categorie_grammaticale", "nature"}
COL_CLASS = {"classe", "noun_class", "classe_nominale"}


def _fold(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s.strip().lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace(" ", "_")


def book_code(s: str | None) -> str | None:
    if not s:
        return None
    k = _fold(s).replace("_", "")
    if k.upper() in USFM:
        return k.upper()
    return BOOKS.get(k)


def book_from_filename(name: str) -> str | None:
    for tok in re.split(r"[_\-. ]+", _fold(name)):
        if tok in BOOKS:
            return BOOKS[tok]
    return None


_REF = re.compile(r"^\s*([1-3]?\s?[A-Za-zÀ-ÿ]+)\.?\s*(\d+)[:.,](\d+)")


_VERSE_TAG = re.compile(r"(\d+)_(?:verset|vers|v)_?(\d+)$")


def parse_ref(s: str) -> tuple[str | None, str | None, str | None]:
    m = _REF.match(s or "")
    if not m:
        return None, None, None
    return book_code(m.group(1)), m.group(2), m.group(3)


# ------------------------------------------------------------------------------------------------ CSV
def parse_csv(src_id: str, entry: dict, aliases: dict) -> list[dict]:
    path = ROOT / entry["path"]
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(raw), dialect))
    if not rows:
        return []
    header = [_fold(h) for h in rows[0]]
    col_lang = {i: aliases[h] for i, h in enumerate(header) if h in aliases}
    fr_cols = [i for i, lg in col_lang.items() if lg == "fr"]
    tgt_cols = {i: lg for i, lg in col_lang.items() if lg != "fr"}
    idx = {h: i for i, h in enumerate(header)}

    def find(names):
        return next((idx[n] for n in names if n in idx), None)

    c_book, c_chap, c_verse, c_ref = find(COL_BOOK), find(COL_CHAP), find(COL_VERSE), find(COL_REF)
    c_turn, c_spk = find(COL_TURN), find(COL_SPEAKER)
    c_sing, c_plur = find(COL_SINGULAR), find(COL_PLURAL)
    c_pos, c_cls = find(COL_POS), find(COL_CLASS)

    hint_lang = aliases.get(_fold(entry.get("lang_hint", "")))
    if not tgt_cols and c_sing is not None:                  # singulier,pluriel,french
        tgt_cols = {c_sing: hint_lang}
    if not tgt_cols:
        # e.g. "number,fang" has the target col; "number" acts as the gloss
        others = [i for i in range(len(header)) if i not in col_lang]
        if hint_lang and len(header) == 2 and others:
            tgt_cols = {1 - others[0]: hint_lang}
    if not fr_cols and "number" in idx:
        fr_cols = [idx["number"]]
    used = set(tgt_cols) | set(fr_cols) | {c for c in (c_book, c_chap, c_verse, c_ref, c_turn, c_spk,
                                                         c_plur, c_pos, c_cls) if c is not None}
    file_book = book_from_filename(path.name)
    kind = entry["kind"]
    out = []
    for r_i, row in enumerate(rows[1:], start=1):
        if not any(c.strip() for c in row):
            continue
        def get(c, row=row):
            return row[c].strip() if c is not None and c < len(row) and row[c].strip() else None

        fr = next((get(c) for c in fr_cols if get(c)), None)
        extra = {header[i]: row[i].strip() for i in range(min(len(row), len(header)))
                 if i not in used and row[i].strip()}
        if c_spk is not None and get(c_spk):
            extra["speaker"] = get(c_spk)
        group = entry.get("group") or src_id
        locator = str(get(c_turn) or r_i) if kind == "dialogue_turn" else str(r_i)
        if kind == "verse":
            b, ch, v = None, get(c_chap), get(c_verse)
            if c_ref is not None:
                b, ch2, v2 = parse_ref(get(c_ref) or "")
                ch, v = ch or ch2, v or v2
            if not (ch and v):                               # e.g. theme = "genese_1_verset_1"
                for val in extra.values():
                    m = _VERSE_TAG.search(_fold(val))
                    if m:
                        b = b or book_from_filename(val)
                        ch, v = m.group(1), m.group(2)
                        break
            b = book_code(get(c_book)) or b or file_book
            if b and ch and v:
                locator = f"{b}.{ch}.{v}"
            elif b:
                locator = f"{b}.r{r_i}"
            group = f"bible/{b}" if b else src_id
        for c, lang in tgt_cols.items():
            text = get(c)
            if not text and not fr:
                continue
            rec = {"lang": lang, "text_raw": text or "", "fr_raw": fr, "kind": kind, "locator": locator,
                   "group": group, "order": r_i, "align": {"method": "verse" if kind == "verse" else "row",
                                                           "score": 1.0},
                   "extra": extra}
            if kind == "lexicon" or c_plur is not None:
                rec["lex"] = {"headword": text, "pos": get(c_pos), "noun_class": get(c_cls),
                              "plural": get(c_plur), "gloss_fr": fr, "examples": []}
            out.append(rec)
    return out


# ------------------------------------------------------------------------------------------------ TXT
def parse_txt(src_id: str, entry: dict, aliases: dict) -> list[dict]:
    """Monolingual text: one segment per non-empty line; blank lines separate paragraphs."""
    path = ROOT / entry["path"]
    lang = aliases.get(_fold(entry.get("lang_hint", "")))
    kind = entry["kind"] if entry["kind"] in ("poem_line", "song_line") else "sentence"
    out, order = [], 0
    for ln, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        order += 1
        out.append({"lang": lang, "text_raw": line.strip(), "fr_raw": None, "kind": kind,
                    "locator": f"L{ln}", "group": entry.get("group") or src_id, "order": order,
                    "align": {"method": "none", "score": None}, "extra": {}})
    return out


# --------------------------------------------------------------------------------------------- v0 JSONL
_V0_REF_KEYS = ("ref", "verse_id", "locator", "id", "verse_ref")
_V0_SCORE_KEYS = ("score", "align_score", "sim", "similarity", "cos")


def parse_v0_jsonl(src_id: str, entry: dict, aliases: dict) -> list[dict]:
    """Adapter for v0 aligned corpora (fang-nt-parallel.scored, punu-bible). Field names are sniffed."""
    path = ROOT / entry["path"]
    out = []
    for i, r in enumerate(read_jsonl(path), start=1):
        lang = aliases.get(_fold(str(r.get("lang", entry.get("lang_hint", "")))))
        text = r.get("text") or r.get("tgt") or r.get(lang) or ""
        fr = r.get("fr") or r.get("french") or r.get("src")
        ref = next((str(r[k]) for k in _V0_REF_KEYS if r.get(k)), None)
        if not ref and r.get("notes"):
            m = re.search(r"([1-3]?[A-Z]{2,3})[ .](\d+)[:.](\d+)", r["notes"])
            ref = m.group(0) if m else None
        b, ch, v = parse_ref(ref.replace(".", " ", 1) if ref and re.match(r"^[1-3]?[A-Z]{2,3}\.", ref) else ref or "")
        b = b or book_code(r.get("book"))
        ch, v = ch or r.get("chapter"), v or r.get("verse")
        locator = f"{b}.{ch}.{v}" if b and ch and v else f"r{i}"
        score = next((float(r[k]) for k in _V0_SCORE_KEYS if isinstance(r.get(k), (int, float))), None)
        out.append({"lang": lang, "text_raw": text, "fr_raw": fr, "kind": "verse", "locator": locator,
                    "group": f"bible/{b}" if b else src_id, "order": i,
                    "align": {"method": "verse" if score is None else "heuristic", "score": score},
                    "extra": {"v0_keys": sorted(r.keys())} if i == 1 else {}})
    return out


# --------------------------------------------------------------------------------------------- Galley
_G_PLURAL = re.compile(r"\b(?:pl|sing)\.\s*([^,.;()]+)")
_G_DERIV = re.compile(r"\((?:vb|cf\.?|voir)\s[^)]*\)")
_G_LEAD = re.compile(r"^[\s,.;:'’\-–]*(?:(?:n|vb|adj|adv|pr|conj|interj|loc|inv|v)\.?\s?[\dlI]?[\s.,]+)*")
_G_SENSE = re.compile(r"^\d+\.\s*")


def parse_galley_entry(entry: str) -> dict:
    """Galley 1964 entry body -> {gloss, plural, deriv}. The body is: [pos info], [pl. X], [(vb ...)], gloss
    sentence(s), then embedded examples ("Fang phrase, French phrase."). The gloss is the first sentence after
    the grammatical prefix. Heuristic (OCR); the full body is kept in fr_note."""
    body = entry.strip()
    plural = None
    m = _G_PLURAL.search(body[:80])
    if m:
        plural = m.group(1).strip()
        body = body[:m.start()] + body[m.end():]
        if m.group(0).startswith("sing"):
            plural = None                                      # headword is the plural; keep body clean
    deriv = None
    m = _G_DERIV.search(body[:120])
    if m:
        deriv = m.group(0).strip("()")
        body = body[:m.start()] + body[m.end():]
    body = _G_SENSE.sub("", _G_LEAD.sub("", body).strip(" ,.;"))     # "n.l," (OCR 1->l), then sense "1."
    body = re.split(r"\s—\s*\d+\.", body, maxsplit=1)[0]                 # keep sense 1 only
    gloss = re.split(r"(?<=[.;])\s", body, maxsplit=1)[0].strip(" .;,")
    return {"gloss": gloss or None, "plural": plural, "deriv": deriv}


def parse_v0_galley(src_id: str, entry: dict, aliases: dict) -> list[dict]:
    """v0 Galley rows: headwords (kind=headword, fr = full entry) or examples (notes headword=X)."""
    path = ROOT / entry["path"]
    out = []
    for i, r in enumerate(read_jsonl(path), start=1):
        lang = aliases.get(_fold(str(r.get("lang", "fang"))))
        notes = r.get("notes", "") or ""
        if r.get("kind") == "headword":
            g = parse_galley_entry(r.get("fr") or "")
            tone = re.search(r"tone=(\w+)", notes)
            out.append({"lang": lang, "text_raw": r["text"], "fr_raw": g["gloss"], "kind": "lexicon",
                        "locator": f"h{i}", "group": entry.get("group") or src_id, "order": i,
                        "align": {"method": "manual", "score": None},
                        "extra": {"fr_note": r.get("fr"), **({"tone": tone.group(1)} if tone else {}),
                                  **({"deriv": g["deriv"]} if g["deriv"] else {})},
                        "lex": {"headword": r["text"], "pos": r.get("pos"), "noun_class": None,
                                "plural": g["plural"], "gloss_fr": g["gloss"], "examples": []}})
        else:
            hw = re.search(r"headword=([^;]+)", notes)
            out.append({"lang": lang, "text_raw": r["text"], "fr_raw": r.get("fr"), "kind": "example",
                        "locator": f"e{i}", "group": entry.get("group") or src_id, "order": i,
                        "align": {"method": "manual", "score": None},
                        "extra": {"headword": hw.group(1).strip()} if hw else {}})
    return out


from .galley import parse_galley_ocr  # noqa: E402


def parse_nt_aligned(src_id: str, entry: dict, aliases: dict) -> list[dict]:
    """Realigned Fang NT (data/interim/fang_nt_aligned.jsonl from nt_align.py): one record per bead. The French
    side is LS1910; the locator is the first French verse of the bead. Books are gated in build by config."""
    out = []
    for i, r in enumerate(read_jsonl(ROOT / entry["path"]), start=1):
        loc = f"{r['book']}.{r['chapter']}.{r['fr_verses'][0]}"
        out.append({"lang": aliases.get("fang", "fan"), "text_raw": r["text"], "fr_raw": r["fr"], "kind": "verse",
                    "locator": loc, "group": f"bible/{r['book']}", "order": i,
                    "align": {"method": "bead_dp", "score": r.get("sim")},
                    "extra": {"bead": r["bead"], "fang_verses": r["fang_verses"], "fr_verses": r["fr_verses"]}})
    return out

PARSERS = {"csv": parse_csv, "txt": parse_txt, "v0_jsonl": parse_v0_jsonl, "v0_galley": parse_v0_galley,
           "galley_ocr": parse_galley_ocr, "nt_aligned": parse_nt_aligned}
