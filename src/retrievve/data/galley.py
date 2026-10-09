"""Galley 1964 Fang-French dictionary: v1 parser straight from the OCR text (replaces the v0 headword parse).

Fixes the systematic errors found by 88's spot check (PR #19):
- line-break dehyphenation: a soft hyphen (U+00AD) at line end is a word split -> joined without hyphen;
  a hard hyphen at line end is a real Fang compound hyphen (balo-nẑoghe) -> kept, joined without space;
- every tone group after the headword is captured ("(bm)", "(h ou m)"), OCR I->l inside tone groups;
- POS OCR misreads normalised: n,4 -> n.4, n.l / n.I -> n.1, n.S -> n.3;
- running headers / section letters / page numbers dropped;
- entries are split on headword lines (ALL-CAPS + tone group), so a missed headword no longer swallows the
  next entry as often as in v0.
Only the Fang->French part is parsed: the French->Fang index (after the "FRANÇAIS-FANG" title) has French
headwords and is out of scope here.
Italic (Fang) vs roman (French) runs are not in the plain text, so embedded examples are NOT split out; the
examples source stays excluded (7d, after #19).
"""
from __future__ import annotations

import re
import unicodedata

_UP = "A-ZÀ-ÖØ-ÞÑƆƐŊẐṼÜÏ"
HEADWORD = re.compile(rf"^([{_UP}][{_UP}'\- ]{{0,40}}?)\s*\(([A-Za-z ]{{1,10}})\)")
TONE = re.compile(r"^[bhmfklIg]{1,6}(?:\s+ou\s+[bhmfklIg]{1,4})?$")
TONE_GROUP = re.compile(r"^\s*\(([bhmfklIg ou]{1,12})\)")
POS = re.compile(r"^[\s,.]*(n\s?[.,]?\s?[\dlISO]|vb\.?\s*(?:r[ée]cip|[ée]tatif|aux|intr|tr)\.?|(?:vb|adj|adv|pr|conj|interj|loc|inv|n))(?=[\s,.;:]|$)")
FR_SECTION = re.compile(r"^\s*FRAN[ÇC]AIS\s*-\s*FANG\s*$")
HEADER = re.compile(rf"^\s*(?:[{_UP}]{{1,3}}|\d{{1,3}}|DICTIONNAIRE.*|FANG\s*-\s*FRAN[ÇC]AIS.*|"
                    rf"[{_UP}\-]+\s*[—-]\s*[{_UP}\-]+|[{_UP}]{{2,6}}\s+\d{{1,3}}\s+[{_UP}]{{2,6}})\s*$")


def fold_upper(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").upper()


def norm_pos(p: str | None) -> str | None:
    if not p:
        return None
    m = re.match(r"vb\.?\s*(r[ée]cip|[ée]tatif|aux|intr|tr)", p)
    if m:
        return "vb." + {"recip": "récip", "etatif": "étatif"}.get(m.group(1).replace("é", "e"), m.group(1))
    p = re.sub(r"\s+", "", p).replace(",", ".")
    p = re.sub(r"^n\.?([lI])$", "n.1", p)
    p = re.sub(r"^n\.?S$", "n.3", p)
    p = re.sub(r"^n\.?O$", "n.0", p)
    p = re.sub(r"^n(\d)$", r"n.\1", p)
    return p.rstrip(".")


# running page header that landed inside an entry: "KAG 147 KAL" (first/last headword stems + page number)
INLINE_HEADER = re.compile(rf"\s(?:[{_UP}]{{2,6}}\s+\d{{1,3}}\s+[{_UP}]{{2,6}}|"
                           rf"(?:[{_UP}]{{2,6}}\s*)?—\s*\d{{1,3}}\s*—(?:\s*[{_UP}]{{2,6}})?)(?=\s)")


def join_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        ln = ln.rstrip()
        if not out:
            out = ln.strip()
        elif out.endswith("­"):                     # soft hyphen: split word
            out = out[:-1] + ln.lstrip()
        elif out.endswith("-"):                          # hard hyphen: real compound hyphen
            out = out + ln.lstrip()
        else:
            out = f"{out} {ln.strip()}"
    out = INLINE_HEADER.sub(" ", f" {out} ")                # running page headers inside an entry
    return re.sub(r"\s+", " ", out.replace("\u00ad", "")).strip()


def split_entries(text: str) -> tuple[list[tuple[str, str]], dict]:
    """OCR text -> [(headword_line_head, body)], plus stats. Body includes tone/pos/gloss text."""
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if HEADWORD.match(ln) and TONE.match(
        HEADWORD.match(ln).group(2).strip())), 0)
    entries, cur_head, cur_body = [], None, []
    stats = {"headers_dropped": 0, "lines": 0}
    for ln in lines[start:]:
        if FR_SECTION.match(ln):
            break
        stats["lines"] += 1
        if HEADER.match(ln):
            stats["headers_dropped"] += 1
            continue
        m = HEADWORD.match(ln)
        if m and TONE.match(m.group(2).strip()):
            if cur_head is not None:
                entries.append((cur_head, join_lines(cur_body)))
            cur_head, cur_body = m.group(1).strip(), [ln[m.end(1):]]
        elif cur_head is not None:
            cur_body.append(ln)
    if cur_head is not None:
        entries.append((cur_head, join_lines(cur_body)))
    stats["entries"] = len(entries)
    return entries, stats


def parse_entry(head: str, body: str) -> dict:
    """-> {headword, tones, pos, rest} where rest is the definition text (plural/deriv/gloss/examples)."""
    tones, rest = [], body
    while True:
        m = TONE_GROUP.match(rest)
        if not m or not TONE.match(m.group(1).strip()):
            break
        tones.append(m.group(1).strip().replace("I", "l"))
        rest = rest[m.end():]
    m = POS.match(rest)
    pos = None
    if m:
        pos = norm_pos(m.group(1))
        rest = rest[m.end():]
    return {"headword": re.sub(r"\s+", " ", head), "tones": tones, "pos": pos, "rest": rest.strip(" ,.;")}


_NOTES = [(re.compile(r"^(?:ss|sans)\s+sing\.?\s*", re.I), "no_singular"),
          (re.compile(r"^(?:ss|sans)\s+pl\.?\s*", re.I), "no_plural"),
          (re.compile(r"^voir\s+le\s+sing\.?\s*", re.I), "see_singular"),
          (re.compile(r"^pl\.?\s+peu\s+employ[ée]\.?\s*", re.I), "plural_rare"),
          (re.compile(r"^inv\.?\s*", re.I), "invariable")]
_SENSE = re.compile(r"^\d+[.,]\s*")                      # "1." (or OCR "1,")
_PLURAL = re.compile(r"^(?:pl|sing)[.,]\s*([^,.;()]+)[,.;]?\s*")
_ABBREV = ("par ex.", "c.-à-d.", "etc.", "cf.", "p. ex.")
_SYN = re.compile(r"[,.;]?\s*\bSyn\s*[.,]?\s*:")
# OCR'd period as comma before an embedded Fang example: ", " + a capitalised word followed by more text
_EXAMPLE_START = re.compile(rf",\s+(?=[{_UP}][^\s,]*\s+[^\s,]+)")
_DERIV = re.compile(r"^\((?:vb|cf\.?|voir)\s[^)]*\)[.,]?\s*")
_LITERAL = re.compile(r"^\(([^)]{2,60})\)[.,]?\s*")
_BASE = re.compile(r"^(?:de|du)\s+([^().,;]{1,30}?)\s*\(([bhmf ]{1,8})\)[.,]?\s*")   # "récip. de zimé (h)."


def definition(rest: str, headword: str | None = None) -> dict:
    """Definition text after headword/tone/pos -> {gloss, plural, deriv, literal, notes}. Strips, in any order,
    the leading grammatical material (plural, 'ss sing.', derivation, literal translation in parentheses, sense
    number) and returns the first sentence of sense 1 as the gloss."""
    s, plural, deriv, literal, notes = rest.strip(" ,.;"), None, None, None, []
    for _ in range(6):
        before = s
        s = s.lstrip(" ,.;")
        m = _PLURAL.match(s)
        if m:
            if m.group(0).lower().startswith("pl"):
                plural = m.group(1).strip()
            s = s[m.end():]
        for rx, tag in _NOTES:
            m = rx.match(s)
            if m:
                notes.append(tag)
                s = s[m.end():]
        m = _DERIV.match(s)
        if m:
            deriv = m.group(0).strip(" .,()")
            s = s[m.end():]
        m = _BASE.match(s)
        if m:
            deriv = f"de {m.group(1).strip()} ({m.group(2).strip()})"
            s = s[m.end():]
        m = _LITERAL.match(s)
        if m:
            literal = m.group(1).strip()
            s = s[m.end():]
        s = _SENSE.sub("", s)
        if s == before:
            break
    s = re.split(r"\s—\s*\d+[.,]", s, maxsplit=1)[0]                      # keep sense 1 only
    s = _SYN.split(s, maxsplit=1)[0]                                      # synonyms list is not the gloss
    protected = s
    for i, ab in enumerate(_ABBREV):                                      # don't split on abbreviations
        protected = protected.replace(ab, f"\x00{i}\x00")
    first = re.split(r"(?<=[.;])\s", protected, maxsplit=1)[0]
    m = _EXAMPLE_START.search(first)
    if m and m.start() >= 3:
        first = first[:m.start()]
    for i, ab in enumerate(_ABBREV):
        first = first.replace(f"\x00{i}\x00", ab)
    gloss = first.strip(" .;,")
    if headword and gloss and fold_upper(gloss.split()[0]) == fold_upper(headword.split()[0]):
        gloss = ""                                   # it's an embedded Fang example, not a definition
    return {"gloss": gloss or None, "plural": plural, "deriv": deriv, "literal": literal, "notes": notes}


def parse_galley_ocr(src_id: str, entry: dict, aliases: dict) -> list[dict]:
    """Parser entry point (PARSERS['galley_ocr']): one lexicon row per Fang->French entry."""
    from .common import ROOT

    text = (ROOT / entry["path"]).read_text(encoding="utf-8")
    entries, _stats = split_entries(text)
    seen: dict[str, int] = {}
    out = []
    for i, (head, body) in enumerate(entries, start=1):
        e = parse_entry(head, body)
        d = definition(e["rest"], e["headword"])
        key = re.sub(r"[^0-9A-Z]+", "_", fold_upper(e["headword"])).strip("_") or f"E{i}"
        seen[key] = seen.get(key, 0) + 1
        locator = f"{key}.{seen[key]}"                    # stable across parser tweaks (headword + homograph #)
        extra = {"fr_note": e["rest"], "tones": e["tones"]}
        for k in ("deriv", "literal"):
            if d[k]:
                extra[k] = d[k]
        if d["notes"]:
            extra["grammar_notes"] = d["notes"]
        out.append({"lang": aliases.get("fang", "fan"), "text_raw": e["headword"], "fr_raw": d["gloss"],
                    "kind": "lexicon", "locator": locator, "group": entry.get("group") or src_id, "order": i,
                    "align": {"method": "manual", "score": None}, "extra": extra,
                    "lex": {"headword": e["headword"], "pos": e["pos"], "noun_class": None,
                            "plural": d["plural"], "gloss_fr": d["gloss"], "examples": []}})
    return out
