"""Text normalization (DATA_SPEC §2). Never strips diacritics or tone marks."""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter

_WS = re.compile(r"\s+")
_FOOTNOTE = re.compile(r"\[\d+\]|\(\d+\)|[*†‡]+(?=\s|$)")
_VERSE_NUM = re.compile(r"^\s*\d{1,3}[a-z]?[\s.:)\-]+(?=\D)")
_HTML_TAG = re.compile(r"<[^>]+>")


class Normalizer:
    def __init__(self, cfg: dict):
        n = cfg["normalize"]
        self.form = n.get("form", "NFC")
        self.lookalikes = dict(n.get("lookalikes") or {})
        self.per_lang = n.get("per_lang_lookalikes") or {}
        self.apos_variants = n.get("apostrophe_variants") or []
        self.apos_canon = n.get("apostrophe_canonical") or {"default": "'"}
        self.strip_verse_numbers = n.get("strip_verse_numbers", True)
        # logging of what was actually changed, for reports/normalization.json
        self.lookalike_hits: Counter = Counter()
        self.apostrophe_seen: Counter = Counter()
        self.case_events: list[dict] = []

    def _common(self, s: str, verse: bool) -> str:
        s = html.unescape(s)
        s = _HTML_TAG.sub(" ", s)
        s = unicodedata.normalize(self.form, s)
        s = _FOOTNOTE.sub(" ", s)
        if verse and self.strip_verse_numbers:
            s = _VERSE_NUM.sub("", s)
        return _WS.sub(" ", s).strip()

    def target(self, s: str | None, lang: str, verse: bool = False) -> str | None:
        if s is None:
            return None
        s = self._common(s, verse)
        table = {**self.lookalikes, **(self.per_lang.get(lang) or {})}
        if table:
            out = []
            for ch in s:
                rep = table.get(ch)
                if rep is not None and rep != ch:
                    self.lookalike_hits[f"{lang}:{ch}->{rep}"] += 1
                    if ch.isupper():                           # case-preserving map; log context (v0.8)
                        i = len(out)
                        prev = "".join(out).rstrip()
                        self.case_events.append({
                            "lang": lang, "char": f"{ch}->{rep}", "context": s[max(0, i - 15):i + 15],
                            "suspect_case": bool(prev) and prev[-1] not in ".!?:;\"«“("})
                    out.append(rep)
                else:
                    out.append(ch)
            s = "".join(out)
        canon = self.apos_canon.get(lang, self.apos_canon.get("default", "'"))
        variants = set(self.apos_variants)
        if variants & set(s):
            s = _elision(s, variants, canon, lang, self.apostrophe_seen)
        # re-compose after substitutions
        return unicodedata.normalize(self.form, s)

    def french(self, s: str | None, verse: bool = False) -> str | None:
        if s is None:
            return None
        s = self._common(s, verse)
        s = s.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
        return s or None

    def report(self) -> dict:
        return {
            "form": self.form,
            "lookalike_table": self.lookalikes,
            "per_lang_lookalikes": self.per_lang,
            "lookalike_hits": dict(self.lookalike_hits.most_common()),
            "uppercase_lookalike_events": self.case_events,
            "apostrophe_occurrences_in_target": dict(self.apostrophe_seen.most_common()),
            "apostrophe_canonical": self.apos_canon,
        }


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("L", "M")


def _elision(s: str, variants: set, canon: str, lang: str, seen: Counter) -> str:
    """Spec v0.6: an apostrophe variant is an elision mark only BETWEEN two letters (combining marks count as
    part of a letter); elsewhere (quotes, leading/trailing) it is left untouched."""
    out = list(s)
    for i, ch in enumerate(s):
        if ch in variants:
            between = 0 < i < len(s) - 1 and _is_letter(s[i - 1]) and _is_letter(s[i + 1])
            seen[f"{lang}:U+{ord(ch):04X}:{'elision' if between else 'other'}"] += 1
            if between and ch != canon:
                out[i] = canon
    return "".join(out)


def text_key(s: str | None) -> str:
    """Conservative key (spec v0.2): NFC + casefold + punctuation/whitespace stripped. KEEPS diacritics/tones.
    Used for within-train dedup and first-pass lexicon lookup. Never embedded or shown to a model."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s).casefold()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    return _WS.sub(" ", s).strip()


def text_fold(s: str | None) -> str:
    """Aggressive key (spec v0.2): text_key with diacritics stripped. Used for cross-split leak removal and
    lexicon lookup fallback only."""
    k = unicodedata.normalize("NFD", text_key(s))
    return unicodedata.normalize("NFC", "".join(ch for ch in k if unicodedata.category(ch) != "Mn"))
