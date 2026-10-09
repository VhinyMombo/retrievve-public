"""Quality filters and dedup (DATA_SPEC §3 cross-split dedup, §4 filters). All thresholds come from config."""
from __future__ import annotations

import re
import zlib
from collections import Counter, defaultdict

from .normalize import text_fold, text_key

_TOK = re.compile(r"[\w']+", re.UNICODE)

BLOCKING = {"empty", "fr_in_target", "len_ratio", "low_align", "exact_dup", "near_dup", "xsplit_dup",
            "ocr_review"}


def lexicon_tokens(records: list[dict]) -> dict[str, set]:
    """Per language: tokens of lexicon headwords and of their linked target-side examples (non-test). Records
    under OCR review never contribute: OCR'd dictionary fields can hold French (e.g. a gloss in the target
    field), which would exempt French words for the whole language."""
    by_id = {r["id"]: r for r in records}
    out = defaultdict(set)
    for r in records:
        if r["kind"] != "lexicon" or r["split"] == "test" or "ocr_review" in r["flags"]:
            continue
        out[r["lang"]].update(_TOK.findall((r["text"] or "").lower()))
        for ex_id in (r.get("lex") or {}).get("examples") or []:
            ex = by_id.get(ex_id)
            if ex and ex["split"] != "test" and "ocr_review" not in ex["flags"]:
                out[r["lang"]].update(_TOK.findall((ex["text"] or "").lower()))
    return out


def per_lang_stopwords(records: list[dict], fcfg: dict,
                       lex_tokens: dict[str, set] | None = None) -> tuple[dict[str, set], dict[str, list]]:
    """Spec v0.4 (final): config list minus words that are lexicon headwords, or tokens of target-side lexicon
    examples, in that language. Frequency in running text is NOT used: French copy-paste rows would exempt
    themselves. Test records never contribute. Returns (lang -> stopword set, lang -> exempted words)."""
    base = {w.lower() for w in fcfg["fr_stopwords"]}
    lex_tokens = lex_tokens if lex_tokens is not None else lexicon_tokens(records)
    out, removed = {}, {}
    for lang in {r["lang"] for r in records}:
        rm = sorted(base & lex_tokens.get(lang, set()))
        out[lang], removed[lang] = base - set(rm), rm
    return out, removed


def french_vocab(fcfg: dict, root) -> set[str]:
    """Spec v0.6 vocabulary detector: LS1910 tokens of >= min_len letters with frequency >= min_freq."""
    import json

    path = root / fcfg["fr_vocab_ref"]
    if not path.exists() or path.stat().st_size == 0:
        return set()
    cnt = Counter()
    for v in json.loads(path.read_text(encoding="utf-8")):
        cnt.update(t.lower() for t in _TOK.findall(v["text"]) if len(t) >= fcfg["fr_vocab_min_token_len"])
    return {t for t, c in cnt.items() if c >= fcfg["fr_vocab_min_freq"]}


def fr_vocab_share(text: str, vocab: set, min_len: int) -> tuple[float, int]:
    toks = [t.lower() for t in _TOK.findall(text or "") if len(t) >= min_len]
    if not toks:
        return 0.0, 0
    return sum(t in vocab for t in toks) / len(toks), len(toks)


def fr_stopword_stats(text: str, fcfg: dict, stop: set) -> tuple[float, int, int]:
    """(ratio, hits, n_tokens). Stopword list (per language) and min token length come from config."""
    min_len = fcfg.get("fr_stopword_min_token_len", 1)
    toks = [t.lower() for t in _TOK.findall(text or "")]
    if not toks:
        return 0.0, 0, 0
    hits = sum(len(t) >= min_len and t in stop for t in toks)
    return hits / len(toks), hits, len(toks)


def flag_record(r: dict, fcfg: dict, align_cut: float | None, stop: set, vocab: set | None = None) -> None:
    """Per-record filters (parallel pairs only, except `empty`)."""
    flags = r["flags"]
    text, fr = r["text"] or "", r["fr"]
    if len(text) < fcfg["min_chars"] or (fr is not None and len(fr) < 1):
        flags.append("empty")
        return
    if fr is None:
        return
    if r["kind"] in ("lexicon", "grammar"):
        return                                              # 1-word glosses: ratio/stopword tests meaningless
    ratio, hits, ntok = fr_stopword_stats(text, fcfg, stop)
    if (ntok >= fcfg["fr_stopword_min_tokens"] and ratio > fcfg["fr_stopword_ratio_max"]
            and hits >= fcfg.get("fr_stopword_min_hits", 1)):
        flags.append("fr_in_target")
    elif vocab:
        share, n = fr_vocab_share(text, vocab, fcfg["fr_vocab_min_token_len"])
        # v0.8 §4.2b precondition: at least fr_vocab_min_qualifying tokens of >= min_token_len letters (a single
        # word like Fang "été"/Punu "mange" is a homograph, not French) — the blind spot is reported separately
        if n >= fcfg.get("fr_vocab_min_qualifying", 2) and share >= fcfg["fr_vocab_share_max"]:
            flags.append("fr_in_target")
    m = fcfg["len_ratio_min_chars"]
    if len(text) >= m and len(fr) >= m:
        lr = len(text) / max(len(fr), 1)
        if not fcfg["len_ratio_min"] <= lr <= fcfg["len_ratio_max"]:
            flags.append("len_ratio")
    s = (r.get("align") or {}).get("score")
    if align_cut is not None and s is not None and s < align_cut:
        flags.append("low_align")


def align_cutoff(records: list[dict], fcfg: dict) -> float | None:
    if fcfg.get("low_align_abs") is not None:
        return float(fcfg["low_align_abs"])
    if fcfg.get("low_align_percentile") is None:
        return None
    scores = sorted(r["align"]["score"] for r in records
                    if r["align"].get("method") == "heuristic" and r["align"].get("score") is not None)
    if not scores:
        return None
    k = int(len(scores) * fcfg["low_align_percentile"] / 100)
    return scores[min(k, len(scores) - 1)]


def usable(r: dict) -> bool:
    """Usable for SFT, LM and eval."""
    return not (set(r["flags"]) & BLOCKING)


def usable_rag(r: dict) -> bool:
    """RAG: like usable(), but OCR-under-review entries are allowed (spec v0.5 §4.6; chunk carries a marker)."""
    return not (set(r["flags"]) & (BLOCKING - {"ocr_review"}))


# --------------------------------------------------------------------------------------------- MinHash
_P = (1 << 61) - 1


class MinHasher:
    def __init__(self, k: int, perms: int, bands: int, seed: int):
        import random
        rnd = random.Random(seed)
        self.k, self.bands, self.rows = k, bands, perms // bands
        self.ab = [(rnd.randrange(1, _P), rnd.randrange(0, _P)) for _ in range(perms)]

    def shingles(self, s: str) -> set[int]:
        s = f" {s} "
        if len(s) <= self.k:
            return {zlib.crc32(s.encode())}
        return {zlib.crc32(s[i:i + self.k].encode()) for i in range(len(s) - self.k + 1)}

    def signature(self, sh: set[int]) -> list[int]:
        return [min((a * x + b) % _P for x in sh) for a, b in self.ab]

    def bands_of(self, sig: list[int]):
        for b in range(self.bands):
            yield b, tuple(sig[b * self.rows:(b + 1) * self.rows])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def near_dup_pairs(keys: dict[str, str], mh: MinHasher, thr: float, min_len: int = 12,
                   report: Counter | None = None, max_bucket: int = 500):
    """keys: id -> folded string. Yields (id1, id2, jaccard) for candidate pairs >= thr.
    Buckets larger than max_bucket are skipped (boilerplate) and counted in `report` (N2)."""
    sh = {i: mh.shingles(s) for i, s in keys.items() if len(s) >= min_len}
    buckets = defaultdict(list)
    for i, s in sh.items():
        for band in mh.bands_of(mh.signature(s)):
            buckets[band].append(i)
    seen = set()
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        if len(ids) > max_bucket:                            # huge buckets = boilerplate; skip but report
            if report is not None:
                report["skipped_buckets"] += 1
                report["skipped_bucket_ids"] += len(ids)
            continue
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                p = (ids[x], ids[y]) if ids[x] < ids[y] else (ids[y], ids[x])
                if p in seen:
                    continue
                seen.add(p)
                j = jaccard(sh[p[0]], sh[p[1]])
                if j >= thr:
                    yield p[0], p[1], j


SPLIT_RANK = {"test": 0, "val": 1, "train": 2, "rag_pool": 2}


def exact_dups(records: list[dict]) -> int:
    """Exact dup on (lang, text_key, fr_key): keep the copy in the highest-ranked split (test > val >
    train/rag_pool), then the first in deterministic order — so a test pair is never lost to a train copy."""
    seen, n = set(), 0
    for r in sorted(records, key=lambda r: SPLIT_RANK.get(r["split"], 3)):   # stable: keeps original order
        if not usable(r):
            continue
        key = (r["lang"], r["text_key"], text_key(r["fr"]))
        if key in seen:
            r["flags"].append("exact_dup")
            n += 1
        else:
            seen.add(key)
    return n


def train_near_dups(records: list[dict], mh: MinHasher, thr: float, keep_kinds: set) -> list[dict]:
    """Near-dup within train (same lang): keep one, unless dialogue/lexicon. Returns clusters for the report."""
    by_id = {r["id"]: r for r in records}
    keys = {r["id"]: r["text_key"] for r in records
            if r["split"] == "train" and usable(r) and r["kind"] not in keep_kinds}
    clusters = defaultdict(set)
    for a, b, _j in near_dup_pairs(keys, mh, thr):
        if by_id[a]["lang"] != by_id[b]["lang"]:
            continue
        drop = max(a, b)                                     # deterministic: keep lexicographically smaller id
        keep = min(a, b)
        if "near_dup" not in by_id[drop]["flags"]:
            by_id[drop]["flags"].append("near_dup")
        clusters[keep].add(drop)
    return sorted(({"keep": k, "dropped": sorted(v)} for k, v in clusters.items()),
                  key=lambda c: -len(c["dropped"]))


def _long_enough(fold: str, min_words: int, min_chars: int) -> bool:
    return len(fold.split()) >= min_words and len(fold) >= min_chars


def cross_split_dedup(records: list[dict], mh: MinHasher, thr: float, fcfg: dict) -> dict:
    """Test wins over val wins over train/rag_pool. Keys are text_fold (aggressive by design, v0.2).
    Target side compared within a language; French side across ALL languages (v0.1).
    Short-segment guard (v0.4): a ONE-sided match is a leak only if that side has >= min_words words and
    >= min_chars chars; shorter formulae are dropped only when BOTH sides match the same held-out pair.
    MinHash only on >= min_chars. Lexicon entries are checked only when example-like (>= 3 target tokens).
    Returns drop counts per lang and rule."""
    rank = SPLIT_RANK
    mw, mc = fcfg["xsplit_min_words"], fcfg["xsplit_min_chars"]
    stats = defaultdict(Counter)
    skipped = Counter()
    held = sorted((r for r in records if r["split"] in ("test", "val")), key=lambda r: rank[r["split"]])
    by_t, by_f, by_pair = {}, {}, {}
    for r in held:
        f = text_fold(r["fr"])
        by_t.setdefault((r["lang"], r["text_fold"]), r["split"])
        if f:
            by_f.setdefault(f, r["split"])
            by_pair.setdefault((r["lang"], r["text_fold"], f), r["split"])

    def checked(r):
        return r["kind"] != "lexicon" or len(r["text_key"].split()) >= 3

    def drop(r, rule):
        if "xsplit_dup" not in r["flags"]:
            r["flags"].append("xsplit_dup")
            stats[r["lang"]][f"{rule}:{r['split']}"] += 1

    for r in records:
        if r["split"] == "test" or not checked(r):
            continue
        f = text_fold(r["fr"])
        hit_p = by_pair.get((r["lang"], r["text_fold"], f)) if f else None
        hit_t = by_t.get((r["lang"], r["text_fold"]))
        hit_f = by_f.get(f) if f else None
        if hit_p and rank[hit_p] < rank[r["split"]]:
            drop(r, "exact_both")
        elif hit_t and rank[hit_t] < rank[r["split"]] and _long_enough(r["text_fold"], mw, mc):
            drop(r, "exact_text")
        elif hit_f and rank[hit_f] < rank[r["split"]] and _long_enough(f, mw, mc):
            drop(r, "exact_fr")

    by_id = {r["id"]: r for r in records}
    for field in ("text", "fr"):
        keys = {}
        for r in records:
            if "xsplit_dup" in r["flags"] or not checked(r):
                continue
            v = r["text_fold"] if field == "text" else text_fold(r["fr"])
            if v and _long_enough(v, mw, mc):
                keys[r["id"]] = v
        for a, b, _j in near_dup_pairs(keys, mh, thr, min_len=mc, report=skipped):
            ra, rb = by_id[a], by_id[b]
            if field == "text" and ra["lang"] != rb["lang"]:
                continue
            if rank[ra["split"]] == rank[rb["split"]]:
                continue
            drop(ra if rank[ra["split"]] > rank[rb["split"]] else rb, f"near_{field}")
    out = {lang: dict(c) for lang, c in sorted(stats.items())}
    out["_minhash_skipped"] = dict(skipped)
    return out


def fr_detector_blind_spot(records: list[dict], vocab_by_lang: dict[str, set], lex_tokens: dict[str, set],
                           min_len: int, max_examples: int = 20) -> dict:
    """v0.8: one-token target rows (parallel, non-lexicon/grammar) whose token is in the French vocabulary and is
    not a lexicon headword of that language — the rows the >= 2-token precondition cannot judge."""
    hits = []
    for r in records:
        if not r["fr"] or r["kind"] in ("lexicon", "grammar"):
            continue
        toks = [t.lower() for t in _TOK.findall(r["text"] or "") if len(t) >= min_len]
        if len(toks) == 1 and toks[0] in vocab_by_lang.get(r["lang"], set()) \
                and toks[0] not in lex_tokens.get(r["lang"], set()):
            hits.append(r)
    by_lang = Counter(r["lang"] for r in hits)
    return {"count": len(hits), "by_lang": dict(by_lang),
            "examples": [{"id": r["id"], "text": r["text"], "fr": r["fr"]} for r in hits[:max_examples]]}
