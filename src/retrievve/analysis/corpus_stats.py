"""DATA_SPEC §8 stats report + [TBD-stats] threshold sweeps + v0 train leakage audit.

Needs pyyaml (filters are read from the pipeline config). Run from the repo root:
    retrievve stats [--config configs/v1_pipeline.yaml] [--out data/v1/reports]

Reads the v0 files as they are (data/corpus/*.jsonl, data/train/*/{train,valid,test}.jsonl).
Files that are still 0-byte Dropbox placeholders are listed as `unhydrated` and skipped.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from retrievve.paths import ROOT  # noqa: E402  (walks up to pyproject.toml)

# --- helpers ---------------------------------------------------------------

FR_STOP = set("""
le la les l un une des du de d et à a au aux en dans sur pour par avec sans ce cet cette ces
il elle ils elles on nous vous je j tu me m te t se s lui leur leurs son sa ses mon ma mes ton ta tes
notre nos votre vos qui que qu quoi dont où ne n pas plus est sont été être avoir ai as avons avez ont
mais ou donc or ni car si y comme tout tous toute toutes aussi très bien
""".split())

TEXT_KEYS = ("text", "target", "tgt", "fang", "punu", "x")
FR_KEYS = ("fr", "french", "src", "source_text")
SCORE_KEYS = ("score", "align_score", "sim", "similarity", "cos")
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Look-alike groups worth flagging in a char inventory.
LOOKALIKES = {
    "open-e": ["ɛ", "ε", "Ɛ", "Ε"],
    "open-o": ["ɔ", "ↄ", "Ɔ", "Ↄ"],
    "schwa": ["ə", "ǝ", "Ə", "Ǝ"],
    "eng": ["ŋ", "Ŋ"],
    "apostrophe": ["'", "’", "‘", "ʼ", "`", "´", "ʻ"],
    "hyphen": ["-", "‐", "‑", "–", "—"],
}


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def words(s):
    return [w.lower() for w in WORD_RE.findall(s or "")]


def fr_stop_ratio(s):
    ws = words(s)
    return sum(w in FR_STOP for w in ws) / len(ws) if ws else 0.0


def pick(rec, keys):
    for k in keys:
        if isinstance(rec.get(k), str) and rec[k].strip():
            return rec[k]
    return None


def pick_score(rec):
    for k in SCORE_KEYS:
        v = rec.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    al = rec.get("align")
    if isinstance(al, dict) and isinstance(al.get("score"), (int, float)):
        return float(al["score"])
    return None


def read_jsonl(path):
    out, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return out, bad


def domain_of(rec, fname):
    # source first: punu-phrases.jsonl is mostly Bible verses with kind=example
    blob = " ".join([str(rec.get("source", "")), fname, str(rec.get("kind", "")),
                     " ".join(map(str, rec.get("tags", []) or []))]).lower()
    for key, dom in (("bible", "religious"), ("wntfan", "religious"), ("genesis", "religious"),
                     ("-nt", "religious"), ("nt-", "religious"), ("verse", "religious"),
                     ("grammaire", "grammar"), ("grammar", "grammar"),
                     ("dictionary", "lexical"), ("lexique", "lexical"), ("vocab", "lexical"), ("headword", "lexical"),
                     ("dialog", "conversational"), ("conv_", "conversational"), ("quizlet", "conversational"),
                     ("tiktok", "conversational"), ("expression", "conversational"), ("phrase", "conversational"),
                     ("poem", "literary"), ("poesie", "literary"), ("proverb", "literary"), ("conte", "literary"),
                     ("song", "oral"), ("monolingual", "other")):
        if key in blob:
            return dom
    return "other"


# v0 files that are unions/copies of other files (measured: share of keys found in the other file = 1.00).
# Kept in per-file stats, excluded from rollups, distributions and sweeps to avoid double counting.
AGGREGATE_FILES = {
    "fang.entries.jsonl": "fang.jsonl + fang-examples + fang-phrases + fang-vocab",
    "punu.entries.jsonl": "deduplicated subset of punu-phrases",
    "fang-nt-parallel.jsonl": "same pairs as fang-nt-parallel.scored.jsonl (minus `sim`)",
}
# MT-pair population for the parallel filters: headwords (fr = dictionary definition) and grammar rules are not
# translation pairs.
NON_MT_KINDS = {"headword", "grammar"}

COMBINING_SEQ = re.compile("." + "[\u0300-\u036f\u1dc0-\u1dff]+")
TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def load_fr_filter(config_path):
    """fr_in_target settings from configs/v1_pipeline.yaml (single source of truth, PR #4)."""
    import yaml
    f = yaml.safe_load(Path(config_path).read_text())["filters"]
    return {"words": {w.lower().replace("’", "'") for w in f["fr_stopwords"]},
            "ratio_max": f["fr_stopword_ratio_max"], "min_tokens": f["fr_stopword_min_tokens"],
            "min_hits": f["fr_stopword_min_hits"], "min_len": f["fr_stopword_min_token_len"],
            "len_min": f["len_ratio_min"], "len_max": f["len_ratio_max"], "len_min_chars": f["len_ratio_min_chars"]}


def fr_hits(s, words_set, min_len):
    toks = [t.lower().replace("’", "'") for t in TOKEN_RE.findall(s or "")]
    return sum(len(t) >= min_len and t in words_set for t in toks), len(toks)


def hist(values, edges):
    counts = [0] * (len(edges) + 1)
    for v in values:
        i = 0
        while i < len(edges) and v >= edges[i]:
            i += 1
        counts[i] += 1
    labels = [f"<{edges[0]}"] + [f"[{a},{b})" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"]
    return dict(zip(labels, counts))


def quantiles(values, qs=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)):
    if not values:
        return {}
    v = sorted(values)
    return {f"p{int(q * 100)}": round(v[min(len(v) - 1, int(q * len(v)))], 4) for q in qs}


# --- near-dup (MinHash + LSH on char 5-grams) -------------------------------

NPERM, BANDS = 64, 16  # 4 rows/band -> ~0.8 Jaccard sweet spot
_rng = random.Random(13)
_SEEDS = [_rng.getrandbits(64) for _ in range(NPERM)]


def norm_for_dup(s):
    s = nfc(s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def shingles(s, k=5):
    s = norm_for_dup(s)
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))} if s else set()


def minhash(sh):
    hs = [int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "big") for x in sh]
    return tuple(min((h ^ seed) for h in hs) for seed in _SEEDS) if hs else None


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def near_dup_clusters(items, thr=0.8):
    """items: list of (key, text). Returns list of clusters (lists of keys), largest first."""
    sh = {k: shingles(t) for k, t in items}
    sig = {k: minhash(s) for k, s in sh.items()}
    rows = NPERM // BANDS
    buckets = collections.defaultdict(list)
    for k, s in sig.items():
        if s is None:
            continue
        for b in range(BANDS):
            buckets[(b, s[b * rows:(b + 1) * rows])].append(k)
    parent = {k: k for k, _ in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = set()
    for ks in buckets.values():
        if len(ks) < 2 or len(ks) > 500:
            continue
        for i in range(1, len(ks)):
            pair = (ks[0], ks[i])
            if pair in seen:
                continue
            seen.add(pair)
            if jaccard(sh[ks[0]], sh[ks[i]]) >= thr:
                parent[find(ks[i])] = find(ks[0])
    groups = collections.defaultdict(list)
    for k, _ in items:
        groups[find(k)].append(k)
    return sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)


# --- corpus stats ------------------------------------------------------------

def corpus_stats(corpus_dir, frf):
    report = {"files": {}, "unhydrated": [], "per_lang": {}, "aggregate_files_excluded": AGGREGATE_FILES}
    pairs = []  # (key, lang, text, fr, fname, domain) — MT pairs only, aggregates excluded
    all_text = collections.defaultdict(list)
    scores = []
    for p in sorted(corpus_dir.glob("*.jsonl")):
        if p.stat().st_size == 0:
            report["unhydrated"].append(p.name)
            continue
        recs, bad = read_jsonl(p)
        keys = collections.Counter(k for r in recs for k in r)
        n_par = n_mono = n_lex = 0
        doms = collections.Counter()
        for i, r in enumerate(recs):
            lang = r.get("lang") or p.stem.split("-")[0].split(".")[0]
            t, fr = pick(r, TEXT_KEYS), pick(r, FR_KEYS)
            dom = domain_of(r, p.name)
            doms[dom] += 1
            if dom == "lexical" or "headword" in r or "lex" in r:
                n_lex += 1
            if t and fr:
                n_par += 1
                if p.name not in AGGREGATE_FILES and r.get("kind") not in NON_MT_KINDS:
                    pairs.append((f"{p.name}:{i}", lang, t, fr, p.name, dom))
            elif t:
                n_mono += 1
            if t and p.name not in AGGREGATE_FILES:
                all_text[lang].append(t)
            if "scored" in p.name:
                s = pick_score(r)
                if s is not None:
                    scores.append(s)
        report["files"][p.name] = {
            "records": len(recs), "bad_lines": bad, "parallel": n_par, "monolingual": n_mono, "lexicon": n_lex,
            "domains": dict(doms), "fields": dict(keys.most_common()),
            "sample": recs[:2],
        }
    # per-language rollup
    for f, s in report["files"].items():
        if f in AGGREGATE_FILES:
            continue
        lang = f.split("-")[0].split(".")[0]
        L = report["per_lang"].setdefault(lang, collections.Counter())
        for k in ("records", "parallel", "monolingual", "lexicon"):
            L[k] += s[k]
        for d, c in s["domains"].items():
            L[f"domain:{d}"] += c
    report["per_lang"] = {k: dict(v) for k, v in report["per_lang"].items()}

    # distributions over MT pairs
    ratios = [len(nfc(t)) / max(1, len(nfc(fr))) for _, _, t, fr, _, _ in pairs]
    hits = [fr_hits(t, frf["words"], 1) for _, _, t, _, _, _ in pairs]          # all list words
    stop = [h / n if n else 0.0 for h, n in hits]
    report["len_ratio"] = {"quantiles": quantiles(ratios),
                           "hist": hist(ratios, [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])}
    report["fr_stopword_ratio_in_text"] = {"list": "config filters.fr_stopwords", "quantiles": quantiles(stop),
                                           "hist": hist(stop, [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7])}
    report["align_score"] = {"n": len(scores), "quantiles": quantiles(scores),
                             "hist": hist(scores, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) if scores else {}}

    # threshold sweeps: % of MT pairs each candidate removes, overall / religious / non-religious / per lang
    groups = {"all": lambda d, l: True, "religious": lambda d, l: d == "religious",
              "non_religious": lambda d, l: d != "religious"}
    for lang in sorted({x[1] for x in pairs}):
        groups[f"lang:{lang}"] = (lambda L: lambda d, l: l == L)(lang)

    def pct(flag):
        out = {}
        for g, sel in groups.items():
            idx = [i for i, x in enumerate(pairs) if sel(x[5], x[1])]
            out[g] = {"n": len(idx), "removed": sum(flag(i) for i in idx),
                      "pct": round(100 * sum(flag(i) for i in idx) / max(1, len(idx)), 2)}
        return out

    def fr_flag(ratio_max, min_hits, min_len, min_tokens=frf["min_tokens"]):
        def f(i):
            h, n = fr_hits(pairs[i][2], frf["words"], min_len)
            return n >= min_tokens and h >= min_hits and h / n > ratio_max
        return f

    def len_flag(lo, hi, min_chars):
        def f(i):
            t, fr = nfc(pairs[i][2]), nfc(pairs[i][3])
            if len(t) < min_chars or len(fr) < min_chars:
                return False
            return not lo <= len(t) / len(fr) <= hi
        return f

    sw = {"fr_in_target": {}, "len_ratio": {}, "n_mt_pairs": len(pairs),
          "config_defaults": {k: v for k, v in frf.items() if k != "words"}}
    for rmax in (0.2, 0.3, 0.4):
        for mh in (1, 2, 3):
            for ml in (2, 3):
                sw["fr_in_target"][f"ratio>{rmax},min_hits={mh},min_len={ml}"] = pct(fr_flag(rmax, mh, ml))
    for lo, hi in ((0.3, 3.0), (0.4, 2.5), (0.5, 2.0)):
        for mc in (0, 15):
            sw["len_ratio"][f"outside[{lo},{hi}],min_chars={mc}"] = pct(len_flag(lo, hi, mc))
    if scores:
        sw["align_below"] = {f"p{q}={quantiles(scores, (q / 100,))[f'p{q}']}": q for q in (5, 10, 20)}
        sw["align_abs"] = {str(th): round(100 * sum(s < th for s in scores) / len(scores), 2)
                           for th in (0.2, 0.3, 0.4, 0.5, 0.6)}
    report["sweeps"] = sw

    # worst offenders for eyeballing (config-default rule)
    def ratio_of(t):
        h, n = fr_hits(t, frf["words"], frf["min_len"])
        return h / n if n else 0.0
    import math
    report["examples"] = {
        "top_fr_in_target": [{"k": k, "text": t[:200], "fr": fr[:200], "ratio": round(ratio_of(t), 2)}
                             for k, _, t, fr, _, _ in sorted(pairs, key=lambda x: -ratio_of(x[2]))[:25]],
        "extreme_len_ratio": [{"k": k, "text": t[:160], "fr": fr[:160], "ratio": round(len(t) / max(1, len(fr)), 2)}
                              for k, _, t, fr, _, _ in sorted(pairs, key=lambda x: abs(
                                  math.log(max(1, len(x[2])) / max(1, len(x[3])))))[-25:]],
    }
    if scores:
        sc = [json.loads(l) for l in open(corpus_dir / "fang-nt-parallel.scored.jsonl", encoding="utf-8") if l.strip()]
        sc.sort(key=lambda r: r["sim"])
        report["examples"]["lowest_align"] = [{"sim": round(r["sim"], 3), "text": r["text"][:160],
                                               "fr": r["fr"][:160]} for r in sc[:15]]
        report["examples"]["align_around_p10"] = [{"sim": round(r["sim"], 3), "text": r["text"][:160],
                                                   "fr": r["fr"][:160]}
                                                  for r in sc[len(sc) // 10 - 5: len(sc) // 10 + 5]]

    # char inventory per language (+ look-alike report, apostrophe variants)
    inv = {}
    for lang, texts in all_text.items():
        c = collections.Counter(ch for t in texts for ch in t)
        nfc_changes = sum(t != nfc(t) for t in texts)
        la = {g: {f"U+{ord(ch):04X} {ch} {unicodedata.name(ch, '?')}": c[ch] for ch in chars if c[ch]}
              for g, chars in LOOKALIKES.items()}
        inv[lang] = {
            "n_texts": len(texts), "not_nfc": nfc_changes,
            "total_chars_nfc": sum(len(nfc(t)) for t in texts),
            "total_words": sum(len(words(t)) for t in texts),
            "mean_chars_per_text": round(sum(len(nfc(t)) for t in texts) / max(1, len(texts)), 1),
            # combining marks that survive NFC (no precomposed form, e.g. tone on ɛ/ɔ)
            "texts_with_combining_after_nfc": sum(any(unicodedata.combining(ch) for ch in nfc(t)) for t in texts),
            "top_base_plus_mark_after_nfc": dict(collections.Counter(
                m.group() for t in texts for m in re.finditer(COMBINING_SEQ, nfc(t))
            ).most_common(30)),
            "non_ascii": {f"U+{ord(ch):04X} {ch} {unicodedata.name(ch, '?')}": n
                          for ch, n in c.most_common() if ord(ch) > 127},
            "combining_marks": sum(n for ch, n in c.items() if unicodedata.combining(ch)),
            "lookalikes": {g: v for g, v in la.items() if len(v) > 1 or g == "apostrophe"},
        }
    report["char_inventory"] = inv

    # near-dup clusters over target text (per lang)
    report["near_dup_top20"] = {}
    for lang in sorted({l for _, l, *_ in pairs}):
        items = [(k, t) for k, l, t, _, _, _ in pairs if l == lang]
        cl = near_dup_clusters(items)
        txt = dict(items)
        report["near_dup_top20"][lang] = {
            "n_clusters": len(cl), "records_in_clusters": sum(map(len, cl)),
            "top": [{"size": len(g), "examples": [f"{k} | {txt[k][:100]}" for k in g[:3]]} for g in cl[:20]],
        }
    return report, pairs


# --- v0 train audit ---------------------------------------------------------

def train_text(rec):
    """Return (prompt_payload, answer) for mlx formats: chat / completions / text.
    The payload is the user turn after the instruction ("Traduis en fang : <payload>")."""
    if "messages" in rec:
        u = " ".join(m["content"] for m in rec["messages"] if m.get("role") == "user")
        a = " ".join(m["content"] for m in rec["messages"] if m.get("role") == "assistant")
    elif "prompt" in rec:
        u, a = rec.get("prompt", ""), rec.get("completion", "")
    else:
        return "", rec.get("text", "")
    return (u.split(":", 1)[1] if ":" in u else u).strip(), a


REF_RE = re.compile(r"(?:ref|bible)=([1-3]?[A-Z]{2,3}) (\d+):(\d+)")
# DATA_SPEC v0.1 §3 held-out Bible books (one global list across languages)
SPEC_HELDOUT = {"test": {"PHP", "1JN", "JUD", "RUT"}, "val": {"COL", "2JN", "JON"}}


def corpus_index_all(corpus_dir):
    """normalized string -> set of (file, book) for every non-aggregate corpus record, both sides."""
    idx = collections.defaultdict(set)
    for p in sorted(corpus_dir.glob("*.jsonl")):
        if p.name in AGGREGATE_FILES or p.stat().st_size == 0:
            continue
        for r in read_jsonl(p)[0]:
            m = REF_RE.search(str(r.get("notes", "")))
            book = m.group(1) if m else (r.get("ref", "").split(" ")[0] if r.get("ref") else None)
            for side in (pick(r, TEXT_KEYS), pick(r, FR_KEYS)):
                if side:
                    idx[norm_for_dup(side)].add((p.name, book))
    return idx


def audit_train(train_root, corpus_dir):
    out = {}
    cidx = corpus_index_all(corpus_dir)
    for d in sorted(p for p in train_root.iterdir() if p.is_dir()):
        info = {"config": None, "splits": {}, "unhydrated": []}
        cfg = d / "mlx-lora-config.yaml"
        if cfg.exists() and cfg.stat().st_size:
            info["config"] = cfg.read_text()
        split_recs = {}
        for split in ("train", "valid", "test"):
            p = d / f"{split}.jsonl"
            if not p.exists():
                continue
            if p.stat().st_size == 0:
                info["unhydrated"].append(p.name)
                continue
            recs, bad = read_jsonl(p)
            split_recs[split] = recs
            fmt = collections.Counter("messages" if "messages" in r else "prompt" if "prompt" in r else "text"
                                      for r in recs)
            instr = collections.Counter(
                re.sub(r"\s*:.*", "", next((m["content"] for m in r["messages"] if m["role"] == "user"), ""))[:60]
                for r in recs if "messages" in r)
            src_files, books, unmatched = collections.Counter(), collections.Counter(), 0
            for r in recs:
                pl, a = train_text(r)
                hit = cidx.get(norm_for_dup(a)) or cidx.get(norm_for_dup(pl)) or set()
                if not hit:
                    unmatched += 1
                for f in {f for f, _ in hit}:
                    src_files[f] += 1
                for b in {b for _, b in hit if b}:
                    books[b] += 1
            info["splits"][split] = {
                "n": len(recs), "bad_lines": bad, "formats": dict(fmt), "instructions_top": dict(instr.most_common(12)),
                "mean_answer_chars": round(sum(len(train_text(r)[1]) for r in recs) / max(1, len(recs)), 1),
                "matched_corpus_files": dict(src_files.most_common()), "unmatched_to_corpus": unmatched,
                "spec_heldout_books_seen": {k: {b: books[b] for b in v if books[b]} for k, v in SPEC_HELDOUT.items()},
                "n_bible_books": len(books), "sample": recs[:2]}
        # leakage train -> valid/test: same pair in either direction, or any shared side; plus near-dup on answers
        if "train" in split_recs:
            tr = [tuple(map(norm_for_dup, train_text(r))) for r in split_recs["train"]]
            tr_pairs = {frozenset(x) for x in tr}
            tr_sides = {s for x in tr for s in x if s}
            for held in ("valid", "test"):
                if held not in split_recs:
                    continue
                hv = [tuple(map(norm_for_dup, train_text(r))) for r in split_recs[held]]
                same_pair = sum(frozenset(x) in tr_pairs for x in hv)
                ans_in_train = sum(x[1] in tr_sides for x in hv)
                any_side = sum(any(s and s in tr_sides for s in x) for x in hv)
                items = [(f"tr:{i}", train_text(r)[1]) for i, r in enumerate(split_recs["train"])] + \
                        [(f"{held}:{i}", train_text(r)[1]) for i, r in enumerate(split_recs[held])]
                leaked = {k for g in near_dup_clusters(items) if any(x.startswith("tr:") for x in g)
                          for k in g if k.startswith(f"{held}:")}
                n = max(1, len(hv))
                info[f"leak_{held}"] = {
                    "n": len(hv),
                    "same_pair_any_direction": same_pair, "pct_same_pair": round(100 * same_pair / n, 1),
                    "answer_seen_in_train(either side)": ans_in_train, "pct_answer_seen": round(100 * ans_in_train / n, 1),
                    "any_side_seen_in_train": any_side, "pct_any_side": round(100 * any_side / n, 1),
                    "near_dup_answer(J>=0.8)": len(leaked), "pct_near_dup": round(100 * len(leaked) / n, 1)}
        out[d.name] = info
    return out


# --- main --------------------------------------------------------------------

def to_md(rep):
    L = ["# retrievve — v0 corpus stats (DATA_SPEC §8)", ""]
    if rep["corpus"]["unhydrated"] or any(v["unhydrated"] for v in rep["train_audit"].values()):
        L += ["> **Partial report**: some files are still Dropbox placeholders: "
              + ", ".join(rep["corpus"]["unhydrated"]), ""]
    L += ["## Per language", "", "| lang | records | parallel | mono | lexicon | domains |", "|---|---|---|---|---|---|"]
    for lang, s in rep["corpus"]["per_lang"].items():
        doms = ", ".join(f"{k[7:]}={v}" for k, v in s.items() if k.startswith("domain:"))
        L.append(f"| {lang} | {s.get('records', 0)} | {s.get('parallel', 0)} | {s.get('monolingual', 0)} | "
                 f"{s.get('lexicon', 0)} | {doms} |")
    L += ["", "## Per file", "", "| file | records | parallel | mono | fields |", "|---|---|---|---|---|"]
    for f, s in rep["corpus"]["files"].items():
        L.append(f"| {f} | {s['records']} | {s['parallel']} | {s['monolingual']} | {', '.join(s['fields'])} |")
    for key in ("len_ratio", "fr_stopword_ratio_in_text", "align_score"):
        L += ["", f"## {key}", "", "```", json.dumps(rep["corpus"][key], ensure_ascii=False, indent=1), "```"]
    L += ["", "## Threshold sweeps (% of parallel pairs removed)", "", "```",
          json.dumps(rep["corpus"]["sweeps"], ensure_ascii=False, indent=1), "```"]
    L += ["", "## Size per language (target text, NFC)", "", "| lang | texts | chars | words | mean chars |",
          "|---|---|---|---|---|"]
    for l, v in rep["corpus"]["char_inventory"].items():
        L.append(f"| {l} | {v['n_texts']} | {v['total_chars_nfc']} | {v['total_words']} | {v['mean_chars_per_text']} |")
    L += ["", "## Look-alikes / apostrophes", "", "```",
          json.dumps({l: v["lookalikes"] for l, v in rep["corpus"]["char_inventory"].items()},
                     ensure_ascii=False, indent=1), "```"]
    L += ["", "## Near-dup clusters", ""]
    for lang, d in rep["corpus"]["near_dup_top20"].items():
        L.append(f"- **{lang}**: {d['n_clusters']} clusters, {d['records_in_clusters']} records")
    L += ["", "## v0 train audit", ""]
    for name, a in rep["train_audit"].items():
        L.append(f"### {name}")
        L.append(f"- splits: { {k: v['n'] for k, v in a['splits'].items()} }; unhydrated: {a['unhydrated']}")
        for held in ("leak_valid", "leak_test"):
            if held in a:
                L.append(f"- {held}: {a[held]}")
    L.append("\nFull details (char inventory, samples, examples): `stats_v0.json`.")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/v1/reports"))
    ap.add_argument("--config", default=str(ROOT / "configs/v1_pipeline.yaml"))
    args = ap.parse_args(argv)
    frf = load_fr_filter(args.config)
    corpus, pairs = corpus_stats(ROOT / "data/corpus", frf)
    rep = {"corpus": corpus, "train_audit": audit_train(ROOT / "data/train", ROOT / "data/corpus")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats_v0.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "stats_v0.md").write_text(to_md(rep))
    print(f"wrote {out}/stats_v0.{{json,md}}; parallel pairs={len(pairs)}; "
          f"unhydrated corpus files={len(corpus['unhydrated'])}")


if __name__ == "__main__":
    main()
