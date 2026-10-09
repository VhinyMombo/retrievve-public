"""Lexicon probe (DATA_SPEC v0.7 §7b): a frozen sample of headwords held out of every SFT lexicon task.

Selection is deterministic (seed), stratified by POS, homographs merged (same text_key), empty glosses
dropped. The sampled headwords stay in RAG. Once written, splits/<lang>.lexprobe.ids is reused on later builds
so the probe never changes.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import stable_hash
from .normalize import text_fold, text_key
from .quality import usable_rag

_DESCRIPTIVE = re.compile(r"\b(sorte|espèce|genre|variété|qui|que|celui|celle|personne|action de|fait de)\b", re.I)


def _strata_alloc(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, capped by stratum size."""
    total = sum(sizes.values())
    if total <= n:
        return dict(sizes)
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: min(sizes[k], int(x)) for k, x in raw.items()}
    rest = n - sum(alloc.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), k)):
        if rest <= 0:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            rest -= 1
    return alloc


def candidates(records: list[dict], lang: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in records:
        if r["lang"] == lang and r["kind"] == "lexicon" and r["text"] and r["fr"] and r["split"] != "test" \
                and usable_rag(r):
            groups[r["text_key"]].append(r)
    return groups


def select(records: list[dict], lang: str, n: int, seed: int, ids_file: Path | None = None) -> list[str]:
    """Returns the probe's headword keys (text_key). Reuses ids_file when it exists (frozen)."""
    groups = candidates(records, lang)
    if ids_file is not None and ids_file.exists() and ids_file.stat().st_size:
        return [k for k in ids_file.read_text(encoding="utf-8").splitlines() if k]
    pos_of = {k: Counter((e.get("lex") or {}).get("pos") or "unknown" for e in ents).most_common(1)[0][0]
              for k, ents in groups.items()}
    strata = defaultdict(list)
    for k, p in pos_of.items():
        strata[p].append(k)
    alloc = _strata_alloc({p: len(ks) for p, ks in strata.items()}, n)
    chosen = []
    for p in sorted(strata):
        ks = sorted(strata[p], key=lambda k: stable_hash(str(seed), "lexprobe", lang, k))
        chosen.extend(ks[:alloc.get(p, 0)])
    return sorted(chosen)


def listed_variants(r: dict) -> list[str]:
    """Spellings an entry itself lists as variants (lex.variants, or a source column named like 'variante')."""
    out = list((r.get("lex") or {}).get("variants") or [])
    for k, v in (r.get("extra") or {}).items():
        if "variant" in k.lower() and v:
            out.extend(x.strip() for x in re.split(r"[;,/]", v) if x.strip())
    return out


def sentence_keys(texts: list[str]) -> tuple[set[str], list[str]]:
    """(token keys, padded sentence keys) for in_sft_sentences: tokens via text_key per token; multiword
    headwords are matched on the whole sentence key with word boundaries."""
    toks, sents = set(), []
    for t in texts:
        k = text_key(t)
        toks.update(k.split())
        sents.append(f" {k} ")
    return toks, sents


def in_sentences(key: str, toks: set[str], sents: list[str]) -> bool:
    if " " not in key:
        return key in toks
    probe = f" {key} "
    return any(probe in s for s in sents)


def rows(records: list[dict], lang: str, keys: list[str], sft_train_texts: list[str]) -> list[dict]:
    groups = candidates(records, lang)
    toks, sents = sentence_keys(sft_train_texts)
    by_fold = defaultdict(set)                       # diagnostics only: never counted as correct answers
    for kk, ents in groups.items():
        by_fold[text_fold(ents[0]["text"])].add(kk)
    out = []
    for i, k in enumerate(keys, start=1):
        ents = sorted(groups.get(k, []), key=lambda r: r["id"])
        if not ents:
            continue
        head = ents[0]["text"]
        glosses = list(dict.fromkeys(e["fr"] for e in ents if e["fr"]))
        pos = Counter((e.get("lex") or {}).get("pos") or "unknown" for e in ents).most_common(1)[0][0]
        # variants: merged entries (same text_key) + spellings the entry lists. NOT same-text_fold headwords:
        # tone-different words are different words (§7b measures EM_key vs EM_fold).
        variants = {e["text"] for e in ents} | {v for e in ents for v in listed_variants(e)}
        out.append({
            "id": f"{lang}:lexprobe:{i:04d}", "lang": lang, "headword": head, "text_key": k,
            "variants": sorted(variants), "gold_glosses": glosses, "pos": pos,
            "orthography": Counter(e.get("orthography") or "unknown" for e in ents).most_common(1)[0][0],
            "fold_neighbours": sorted(groups[n][0]["text"] for n in by_fold[text_fold(head)] - {k}),
            "in_sft_sentences": in_sentences(k, toks, sents),
            "definition_gloss": any(len(g.split()) > 4 or _DESCRIPTIVE.search(g) for g in glosses),
            "src_ids": [e["id"] for e in ents],
        })
    return out
