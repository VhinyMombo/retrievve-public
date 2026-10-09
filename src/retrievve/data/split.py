"""Grouped, leak-proof splits (DATA_SPEC §3, v0.7).

Splits are assigned per group_id across ALL languages at once (group_id is language-independent).
- Bible: groups are `bible/<BOOK>`; configured held-out books if present, otherwise the smallest
  non-synoptic books actually present per language (choice is reported). v0.5: if that smallest book is
  over 2x the target, hold out configured chapter ranges instead (`bible/<BOOK>/tail` groups), e.g. Punu
  test = DEU 31-34, val = GEN tail from ch.50 back to ~2%.
- Other domains: ~doc_test_frac / doc_val_frac of documents, stratified by domain, by stable hash.
  `never_heldout_domains` (lexical, grammar) stay in train.
- Eval-only languages: their groups go to rag_pool / test (lexical, grammar -> rag_pool).
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .common import stable_hash


def chapter_of(r: dict) -> int | None:
    """Chapter from a Bible id locator (`puu:src:DEU.31.5` -> 31)."""
    parts = r["id"].rsplit(":", 1)[-1].split("#")[0].split(".")
    return int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else None


def _tail_choice(lang, split, present_ch, total, assign, sc) -> tuple[str, set[int]] | None:
    """Chapter-range fallback (v0.5). present_ch: book -> {chapter: parallel verses} for this language."""
    for spec in (sc.get("bible_tail_fallback") or {}).get(split, []):
        book = spec["book"]
        if book not in present_ch or f"bible/{book}" in assign or f"bible/{book}/tail" in assign:
            continue
        chs = present_ch[book]
        if "chapters" in spec:
            lo, hi = spec["chapters"]
            chosen = {c for c in chs if lo <= c <= hi}
        else:
            target, got, chosen = spec["tail_frac"] * total, 0, set()
            for c in sorted(chs, reverse=True):
                if got >= target:
                    break
                chosen.add(c)
                got += chs[c]
        if chosen and len(chosen) < len(chs):                 # never the whole book
            return book, chosen
    return None


def _bible_choice(records, cfg) -> tuple[dict[str, str], dict]:
    sc = cfg["splits"]
    # Book sizes count PARALLEL verses only: a held-out book must yield eval pairs, so books that are
    # monolingual in a language (e.g. Punu NT from the HTML scrape) are never candidates for it.
    size = Counter()                          # (lang, book) -> parallel records
    chap = defaultdict(lambda: defaultdict(Counter))   # lang -> book -> chapter -> parallel records
    for r in records:
        if r["group_id"].startswith("bible/") and r["fr"]:
            book = r["group_id"][6:]
            size[(r["lang"], book)] += 1
            c = chapter_of(r)
            if c is not None:
                chap[r["lang"]][book][c] += 1
    tails: dict[str, set[int]] = {}           # book -> held-out chapters (group bible/<BOOK>/tail)
    langs = sorted({lg for lg, _ in size})
    assign: dict[str, str] = {}
    report = {}
    for split, key, frac_key in (("test", "bible_test_books", "bible_fallback_test_frac"),
                                 ("val", "bible_val_books", "bible_fallback_val_frac")):
        for lang in langs:
            present = {b: n for (lg, b), n in size.items() if lg == lang}
            total = sum(present.values())
            free = [b for b in present if f"bible/{b}" not in assign]
            wanted = [b for b in (sc.get(key, {}) or {}).get(lang, []) if b in free]
            how = "configured"
            if not wanted:
                how = "fallback:smallest-present-non-synoptic"
                target = sc[frac_key] * total
                got = 0
                avoid = set(sc.get("bible_avoid_books") or [])
                cands = [b for b in sorted(free, key=lambda b: (present[b], b))
                         if b not in sc["synoptic_books"] and b not in avoid]
                tail = None
                if cands and present[cands[0]] > 2 * target:
                    tail = _tail_choice(lang, split, chap[lang], total, assign, sc)
                if tail:
                    book, chs = tail
                    assign[f"bible/{book}/tail"] = split
                    tails[book] = chs
                    cands, how = [], f"tail:{book} ch.{min(chs)}-{max(chs)}"
                for b in cands:
                    if got and got + present[b] > 2 * target:
                        break
                    wanted.append(b)
                    got += present[b]
                    if got >= target:
                        break
                if len(free) <= 1:
                    wanted = []                                   # never hold out a language's last free book
                    how = "none:only-one-book"
            for b in wanted:
                assign[f"bible/{b}"] = split
            # what this language actually loses to `split`, including books chosen jointly by another language
            wanted = sorted(b for b in present if assign.get(f"bible/{b}") == split)
            held = sum(present.get(b, 0) for b in wanted)
            for b, chs in tails.items():
                if assign.get(f"bible/{b}/tail") == split and b in chap[lang]:
                    wanted.append(f"{b}/tail:{min(chs)}-{max(chs)}")
                    held += sum(n for c, n in chap[lang][b].items() if c in chs)
            target = sc[frac_key] * total
            report[f"{lang}:{split}"] = {"books": wanted, "how": how, "records": held, "of": total,
                                         "frac": round(held / total, 4) if total else 0.0,
                                         "target_frac": sc[frac_key],
                                         "over_2x_target": bool(total and held > 2 * target)}  # N3
    report["books_present"] = {lg: sorted(b for (x, b) in size if x == lg) for lg in langs}
    report["tails"] = {b: sorted(chs) for b, chs in tails.items()}
    return assign, tails, report


def _doc_choice(groups: dict[str, int], frac_test: float, frac_val: float, seed: int) -> dict[str, str]:
    """groups: group_id -> record count, all in one stratum."""
    order = sorted(groups, key=lambda g: stable_hash(str(seed), g))
    total = sum(groups.values())
    out, got = {}, {"test": 0, "val": 0}
    if len(order) < 3:                                  # too few documents to hold any out
        return {g: "train" for g in order}
    for g in order:
        n = groups[g]
        for split, frac in (("test", frac_test), ("val", frac_val)):
            target = frac * total
            if got[split] < target and (got[split] + n <= 1.5 * target or got[split] == 0 and n <= 0.5 * total):
                out[g] = split
                got[split] += n
                break
        else:
            out[g] = "train"
    return out


def assign_splits(records: list[dict], cfg: dict) -> dict:
    sc, seed = cfg["splits"], cfg["seed"]
    eval_only = set(cfg["languages"]["eval_only"])
    never = set(sc["never_heldout_domains"])
    bible, tails, bible_report = _bible_choice(records, cfg)
    for r in records:                                   # relabel held-out chapter ranges, in every language
        if r["group_id"].startswith("bible/") and r["group_id"][6:] in tails and chapter_of(r) in tails[
                r["group_id"][6:]]:
            r["group_id"] = f"{r['group_id']}/tail"

    # group -> domain (majority) and size, over train/eval languages; eval-only handled separately
    g_dom, g_size = defaultdict(Counter), Counter()
    for r in records:
        if r["lang"] not in eval_only:
            g_dom[r["group_id"]][r["domain"]] += 1
            g_size[r["group_id"]] += 1
    strata = defaultdict(dict)
    for g, dom in g_dom.items():
        d = dom.most_common(1)[0][0]
        if g.startswith("bible/") or d in never:
            continue
        strata[d][g] = g_size[g]
    doc = {}
    for _d, groups in sorted(strata.items()):
        doc.update(_doc_choice(groups, sc["doc_test_frac"], sc["doc_val_frac"], seed))

    # eval-only languages: rag_pool / test, by group hash (lexical, grammar always rag_pool)
    eo_groups = defaultdict(Counter)
    for r in records:
        if r["lang"] in eval_only:
            eo_groups[r["domain"]][r["group_id"]] += 1
    eo, held = {}, Counter()                            # held: test/rag_pool counts over held-out-eligible groups
    for d, groups in sorted(eo_groups.items()):
        for g in sorted(groups, key=lambda g: stable_hash(str(seed), "eo", g)):
            if g in g_size or g in bible or g in doc:
                continue                                 # shared group: follows the train/eval decision (N4)
            if d in never:
                eo.setdefault(g, "rag_pool")
            else:
                if g in eo:
                    continue
                frac = sc["eval_only_rag_pool_frac"]
                eo[g] = "test" if held["test"] < (sum(held.values()) + 1) * (1 - frac) else "rag_pool"
                held[eo[g]] += 1

    for r in records:
        g = r["group_id"]
        if r["lang"] in eval_only:
            # a group shared with train/eval langs follows their decision (test stays test, train/val -> rag_pool)
            shared = bible.get(g) or doc.get(g) or ("train" if g in g_size else None)
            if shared:
                r["split"] = "test" if shared == "test" else "rag_pool"
            else:
                r["split"] = eo.get(g, "rag_pool")
        else:
            r["split"] = bible.get(g) or doc.get(g) or "train"

    counts = Counter((r["lang"], r["split"]) for r in records)
    return {"bible": bible_report,
            "doc_groups": {s: sorted(g for g, x in doc.items() if x == s) for s in ("test", "val")},
            "eval_only_groups": {s: sorted(g for g, x in eo.items() if x == s) for s in ("test", "rag_pool")},
            "records": {f"{lg}:{sp}": n for (lg, sp), n in sorted(counts.items())}}
