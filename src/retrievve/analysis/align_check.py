"""Proper-noun agreement check for verse/sentence alignments (acceptance test for Bible parallels).

For every pair whose French side contains a proper noun, check whether the target side contains a token with the
same consonant skeleton. The names are found automatically (capitalised in >= 80% of their French occurrences,
>= 4 letters, >= 3 occurrences), so no per-language name table is needed. Agreement is reported next to a
shuffled baseline (each target paired with the French of the record n/2 further on in the same group), because
the absolute level depends on how a language spells names. A correctly aligned book sits far above its baseline.

    python -m retrievve.analysis.align_check data/v1/corpus/puu.jsonl [--min-agree 0.8] [--by book|chapter|decile]
    python -m retrievve.analysis.align_check data/v1/corpus/puu.jsonl --domain religious --split test,val --by group_id --strict  # freeze gate
    retrievve align-check data/corpus/fang-nt-parallel.scored.jsonl --json out.json

Pass = agree >= --min-agree AND agree - baseline >= --min-margin, for groups with >= --min-names name occurrences.
Exit 1 if any judged group fails. With --strict, exit 2 ("unverifiable") if any group is unjudged or none is judged;
unjudged groups are always listed. Freeze gate: --domain religious --split test,val --by group_id --strict.
Train-side check (report-only, feeds low_align): --domain religious --split train --by group_id [--scan-offsets].
Stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
import unicodedata
from pathlib import Path

# v1 ids end with BOOK.chapter.verse; v0 records carry "ref=MAT 1:1" or "bible=GEN 37:19" in notes
V1_LOC = re.compile(r":([1-3]?[A-Z]{2,3})\.(\d+)\.(\d+)$")
V0_LOC = re.compile(r"(?:ref|bible)=([1-3]?[A-Z]{2,3}) (\d+):(\d+)")
CAP_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
NOT_NAMES = {"dieu", "éternel", "seigneur", "saint", "esprit", "fils", "père", "roi", "christ", "ciel", "cieux",
             "loi", "maître", "amen", "alléluia", "très", "haut", "tout", "puissant",
             # capitalised common nouns in LS1910 that are translated, not transliterated
             "église", "églises", "évangile", "sauveur", "juifs", "juif", "parole", "agneau", "temple",
             "écriture", "écritures"}
# LS1910 capitalises sentence-initial 2pl imperatives ("Saluez Prisca", "Servez le Seigneur"): never names.
IMPERATIVE_2PL = re.compile(r"[a-zéèêàâîôûç]{3,}ez$")
# French soft c (Macédoine, César, Cilicie) is rendered with s in the targets.
SOFT_C = re.compile(r"c(?=[eéèêëiîïy])")


def skeleton(w):
    """Consonant skeleton, tolerant to common spelling shifts across French / Bantu orthographies."""
    w = SOFT_C.sub("s", w.lower())
    w = "".join(c for c in unicodedata.normalize("NFD", w) if not unicodedata.combining(c))
    for a, b in (("ph", "f"), ("th", "t"), ("ch", "s"), ("qu", "k"), ("c", "k"), ("q", "k"), ("j", "y"),
                 ("z", "s"), ("x", "ks"), ("w", "v"), ("ñ", "n"), ("ŋ", "n"), ("ɲ", "n"), ("ẑ", "s")):
        w = w.replace(a, b)
    cons = [c for c in w if c.isalpha() and c not in "aeiouyhɔɛəô"]
    return "".join(c for i, c in enumerate(cons) if i == 0 or c != cons[i - 1])


def locator(rec):
    m = V1_LOC.search(rec.get("id", "")) or V0_LOC.search(str(rec.get("notes", "")))
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def load(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("text") and r.get("fr"):
                        recs.append(r)
    return recs


def find_names(recs, min_count=3):
    cap, total = collections.Counter(), collections.Counter()
    for r in recs:
        for w in CAP_WORD.findall(r["fr"]):
            lw = w.lower()
            total[lw] += 1
            cap[lw] += w[0].isupper()
    return {w for w, n in total.items()
            if n >= min_count and len(w) >= 4 and cap[w] / n >= 0.8 and w not in NOT_NAMES
            and not IMPERATIVE_2PL.match(w) and len(skeleton(w)) >= 2}


def _subseq(a, b, max_gap=1):
    """a's letters appear in b in order, with at most max_gap extra letters in total (Paul pl ~ Pablo pbl)."""
    for start in range(len(b)):
        if b[start] != a[0]:
            continue
        i, gaps = 1, 0
        for c in b[start + 1:]:
            if i == len(a):
                break
            if c == a[i]:
                i += 1
            elif gaps < max_gap:
                gaps += 1
            else:
                break
        if i == len(a):
            return True
    return False


# French ethnonyms (Amoréens, Moabites, Horiens, Anakim) are rendered in Bantu targets as prefix + root (baAmor, bisi Moab,
# baHori, baAnac): match on the root.
ETHNONYM_SUFFIX = re.compile(r"(éens|eens|iens|ites|ite|im|ens)$")


def name_root(w):
    w = w.lower()
    return ETHNONYM_SUFFIX.sub("", w) if len(w) > 6 else w


def agree(text, fr, names):
    """(hits, n) over name occurrences in fr."""
    tsk = [skeleton(t) for t in CAP_WORD.findall(text)]
    hits = n = 0
    for w in {w.lower() for w in CAP_WORD.findall(fr)} & names:
        sk = skeleton(name_root(w))[:3]
        n += 1
        hits += any(_subseq(sk, t) for t in tsk if t)
    return hits, n


def group_key(rec, by, pos=None):
    loc = locator(rec)
    book = loc[0] if loc else rec.get("source_id") or rec.get("source") or "?"
    if by == "book":
        return book
    if by == "chapter":
        return f"{book} {loc[1]}" if loc else book
    if by == "group_id":
        return rec.get("group_id") or book
    return f"{book} d{pos}"  # decile within book


def run(recs, by="book", min_names=10, names=None):
    names = find_names(recs) if names is None else names
    books = collections.defaultdict(list)
    for r in recs:
        loc = locator(r)
        books[loc[0] if loc else r.get("source_id") or r.get("source") or "?"].append(r)
    groups = collections.defaultdict(list)
    for b, rs in books.items():
        rs.sort(key=lambda r: locator(r)[1:] if locator(r) else (0, 0))
        for i, r in enumerate(rs):
            groups[group_key(r, by, 10 * i // len(rs))].append((r, rs[(i + len(rs) // 2) % len(rs)]))
    out = {}
    for g, pairs in sorted(groups.items()):
        h = n = bh = bn = 0
        for r, other in pairs:
            a, k = agree(r["text"], r["fr"], names)
            h, n = h + a, n + k
            a, k = agree(r["text"], other["fr"], names)
            bh, bn = bh + a, bn + k
        ag, bl = (h / n if n else None), (bh / bn if bn else None)
        out[g] = {"pairs": len(pairs), "name_occ": n, "agree": None if ag is None else round(ag, 3),
                  "baseline": None if bl is None else round(bl, 3),
                  "margin": None if ag is None or bl is None else round(ag - bl, 3)}
    return {"n_names": len(names), "groups": out}


def scan_offsets(recs, max_off=3, min_names=4, margin=0.15, names=None):
    """Per chapter: agreement when target verse v is paired with French verse v+k. Flags chapters where some k != 0
    beats k = 0 by > margin (versification shift, e.g. Hebrew vs English numbering of DEU 23 / 29)."""
    names = find_names(recs) if names is None else names
    chap = collections.defaultdict(dict)
    for r in recs:
        loc = locator(r)
        if loc:
            chap[(loc[0], loc[1])][loc[2]] = r
    out = {}
    for (b, c), vs in sorted(chap.items()):
        res = {}
        for k in range(-max_off, max_off + 1):
            h = n = 0
            for v, r in vs.items():
                o = vs.get(v + k)
                if o:
                    a, m = agree(r["text"], o["fr"], names)
                    h, n = h + a, n + m
            res[k] = (h / n if n >= min_names else None, n)
        base = res[0][0]
        best = max((k for k in res if res[k][0] is not None), key=lambda k: res[k][0], default=0)
        if res[best][0] is not None and best != 0 and (base is None or res[best][0] - base > margin):
            out[f"{b} {c}"] = {"best_offset": best, "agree_best": round(res[best][0], 3),
                               "agree_0": None if base is None else round(base, 3), "verses": len(vs)}
    return out


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else 0.0


def length_check(recs, by="group_id", perms=999, seed=20261013):
    """Name-independent second check: correlation of log char lengths (target vs French) over the verses of a group,
    against a permutation null that shuffles the French within each chapter (999 draws), plus a shift-by-one control
    (target verse v paired with French verse v+1 of the same chapter) showing that the statistic sees verse offsets."""
    rng = random.Random(seed)
    L = lambda s: math.log(len(s) + 1)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        loc = locator(r)
        g = group_key(r, by) if by in ("book", "group_id") else group_key(r, "book")
        groups[g][loc[1] if loc else 0].append((loc[2] if loc else 0, r))
    out = {}
    for g, chs in sorted(groups.items()):
        chs = [[r for _, r in sorted(v, key=lambda t: t[0])] for v in chs.values()]
        x = [L(r["text"]) for c in chs for r in c]
        y = [L(r["fr"]) for c in chs for r in c]
        if len(x) < 3:
            out[g] = {"verses": len(x), "r": None, "r_shift1": None, "perm_p": None, "perm_max": None}
            continue
        r0 = _pearson(x, y)
        xs = [L(c[i]["text"]) for c in chs for i in range(len(c) - 1)]
        ys = [L(c[i + 1]["fr"]) for c in chs for i in range(len(c) - 1)]
        ge, mx = 0, -1.0
        for _ in range(perms):
            yy = []
            for c in chs:
                f = [L(r["fr"]) for r in c]
                rng.shuffle(f)
                yy += f
            rp = _pearson(x, yy)
            mx, ge = max(mx, rp), ge + (rp >= r0)
        out[g] = {"verses": len(x), "r": round(r0, 3), "r_shift1": round(_pearson(xs, ys), 3) if len(xs) > 2 else None,
                  "perm_p": round((ge + 1) / (perms + 1), 4), "perm_max": round(mx, 3)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", choices=["book", "chapter", "decile", "group_id"], default="book")
    ap.add_argument("--min-agree", type=float, default=0.8)
    ap.add_argument("--min-margin", type=float, default=0.4, help="required agree - shuffled baseline")
    ap.add_argument("--min-names", type=int, default=10, help="groups with fewer name occurrences are not judged")
    ap.add_argument("--source", help="only records whose source_id/source contains this string")
    ap.add_argument("--split", help="comma-separated v1 splits to check, e.g. test,val (freeze gate) or train (report-only)")
    ap.add_argument("--domain", help="only records with this v1 `domain` (e.g. religious)")
    ap.add_argument("--strict", action="store_true",
                    help="unjudged groups make the run UNVERIFIABLE, exit 2 (use when freezing)")
    ap.add_argument("--scan-offsets", action="store_true", help="per chapter, look for a verse offset that fits better")
    ap.add_argument("--length", action="store_true",
                    help="run the name-independent verse-length check instead (pass: r >= --min-r, perm p <= --max-p)")
    ap.add_argument("--min-r", type=float, default=0.70)
    ap.add_argument("--max-p", type=float, default=0.01)
    ap.add_argument("--min-verses", type=int, default=10, help="length check: smaller groups are not judged")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    recs = load(a.files)
    names = find_names(recs)  # from the whole file, so filtering to a small slice doesn't shrink the name list
    if a.source:
        recs = [r for r in recs if a.source in str(r.get("source_id") or r.get("source"))]
    if a.domain:
        recs = [r for r in recs if r.get("domain") == a.domain]
    if a.split:
        keep = set(a.split.split(","))
        recs = [r for r in recs if r.get("split") in keep]
    if a.length:
        res = length_check(recs, a.by)
        fails, na = [], []
        print(f"{'group':28} {'verses':>6} {'r':>6} {'shift1':>6} {'perm_p':>7} {'p_max':>6}")
        for g, v in res.items():
            judged = v["verses"] >= a.min_verses and v["r"] is not None
            bad = judged and (v["r"] < a.min_r or v["perm_p"] > a.max_p)
            fails += [g] if bad else []
            na += [] if judged else [g]
            st = "FAIL" if bad else ("" if judged else ("UNVERIFIABLE" if a.strict else "n/a"))
            print(f"{g:28} {v['verses']:6} {v['r']!s:>6} {v['r_shift1']!s:>6} {v['perm_p']!s:>7} {v['perm_max']!s:>6} {st}")
        if a.json:
            Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"judged {len(res) - len(na)}/{len(res)}; pass = r >= {a.min_r} and perm p <= {a.max_p}")
        if fails:
            print(f"FAIL {len(fails)}: {fails}")
            return 1
        if a.strict and (na or len(res) == len(na)):
            print(f"UNVERIFIABLE {len(na)}: {na}")
            return 2
        print("PASS")
        return 0
    if a.scan_offsets:
        sh = scan_offsets(recs, names=names)
        print(f"{len(sh)} chapter(s) with a verse-offset shift:")
        for g, v in sh.items():
            print(f"  {g:12} offset {v['best_offset']:+d}  agree {v['agree_0']} -> {v['agree_best']}  ({v['verses']} verses)")
        if a.json:
            Path(a.json).write_text(json.dumps(sh, ensure_ascii=False, indent=1))
        return 1 if sh else 0
    res = run(recs, a.by, a.min_names, names=names)
    fails, na = [], []
    print(f"{len(recs)} pairs, {res['n_names']} names detected")
    print(f"{'group':28} {'pairs':>6} {'names':>6} {'agree':>6} {'base':>6} {'margin':>6}")
    fmt = lambda x: "-" if x is None else x
    for g, v in res["groups"].items():
        judged = v["name_occ"] >= a.min_names and v["margin"] is not None
        bad = judged and (v["agree"] < a.min_agree or v["margin"] < a.min_margin)
        fails += [g] if bad else []
        na += [] if judged else [g]
        status = "FAIL" if bad else ("" if judged else ("UNVERIFIABLE" if a.strict else "n/a"))
        print(f"{g:28} {v['pairs']:6} {v['name_occ']:6} {fmt(v['agree']):>6} {fmt(v['baseline']):>6} "
              f"{fmt(v['margin']):>6} {status}")
    n_judged = len(res["groups"]) - len(na)
    res.update({"n_groups": len(res["groups"]), "n_judged": n_judged, "failed": fails, "unjudged": na,
                "criteria": {"min_agree": a.min_agree, "min_margin": a.min_margin, "min_names": a.min_names}})
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"judged {n_judged}/{len(res['groups'])} group(s); pass = agree >= {a.min_agree} and margin >= {a.min_margin}")
    if fails:
        print(f"FAIL {len(fails)}: {fails}")
    if na:
        print(f"{'UNVERIFIABLE' if a.strict else 'not judged'} {len(na)} (< {a.min_names} name occurrences): {na}")
    if fails:
        return 1
    if a.strict and (na or n_judged == 0):
        return 2
    print("PASS" if n_judged else "nothing judged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
