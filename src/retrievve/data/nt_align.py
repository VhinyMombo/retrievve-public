"""Fang NT realignment (replaces the v0 alignment, which 88 found wrong for most books).

1. `parse_pdf`  — verse segmentation from the PDF's TYPOGRAPHY, not from number sequences:
   verse numbers are 7pt bold spans, chapter numbers are ~28pt drop caps, book titles are 24pt,
   section headings 9pt bold, running headers sit above y=50, footnotes are < 6.5pt. Two-column pages
   are read column by column. (The v0 parser split on any digit sequence and took wrapped lines as verses.)
2. `align_chapter` — monotonic DP per chapter between parsed Fang verses and LS1910 French verses with
   1-1 / 1-2 / 2-1 beads (plus penalised 1-0 / 0-1 skips). Cost = (1 - sim) of the bead, where a merged
   side is embedded as the normalised sum of its verse vectors (Vecalign-style), plus a small penalty when the
   bead's verse numbers disagree. Proper nouns are deliberately NOT used in the cost: they are 88's acceptance
   metric and must stay independent.
3. `run` — writes data/interim/fang_nt_aligned.jsonl + data/v1/reports/fang_nt_alignment.json (per book).
   Books enter the v1 build only once listed in config `nt_books_accepted` (after 88's check, >= 0.8 per book).

    python -m retrievve.data.nt_align [--pdf ...] [--ref ...] [--scorer ollama:embeddinggemma] [--books MRK,ACT]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .common import ROOT, write_json, write_jsonl

PDF = ROOT / "data" / "raw" / "fang" / "00-WNTfan-web-2.pdf"
REF = ROOT / "data" / "raw" / "ref" / "ls1910-nt.json"
OUT = ROOT / "data" / "interim" / "fang_nt_aligned.jsonl"
REPORT = ROOT / "data" / "v1" / "reports" / "fang_nt_alignment.json"

# Book titles (24pt) in this edition, in canonical order -> USFM
TITLES = {
    "MATEO": "MAT", "MARKOS": "MRK", "LUKAS": "LUK", "JUAN": "JHN", "MIBƆANN": "ACT", "BEROMANO": "ROM",
    "1 BEKORINTIO": "1CO", "2 BEKORINTIO": "2CO", "BEGALATA": "GAL", "BEEFESIO": "EPH", "BEFILIPENSE": "PHP",
    "BEKOLOSENSE": "COL", "1 BETESALONIKA": "1TH", "2 BETESALONIKA": "2TH", "1 TIMOTEO": "1TI",
    "2 TIMOTEO": "2TI", "TITO": "TIT", "FILEMÓN": "PHM", "BEEBREO": "HEB", "SANTIAGO": "JAS", "1 PEDRO": "1PE",
    "2 PEDRO": "2PE", "1 JUAN": "1JN", "2 JUAN": "2JN", "3 JUAN": "3JN", "JUDAS": "JUD", "NLÉRÁNN": "REV",
}

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f​   ]")


@dataclass
class Typo:
    """Typographic thresholds (pt / page coords). Defaults measured on 00-WNTfan-web-2.pdf."""
    header_max_y: float = 50.0
    footnote_max_size: float = 6.5
    verse_size: tuple[float, float] = (6.6, 7.4)
    heading_size: tuple[float, float] = (8.6, 9.4)
    title_size: tuple[float, float] = (23.0, 25.0)
    subtitle_size: tuple[float, float] = (19.0, 21.0)
    dropcap_min_size: float = 26.0


def _in(x: float, rng: tuple[float, float]) -> bool:
    return rng[0] <= x <= rng[1]


def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", _CTRL.sub(" ", s)).strip().upper()


@dataclass
class Verse:
    book: str
    chapter: int
    verse: int
    text: str = ""
    page: int = 0
    parts: list[str] = field(default_factory=list)


def classify_spans(spans: list[dict], typo: Typo | None = None) -> list[tuple[str, str]]:
    """spans (already in reading order) -> [(kind, text)]; kind in title|chapter|verse|text|skip."""
    typo = typo or Typo()
    out = []
    for s in spans:
        t = _CTRL.sub(" ", s["text"]).strip() if s["text"] else ""
        if not t:
            continue
        size, bold, y0 = s["size"], "Bold" in s.get("font", "") or bool(s.get("flags", 0) & 16), s["bbox"][1]
        if y0 < typo.header_max_y:
            out.append(("skip", t))                              # running header / page number
        elif size >= typo.dropcap_min_size and t.isdigit():
            out.append(("chapter", t))
        elif _in(size, typo.title_size):
            out.append(("title", t))
        elif _in(size, typo.subtitle_size) or size < typo.footnote_max_size:
            out.append(("skip", t))                              # book subtitle / footnotes
        elif _in(size, typo.verse_size) and bold and t.isdigit():
            out.append(("verse", t))
        elif _in(size, typo.heading_size) and bold:
            out.append(("skip", t))                              # section heading
        else:
            out.append(("text", s["text"]))
    return out


def assemble(pages: list[list[tuple[str, str]]]) -> tuple[list[Verse], list[str]]:
    """Classified spans per page -> verses. Returns (verses, warnings)."""
    verses: list[Verse] = []
    warnings: list[str] = []
    book, chap, cur = None, 0, None
    for pno, items in enumerate(pages):
        for kind, t in items:
            if kind == "title":
                name = _norm_title(t)
                if name in TITLES:
                    book, chap, cur = TITLES[name], 0, None
            elif kind == "chapter" and book:
                chap = int(t)
                cur = Verse(book, chap, 1, page=pno)            # verse 1 is often unnumbered after a drop cap
                verses.append(cur)
            elif kind == "verse" and book:
                n = int(t)
                if chap == 0:                                   # one-chapter books may have no drop cap
                    chap = 1
                if cur and cur.chapter == chap and n == 1 and cur.verse == 1 and not cur.parts:
                    continue                                    # explicit "1" right after the drop cap
                if cur and cur.chapter == chap and n != cur.verse + 1:
                    warnings.append(f"{book} {chap}: verse {n} after {cur.verse} (p.{pno})")
                cur = Verse(book, chap, n, page=pno)
                verses.append(cur)
            elif kind == "text" and cur is not None:
                cur.parts.append(t)
    for v in verses:
        v.text = join_parts(v.parts)
    return [v for v in verses if v.text], warnings


def join_parts(parts: list[str]) -> str:
    s = ""
    for p in parts:
        p = _CTRL.sub(" ", p)
        if s.endswith("­") or s.endswith("-­"):        # soft hyphen at line end: join the word
            s = s.rstrip("­") + p.lstrip()
        else:
            s = f"{s} {p}"
    s = s.replace("­", "")
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", s).strip())


def order_spans(spans: list[dict], width: float, typo: Typo | None = None, row_tol: float = 3.5) -> list[dict]:
    """Reading order. Pages are single- or two-column (book openings are single-column): a page is two-column
    unless several body spans straddle the middle. Within a column, spans are grouped into rows by BASELINE
    (verse numbers sit ~2pt high, so they are lowered to the text baseline; drop caps span several lines, so
    they sort by their top edge) and read left to right."""
    typo = typo or Typo()
    mid = width / 2
    body = [s for s in spans if s["text"].strip() and s["bbox"][1] >= typo.header_max_y]
    straddle = sum(1 for s in body if s["bbox"][0] < mid - 20 and s["bbox"][2] > mid + 20)
    two_col = straddle < 3

    def row_y(s):
        if s["size"] >= typo.dropcap_min_size:
            return s["bbox"][1]
        y = s.get("origin", (0, s["bbox"][3]))[1]
        return y + 2 if _in(s["size"], typo.verse_size) else y

    def col(s):
        return 1 if two_col and s["bbox"][0] >= mid - 10 else 0

    out = []
    for c in (0, 1):
        cs = sorted((s for s in spans if col(s) == c), key=row_y)
        row, last = [], None
        for s in cs:
            y = row_y(s)
            if last is not None and y - last > row_tol:
                out.extend(sorted(row, key=lambda s: s["bbox"][0]))
                row = []
            row.append(s)
            last = y
        out.extend(sorted(row, key=lambda s: s["bbox"][0]))
    return out


def page_spans(page) -> list[dict]:
    d = page.get_text("dict")
    spans = [s for b in d["blocks"] for line in b.get("lines", []) for s in line["spans"]]
    return order_spans(spans, page.rect.width)


def parse_pdf(path: Path = PDF, typo: Typo | None = None) -> tuple[list[Verse], list[str]]:
    import fitz  # PyMuPDF (extra: ingest)

    doc = fitz.open(str(path))
    pages = [classify_spans(page_spans(p), typo) for p in doc]
    return assemble(pages)


# --------------------------------------------------------------------------------------------- align
class Scorer(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _add(a, b):
    return [x + y for x, y in zip(a, b, strict=True)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


BEADS = ((1, 1), (1, 2), (2, 1), (1, 0), (0, 1))


@dataclass
class AlignCfg:
    skip_cost: float = 0.9          # 1-0 / 0-1 bead
    # Merged sides embed with systematically higher cosine (more text), so merges need a real penalty, and an
    # exact verse-number match on a 1-1 bead earns a bonus (the typography parser numbers verses reliably).
    merge_penalty: float = 0.15     # extra cost for 1-2 / 2-1
    verse_mismatch_penalty: float = 0.10
    verse_match_bonus: float = 0.15  # 1-1 bead whose Fang and French verse numbers are equal


def align_chapter(fang: list[Verse], fr: list[dict], fv: list[list[float]], rv: list[list[float]],
                  cfg: AlignCfg | None = None) -> list[dict]:
    """Monotonic DP; fv/rv are unit vectors for fang/fr verses. Returns beads in order."""
    cfg = cfg or AlignCfg()
    n, m = len(fang), len(fr)
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    def bead_cost(i, j, di, dj):
        if di == 0 or dj == 0:
            return cfg.skip_cost
        a = _unit(fv[i]) if di == 1 else _unit(_add(fv[i], fv[i + 1]))
        b = _unit(rv[j]) if dj == 1 else _unit(_add(rv[j], rv[j + 1]))
        c = 1.0 - _dot(a, b)
        if (di, dj) != (1, 1):
            c += cfg.merge_penalty
        fnums = {fang[i + k].verse for k in range(di)}
        rnums = {int(fr[j + k]["verse"]) for k in range(dj)}
        if not fnums & rnums:
            c += cfg.verse_mismatch_penalty
        elif (di, dj) == (1, 1):
            c -= cfg.verse_match_bonus
        return c

    for i in range(n + 1):
        for j in range(m + 1):
            if cost[i][j] == INF:
                continue
            for di, dj in BEADS:
                if i + di > n or j + dj > m:
                    continue
                c = cost[i][j] + bead_cost(i, j, di, dj)
                if c < cost[i + di][j + dj]:
                    cost[i + di][j + dj] = c
                    back[i + di][j + dj] = (di, dj)
    beads, i, j = [], n, m
    while (i, j) != (0, 0):
        di, dj = back[i][j]
        i, j = i - di, j - dj
        sim = None
        if di and dj:
            a = _unit(fv[i]) if di == 1 else _unit(_add(fv[i], fv[i + 1]))
            b = _unit(rv[j]) if dj == 1 else _unit(_add(rv[j], rv[j + 1]))
            sim = round(_dot(a, b), 4)
        beads.append({"fang": fang[i:i + di], "fr": fr[j:j + dj], "bead": f"{di}-{dj}", "sim": sim})
    return beads[::-1]


def load_ref(path: Path = REF) -> dict[tuple[str, int], list[dict]]:
    by_ch = defaultdict(list)
    for r in json.loads(path.read_text(encoding="utf-8")):
        by_ch[(r["book"], int(r["chapter"]))].append(r)
    for v in by_ch.values():
        v.sort(key=lambda r: int(r["verse"]))
    return by_ch


class OllamaScorer:
    """The v0 'sim' scorer: EmbeddingGemma via the local Ollama HTTP API, with the similarity task prefix.
    Embeddings are cached on disk (sha of model+text) so reruns never re-embed (cache outside the repo)."""

    def __init__(self, model: str = "embeddinggemma", batch: int = 64, url: str = "http://127.0.0.1:11434",
                 cache_dir: Path | None = None):
        self.model, self.batch, self.url = model, batch, url.rstrip("/")
        self.cache_dir = cache_dir or Path.home() / "retrievve-runs" / "emb_cache" / model.replace(":", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> Path:
        import hashlib
        h = hashlib.sha256(f"{self.model}\x1f{text}".encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def _post(self, texts: list[str]) -> list[list[float]]:
        import urllib.request
        body = json.dumps({"model": self.model,
                           "input": [f"task: sentence similarity | query: {t}" for t in texts]}).encode()
        req = urllib.request.Request(f"{self.url}/api/embed", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())["embeddings"]

    def embed(self, texts):
        out: list = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            k = self._key(t)
            if k.exists():
                out[i] = json.loads(k.read_text())
            else:
                todo.append(i)
        for s in range(0, len(todo), self.batch):
            idx = todo[s:s + self.batch]
            for i, v in zip(idx, self._post([texts[i] for i in idx]), strict=True):
                k = self._key(texts[i])
                k.parent.mkdir(parents=True, exist_ok=True)
                k.write_text(json.dumps(v))
                out[i] = v
        return out


def run(scorer: Scorer, books: set[str] | None = None, pdf: Path = PDF, ref: Path = REF,
        out: Path = OUT, report_path: Path = REPORT, log: Callable[[str], None] = print,
        cfg: AlignCfg | None = None) -> dict:
    verses, warnings = parse_pdf(pdf)
    refs = load_ref(ref)
    by_ch = defaultdict(list)
    for v in verses:
        if books is None or v.book in books:
            by_ch[(v.book, v.chapter)].append(v)
    rows, per_book = [], defaultdict(Counter)
    sims = defaultdict(list)
    for (book, ch), fang in sorted(by_ch.items(), key=lambda kv: (list(TITLES.values()).index(kv[0][0]),
                                                                     kv[0][1])):
        fr = refs.get((book, ch), [])
        per_book[book]["fang_verses"] += len(fang)
        per_book[book]["fr_verses"] += len(fr)
        per_book[book]["chapters"] += 1
        if len(fang) != len(fr):
            per_book[book]["chapters_count_mismatch"] += 1
        if not fr:
            per_book[book]["chapters_missing_ref"] += 1
            continue
        fv = [_unit(x) for x in scorer.embed([v.text for v in fang])]
        rv = [_unit(x) for x in scorer.embed([r["text"] for r in fr])]
        for b in align_chapter(fang, fr, fv, rv, cfg):
            per_book[book][f"bead_{b['bead']}"] += 1
            if not b["fang"] or not b["fr"]:
                continue
            sims[book].append(b["sim"])
            fvs, rvs = b["fang"], b["fr"]
            rows.append({
                "lang": "fang", "book": book, "chapter": ch,
                "fang_verses": [v.verse for v in fvs], "fr_verses": [int(r["verse"]) for r in rvs],
                "ref": f"{book} {ch}:{rvs[0]['verse']}" + (f"-{rvs[-1]['verse']}" if len(rvs) > 1 else ""),
                "text": " ".join(v.text for v in fvs), "fr": " ".join(r["text"] for r in rvs),
                "bead": b["bead"], "sim": b["sim"], "page": fvs[0].page,
            })
        log(f"  {book} {ch}: fang={len(fang)} fr={len(fr)}")
    for book, s in sims.items():
        s = sorted(s)
        per_book[book]["sim_median"] = s[len(s) // 2] if s else None
        per_book[book]["sim_p10"] = s[len(s) // 10] if s else None
    write_jsonl(out, rows)
    report = {"pdf": str(pdf.name), "ref": str(ref.name), "verses_parsed": len(verses),
              "pairs": len(rows), "parse_warnings": len(warnings), "parse_warning_samples": warnings[:50],
              "per_book": {b: dict(c) for b, c in per_book.items()}, "align_cfg": vars(cfg or AlignCfg()),
              "acceptance": "88's proper-noun agreement >= 0.8 per book; accepted books go to "
                            "config nt_books_accepted"}
    write_json(report_path, report)
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf", type=Path, default=PDF)
    ap.add_argument("--ref", type=Path, default=REF)
    ap.add_argument("--books", default="", help="comma-separated USFM codes (default: all)")
    ap.add_argument("--scorer", default="ollama:embeddinggemma")
    ap.add_argument("--parse-only", action="store_true", help="parse verses and report counts, no alignment")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--merge-penalty", type=float, default=AlignCfg.merge_penalty)
    ap.add_argument("--verse-bonus", type=float, default=AlignCfg.verse_match_bonus)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args(argv)
    if args.parse_only:
        verses, warnings = parse_pdf(args.pdf)
        refs = load_ref(args.ref)
        cnt = Counter(v.book for v in verses)
        ref_cnt = Counter(b for (b, _c), vs in refs.items() for _ in vs)
        for b in TITLES.values():
            print(f"{b}: fang={cnt[b]} ls1910={ref_cnt[b]}")
        print(f"warnings={len(warnings)}")
        return
    kind, _, model = args.scorer.partition(":")
    if kind != "ollama":
        raise SystemExit(f"unknown scorer {args.scorer}")
    books = {b.strip() for b in args.books.split(",") if b.strip()} or None
    cfg = AlignCfg(merge_penalty=args.merge_penalty, verse_match_bonus=args.verse_bonus)
    rep = run(OllamaScorer(model or "embeddinggemma"), books, args.pdf, args.ref, args.out, args.report, cfg=cfg)
    print(f"pairs={rep['pairs']} verses={rep['verses_parsed']} warnings={rep['parse_warnings']}")


if __name__ == "__main__":
    main()
