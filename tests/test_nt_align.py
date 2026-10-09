"""Fang NT realignment: typography-based verse parsing and the monotonic bead DP.

All fixtures are SYNTHETIC (invented Fang-like strings): the Fang NT is release:false, no real verse text in the repo.
"""
from __future__ import annotations

from retrievve.data.nt_align import AlignCfg, Verse, align_chapter, assemble, classify_spans, join_parts


def span(text, size=10.0, bold=False, y=100.0, x=40.0):
    return {"text": text, "size": size, "font": "CharisSIL-Bold" if bold else "CharisSIL", "flags": 0,
            "bbox": [x, y, x + 10, y + 10]}


def test_typography_classes():
    kinds = [k for k, _ in classify_spans([
        span("MATEO 2, 3", 9.0, True, y=29),            # running header
        span("MATEO", 24.0),                              # book title
        span("É Kálara Ntoma", 20.0),                      # subtitle
        span("3", 27.9),                                  # chapter drop cap
        span("Ndzeñ Ntoma", 9.0, True),                # section heading
        span("12", 7.0, True),                            # verse number
        span("Mi tobe ñkó", 10.0),                       # text
        span("note", 6.0),                                # footnote
    ])]
    assert kinds == ["skip", "title", "skip", "chapter", "skip", "verse", "text", "skip"]


def test_wrapped_lines_are_not_verses_and_dropcap_starts_verse_1():
    page = classify_spans([
        span("MATEO", 24.0), span("1", 27.9), span("Ntoma a ngá tobe;", y=120),
        span("é nda dzam.", y=132),                     # wrapped line: same verse
        span("2", 7.0, True, y=144), span("Zila a ngá kene,", y=144),
        span("2", 27.9, y=200), span("1", 7.0, True, y=210), span("Ntoma a ngá dzo", y=210),
    ])
    verses, warnings = assemble([page])
    assert [(v.book, v.chapter, v.verse) for v in verses] == [("MAT", 1, 1), ("MAT", 1, 2), ("MAT", 2, 1)]
    assert verses[0].text == "Ntoma a ngá tobe; é nda dzam."
    assert not warnings


def test_soft_hyphen_join():
    assert join_parts(["chan­", "ter. Voir"]) == "chanter. Voir"


def _vec(i, dim=8):
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def test_dp_finds_2_1_bead():
    # Fang verses 1,2 together translate French verse 1; then 1-1 for the rest.
    fang = [Verse("MAT", 1, k, text=f"f{k}") for k in (1, 2, 3, 4)]
    fr = [{"verse": k, "text": f"r{k}"} for k in (1, 2, 3)]
    fv = [[1, 1, 0, 0, 0, 0, 0, 0], [1, -0.2, 0, 0, 0, 0, 0, 0], _vec(2), _vec(3)]
    rv = [[1, 0.4, 0, 0, 0, 0, 0, 0], _vec(2), _vec(3)]
    beads = align_chapter(fang, fr, fv, rv, AlignCfg(verse_mismatch_penalty=0.0))
    assert [b["bead"] for b in beads] == ["2-1", "1-1", "1-1"]
    assert [[v.verse for v in b["fang"]] for b in beads] == [[1, 2], [3], [4]]


def test_dp_identity_is_all_1_1():
    fang = [Verse("MRK", 1, k, text="x") for k in range(1, 6)]
    fr = [{"verse": k, "text": "y"} for k in range(1, 6)]
    vecs = [_vec(k) for k in range(5)]
    beads = align_chapter(fang, fr, vecs, vecs)
    assert {b["bead"] for b in beads} == {"1-1"} and all(b["sim"] == 1.0 for b in beads)


def test_reading_order_dropcap_before_first_line_and_raised_verse_numbers():
    from retrievve.data.nt_align import order_spans

    def sp(t, size, x, top, base, bold=False):
        return {"text": t, "size": size, "font": "CharisSIL-Bold" if bold else "CharisSIL", "flags": 0,
                "bbox": [x, top, x + 300 if size == 10.0 else x + 5, top + 12], "origin": (x, base)}
    spans = [  # single-column book opening (body spans straddle the middle of a 420pt page)
        sp("Zila a ngá kene", 10.0, 92, 244, 257), sp("2", 7.0, 86, 246, 255, True),
        sp("1", 27.9, 52, 220, 257), sp("Ntoma a ngá tobe;", 10.0, 81, 232, 245),
        sp("1", 7.0, 76, 234, 243, True), sp("Okó bɔdzaŋ.", 10.0, 54, 256, 269),
        sp("3", 7.0, 121, 258, 267, True), sp("Okó bá Nzila", 10.0, 127, 256, 269),
    ]
    assert [s["text"] for s in order_spans(spans, 420)] == [
        "1", "1", "Ntoma a ngá tobe;", "2", "Zila a ngá kene", "Okó bɔdzaŋ.", "3", "Okó bá Nzila"]


def test_nt_books_gate(tmp_path):
    import copy
    import json

    from retrievve.data.build import build_records
    from retrievve.data.common import load_config
    from retrievve.data.normalize import Normalizer

    p = tmp_path / "aligned.jsonl"
    rows = [{"book": b, "chapter": 1, "fang_verses": [1], "fr_verses": [1], "text": f"fang {b}", "fr": f"fr {b}",
             "bead": "1-1", "sim": 0.6} for b in ("MAT", "ROM")]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    cfg = copy.deepcopy(load_config())
    cfg["sources_disabled"] = {}
    cfg["nt_books_accepted"] = ["MAT"]
    entry = {"path": str(p), "lang_hint": "fang", "parser": "nt_aligned", "kind": "verse", "domain": "religious",
             "release": False, "enabled": True, "accept_books_key": "nt_books_accepted"}
    recs, per = build_records({"fang_nt_v1": entry}, {"fang_nt_v1": True}, cfg, Normalizer(cfg))
    assert [r["group_id"] for r in recs] == ["bible/MAT"]
    assert per["fang_nt_v1"]["book_gate"] == {"accepted": ["MAT"], "excluded": ["ROM"]}
