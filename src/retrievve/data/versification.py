"""Hebrew (BHS / Leningrad) -> English (KJV, = LS1910) versification for the Old Testament.

Standard chapter/verse differences (cf. the CrossWire/OSIS "Leningrad" vs "KJV" versification mappings).
Each rule maps a Hebrew range (book, chapter, v_from, v_to) to an English start (chapter, verse); verses map
one-to-one inside a rule. Psalms superscriptions are NOT covered (they change verse counts per psalm) —
a source with versification: hebrew in PSA is rejected rather than silently mis-mapped.

Applied per source via sources.yaml (`versification: hebrew`, optional `versification_scope: [DEU.13, ...]`
when a source follows Hebrew numbering only in some chapters, as established by align_check --scan-offsets).
"""
from __future__ import annotations

# (book, heb_ch, heb_from, heb_to) -> (eng_ch, eng_from)
HEB_TO_ENG: list[tuple[tuple[str, int, int, int], tuple[int, int]]] = [
    (("GEN", 32, 1, 1), (31, 55)), (("GEN", 32, 2, 33), (32, 1)),
    (("EXO", 7, 26, 29), (8, 1)), (("EXO", 8, 1, 28), (8, 5)),
    (("EXO", 21, 37, 37), (22, 1)), (("EXO", 22, 1, 30), (22, 2)),
    (("LEV", 5, 20, 26), (6, 1)), (("LEV", 6, 1, 23), (6, 8)),
    (("NUM", 17, 1, 15), (16, 36)), (("NUM", 17, 16, 28), (17, 1)),
    (("NUM", 30, 1, 1), (29, 40)), (("NUM", 30, 2, 17), (30, 1)),
    (("DEU", 13, 1, 1), (12, 32)), (("DEU", 13, 2, 19), (13, 1)),
    (("DEU", 23, 1, 1), (22, 30)), (("DEU", 23, 2, 26), (23, 1)),
    (("DEU", 28, 69, 69), (29, 1)), (("DEU", 29, 1, 28), (29, 2)),
    (("1SA", 21, 1, 1), (20, 42)), (("1SA", 21, 2, 16), (21, 1)),
    (("1SA", 24, 1, 1), (23, 29)), (("1SA", 24, 2, 23), (24, 1)),
    (("2SA", 19, 1, 1), (18, 33)), (("2SA", 19, 2, 44), (19, 1)),
    (("1KI", 5, 1, 14), (4, 21)), (("1KI", 5, 15, 32), (5, 1)),
    (("2KI", 12, 1, 1), (11, 21)), (("2KI", 12, 2, 22), (12, 1)),
    (("1CH", 5, 27, 41), (6, 1)), (("1CH", 6, 1, 66), (6, 16)),
    (("2CH", 1, 18, 18), (2, 1)), (("2CH", 2, 1, 17), (2, 2)),
    (("2CH", 13, 23, 23), (14, 1)), (("2CH", 14, 1, 14), (14, 2)),
    (("NEH", 3, 33, 38), (4, 1)), (("NEH", 4, 1, 17), (4, 7)),
    (("NEH", 10, 1, 1), (9, 38)), (("NEH", 10, 2, 40), (10, 1)),
    (("JOB", 40, 25, 32), (41, 1)), (("JOB", 41, 1, 26), (41, 9)),
    (("ECC", 4, 17, 17), (5, 1)), (("ECC", 5, 1, 19), (5, 2)),
    (("SNG", 7, 1, 1), (6, 13)), (("SNG", 7, 2, 14), (7, 1)),
    (("ISA", 8, 23, 23), (9, 1)), (("ISA", 9, 1, 20), (9, 2)),
    (("JER", 8, 23, 23), (9, 1)), (("JER", 9, 1, 25), (9, 2)),
    (("EZK", 21, 1, 5), (20, 45)), (("EZK", 21, 6, 37), (21, 1)),
    (("DAN", 3, 31, 33), (4, 1)), (("DAN", 4, 1, 34), (4, 4)),
    (("DAN", 6, 1, 1), (5, 31)), (("DAN", 6, 2, 29), (6, 1)),
    (("HOS", 2, 1, 2), (1, 10)), (("HOS", 2, 3, 25), (2, 1)),
    (("HOS", 12, 1, 1), (11, 12)), (("HOS", 12, 2, 15), (12, 1)),
    (("HOS", 14, 1, 1), (13, 16)), (("HOS", 14, 2, 10), (14, 1)),
    (("JOL", 3, 1, 5), (2, 28)), (("JOL", 4, 1, 21), (3, 1)),
    (("JON", 2, 1, 1), (1, 17)), (("JON", 2, 2, 11), (2, 1)),
    (("MIC", 4, 14, 14), (5, 1)), (("MIC", 5, 1, 14), (5, 2)),
    (("NAM", 2, 1, 1), (1, 15)), (("NAM", 2, 2, 14), (2, 1)),
    (("ZEC", 2, 1, 4), (1, 18)), (("ZEC", 2, 5, 17), (2, 1)),
    (("MAL", 3, 19, 24), (4, 1)),
]
UNSUPPORTED_BOOKS = {"PSA"}


def heb_to_eng(book: str, ch: int, v: int) -> tuple[int, int]:
    """Map a Hebrew-numbered verse to English (KJV/LS1910) numbering; identity outside the table."""
    if book in UNSUPPORTED_BOOKS:
        raise ValueError(f"{book}: Hebrew->English versification not supported (superscriptions)")
    for (b, hch, vf, vt), (ech, ev) in HEB_TO_ENG:
        if b == book and hch == ch and vf <= v <= vt:
            return ech, ev + (v - vf)
    return ch, v


def remap_rows(rows: list[dict], scope: set[str] | None = None) -> tuple[list[dict], dict]:
    """Re-pair a verse-keyed parallel source whose TARGET side follows Hebrew numbering while the French side
    (LS1910) follows English numbering, both stored under the same row label.

    rows: parser rows with locator "BOOK.ch.v", text_raw (target, Hebrew-numbered) and fr_raw (French,
    English-numbered). scope: chapters ("BOOK.ch", Hebrew numbering) where the target is Hebrew-numbered;
    None = whole source. Returns (rows with English locators and re-paired French, report)."""
    fr_by_loc = {r["locator"]: r["fr_raw"] for r in rows}
    out, moved, unpaired, collisions = [], 0, 0, 0
    taken: dict[str, int] = {}
    for r in rows:
        parts = r["locator"].split(".")
        if len(parts) != 3 or not (parts[1].isdigit() and parts[2].isdigit()):
            out.append(r)
            continue
        book, ch, v = parts[0], int(parts[1]), int(parts[2])
        if scope is not None and f"{book}.{ch}" not in scope:
            out.append(r)
            continue
        ech, ev = heb_to_eng(book, ch, v)
        loc = f"{book}.{ech}.{ev}"
        if loc == r["locator"]:
            out.append(r)
            continue
        moved += 1
        fr = fr_by_loc.get(loc)
        unpaired += fr is None
        out.append({**r, "locator": loc, "fr_raw": fr,
                    "extra": {**(r.get("extra") or {}), "heb_ref": r["locator"]}})
    # A moved verse may now share an English locator with an unmoved one (e.g. DEU 13:1 -> 12:32 while the
    # source also has a 12:32 row). Keep the unmoved row's pairing; the moved one loses its French side.
    for r in out:
        if "heb_ref" not in (r.get("extra") or {}):
            taken[r["locator"]] = 1
    for r in out:
        if "heb_ref" in (r.get("extra") or {}) and r["locator"] in taken:
            r["fr_raw"] = None
            collisions += 1
    return out, {"moved": moved, "unpaired_after_move": unpaired, "locator_collisions": collisions}
