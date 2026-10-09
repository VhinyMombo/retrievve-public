"""Galley v1 OCR parser: dehyphenation, tone groups, POS fixes, headers, French section cut-off.

All fixtures are SYNTHETIC: invented headwords and French glosses shaped like the dictionary's layout (the
dictionary itself is copyrighted and release:false; no real entry text may live in the repo).
"""
from __future__ import annotations

from retrievve.data.galley import definition, join_lines, norm_pos, parse_entry, split_entries

OCR = "\n".join([
    "DICTIONNAIRE", "A",
    "BEKO-NDAMA (hh) n.l, pl. mebeko-",
    "ndama (pied de pierre). Petite plan­",
    "te grise des bords de ri­",
    "vière.",
    "TOLÔZA (h) n,4, ss sing. (vb tolô h). Récolte, partage.",
    "PIKA (h) (Im) vb 1. Ranger. — 2. Plier.",
    "PIK 147 PIL",
    "FRANÇAIS-FANG",
    "ARRANGER (se), se mettre en ordre, pika",
])


def test_split_and_section_cutoff():
    ents, stats = split_entries(OCR)
    assert [h for h, _ in ents] == ["BEKO-NDAMA", "TOLÔZA", "PIKA"]   # French index (ARRANGER) excluded
    assert stats["headers_dropped"] >= 1                              # the "PIK 147 PIL" running header
    assert "PIK" not in dict(ents)["PIKA"]


def test_dehyphenation_keeps_real_hyphen():
    assert join_lines(["pl. mebeko-", "ndama. Petite plan­", "te"]) == "pl. mebeko-ndama. Petite plante"


def test_entry_fields():
    ents = dict(split_entries(OCR)[0])
    e = parse_entry("BEKO-NDAMA", ents["BEKO-NDAMA"])
    d = definition(e["rest"])
    assert e["tones"] == ["hh"] and e["pos"] == "n.1"
    assert d == {"gloss": "Petite plante grise des bords de rivière", "plural": "mebeko-ndama",
                 "deriv": None, "literal": "pied de pierre", "notes": []}
    m = parse_entry("TOLÔZA", ents["TOLÔZA"])
    assert m["pos"] == "n.4" and definition(m["rest"])["notes"] == ["no_singular"]
    a = parse_entry("PIKA", ents["PIKA"])
    assert a["tones"] == ["h", "lm"] and a["pos"] == "vb"           # second group captured, OCR I -> l
    assert definition(a["rest"])["gloss"] == "Ranger"


def test_pos_normalisation():
    assert [norm_pos(p) for p in ("n.l", "n.I", "n,4", "n.S", "vb")] == ["n.1", "n.1", "n.4", "n.3", "vb"]


def test_every_galley_source_is_release_false(tmp_path):
    from retrievve.data.sources import sync_registry
    reg = sync_registry(tmp_path / "sources.yaml", tmp_path / "raw_empty")
    galley = {sid: e for sid, e in reg.items() if "galley" in sid}
    assert galley and all(e["release"] is False for e in galley.values())
    assert all("Galley" in e["license"] for e in galley.values())


def test_recheck_fixes():
    from retrievve.data.galley import TONE, norm_pos
    assert TONE.match("bf")                                             # (bf) tone group
    e = parse_entry("DUMA", "(b) vb récip. Se croiser. Ba duma, ils se croisent.")
    assert e["pos"] == "vb.récip" and definition(e["rest"])["gloss"] == "Se croiser"
    assert norm_pos("vb. aux.") == "vb.aux"
    assert definition("pl, beto. Objet")["plural"] == "beto"                # "pl," OCR comma
    assert definition("1, Marcher, Duk za me, il marche")["gloss"] == "Marcher"   # comma-for-period
    assert definition("Briller, luire, Syn,: kong")["gloss"] == "Briller, luire"      # Syn cut
    assert definition("Corde, par ex. de liane. Suite")["gloss"] == "Corde, par ex. de liane"  # abbreviation
    assert "no_plural" in definition("ss pl. Sable")["notes"]


def test_em_dash_inline_header_removed():
    assert join_lines(["Outil en TAK — 312 — TAM bois"]) == "Outil en bois"
    assert join_lines(["la DOR — 81 — DUN suite"]) == "la suite"


def test_derived_verb_base_and_example_only_entries():
    d = definition("de dumé (h). Se rencontrer récip.", "DUMA")
    assert d["deriv"] == "de dumé (h)" and d["gloss"] == "Se rencontrer récip"
    assert definition("de kobe (b). Koba nzok, se partager la viande.", "KOBA")["gloss"] is None
    assert definition("pl, meto (vb to b). Ato mvem, sac de sel", "ATO")["gloss"] is None
