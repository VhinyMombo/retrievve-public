"""Lexicon probe: deterministic, POS-stratified, homographs merged, held out of SFT lex tasks."""
from __future__ import annotations

import copy

from retrievve.data import derive, lexprobe
from retrievve.data.common import load_config
from retrievve.data.normalize import text_fold, text_key

CFG = load_config()


def lex(i, text, fr, pos):
    return {"id": f"puu:l:{i}", "lang": "puu", "text": text, "text_raw": text, "fr": fr, "kind": "lexicon",
            "domain": "lexical", "source_id": "l", "group_id": "l", "order": i, "orthography": "", "align": {},
            "ocr": False, "flags": [], "split": "train", "release": True, "text_key": text_key(text),
            "text_fold": text_fold(text), "lex": {"headword": text, "pos": pos, "gloss_fr": fr, "examples": []}}


def test_selection_deterministic_stratified_and_held_out_of_sft():
    recs = [lex(i, f"mot{i}", f"sens {i}", "n" if i % 3 else "vb") for i in range(30)]
    recs.append(lex(99, "mot1", "autre sens", "n"))           # homograph of mot1 -> merged
    recs.append(lex(98, "vide", "", "n"))                     # empty gloss -> dropped
    a = lexprobe.select(recs, "puu", 9, 1729)
    assert a == lexprobe.select(copy.deepcopy(recs), "puu", 9, 1729) and len(a) == 9
    pos = {k: ("vb" if int(k[3:]) % 3 == 0 else "n") for k in a}
    assert sorted(pos.values()).count("vb") == 3              # 10 of 30 are vb -> 3 of 9
    assert "vide" not in a
    ex = derive.sft_examples(recs, "puu", "train", CFG, exclude_lex_keys=set(a))
    lex_targets = {m["content"] for e in ex if e["meta"]["task"].startswith("lex") for m in e["messages"]}
    assert not any(k in t.split(" ; ") or k == t for k in a for t in lex_targets)


def test_tonal_minimal_pair_is_not_a_variant_and_in_sft_matching():
    recs = [lex(1, "bàla", "enfant", "n"), lex(2, "bálá", "maison", "n"), lex(3, "mu kento", "femme", "n")]
    keys = [text_key("bàla"), text_key("mu kento")]
    rows = {r["headword"]: r for r in lexprobe.rows(recs, "puu", keys, ["Bàla, wa ma ghenda.", "O mu kento!"])}
    assert rows["bàla"]["variants"] == ["bàla"]               # tone-different word is NOT a variant
    assert rows["bàla"]["fold_neighbours"] == ["bálá"]        # kept for diagnostics only
    assert rows["bàla"]["in_sft_sentences"]                   # "Bàla," (punctuation-attached) matches
    assert rows["mu kento"]["in_sft_sentences"]               # two-word headword matches with boundaries


def test_listed_variant_spelling_counts():
    r = lex(1, "dibaku", "chance", "n")
    r["extra"] = {"variante": "dibagu; dibaghu"}
    row = lexprobe.rows([r], "puu", [text_key("dibaku")], [])[0]
    assert set(row["variants"]) == {"dibaku", "dibagu", "dibaghu"}
