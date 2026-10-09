"""Regression tests for the PR #4 review blockers B2 (test wins exact dups) and B3 (no held-out text in
dialogue context). Synthetic records, run through the same quality pass as the build."""
from __future__ import annotations

import copy

import pytest

from retrievve.data import derive
from retrievve.data.build import quality_pass
from retrievve.data.common import load_config
from retrievve.data.normalize import text_fold, text_key
from retrievve.data.quality import usable

CFG = load_config()


def rec(rid, text, fr, split, *, kind="sentence", domain="conversational", group="g", order=1, source="src"):
    return {"id": rid, "lang": "fan", "text": text, "text_raw": text, "fr": fr, "kind": kind, "domain": domain,
            "source_id": source, "group_id": group, "order": order, "orthography": "",
            "align": {"method": "row", "score": 1.0}, "ocr": False, "flags": [], "split": split, "release": True,
            "text_key": text_key(text), "text_fold": text_fold(text)}


# ------------------------------------------------------------------------------------------------- B2
PAIR = ("Mə ne nda ya bisi bəse", "Je suis à la maison avec tout le monde")


@pytest.mark.parametrize("train_first", [True, False])
def test_b2_test_copy_survives_exact_duplicate(train_first):
    train = rec("fan:a:1", *PAIR, "train", source="a", group="ga")
    test = rec("fan:b:1", *PAIR, "test", source="b", group="gb")
    filler = [rec(f"fan:c:{i}", f"Ma kɔ̂ ya nda {i} ening", f"Je pars à la maison numéro {i}", "train",
                  source="c", group="gc", order=i) for i in range(3)]
    records = ([train, test] if train_first else [test, train]) + filler
    quality_pass(records, copy.deepcopy(CFG))
    assert usable(test), test["flags"]
    assert not usable(train)
    assert {"exact_dup", "xsplit_dup"} & set(train["flags"])


# ------------------------------------------------------------------------------------------------- B3
def test_b3_held_out_turn_never_in_train_prompt():
    held_text, held_fr = "Wa yiane ke dzam ete ening ya", "Est-ce que tu veux faire cela ce soir"
    dialogue = [
        rec("fan:d:1", "Mbolo ah mot wam", "Bonjour mon ami", "train", kind="dialogue_turn", source="d",
            group="gd", order=1),
        rec("fan:d:2", held_text, held_fr, "train", kind="dialogue_turn", source="d", group="gd", order=2),
        rec("fan:d:3", "Ɛ̂ɛ, ma yiane dzo", "Oui, je le veux bien", "train", kind="dialogue_turn", source="d",
            group="gd", order=3),
    ]
    test = rec("fan:t:1", held_text, held_fr, "test", source="t", group="gt")
    records = dialogue + [test]
    quality_pass(records, copy.deepcopy(CFG))
    assert "xsplit_dup" in dialogue[1]["flags"]
    examples = derive.sft_examples(records, "fan", "train", CFG)
    prompts = [m["content"] for e in examples for m in e["messages"] if m["role"] == "user"]
    assert prompts, "expected some train examples"
    assert not any(held_text in p for p in prompts)
    assert not any(held_text == e["messages"][-1]["content"] for e in examples)


# ------------------------------------------------------------------------------------------------- N5
def test_n5_shared_gloss_gives_one_fr2x_example():
    records = [rec("fan:l:1", "akiba", "merci", "train", kind="lexicon", domain="lexical", source="l", order=1),
               rec("fan:l:2", "abora", "merci", "train", kind="lexicon", domain="lexical", source="l", order=2)]
    quality_pass(records, copy.deepcopy(CFG))
    ex = [e for e in derive.sft_examples(records, "fan", "train", CFG) if e["meta"]["task"] == "lex_fr2x"]
    assert len(ex) == 1
    assert ex[0]["messages"][-1]["content"] == "akiba ; abora"


# --------------------------------------------------------------------------------- sources_disabled
def test_fang_nt_v0_alignment_is_disabled_by_config():
    assert "fang_nt_web_v0" in (CFG.get("sources_disabled") or {})


# --------------------------------------------------------------------------- disabled source (behavioural)
def test_disabled_source_produces_nothing(tmp_path):
    from retrievve.data.build import build_records
    from retrievve.data.normalize import Normalizer

    csv_path = tmp_path / "fang_demo.csv"
    csv_path.write_text("fang,french\nMbolo ah mot wam,Bonjour mon ami\n", encoding="utf-8")
    entry = {"path": str(csv_path), "lang_hint": "fang", "parser": "csv", "kind": "sentence",
             "domain": "conversational", "release": True, "enabled": True}
    cfg = copy.deepcopy(CFG)
    cfg["sources_disabled"] = {"fang_demo": "test"}
    records, per_source = build_records({"fang_demo": entry}, {"fang_demo": True}, cfg, Normalizer(cfg))
    assert records == [] and per_source["fang_demo"]["status"] == "disabled by config"
    cfg["sources_disabled"] = {}
    records, _ = build_records({"fang_demo": entry}, {"fang_demo": True}, cfg, Normalizer(cfg))
    assert len(records) == 1                                  # control: same source enabled yields a record


def test_ocr_review_records_never_exempt_french_words():
    from retrievve.data.quality import lexicon_tokens
    ocr = rec("fan:g:1", "Grande pierre du chemin", "x", "train", kind="lexicon", domain="lexical")
    ocr["flags"] = ["ocr_review"]
    clean = rec("fan:t:1", "nzok", "éléphant", "train", kind="lexicon", domain="lexical")
    toks = lexicon_tokens([ocr, clean])["fan"]
    assert "pierre" not in toks and "nzok" in toks


def test_vocab_detector_needs_two_qualifying_tokens_and_reports_blind_spot():
    from retrievve.data.quality import flag_record, fr_detector_blind_spot
    fcfg = copy.deepcopy(CFG["filters"])
    vocab = {"été", "dans", "vieille", "route"}
    one = rec("fan:x:1", "été", "dans", "train")                       # Fang "été" (= dans): homograph
    two = rec("fan:x:2", "vieille route", "le chemin du village", "train")
    for r in (one, two):
        flag_record(r, fcfg, None, set(), vocab)
    assert "fr_in_target" not in one["flags"] and "fr_in_target" in two["flags"]
    blind = fr_detector_blind_spot([one, two], {"fan": vocab}, {"fan": set()}, 3)
    assert blind["count"] == 1 and blind["examples"][0]["id"] == "fan:x:1"
