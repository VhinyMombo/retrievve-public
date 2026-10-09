"""DATA_SPEC v0.8 §3 freeze guard: frozen slices never change or disappear without --unfreeze + reason."""
from __future__ import annotations

from retrievve.data.build import check_frozen, promote, protected_files
from retrievve.data.common import sha256_file


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _manifest(live, frozen):
    files = {p.relative_to(live).as_posix(): {"sha256": sha256_file(p)} for p in live.rglob("*") if p.is_file()}
    return {"frozen_slices": frozen, "files": files,
            "slices": {"puu.general": {"files": ["eval/puu.mt_fr2x.general.test.jsonl"]}}}


def _setup(tmp_path):
    live, stage = tmp_path / "v1", tmp_path / "v1.staging"
    _write(live, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1}\n')
    _write(live, "eval/fan.mt_fr2x.general.test.jsonl", '{"id": 2}\n')
    prev = _manifest(live, {"puu.general": True, "fan.general": False})
    return live, stage, prev


def test_identical_rebuild_passes(tmp_path):
    live, stage, prev = _setup(tmp_path)
    _write(stage, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1}\n')
    _write(stage, "eval/fan.mt_fr2x.general.test.jsonl", '{"id": 2}\n')
    assert check_frozen(prev, stage, {}) == ([], [])


def test_modified_frozen_file_aborts_and_unfrozen_slice_may_change(tmp_path):
    live, stage, prev = _setup(tmp_path)
    _write(stage, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1, "x": "changed"}\n')
    _write(stage, "eval/fan.mt_fr2x.general.test.jsonl", '{"id": 999}\n')   # fan.general not frozen
    problems, log = check_frozen(prev, stage, {})
    assert len(problems) == 1 and "puu.general" in problems[0] and "would change" in problems[0]
    assert log == []


def test_deleted_frozen_file_aborts(tmp_path):
    live, stage, prev = _setup(tmp_path)
    _write(stage, "eval/fan.mt_fr2x.general.test.jsonl", '{"id": 2}\n')
    problems, _ = check_frozen(prev, stage, {})
    assert problems and "would be deleted" in problems[0]


def test_unfreeze_with_reason_is_logged(tmp_path):
    live, stage, prev = _setup(tmp_path)
    _write(stage, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1, "x": "fixed"}\n')
    problems, log = check_frozen(prev, stage, {"puu.general": "typo fix in reference"})
    assert problems == [] and log[0]["slice"] == "puu.general" and log[0]["reason"] == "typo fix in reference"
    assert log[0]["changes"][0]["new_sha256"] != log[0]["changes"][0]["old_sha256"]


def test_promote_never_deletes_protected_files(tmp_path):
    live, stage, prev = _setup(tmp_path)
    _write(live, "eval/stale.test.jsonl", "{}\n")
    _write(stage, "eval/fan.mt_fr2x.general.test.jsonl", '{"id": 2}\n')       # staged build lacks puu file
    removed = promote(stage, live, protected_files(prev, {}))
    assert (live / "eval/puu.mt_fr2x.general.test.jsonl").exists()             # frozen: kept
    assert removed == ["eval/stale.test.jsonl"] and not stage.exists()


def test_first_build_has_no_guard(tmp_path):
    assert check_frozen(None, tmp_path, {}) == ([], [])


def test_silent_defreeze_aborts_unless_unfrozen(tmp_path):
    live, stage, prev = _setup(tmp_path)
    _write(stage, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1}\n')     # files identical...
    _write(stage, "eval/fan.mt_fr2x.general.test.jsonl", '{"id": 2}\n')
    new = {"puu.general": False, "fan.general": False}                     # ...but a config flip de-freezes
    problems, _ = check_frozen(prev, stage, {}, new)
    assert problems and "de-frozen" in problems[0]
    assert check_frozen(prev, stage, {}, {"fan.general": False})[0][0].endswith("without --unfreeze")
    assert check_frozen(prev, stage, {"puu.general": "gate re-opened"}, new)[0] == []
    assert check_frozen(prev, stage, {}, {"puu.general": True, "fan.general": False})[0] == []


def test_frozen_slice_with_no_tracked_files_is_unverifiable(tmp_path):
    live, stage, prev = _setup(tmp_path)
    prev["frozen_slices"]["mye.general"] = True                            # nothing tracked for it
    problems, _ = check_frozen(prev, stage, {})
    assert any("mye.general: unverifiable" in p for p in problems)
    problems, log = check_frozen({**prev, "frozen_slices": {"mye.general": True}}, stage, {"mye.general": "new"})
    assert problems == [] and log[0]["note"] == "unverifiable"


def test_unfreeze_allow_added_fields(tmp_path):
    live, stage, prev = _setup(tmp_path)
    _write(live, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1, "ref": "a"}\n')
    prev = _manifest(live, {"puu.general": True})
    _write(stage, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1, "ref": "a", "orthography": "x"}\n')
    ok, log = check_frozen(prev, stage, {"puu.general": "add field"}, live=live, allow_added_fields=["orthography"])
    assert ok == [] and "only added fields" in log[0]["verified"]
    _write(stage, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1, "ref": "B", "orthography": "x"}\n')
    bad, _ = check_frozen(prev, stage, {"puu.general": "add field"}, live=live, allow_added_fields=["orthography"])
    assert bad and "existing field values changed" in bad[0]


def test_removed_field_transition_only_when_release_is_the_sole_difference(tmp_path):
    live, stage, prev = _setup(tmp_path)
    _write(live, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1, "ref": "a", "release": true}\n')
    prev = _manifest(live, {"puu.general": True})
    _write(stage, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1, "ref": "a"}\n')
    ok, log = check_frozen(prev, stage, {"puu.general": "move release"}, live=live, allow_removed_fields=["release"])
    assert ok == [] and "removed fields ['release']" in log[0]["verified"]
    _write(stage, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1, "ref": "CHANGED"}\n')
    bad, _ = check_frozen(prev, stage, {"puu.general": "move release"}, live=live, allow_removed_fields=["release"])
    assert bad and "existing field values changed" in bad[0]
    _write(stage, "eval/puu.mt_fr2x.general.test.jsonl", '{"id": 1}\n')            # removed more than release
    bad, _ = check_frozen(prev, stage, {"puu.general": "move release"}, live=live, allow_removed_fields=["release"])
    assert bad and "fields changed beyond" in bad[0]
