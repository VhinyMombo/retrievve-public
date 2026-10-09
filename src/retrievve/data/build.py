"""Build data/v1 from raw sources (DATA_SPEC v0.8).

    retrievve build-data [--config configs/v1_pipeline.yaml] [--out data/v1]

Steps: registry -> parse -> normalize -> canonical records -> filters -> splits -> dedup (exact, cross-split,
within-train) -> corpus/splits -> SFT/LM/RAG/eval -> reports + manifest.
Missing (Dropbox online-only) sources are skipped and listed; the build is then marked frozen:false.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import derive, lexprobe, quality
from .common import CONFIG_PATH, ROOT, load_config, sha256_file, write_json, write_jsonl
from .normalize import Normalizer, text_fold, text_key
from .parsers import PARSERS
from .sources import availability, sync_registry
from .split import assign_splits, chapter_of
from .versification import remap_rows

_PARENS = re.compile(r"^(.*\S)\s*\(([^()]+)\)\s*[.!]?\s*$")

DERIVED_DIRS = ("corpus", "splits", "lm", "sft", "rag", "eval", "mlx")

RECORD_KEYS = ["id", "lang", "text", "text_raw", "fr", "kind", "domain", "source_id", "group_id", "order",
               "orthography", "align", "ocr", "flags", "split", "release", "origin", "verified_by",
               "text_key", "text_fold", "rag_marker"]


def _code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def build_records(sources: dict, avail: dict, cfg: dict, norm: Normalizer) -> tuple[list[dict], dict]:
    aliases = {k.lower(): v for k, v in cfg["lang_aliases"].items()}
    known = set(cfg["languages"]["train_eval"]) | set(cfg["languages"]["eval_only"])
    records, per_source, seen_ids = [], {}, Counter()
    proper_nouns = {text_key(w) for w in cfg["normalize"].get("headword_proper_nouns") or []}
    for sid, e in sources.items():
        if sid in (cfg.get("sources_disabled") or {}):
            per_source[sid] = {"status": "disabled by config", "reason": cfg["sources_disabled"][sid]}
            continue
        if not e.get("enabled", True):
            per_source[sid] = {"status": "disabled"}
            continue
        if not avail.get(sid):
            per_source[sid] = {"status": "missing (online-only placeholder)", "path": e["path"]}
            continue
        parser = PARSERS.get(e.get("parser"))
        if parser is None:
            per_source[sid] = {"status": f"unsupported parser {e.get('parser')}"}
            continue
        try:
            rows = parser(sid, e, aliases)
            vers_report = None
            if e.get("versification") == "hebrew":
                scope = set(e["versification_scope"]) if e.get("versification_scope") else None
                rows, vers_report = remap_rows(rows, scope)
        except Exception as ex:                                    # keep building; report the failure
            per_source[sid] = {"status": f"parse error: {type(ex).__name__}: {ex}"}
            continue
        gate_report = None
        if e.get("accept_books_key"):                              # per-book acceptance gate (NT realignment)
            allowed = set(cfg.get(e["accept_books_key"]) or [])
            books = Counter(r["group"].split("/", 1)[1] for r in rows if r["group"].startswith("bible/"))
            rows = [r for r in rows if r["group"].split("/", 1)[-1] in allowed]
            gate_report = {"accepted": sorted(b for b in books if b in allowed),
                           "excluded": sorted(b for b in books if b not in allowed)}
        n_bad_lang = 0
        for row in rows:
            lang = row["lang"]
            if lang not in known:
                n_bad_lang += 1
                continue
            verse = row["kind"] == "verse"
            text = norm.target(row["text_raw"], lang, verse=verse) or ""
            if e.get("headword_case") == "upper" and row["kind"] == "lexicon" and text.isupper() \
                    and text_key(text) not in proper_nouns:
                text = text.lower()                                # Unicode-aware; text_raw keeps the caps
            fr = norm.french(row["fr_raw"], verse=verse)
            fr_alt = fr_note = None
            mode = e.get("fr_parens")                              # declared per source (spec v0.4), never guessed
            if mode and fr:
                m = _PARENS.match(fr)
                if m:
                    outside, inside = m.group(1).strip(), m.group(2).strip()
                    if mode == "literal":
                        fr, fr_alt = inside, outside
                    elif mode == "note":
                        fr, fr_note = outside, inside
            rid = f"{lang}:{sid}:{row['locator']}"
            seen_ids[rid] += 1
            if seen_ids[rid] > 1:                                  # duplicate locator inside a source
                rid = f"{rid}#{seen_ids[rid]}"
            rec = {
                "id": rid, "lang": lang, "text": text, "text_raw": row["text_raw"], "fr": fr,
                "kind": row["kind"], "domain": e["domain"], "source_id": sid, "group_id": row["group"],
                "order": row["order"], "orthography": e.get("orthography", ""), "align": row["align"],
                "ocr": bool(e.get("ocr", False)), "flags": [], "split": None, "release": bool(e["release"]),
                "text_key": text_key(text), "text_fold": text_fold(text),
                "origin": e.get("record_origin", "human"), "verified_by": None,      # v0.5
            }
            if e.get("ocr") and not e.get("ocr_review_lifted"):
                rec["flags"].append("ocr_review")                 # §4.6: blocks SFT/LM/eval, RAG allowed
                if e.get("rag_marker"):
                    rec["rag_marker"] = e["rag_marker"]
            if row.get("lex"):
                lex = dict(row["lex"])
                lex["headword"] = text
                lex["gloss_fr"] = fr
                if lex.get("plural"):
                    lex["plural"] = norm.target(lex["plural"], lang)
                rec["lex"] = lex
            if fr_alt:
                rec["fr_alt"] = fr_alt
            if fr_note:
                rec["fr_note"] = fr_note
            if row.get("extra"):
                rec["extra"] = row["extra"]
            records.append(rec)
        per_source[sid] = {"status": "ok", "records": len(rows) - n_bad_lang,
                           **({"dropped_unknown_lang": n_bad_lang} if n_bad_lang else {}),
                           **({"versification": vers_report} if vers_report else {}),
                           **({"book_gate": gate_report} if gate_report else {})}
    records.sort(key=lambda r: (r["lang"], r["source_id"], r["order"], r["id"]))
    link_lexicon_examples(records)
    return records, per_source


def link_lexicon_examples(records: list[dict]) -> int:
    """Example records carrying extra.headword (Galley) are linked into the matching headword's lex.examples
    (same language; matched on text_raw, i.e. the dictionary's own spelling)."""
    heads = defaultdict(list)
    for r in records:
        if r["kind"] == "lexicon" and r.get("lex"):
            heads[(r["lang"], r["text_raw"].strip())].append(r)
    n = 0
    for r in records:
        hw = (r.get("extra") or {}).get("headword")
        if r["kind"] == "example" and hw:
            for h in heads.get((r["lang"], hw.strip()), []):
                h["lex"]["examples"].append(r["id"])
                n += 1
    return n


def missing_sources(per_source: dict) -> dict:
    return {s: v["path"] for s, v in per_source.items() if v.get("status", "").startswith("missing")}


def quality_pass(records: list[dict], cfg: dict) -> dict:
    """Filters + dedup on split-assigned records, in the order the build uses (tested as a unit)."""
    fcfg, scfg = cfg["filters"], cfg["splits"]
    cut = quality.align_cutoff(records, fcfg)
    lex_tok = quality.lexicon_tokens(records)
    stopwords, exempted = quality.per_lang_stopwords(records, fcfg, lex_tok)
    vocab = quality.french_vocab(fcfg, ROOT)
    vocab_by_lang = {lang: vocab - toks for lang, toks in lex_tok.items()}   # same exemptions (v0.6)
    for r in records:
        lang = r["lang"]
        quality.flag_record(r, fcfg, cut, stopwords.get(lang, set()), vocab_by_lang.get(lang, vocab))
    low_ch = {(lang, ch) for lang, chs in (fcfg.get("low_align_chapters") or {}).items() for ch in chs}
    for r in records:
        if low_ch and r["group_id"].startswith("bible/"):
            c = chapter_of(r)
            book = r["group_id"].split("/")[1]
            if (r["lang"], f"{book}.{c}") in low_ch and "low_align" not in r["flags"]:
                r["flags"].append("low_align")
    n_exact = quality.exact_dups(records)
    mh = quality.MinHasher(scfg["shingle_k"], scfg["minhash_perms"], scfg["minhash_bands"], cfg["seed"])
    xsplit = quality.cross_split_dedup(records, mh, scfg["near_dup_jaccard"], fcfg)
    clusters = quality.train_near_dups(records, mh, scfg["near_dup_jaccard"], set(fcfg["keep_near_dup_kinds"]))
    blind = quality.fr_detector_blind_spot(records, vocab_by_lang, lex_tok, fcfg["fr_vocab_min_token_len"])
    return {"align_cutoff": cut, "stopwords_exempted": exempted, "fr_vocab_size": len(vocab),
            "fr_detector_blind_spot": blind,
            "exact_dups": n_exact, "cross_split": xsplit, "near_dup_clusters": clusters}


def build(out: Path, cfg: dict, args) -> dict:
    """Run the whole pipeline into `out` (a fresh staging dir). Writes everything except manifest.json."""
    raw = ROOT / cfg["paths"]["raw"]

    sources = sync_registry(out / "sources.yaml", raw)
    avail = availability(sources)
    norm = Normalizer(cfg)
    records, per_source = build_records(sources, avail, cfg, norm)

    split_report = assign_splits(records, cfg)              # splits don't depend on flags; stopwords need splits
    qrep = quality_pass(records, cfg)
    cut, stop_removed, n_exact, xsplit, clusters = (qrep[k] for k in (
        "align_cutoff", "stopwords_exempted", "exact_dups", "cross_split", "near_dup_clusters"))

    langs = sorted({r["lang"] for r in records})
    by_lang = defaultdict(list)
    for r in records:
        by_lang[r["lang"]].append({k: r[k] for k in RECORD_KEYS if k in r} |
                                  {k: r[k] for k in ("fr_alt", "fr_note", "lex", "extra") if k in r})
    for lang in langs:
        write_jsonl(out / "corpus" / f"{lang}.jsonl", by_lang[lang])
        for s in ("train", "val", "test", "rag_pool"):
            ids = [r["id"] for r in by_lang[lang] if r["split"] == s and quality.usable(r)]
            if ids:
                p = out / "splits" / f"{lang}.{s}.ids"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("\n".join(ids) + "\n", encoding="utf-8")

    # ---- derived artifacts
    counter = derive.TokenCounter(cfg)
    if counter.approx and not args.allow_approx_tokens:
        raise SystemExit(f"token counter unavailable ({counter.desc}); SFT exclusion and LM packing would use an "
                         "approximate chars/token ratio. Install the tokenizer or pass --allow-approx-tokens.")
    budget, min_tok = derive.lm_budget(cfg)
    max_seq = int(cfg["lm"]["max_seq_length"])
    sft_report, lm_report, rag_report, eval_report = {}, {}, {}, {}
    mlx_release = defaultdict(dict)                 # lang -> {mlx file: counts incl. release_false}
    # lexicon probe (v0.7 §7b): frozen headword sample, held out of every SFT lexicon task, kept in RAG
    probe_keys, probe_report = {}, {}
    for lang, n in (cfg.get("lexprobe") or {}).get("sizes", {}).items():
        if not n or lang not in by_lang:
            continue
        ids_file = out / "splits" / f"{lang}.lexprobe.ids"
        keys = lexprobe.select(records, lang, int(n), cfg["seed"], ids_file)
        ids_file.parent.mkdir(parents=True, exist_ok=True)
        ids_file.write_text("\n".join(keys) + "\n", encoding="utf-8")
        probe_keys[lang] = set(keys)
        train_texts = [r["text"] for r in records if r["lang"] == lang and r["split"] == "train" and r["fr"]
                       and r["kind"] in derive.PARALLEL_KINDS and quality.usable(r)]
        prows = lexprobe.rows(records, lang, keys, train_texts)
        write_jsonl(out / "eval" / f"{lang}.lexprobe.jsonl", prows)
        probe_report[lang] = {"headwords": len(prows), "in_sft_sentences": sum(r["in_sft_sentences"] for r in prows),
                              "definition_gloss": sum(r["definition_gloss"] for r in prows),
                              "pos": dict(Counter(r["pos"] for r in prows))}
    for lang in cfg["languages"]["train_eval"]:
        if lang not in by_lang:
            continue
        for split in ("train", "val"):
            ex = derive.sft_examples(records, lang, split, cfg, exclude_lex_keys=probe_keys.get(lang))
            mix = None
            if split == "train":
                ex, mix = derive.apply_domain_cap(ex, cfg["sft"]["religious_cap"], cfg["sft"]["max_upsample"],
                                                 set(cfg["sft"].get("upsample_domains", [])))
            too_long, kept = [], []
            over = defaultdict(Counter)                          # task -> {seq_len: n_over}, plus total
            for e in ex:
                n = counter.chat(e["messages"])
                (too_long if n > max_seq else kept).append((e, n))
                t = e["meta"]["task"]
                over[t]["total"] += 1
                for L in cfg["lm"].get("report_seq_lengths", [512, 1024]):
                    over[t][f"over_{L}"] += n > L
            write_jsonl(out / "sft" / f"{lang}.{split}.jsonl", (e for e, _ in kept))
            mlx_name = f"{'valid' if split == 'val' else 'train'}.jsonl"
            write_jsonl(out / "mlx" / "sft" / lang / mlx_name, ({"messages": e["messages"]} for e, _ in kept))
            mlx_release[lang][f"sft/{lang}/{mlx_name}"] = {
                "examples": len(kept), "release_false": sum(not e["meta"]["release"] for e, _ in kept)}
            sft_report[f"{lang}:{split}"] = {
                "examples": len(kept), "mix": derive.mix_report([e for e, _ in kept]), "domain_cap": mix,
                "lex_fr2x_merged_glosses": sum(1 for e, _ in kept if e["meta"].get("merged_headwords")),
                "excluded_over_max_seq": [{"src_id": e["meta"]["src_id"], "task": e["meta"]["task"],
                                           "tokens": n} for e, n in too_long],
                "max_tokens": max((n for _, n in kept), default=0),
                "over_length_rate": {t: {k: (round(v / c["total"], 4) if k != "total" else v) for k, v in c.items()}
                                     for t, c in over.items()}}
            chunks = derive.lm_chunks(records, lang, split, counter, budget, min_tok)
            write_jsonl(out / "lm" / f"{lang}.{split}.jsonl", ({"text": c["text"]} for c in chunks))
            write_jsonl(out / "mlx" / "lm" / lang / mlx_name, ({"text": c["text"]} for c in chunks))
            mlx_release[lang][f"lm/{lang}/{mlx_name}"] = {
                "chunks": len(chunks), "release_false": sum(not c["release"] for c in chunks)}
            toks = [c["n_tokens"] for c in chunks]
            lm_report[f"{lang}:{split}"] = {"chunks": len(chunks), "tokens": sum(toks),
                                            "below_min": sum(t < min_tok for t in toks),
                                            "max": max(toks, default=0)}
    rec_by_id = {r["id"]: r for r in records}

    def release_info(src_ids: list[str]) -> dict:
        recs = [rec_by_id[i] for i in src_ids if i in rec_by_id]
        srcs = sorted({r["source_id"] for r in recs})
        return {"source_ids": srcs, "release": all(r["release"] for r in recs),
                "license": sorted({str(sources.get(s, {}).get("license", "unknown")) for s in srcs})}

    for lang in langs:
        chunks = derive.rag_chunks(records, lang, cfg)
        write_jsonl(out / "rag" / f"{lang}.jsonl", chunks)
        rag_report[lang] = dict(Counter(c["type"] for c in chunks))
        sidecar = {}
        for name, rows in derive.eval_sets(records, lang).items():
            write_jsonl(out / "eval" / f"{lang}.{name}.test.jsonl", rows)
            eval_report[f"{lang}.{name}"] = len(rows)
            for row in rows:
                sidecar[row["id"]] = release_info([row["id"]])
        probe_p = out / "eval" / f"{lang}.lexprobe.jsonl"
        if lang in probe_report and probe_p.exists():
            for line in probe_p.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                sidecar[row["id"]] = release_info(row["src_ids"])
        if sidecar:
            # v0.9: licence/provenance is a property of the SOURCE and may change; it never lives inside frozen
            # test files. Regenerated every build, not part of any frozen slice.
            write_json(out / "eval" / f"RELEASE.{lang}.json", dict(sorted(sidecar.items())))

    # mlx exports drop `meta`; RELEASE.<lang>.json keeps provenance visible next to them (DATA_SPEC §10).
    for lang, files in mlx_release.items():
        internal = any(v["release_false"] for v in files.values())
        write_json(out / "mlx" / f"RELEASE.{lang}.json", {
            "lang": lang, "files": files, "releasable": not internal,
            "note": "Contains release:false sources: internal use only (private storage); adapters trained on "
                    "these files are internal-only." if internal else "All examples come from release:true sources."})

    # ---- per-slice freeze (v0.6 §3): lang x {general, religious, lexprobe}
    base_ok = not missing_sources(per_source) and not counter.approx and bool(cfg.get("thresholds_final"))
    gates = cfg.get("religious_gate_passed") or {}
    eval_only = set(cfg["languages"]["eval_only"])
    min_pairs = int((cfg.get("eval") or {}).get("descriptive_only_below", 50))
    slices, frozen_slices = {}, {}
    for key, n in eval_report.items():
        lang, _task, dom = key.split(".")
        name = f"{lang}.{dom}"
        sl = slices.setdefault(name, {"pairs_per_direction": n, "files": []})
        sl["pairs_per_direction"] = max(sl["pairs_per_direction"], n)
        sl["files"] = sorted({*sl["files"], f"eval/{key}.test.jsonl"})
    for name, sl in slices.items():
        lang, dom = name.split(".")
        ok = base_ok and (dom != "religious" or bool(gates.get(lang)))
        sl["descriptive_only"] = lang in eval_only and sl["pairs_per_direction"] < min_pairs
        sl["gate"] = "align_check (88) >= 0.8 per book/tail" if dom == "religious" else "thresholds_final"
        frozen_slices[name] = ok
    for lang, passed in gates.items():                   # a gated slice with no data yet is still pending
        name = f"{lang}.religious"
        if name not in slices and not passed:
            slices[name] = {"pairs_per_direction": 0, "descriptive_only": False,
                            "gate": "align_check (88) >= 0.8 per book/tail", "note": "no eval data yet (pending)"}
            frozen_slices[name] = False
    for lang, rep in probe_report.items():
        p = out / "eval" / f"{lang}.lexprobe.jsonl"
        strata = "none (no POS)" if set(rep["pos"]) == {"unknown"} else "pos"
        slices[f"{lang}.lexprobe"] = {"headwords": rep["headwords"], "sha256": sha256_file(p), "strata": strata,
                                      "ids_file": f"splits/{lang}.lexprobe.ids",
                                      "files": [f"eval/{lang}.lexprobe.jsonl", f"splits/{lang}.lexprobe.ids"]}
        frozen_slices[f"{lang}.lexprobe"] = base_ok

    # ---- reports
    missing = missing_sources(per_source)
    flags = Counter(f"{r['lang']}:{f}" for r in records for f in r["flags"])
    build = {
        "frozen": bool(frozen_slices) and all(frozen_slices.values()),
        "frozen_slices": frozen_slices, "slices": slices,
        "frozen_note": "; ".join(n for n in (
            "test split may change: sources missing" if missing else "",
            "approximate token counts" if counter.approx else "",
            "filter thresholds not final (DATA_SPEC §8 TBD-stats)" if not cfg.get("thresholds_final") else "",
            "religious slices pending (align_check gate / realignment)" if any(
                not v for k, v in frozen_slices.items() if k.endswith(".religious")) else "")
            if n) or "all slices frozen",
        "lexprobe": probe_report,
        "token_counter": counter.desc, "lm_budget_tokens": budget, "max_seq_length": max_seq,
        "align_cutoff": cut, "exact_dups": n_exact, "cross_split_dedup": xsplit,
        "fr_stopwords_exempted_per_lang": stop_removed, "fr_vocab_size": qrep["fr_vocab_size"],
        "fr_detector_blind_spot": qrep["fr_detector_blind_spot"],
        "records_by_lang_kind_domain": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in sorted(
            Counter((r["lang"], r["kind"], r["domain"]) for r in records).items())},
        "parallel_vs_mono": {f"{k[0]}:{k[1]}": v for k, v in sorted(
            Counter((r["lang"], "parallel" if r["fr"] else "mono") for r in records).items())},
        "flags": dict(sorted(flags.items())), "splits": split_report,
        "sft": sft_report, "lm": lm_report, "rag": rag_report, "eval": eval_report,
        "sources": per_source, "missing_sources": missing,
        "release_false_records": sum(not r["release"] for r in records),
        "release_true_sources": sorted(sid for sid, e in sources.items() if e.get("release")
                                       and e.get("enabled", True) and sid not in (cfg.get("sources_disabled") or {})),
        "release_true_sources_all": sorted(sid for sid, e in sources.items() if e.get("release")),
    }
    write_json(out / "reports" / "build_report.json", build)
    write_json(out / "reports" / "normalization.json", norm.report())
    write_json(out / "reports" / "near_dup_clusters.json", clusters[:50])

    print(f"records={len(records)} langs={langs} missing_sources={len(missing)} frozen={build['frozen']}")
    print(f"release:true sources ({len(build['release_true_sources_all'])}): {build['release_true_sources_all']}")
    print(f"token_counter={counter.desc}")
    for k, v in split_report["records"].items():
        print(f"  {k}: {v}")
    return {"build": build, "avail": avail}


# ------------------------------------------------------------------------------------------- freeze guard
def slice_files(name: str, prev_slices: dict) -> list[str]:
    """Files of a slice: from the manifest when recorded, else by naming convention (older manifests)."""
    files = (prev_slices.get(name) or {}).get("files")
    if files:
        return list(files)
    lang, dom = name.split(".", 1)
    if dom == "lexprobe":
        return [f"eval/{lang}.lexprobe.jsonl", f"splits/{lang}.lexprobe.ids"]
    return [f"eval/{lang}.mt_fr2x.{dom}.test.jsonl", f"eval/{lang}.mt_x2fr.{dom}.test.jsonl"]


def verify_field_changes(live: Path | None, stage: Path, changes: list[dict], added: set[str],
                         removed: set[str]) -> list[str]:
    """For an unfreeze restricted to schema transitions: every changed file must be a JSONL whose rows (same count,
    same order) differ from the live file only by ADDING fields in `added` and/or REMOVING fields in `removed`;
    every other field must be byte-identical. Any other changed file (e.g. an ids file) is an error."""
    errs = []
    for c in changes:
        rel = c["file"]
        old_p, new_p = (live / rel) if live else None, stage / rel
        if not rel.endswith(".jsonl") or old_p is None or not old_p.exists() or not new_p.exists():
            errs.append(f"{rel}: changed but not verifiable as a field-only change")
            continue
        old_rows = [json.loads(x) for x in old_p.read_text(encoding="utf-8").splitlines() if x.strip()]
        new_rows = [json.loads(x) for x in new_p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(old_rows) != len(new_rows):
            errs.append(f"{rel}: row count {len(old_rows)} -> {len(new_rows)}")
            continue
        for i, (o, n) in enumerate(zip(old_rows, new_rows, strict=True)):
            if (set(n) - set(o)) - added or (set(o) - set(n)) - removed:
                errs.append(f"{rel}: row {i}: fields changed beyond added={sorted(added)} removed={sorted(removed)}")
                break
            common_o = {k: v for k, v in o.items() if k in n}
            common_n = {k: v for k, v in n.items() if k in o}
            if json.dumps(common_o, sort_keys=True, ensure_ascii=False) != json.dumps(common_n, sort_keys=True,
                                                                                     ensure_ascii=False):
                errs.append(f"{rel}: row {i}: existing field values changed")
                break
    return errs


def check_frozen(prev: dict | None, stage: Path, unfreeze: dict[str, str],
                 new_frozen: dict[str, bool] | None = None, live: Path | None = None,
                 allow_added_fields: list[str] | None = None,
                 allow_removed_fields: list[str] | None = None) -> tuple[list[str], list[dict]]:
    """DATA_SPEC v0.8 §3 freeze guard. For every slice frozen in the PREVIOUS manifest:
    - each of its files must exist in the new build with the same sha256, and
    - it must still be frozen in the new build (no silent de-freezing through a config flip),
    unless the slice is explicitly unfrozen (with a reason). Returns (problems, unfreeze log entries)."""
    if not prev:
        return [], []
    problems, log = [], []
    if new_frozen is not None:
        for name, frozen in sorted((prev.get("frozen_slices") or {}).items()):
            if frozen and name not in unfreeze and new_frozen.get(name) is not True:
                state = "would disappear" if name not in new_frozen else "would be de-frozen"
                problems.append(f"frozen slice {name}: {state} (frozen_slices) without --unfreeze")
    prev_files = prev.get("files") or {}
    for name, frozen in sorted((prev.get("frozen_slices") or {}).items()):
        if not frozen:
            continue
        changes, compared = [], 0
        for rel in slice_files(name, prev.get("slices") or {}):
            old = (prev_files.get(rel) or {}).get("sha256")
            if old is None:
                continue
            compared += 1
            new_p = stage / rel
            new = sha256_file(new_p) if new_p.exists() else None
            if new != old:
                changes.append({"file": rel, "old_sha256": old, "new_sha256": new})
        if compared == 0:                                    # F2: frozen but checked against nothing
            if name not in unfreeze:
                problems.append(f"frozen slice {name}: unverifiable (no tracked files in the previous manifest)")
            else:
                log.append({"slice": name, "reason": unfreeze[name], "changes": [], "note": "unverifiable",
                            "at": dt.datetime.now().isoformat(timespec="seconds")})
            continue
        if not changes:
            continue
        if name in unfreeze:
            entry = {"slice": name, "reason": unfreeze[name], "changes": changes,
                     "at": dt.datetime.now().isoformat(timespec="seconds")}
            if allow_added_fields or allow_removed_fields:
                add, rem = set(allow_added_fields or []), set(allow_removed_fields or [])
                errs = verify_field_changes(live, stage, changes, add, rem)
                if errs:
                    problems.extend(f"frozen slice {name}: {e}" for e in errs)
                    continue
                entry["verified"] = (f"only added fields {sorted(add)} / removed fields {sorted(rem)}; "
                                     "all other fields byte-identical")
            log.append(entry)
        else:
            for c in changes:
                what = "would be deleted" if c["new_sha256"] is None else "would change"
                problems.append(f"frozen slice {name}: {c['file']} {what}")
    return problems, log


def protected_files(prev: dict | None, unfreeze: dict[str, str]) -> set[str]:
    if not prev:
        return set()
    return {rel for name, fr in (prev.get("frozen_slices") or {}).items() if fr and name not in unfreeze
            for rel in slice_files(name, prev.get("slices") or {})}


def promote(stage: Path, live: Path, protected: set[str]) -> list[str]:
    """Move the staged build into `live`; delete stale derived files, never a protected (frozen) one."""
    staged = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    for rel in sorted(staged):
        dest = live / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage / rel, dest)
    removed = []
    for sub in DERIVED_DIRS:
        for p in sorted((live / sub).rglob("*")) if (live / sub).exists() else []:
            rel = p.relative_to(live).as_posix()
            if p.is_file() and rel not in staged and rel not in protected:
                p.unlink()
                removed.append(rel)
    shutil.rmtree(stage, ignore_errors=True)
    return removed


def write_manifest(live: Path, cfg: dict, args, info: dict, prev: dict | None, log: list[dict]) -> None:
    build_ = info["build"]
    files = {}
    for p in sorted(live.rglob("*")):
        if p.is_file() and p.name != "manifest.json" and p.suffix in (".jsonl", ".ids", ".yaml", ".json"):
            rel = p.relative_to(live).as_posix()
            n = sum(1 for _ in open(p, encoding="utf-8")) if p.suffix in (".jsonl", ".ids") else None
            files[rel] = {"sha256": sha256_file(p), **({"lines": n} if n is not None else {})}
    write_json(live / "manifest.json", {
        "spec_version": cfg["spec_version"], "build_date": dt.datetime.now().isoformat(timespec="seconds"),
        "frozen": build_["frozen"], "frozen_slices": build_["frozen_slices"], "slices": build_["slices"],
        "frozen_note": build_["frozen_note"], "religious_gate_passed": cfg.get("religious_gate_passed") or {},
        "unfreeze_log": ((prev or {}).get("unfreeze_log") or []) + log,
        "release_sidecars": {k: v["sha256"] for k, v in files.items() if k.startswith("eval/RELEASE.")},
        "code_sha256": _code_hash(), "config_sha256": sha256_file(args.config or CONFIG_PATH),
        "sources_available": sum(info["avail"].values()), "sources_total": len(info["avail"]), "files": files})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-approx-tokens", action="store_true",
                    help="build even if the base tokenizer is unavailable (chars/token estimate; frozen:false)")
    ap.add_argument("--unfreeze", action="append", default=[], metavar="SLICE",
                    help="allow a frozen slice to change (repeatable; requires --reason; logged in the manifest)")
    ap.add_argument("--reason", default=None, help="why the --unfreeze slices may change")
    ap.add_argument("--unfreeze-allow-added-fields", action="append", default=[], metavar="FIELD",
                    help="restrict --unfreeze to adding these JSONL fields: every other field must stay byte-identical")
    ap.add_argument("--unfreeze-allow-removed-fields", action="append", default=[], metavar="FIELD",
                    help="restrict --unfreeze to removing these JSONL fields: every other field must stay byte-identical")
    args = ap.parse_args(argv)
    if args.unfreeze and not (args.reason and args.reason.strip()):
        ap.error("--unfreeze requires --reason")
    if (args.unfreeze_allow_added_fields or args.unfreeze_allow_removed_fields) and not args.unfreeze:
        ap.error("--unfreeze-allow-added/removed-fields only apply together with --unfreeze")
    unfreeze = {s: args.reason.strip() for s in args.unfreeze}
    cfg = load_config(args.config)
    live = (args.out or ROOT / cfg["paths"]["out"]).resolve()
    marker = live / ".promoting"
    if marker.exists():
        raise SystemExit(f"{marker} exists: a previous build was interrupted while promoting, so {live} may be a mix "
                         "of two builds. Inspect it (manifest.json is the older one), then delete the marker.")
    prev_path = live / "manifest.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else None

    # Build into a staging dir; the live dir is only touched if the freeze guard passes.
    stage = live.with_name(live.name + ".staging")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for rel in ["sources.yaml", *[p.relative_to(live).as_posix() for p in (live / "splits").glob("*.lexprobe.ids")]] \
            if live.exists() else []:
        if (live / rel).exists():                          # human-edited registry + frozen probe ids carry over
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live / rel, stage / rel)
    try:
        info = build(stage, cfg, args)
        problems, log = check_frozen(prev, stage, unfreeze, info["build"]["frozen_slices"], live,
                                     args.unfreeze_allow_added_fields, args.unfreeze_allow_removed_fields)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if problems:
        shutil.rmtree(stage, ignore_errors=True)
        msg = "\n  ".join(problems)
        raise SystemExit(f"FREEZE GUARD: aborting, live {live} left untouched.\n  {msg}\n"
                         "Pass --unfreeze <slice> --reason '...' to change a frozen slice deliberately.")
    live.mkdir(parents=True, exist_ok=True)
    marker.write_text(dt.datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")
    removed = promote(stage, live, protected_files(prev, unfreeze))
    if removed:
        print(f"removed {len(removed)} stale file(s): {removed[:10]}")
    for entry in log:
        print(f"UNFROZEN {entry['slice']}: {entry['reason']} ({len(entry['changes'])} file(s) changed)")
    write_manifest(live, cfg, args, info, prev, log)
    marker.unlink()


if __name__ == "__main__":
    main()
