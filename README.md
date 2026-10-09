# retrievve

A language model that can **converse in**, **translate** and help **preserve** Gabonese languages,
starting with **Fang** and **Punu (Yipunu)**.

These languages have very little digital text, so we don't train from scratch. We build a clean, verified
parallel corpus, adapt a French-strong multilingual model to it, and evaluate with native speakers. The corpus
itself is a goal: it is meant to be **published as a preservation artifact**, limited to the sources whose
`release` flag allows it (DATA_SPEC §10).

| language | ISO 639-3 | role |
|---|---|---|
| Fang | `fan` | train + eval |
| Punu (yipunu) | `puu` | train + eval |
| Nzebi | `nzb` | eval only (RAG + few-shot) |
| Myene | `mye` | eval only |
| Lembaama | `lev` | eval only |

French is the pivot language. Eval-only languages get fine-tuned once they reach about 2k parallel
sentences (DATA_SPEC §0).

## What it does

1. **Translate** French ↔ Fang/Punu, with a LoRA adapter on an instruction-tuned LLM (v0: `Qwen2.5-7B-Instruct-4bit`
   with `mlx-lm` on Apple silicon). Training has two stages: continued pretraining on target-language text, then
   SFT on translation, lexicon, grammar and dialogue tasks.
2. **Converse**, through a grounded assistant (RAG) that answers lexicon and grammar questions from an index of
   headwords, grammar chunks and example sentence pairs. Answers stay tied to sources.
3. **Preserve**: a documented, leak-free, versioned corpus, publishable for the sources whose `release` flag allows it.

## Status

| version | state |
|---|---|
| v0 | Corpus in `data/corpus`, training sets in `data/train`, LoRA adapters in `models/`. All built before the data contract existed, and kept for comparison. Not in git. The original v0 README is in [`legacy/v0/README.md`](legacy/v0/README.md). v0 roadmap: [`PLAN.md`](PLAN.md). |
| v1 | Pipeline under construction against [`DATA_SPEC.md`](DATA_SPEC.md): normalization, quality filters and the source registry are in `src/v1/`; the build step, splits and training artifacts are in progress. |

## How the v1 data pipeline works

All rules and thresholds are in [`DATA_SPEC.md`](DATA_SPEC.md) (the data contract), and their values in
[`configs/v1_pipeline.yaml`](configs/v1_pipeline.yaml).

```
data/raw  ─►  parse  ─►  normalize  ─►  quality filters  ─►  grouped split  ─►  derived artifacts
             (§1)         (§2)            (§4)                (§3)               lm/ sft/ rag/ eval/
```

- **One canonical record per segment** (`data/v1/corpus/<lang>.jsonl`). Every LM, SFT, RAG and eval file is
  derived from these records only.
- **Normalization never strips tone marks or diacritics.** Unicode NFC, look-alike letters unified (ε→ɛ, ↄ→ɔ),
  one canonical apostrophe per language. The raw text is kept in `text_raw`.
- **Leak-proof splits.** Split by document or Bible book, never by line. Test sets are frozen, excluded
  from the RAG index, and deduplicated against train (exact and MinHash near-duplicates). Synoptic gospels
  are never held out, since their near-parallel verses would leak.
- **Domain balance.** The corpus is Bible-heavy, so religious examples are capped at 50% of SFT, and
  religious and non-religious test scores are reported separately.
- **Provenance.** Every source has an entry in `data/v1/sources.yaml`. Sources marked `release: false`
  (DATA_SPEC §10) may be used for internal experiments only, never in a released dataset, adapter or paper example.

## Repository layout

[`CONTRIBUTING.md`](CONTRIBUTING.md) lists who owns what.

```
DATA_SPEC.md          data contract
configs/              v1_pipeline.yaml (pipeline parameters), train/ (training configs)
src/retrievve/        installable package: data/ (v1 pipeline), analysis/, rag/, train/, eval/, api/, cli.py
tests/                pytest
data/                 not in the public repo (sources are not redistributable; see DATA_SPEC §10)
models/               trained adapters (not in the public repo)
```

## Getting started

Python ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/). **One virtualenv per worktree**, at
`<worktree>/.venv` (the default for `uv sync`): the package is installed in editable mode, so a shared venv
would run whichever checkout it was synced from. Never create a venv in the Dropbox checkout (Dropbox turns
files into online-only placeholders, which breaks installed packages).

```sh
cd ~/code/retrievve-wt/<role>          # your worktree (see CONTRIBUTING.md), outside Dropbox
make setup                             # = uv sync: base deps + dev tools, adds .venv/bin/retrievve (~80 MB)
uv sync --extra mlx                    # + mlx / mlx-lm for training on Apple silicon
make test lint                         # pytest + ruff
make data                              # rebuild the shared Dropbox data/v1 (freeze-guarded); DATA_OUT=... for scratch
```

The data lives in the shared Dropbox folder, not in git. To work on your own copy, link it in as described in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Make the Dropbox folder **available offline** first: online-only files
read as 0 bytes, and the scripts list them as `unhydrated` and skip them.

Examples, run from the repo root:

```sh
retrievve stats                                                    # stats report -> data/v1/reports/
retrievve token-lengths corpus data/corpus/fang*.jsonl              # chars/token per field and orthography
retrievve token-lengths sft data/train/fang-sft.train.jsonl         # SFT lengths vs max_seq_length
```

## Things worth knowing

- **Token budget.** On the Qwen2.5 tokenizer, Fang and Punu use about 1.5–1.7× more tokens than French for
  the same content (early estimate, n small: being measured on the full corpus). The letters ɛ, ɔ and ŋ cost 2 tokens each and a combining tone mark adds 1. Size chunks
  and `max_seq_length` in **tokens**, never in characters.
- **Evaluation.** chrF++ on frozen test sets, compared between runs with paired bootstrap resampling. With
  about 500 test pairs, differences under about 1–1.5 chrF are usually noise.

## Contributing

`main` changes only through pull requests, and commits are authored as VhinyMombo with no AI attribution.
Work in your own worktree outside Dropbox. The full rules and the team ownership table are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).
