# Baseline grid: plan (E0)

Owner: AI scientist. Harness: deputy (Codex, `src/retrievve/eval/`). Status: plan, nothing run yet.
Runs on the frozen v1 test set (DATA_SPEC §3, §7), once PR #4 is merged and the build reports `frozen: true`.

## 1. Question
Before we pick a v1 architecture: **what do retrieval, the v0 adapters and a dedicated MT model each give us,
over the base model and over trivial baselines, and how much of that is memorization?**

## 2. What we already know about v0 (checked 2026-10-10, after hydration)
| artifact | facts |
|---|---|
| `models/fang-mlx-lora/0002200_adapters.safetensors` | **Intact.** 224 F32 tensors; 16 layers × 7 modules (q,k,v,o,gate,up,down), r=16; finite, non-zero B. From the 12k-iter run (`data/train/fang-mlx`, 23,153 train / 472 valid), stopped at 2,200 iters × bs 4 ≈ **0.38 epoch**, with the LR still near peak (cosine over 12k). Its config lives only in `data/train/fang-mlx/mlx-lora-config.yaml` (the adapter_config.json was overwritten). |
| `models/fang-mlx-lora/adapters.safetensors` | sha256 identical to `0000200`: a 200-iter SFT from base (`fang-mlx-sft`, 25,288 train), ≈ **0.03 epoch**. Practically untrained. |
| both Fang runs | `mask_prompt: false`. The assistant turn is only **14%** of Fang SFT tokens (27.5% Punu), so about 86% of the loss went to the prompt. lr 1e-5, scale 20, max_seq 512, LoRA on layers 12–27 only (23.07M params). |
| passes over train | fang-mlx 0.38, fang-mlx-sft 0.03, punu-mlx-lm 0.09, punu-mlx-sft 0.17. **Every v0 adapter is badly undertrained**, so S2–S4 measure the v0 runs, not the fine-tuning method. |
| `data/corpus/{fang,punu}.vecs.npy` | **Not built from `text_norm`.** `src/rag/index.py` embeds `text \| fr \| en` from the `text` field, which keeps tones (fang: 10,970 of 16,923 entries carry diacritics). Embedder: Ollama **embeddinggemma** (dim 768, rows align with `*.entries.jsonl`: 16,923 fang, 2,859 punu). **But:** the index covers all v0 data (no test exclusion), and dictionary entries embed long gloss blobs. So it is **not reused**; the v1 RAG index is rebuilt from `data/v1/rag/`. |

## 2b. Test slices available for E0 (2026-10-10)
- **fan:** general only (115 pairs per direction in the frozen:false build). The Fang religious slice is **empty**: the Fang
  NT is misaligned and excluded (DATA_SPEC v0.6 §3). It's added after realignment as its own frozen slice. With n≈115 the
  CIs will be wide, so Fang conclusions need either a large effect or confirmation from the native spot check.
- **puu:** general (168) and religious (DEU 31–34, about 120 after v0.5).
- **v0 valid sets are not used anywhere:** 27–40% of their pairs are reversed train pairs.
- The Fang v0 adapters never saw the NT (1 match), so the misalignment didn't contaminate them.

## 3. Systems
Every system is run on fan and puu, both directions (fr→x, x→fr), with religious and general scored separately.

| id | system | why |
|---|---|---|
| B0 | **copy source** | chrF floor (shared names, numbers, loanwords). |
| B1 | **translation memory**: return the target of the most similar **train** pair (chrF on the source side) | Measures how far plain lookup gets. Expected to be strong on religious data. |
| S0 | Qwen2.5-7B-Instruct-4bit, zero-shot | The base model. |
| S1 | S0 + RAG (v1 index, embeddinggemma, top-k=8: lexicon + example pairs) | Retrieval without training. |
| S2 | S0 + fang ckpt 0002200 | The best surviving v0 Fang adapter. |
| S3 | S0 + fang final (200-iter) | Completeness; expected to be close to S0. |
| S4 | S0 + punu LM→SFT adapter | The v0 Punu adapter. |
| S5 | best of S2/S4 + RAG | Do the adapter and retrieval add up? |
| S7 | `vmombo/bloom-560m-fang-translation` (HF cache), fr↔fan only | An earlier Fang MT model by the user. Its training data is unknown (asking the user), so it is **scored on the v0-unseen subset only**. |
| S6 | NLLB-200-distilled-600M, **fine-tuned** on v1 train MT pairs, with new language tags initialised from `kon_Latn` | NLLB doesn't cover fan/puu. This is a dedicated-MT reference, run in phase 2 (it needs training). |

Prompts: the system prompt and **template 0** of each SFT task (`derive.TEMPLATES`), fixed across systems. For RAG
the context goes in the user turn before the instruction, in a fixed format.
Decoding: greedy, `max_new_tokens = min(256, 3 × source tokens)`, stop at `<|im_end|>`. Post-processing:
strip whitespace and NFC normalize only. **Never** strip diacritics before scoring.

## 4. Contamination control (mandatory)
The v0 adapters were trained on v0 data with no split, so the v1 test **may be inside v0 train**.
- For every v1 test item, flag `seen_in_v0` if its `text_fold` or French `text_fold` (with the §3 short-segment
  guard) occurs in `data/train/**/train.jsonl` or `valid.jsonl`.
- For S2–S5, report scores on the **v0-unseen subset**, with the full-set score beside it. A large gap between the
  two means memorization.
- B1 uses the v1 train only, so it is clean by construction.

## 5. Metrics and statistics
Every score is reported with its **n** (lang × direction × domain slice). The general-domain slices will be well
below 500, and the noise rule below scales with n. Trust the bootstrap CI, not the rule of thumb.
chrF++ (primary) and BLEU via sacrebleu, with the signature logged. 95% CI by bootstrap. Paired bootstrap
(1,000 resamples) of every system vs **S0** and vs **B1**, on identical `id`s. With n ≈ 500, a difference under about
1–1.5 chrF++ is treated as noise unless the bootstrap says otherwise.
**Diagnostic, never primary:** chrF++ on `text_fold` of both hypothesis and reference (tones and diacritics stripped).
A large gap between the folded and the real chrF++ means the errors are mostly tones and orthography; a small gap
means the errors are about content.
Also reported: the share of outputs in the wrong language (French in fr→x output, via the §4 stopword test) and the
empty/runaway rate (output longer than 3× the reference).

## 5b. Native-speaker spot check (before the grid drives any decision)
chrF has not been validated for these languages, so automatic scores need one human sanity check.
- Per language: **50 test items**, stratified across directions and domains, from the **two best systems**, plus the
  reference, shown **blind and in random order**.
- The rater scores each output for adequacy (1–5) and fluency in the target language (1–5), and gives a pairwise
  preference. Tones/orthography errors are noted separately.
- Decision use: if the human preference contradicts the chrF++ ranking, the grid's decision is held until we
  understand why.
- Needs: one native speaker per language (to be found with the user), about 2 h each.

## 6. What the grid decides
| observation | consequence for v1 |
|---|---|
| any v0 adapter (S2–S4) ≤ S0 or S1 | **Does not** mean fine-tuning fails. v0 had mask_prompt false (86% of the Fang loss on the prompt) and 0.03–0.38 passes. Fine-tuning is judged only in E1 (§9). |
| S1 ≫ S0 and S1 ≈ S2 | Retrieval does most of the work: keep v1 SFT light and invest in the RAG index and lexicon quality. |
| S2 ≫ S1 on the v0-unseen subset | Fine-tuning generalises: full v1 SFT is worth it (mask_prompt, all-linear, checkpoint averaging). |
| B1 ≈ best system on religious only | The religious test measures lookup. Decide on **general** scores. |
| S5 > max(S2, S1) | Ship adapter + RAG together; evaluate them jointly from then on. |
| S6 > all LLM systems on fr↔x | Use a dedicated MT model for translation and the LLM for chat (two-model design). |
| fr→x ≪ x→fr everywhere | Generation in the target language is the bottleneck: target-side LM stage and tokenizer work (the parked vocabulary extension) move up. |

**Caveat on S2–S4:** since the v0 adapters saw ≤ 0.38 of an epoch, "S1 ≥ S2" does **not** show that fine-tuning is
useless. That question is answered by the first v1 run (E1, §9), not by this grid.

## 7. Out of scope here
The chat/dialogue quality evaluation (native-speaker rubric, DATA_SPEC v0.5) runs separately. Gemma 2 and Aya
Expanse are v1 base-model candidates, compared after this grid on the same harness.

## 8. Cost and disk
**Disk:** about 9 GB free on the machine (98% full). Qwen2.5-7B-4bit is already cached (about 4 GB). NLLB-600M plus
fine-tuning needs 3–5 GB. All outputs (hypotheses, the S6 checkpoints) go **outside Dropbox**, to `~/retrievve-runs/E0/`,
and only the final S6 checkpoint is kept. Free space is checked before each phase, and a phase aborts below 5 GB.

About 500 test pairs × 2 directions × 2 languages × 7 LLM systems ≈ 14k generations of ~40 tokens on an M-series
Mac with 7B 4-bit. That's about 5–8 h of wall clock, so run it overnight. S6 fine-tuning is a separate run.

## 9. v1 training config: decided so far (from the applied-maths measurements, 2026-10-10)
| setting | v0 | v1 | status |
|---|---|---|---|
| `mask_prompt` | false | **true** | decided: 86% of the Fang loss was on the prompt |
| `max_seq_length` (SFT) | 512 | **512** | decided: Fang max 206 tokens; Punu p99 240, 1 example of 13,862 over 512 |
| `iters` | hard-coded | **derived**: `ceil(E × n_train / batch_size)`, E = 2–3 passes | decided |
| `adapter_path` | shared | **unique per run**: `models/<lang>-<stage>-<YYYYMMDD>-<shortsha>/` | decided |
| `num_layers` | 16 (layers 12–27) | 16 vs **-1** (all 28, 40.4M params) | E1 ablation |
| lr / LoRA scale | 1e-5 / 20 | 1e-5 vs 5e-5 vs 1e-4 (short runs, val chrF++) | E1 ablation |
| checkpoint selection | last | best val chrF++, plus averaging the top-k checkpoints | E1 |
| batch / LR | B=4 / 1e-5 | **B=4 × grad_accumulation 4** (effective 16; B=16 swapped 35 GB and filled the disk on 2026-10-10); LR grid **{2e-5, 4e-5, 1e-4}** at scale 20 (sqrt rule from B=4; scale 20 at 1e-5 ≈ PEFT α/r=2 at 1e-4) | E1 ablation |
| schedule | cosine | **WSD**: warmup clamp(3%, 10, 300), constant until 80%, linear decay to 0.1× | decided |
| passes | 0.03–0.38 | SFT **E=3**, LM **E=3** | decided |
| early stopping | none | val chrF++ (~200 sentences) every max(100, iters/20); patience 3, Δ 0.3 | decided |
| task loss weighting | per token | (a) per token vs (b) per-example mean. Under masking, MT is ~91% of the gradient but 61–64% of the examples. | E1 ablation |

Measured on the frozen:false v1 build (n_train will shift after the spec v0.5 rebuild): SFT fan 11,676 / puu 19,137
examples; max 307 tokens, 0 over 512; LM fill about 92%. Answer-token share is fan 0.33 and puu 0.29.

**Compute is the binding constraint.** v0 throughput was about 52 tokens/s, which makes one E=3 SFT run about 17 h (fan)
or 27 h (puu). The E1 schedule therefore runs in stages, each one fixing a single choice:
1. Timing: 100 iters at B=16, num_layers 16 vs −1, giving the real tokens/s.
2. LR grid on **puu only, E=1**, at num_layers 16: 3 runs.
3. At the winning LR, puu E=1: num_layers −1 vs 16, and loss weighting (a) vs (b), in 2 more runs.
4. Final runs at E=3 for puu, plus the LM→SFT variant. **Fang E1 waits** until its training data is restored: after
   excluding the misaligned NT and gating the Galley OCR, Fang SFT train is only **1,946 examples**. It comes back once
   the Galley OCR check (≥ 95% correct headword+gloss) and/or the NT realignment pass.
**Safety for every run:** an MLX memory cap, abort if swap goes above 4 GB, abort if free disk drops below 10 GB, and
outputs under `~/retrievve-runs/`. Each run starts only with the lead's go-ahead on the GPU.
Lexicon accuracy (for the weighting ablation) needs a held-out lexicon probe. That's a spec item for v0.6:
200 headwords per language, excluded from SFT but kept in RAG, so only no-RAG systems are scored on it.
