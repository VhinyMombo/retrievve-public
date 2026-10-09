# Cloud compute plan and budget (E0/E1)

Owner: AI scientist. Status: plan. Nothing runs until the RunPod account and credit exist (user) and the lead gives the go.

## 1. Constraints (user decisions, 2026-10-10)
- **Hard cap: $30 total**, RunPod prepaid credit (the prepayment is the cap). No HF Pro, so no HF Jobs.
- A100 at about $1.79/h ≈ **16 A100-hours** for everything. Cheaper GPUs are used where a job doesn't need an A100.
- `release: false` sources may be uploaded to a **private** Hugging Face dataset, the only way data reaches a pod.
  It is never public. The HF token is passed as a pod environment variable, never in the repo or a script.
- The Mac (36 GB, mlx) stays the free compute for inference and for anything mlx-specific, once the machine freeze is lifted.

## 2. What runs where
| work | where | why |
|---|---|---|
| E0 inference with the **v0 mlx adapters** (S2–S5) and an **S0-mlx** reference | **Mac** | The adapters are mlx-format on a 4-bit mlx base. Porting them to CUDA adds a quantization mismatch, and inference is small (about 400 test pairs × 2 directions + the probe). Cost $0, about 30 min per system. |
| E0 floors B0/B1, S7 (bloom-560m) | **Mac (CPU/mlx)** | Tiny. |
| E0 **S0-cuda** and **S1-cuda** (base Qwen2.5-7B-Instruct bf16, ± RAG) | **Cloud**, batched with E1 | E1 models are trained on the bf16 base, so they must be compared with a base on the **same** precision. S0-mlx vs S0-cuda also measures the 4-bit effect. |
| S6: NLLB-200-distilled-600M fine-tune | **Cloud, cheap GPU** (24 GB class, e.g. RTX 4090/A5000) | 600M parameters don't need an A100. |
| E1 training (Punu first, Fang once its data is restored) | **Cloud, A100** | The Mac needs about 27 h per E=3 run and swaps at B=16. |

**Comparability rule:** every E1 number is compared against **S0-cuda/S1-cuda** (same base and precision). v0
adapters are only compared against **S0-mlx**. Paired bootstrap is never run across the mlx/cuda boundary.

## 3. Budget allocation (A100-hours; 1 h ≈ $1.79)
Training throughput for LoRA on 7B bf16 is expected to be about 2–4k tokens/s on an A100. With Punu SFT at about 1.7M
tokens per pass, that's about 10–15 min per pass. **The calibration step (C) measures this, and every later row is
re-estimated from it before launch.**

| step | content | est. h | $ |
|---|---|---|---|
| C | Calibration: pod setup, smoke test (20 iters), measured tokens/s, val-eval time; S0-cuda/S1-cuda inference on all frozen slices + probe | 1.0 | 1.8 |
| E1-a | Punu LR grid {2e-5, 4e-5, 1e-4} (scale 20, eff. batch 16), **E=1 on full train** (not a subsample, if C confirms ≤ 15 min/pass) | 1.5 | 2.7 |
| E1-b | At the best LR, E=1: num_layers 16 vs all; loss weighting per-token vs per-example (2 runs) | 1.0 | 1.8 |
| E1-c | Final Punu E=3 (WSD, early stopping) + LM→SFT variant; checkpoint averaging; eval | 2.5 | 4.5 |
| E1-f | Fang E=3 with the E1 settings (only after the Galley OCR gate / NT realignment restores its data) | 2.0 | 3.6 |
| S6 | NLLB-600M fine-tune on a 24 GB GPU (~$0.4/h; about 2 h) | (≈0.5 A100-eq.) | 0.8 |
| R | **Reserve**: reruns, failures, a second seed for the final Punu run | 4.0 | 7.2 |
| | **Total planned** | **12.0 A100-h + S6** | **≈ $22.3** |
| | Unallocated margin | | ≈ $7.7 |

If C shows throughput below 1k tokens/s, E1-a moves to a **25% stratified subsample** at E=1 (as the lead
proposed), and E1-f is postponed. The reserve is never spent on new experiments without updating this table first.

## 4. Operating rules (every pod)
1. **No cloud run without a local smoke test** of the exact script and config: 20 iters on CPU or mlx with a tiny
   model (e.g. Qwen2.5-0.5B) and 50 examples, ending with a non-NaN loss and a written adapter. The PEFT/TRL port also
   needs the **parity check** (applied maths): mlx vs PEFT loss curves over 50 iters on the same data and settings.
   They should match within quantization noise; a gross mismatch blocks the cloud run.
2. **Hard time limit per pod:** the launcher wraps the job as `timeout <budgeted>m <job>; <push results>; runpodctl remove pod $RUNPOD_POD_ID`,
   plus an independent watchdog (`sleep <limit+10m>; runpodctl remove pod $RUNPOD_POD_ID &`) started first.
   **Terminate (`remove`), don't just stop:** a *stopped* RunPod pod still bills for its volume disk. Only a
   terminated pod costs nothing.
3. **Terminate the pod the moment the job ends**, success or failure. Results (adapters, hypotheses, logs) are pushed
   to a **private** HF model repo or downloaded **before** termination. Nothing is left on a pod, and no network volume
   or stopped pod survives a session. After each session, check the RunPod console: 0 pods, 0 volumes.
4. **One job queue per pod session:** batch C + E1-a in one session to pay the model download (about 15 GB, a few minutes) once.
   No network volume (it bills even when stopped), unless measured setup time exceeds about 15 min per session.
5. Seeds are fixed. Every run writes `run.json` (git sha, config hash, data manifest sha, GPU type, tokens/s, wall time,
   cost) next to its outputs.
6. **Cost log** (below) is updated after every pod, from the RunPod billing page, not from estimates.
7. Abort rule: if spend reaches $25 before E1-c is done, stop and re-plan with the lead and the user.
8. **Cost log from the actual RunPod balance** (before and after each session), not from rate × hours.

## 4b. Lead's conditions (PR #18 review, 2026-10-10)
- **Mac E0 inference only** (a partial freeze lift, no restart needed): `mx.set_memory_limit` ≤ 12 GB, batch 1–4,
  the swap-delta + free-disk watchdog (abort if swap grows by > 4 GB or free disk drops below 5 GB), and a **20-item smoke
  test** before each system's full run. **Never mlx training on the Mac.**
- **E0 starts only after #17 is merged and `puu.religious` is frozen** (gate passed 2026-10-10: DEU tail 0.955, GEN tail 0.878).
- No cloud session before the lead confirms that the RunPod account and key are ready.
- **Mac GPU lock (all Mac GPU jobs: mlx, ollama, embeddings):** use the shared helper `retrievve.gpulock`, never a hand-rolled
  lock. Python: `with gpu_lock(owner, job):`. Shell: `python -m retrievve.gpulock run --owner X --job Y -- <cmd>` (holds the lock for
  the child's whole lifetime). The mechanism is a kernel `flock` on `~/retrievve-runs/GPU.lock`, released by the OS when the holder exits
  for any reason (SIGKILL included), so there are no stale locks, no break races and no deadlock on a half-written file. The file's JSON
  (owner, job, pid, host, started_at) is informational only. A live legacy `GPU.lock/` directory lock is respected, and a dead one
  is removed. Status: `python -m retrievve.gpulock status`. Never two GPU jobs at once.
- **One-time exception (lead, 2026-10-10):** the mlx-vs-PEFT parity check at Qwen2.5-0.5B **bf16** on the Mac (≤ 50 iters, under
  the cap, watchdog and lock), with identical LoRA init (PEFT init exported to mlx), identical data order and constant LR.
  Pass: step-0 loss within 1e-3 and per-step loss within ~1% relative. **7B training on the Mac stays banned.**

## 4c. RunPod access (user rule, 2026-10-10)
- **Only the lead (retrievve-60) and the AI scientist (retrievve-7d) may launch, stop, remove or manage anything on
  RunPod**, or touch the RunPod account. Nobody else does, including the deputy (Codex), the data engineer, the data
  scientist and applied maths.
- Applied maths writes and tests the launcher (`scripts/runpod_job.sh`) locally. The lead or the AI scientist runs it.
- **Credentials:** `RUNPOD_API_KEY` (and the HF token) are set by the user in their own shell. Scripts **only read them
  from the environment**: never write them to files, logs, `run.json`, the repo, pod templates or command lines that
  are echoed. The launcher fails fast with a clear message if a variable is missing.

## 5. Porting requirements (owners)
- **Training** (applied maths): HF PEFT + TRL `SFTTrainer` on CUDA, reproducing the decided E1 config: assistant-only
  loss (completion-only collator, the PEFT equivalent of `mask_prompt`), LoRA r=16 with α chosen so the update
  scale matches mlx scale 20 (document the mapping), all-linear target modules, WSD schedule, eval/early stopping on val
  chrF++, unique output dir per run. It runs unchanged on CPU for the smoke test.
- **Harness** (deputy): runs on CUDA (vLLM or HF `generate`, batched) **and** on the Mac (mlx), with identical
  prompts, decoding and post-processing; `run.json` records the backend.
- **Data:** the data engineer pushes the frozen `data/v1` artifacts (sft, eval, rag, manifest) to a **private** HF
  dataset with the manifest sha in the commit message. Pods read only that.

## 6. Cost log
| date | pod / GPU | job | wall h | $ | cumulative $ | notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | 0.00 | nothing run yet |
