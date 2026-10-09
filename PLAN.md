# PLAN — Gabon LLM (Fang & Punu)

## Why this approach

Fang and Punu are **low-resource**: no large Wikipedias, few digitized books, almost no
parallel data. Training an LLM from scratch needs billions of tokens we don't have.

French, however, is fluently modeled by every modern multilingual LLM, and it is the
bridge language of Gabon. So we **adapt** a French-strong base model rather than build one.
The scarce resource is *data*, not compute — so the bulk of the work is corpus building.

## Phases

### Phase 0 — Data pipeline  (the real work)
- [ ] Collect source docs into `data/raw/<lang>/` (dictionaries, grammars, religious texts, primers).
- [ ] **Digital PDFs** → `src.ingest.parse_pdf` (direct text extraction).
- [ ] **Scanned/image PDFs** → `src.ingest.ocr` (Tesseract; diacritic-aware: ɛ ɔ ŋ + tones).
- [ ] Clean text → `data/interim/`.
- [ ] Build parallel corpus `{src_lang, text_src, fr, en, kind, source}` → `data/corpus/<lang>.jsonl`.
- [ ] **Augment**: use Claude to expand seed entries into conversational variants.
- [ ] **Verify**: native speakers correct augmented data (the only quality gate that matters).

### Phase 1 — RAG prototype  (fast win, no training)
- [ ] Embed corpus, build a vector index (`src.rag.index`).
- [ ] Retrieval-augmented chat: dictionary entries injected as context, Claude composes replies (`src.rag.chat`).
- [ ] Doubles as a **data flywheel** — every interaction is a candidate training example.

### Phase 2 — Fine-tune
- [ ] Format corpus into instruction pairs (translate / converse / explain).
- [ ] **Translation**: evaluate NLLB-200 (already has some Bantu) + fine-tune.
- [ ] **Chat**: QLoRA on Gemma 2 / Qwen 2.5 / Aya Expanse (`src.train.finetune_qlora`).
- [ ] One LoRA adapter per language to start.

### Phase 3 — Evaluate & preserve
- [ ] Held-out test set; chrF / BLEU for translation; native-speaker rubric for chat.
- [ ] Publish the cleaned corpus as a Hugging Face dataset — a lasting preservation artifact.
- [ ] Document orthography/tone decisions in `config/languages.yaml`.

## Base model shortlist
| Use | Candidate | Note |
|-----|-----------|------|
| Chat | Gemma 2 9B/27B-it | Strong French, good for low-resource LoRA |
| Chat | Qwen 2.5 7B/14B-it | Excellent instruction following |
| Chat | Aya Expanse 8B/32B | Built for multilingual incl. African languages |
| MT  | NLLB-200 | Translation-only; some Bantu coverage already |

## Open questions / decisions to record
- Orthography: which standard for Fang? (regional variants differ). Capture in config.
- Tone marking: preserve in corpus or normalize? (Recommend: preserve, normalize a copy.)
- License of source dictionaries — confirm we can redistribute the derived corpus.
