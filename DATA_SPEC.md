# retrievve — Data Spec v0.9 (provisional; pending lead sign-off)

Owner: AI scientist (model/architecture). Implementer: data engineer.
Status: v0, written before the corpus was readable (Dropbox placeholders). Thresholds marked **[TBD-stats]**
get fixed once the engineer sends the distributions in §8.

## 0. Working assumptions (to confirm with the user)
- Primary task: **French ↔ target-language translation**, plus a grounded assistant (RAG) for lexicon/grammar questions.
- Priority: **Fang (`fan`) and Punu (`puu`)** = train + eval. **Nzebi (`nzb`), Myene (`mye`), Lembaama (`lev`)** = eval-only
  (RAG + few-shot), no fine-tuning until they have at least about 2k parallel sentences.
- Pivot language: French. Prompts in French.

## 1. Canonical corpus record (`data/v1/corpus/<lang>.jsonl`)
One segment per line. Every downstream artifact (SFT, LM, RAG, eval) is derived from these records only.

```json
{
  "id": "fan:nt:MRK.4.3",            // stable, deterministic: <lang>:<source_id>:<locator>
  "lang": "fan",                      // ISO 639-3
  "text": "…",                        // target language, normalized (§2)
  "text_raw": "…",                    // as extracted, untouched
  "fr": "…",                          // French pivot = MT reference, or null if monolingual
  "fr_alt": "…",                      // optional: idiomatic French equivalent (proverbs); not an MT reference
  "fr_note": "…",                     // optional: explanation/commentary; never an MT reference
  "kind": "verse|sentence|dialogue_turn|lexicon|grammar|example|poem_line|song_line",
  "domain": "religious|conversational|lexical|grammar|literary|oral",
  "source_id": "fang_nt_web",         // key into data/v1/sources.yaml
  "group_id": "fang_nt_web/MRK",      // unit for splitting (§3): book, document, song, dialogue
  "orthography": "…",                 // free tag if the source uses a distinct spelling convention
  "align": {"method": "verse|manual|row|heuristic", "score": 0.0},
  "ocr": false,
  "origin": "human",                  // human | synthetic | user_interaction (§10b); default human for curated sources
  "verified_by": null,                // native-speaker check, e.g. {"who": "<role/initials>", "date": "…"}; null if unchecked
  "flags": [],                        // e.g. ["fr_in_target", "len_ratio", "near_dup"]
  "split": "train|val|test"
}
```
Lexicon records additionally carry `"lex": {"headword", "pos", "noun_class", "plural", "gloss_fr", "examples": [ids]}`.

`data/v1/sources.yaml`, one entry per source: `title, author, url/origin, acquired_on, license, release: true|false, notes`,
`origin` (default `human`), `orthography` (required for every target-language source; §2),
plus an optional `fr_parens: literal|note` for sources whose French field holds a parenthetical.
**Proverbs and other two-part French fields:** the parse rule is declared **per source**, never guessed per row.
- `fr_parens: literal` (e.g. `myene_proverbes_mandji`): the parenthetical is a literal gloss. `fr` = literal gloss and
  `fr_alt` = idiomatic equivalent.
- `fr_parens: note` (e.g. the Punu proverbs): `fr` = the translation outside the parentheses and `fr_note` = the explanation.
Quality filters and MT eval use `fr` only. RAG `display` shows `fr` + `fr_alt` + `fr_note`. SFT may add a
`proverb_meaning` task (x → `fr_alt`/`fr_note`) that is never used as an MT reference.

## 2. Normalization (applies to `text` and `fr`)
- Unicode **NFC**. **Never strip diacritics or tone marks.**
- Unify look-alikes to IPA-Africa letters: Greek ε→ɛ, ↄ→ɔ, Latin-script ǝ/ə consistently (pick per language and log the mapping table).
- Apostrophes and elision marks (measured 2026-10-10): a variant is an **elision mark only between two letters**.
  Anywhere else it is punctuation and is left as is (the Fang NT's ’/‘ are quotation marks around speech, ×673/672).
  Canonical elision mark: **Punu `ʼ` U+02BC** (×727, so ' between letters → ʼ); **Fang `'` U+0027** (×408).
- Collapse whitespace, strip verse numbers, footnote markers, and HTML entities. Keep sentence-final punctuation.
- Keep `text_raw` so any normalization can be revisited.
- Two derived keys, used **only** for dedup and lookup, never for SFT, LM, embeddings or display:
  - `text_key` = NFC + casefold + punctuation/whitespace stripped, **diacritics and tones kept**.
  - `text_fold` = `text_key` with diacritics and tones stripped.
  Usage, chosen by the cost of each error:
  - **Cross-split leak removal uses `text_fold`.** It is aggressive: a false merge only drops a train example, while a
    missed leak corrupts the test.
  - **Within-train dedup uses `text_key`.** It is conservative: in a tonal language, a false merge loses a real
    minimal pair.
  - **Lexicon lookup** tries `text_key` first and falls back to `text_fold`, since users often type without tones.
- The v0 field `text_norm` stripped tones (e.g. "nyè-nyè dzènye" → "nye-nye dzenye"),
  so it is **deprecated**: do not carry it into v1.
- **Orthography tags are mandatory per source** and declared in `sources.yaml`, never guessed per row. Fang, as of
  2026-10-10: `galley1964` (fang-dictionary.txt, fang-dictionary.pdf / fang-examples; letters ñ, ô, ẑ),
  `nt_ipa` (Fang NT: ɔ, ŋ, combining tones). The grammar sources are galley1964 (ô); fang-phrases mixes systems and is
  tagged `mixed` (excluded from per-tag stats, kept in data), except the nkobefang TikTok sources, which use a third system:
  **`schwa_untoned`** (ə + ɔ, no tone marks). Look-alike Ǝ U+018E → Ə U+018F is mapped **case-preserving**. Case is never
  changed to "correct" a sentence; a mid-sentence capital is logged as `suspect_case` for native review. **`orthography` describes the spelling system only.** The
  dialect/region goes in a separate optional `variety` field in sources.yaml (e.g. the NT's Spanish-form names
  suggest an Equatorial Guinea edition). It is filled in only when confirmed (a native speaker or the source's front
  matter), never inferred. Fang ə ×60 and Ǝ ×4 are to be traced to their sources before any mapping. Different orthographies are
  **never** unified into one another without a mapping table signed off by the AI scientist. Token stats, eval
  and RAG all report per tag.
- **All-caps lexicon headwords** (typographic, not orthographic): for sources declared `headword_case: upper` in
  `sources.yaml`, lowercase `text` for `kind=lexicon` records where `text == text.upper()`. Unicode-aware, so Ñ→ñ and
  Ô→ô. `text_raw` keeps the caps. Proper nouns are rare among headwords; any found go on an exception list in config.
  (Measured: fang-dictionary.txt is 100% caps, and lowercasing alone moves it from 1.57 to 1.78 chars/token.)
- `lang` uses ISO 639-3 codes. Map the v0 names: fang→fan, punu→puu, nzebi→nzb, myene→mye, lembaama→lev.

## 3. Splits: grouped, frozen, leak-proof
- Split by **`group_id`, never by line**.
- **Fang NT (`fang-nt-parallel`) is EXCLUDED from SFT, LM pairs, eval and RAG** until re-parsed and realigned
  (data-scientist finding, 2026-10-10: proper-noun agreement is 0–3% in most books, and wrapped PDF lines were parsed as
  verses). Realignment (data engineer): re-parse the verses, then a monotonic DP per chapter against
  `data/raw/ref/ls1910-nt.json` with the sim scorer as cost. **Acceptance: proper-noun agreement ≥ 0.8 per book**
  (data scientist validates). Books failing acceptance stay excluded. There is **no interim sim-floor stopgap**:
  misaligned references would corrupt eval, and the 219 pairs at sim ≥ 0.65 are not worth the noise in train.
  Until then the Fang religious test slice is empty. When realigned, it is frozen as its **own slice** (`fan.religious`,
  frozen separately) and never merged retroactively into already-reported numbers.
- **Bible/NT:** test = whole held-out books. **Fang NT (v0.9): test = {REV}, val = {HEB}**, both among the books accepted
  by the alignment gate (realigned NT, 2026-10-10: REV 0.932 / margin 0.89, HEB 0.853 / 0.73). Rest of the accepted books = train.
  No gospel is held out (synoptic leakage), and ACT isn't either (it shares narrative with LUK). REV is lexically atypical
  (apocalyptic), so Fang religious scores are **always reported separately and never pooled** with Punu religious or with
  general slices.
  **NT book acceptance:** a book enters the corpus only if it's listed in `nt_books_accepted`, set from the data scientist's
  official `--strict` run. Name-poor books (unverifiable, or failing agree with a high margin) need an **independent second check**
  (e.g. verse-length correlation vs a shuffled baseline, numerals, a bilingual-lexicon hit rate) that the misaligned v0 NT fails.
  The bar is never lowered. Until then they stay out, as does MRK (0.798).
  **Do not** put a synoptic gospel in test while the others are in train: near-parallel verses leak.
  Punu OT: test = {RUT}, val = {JON} (or the nearest available short books).
  **Only books with parallel verses count** (a monolingual book yields no eval pairs).
  **Oversized fallback → tail chapter ranges.** If the smallest available book with parallel verses is > 2× the
  target fraction, hold out **whole chapters from the end of a narrative book** until the target is reached
  (`group_id = bible/<BOOK>/tail`, with the rest of the book in `bible/<BOOK>`). Avoid legal or duplicated passages
  (LEV, EXO 35–40, which repeat EXO 25–31). Cross-split near-dup removal then cleans formulaic repeats.
  Punu, decided 2026-10-10: **test = DEU 31–34 exactly** (the narrative ending, about 120 verses). Earlier DEU restates
  EXO/LEV/NUM law and would near-duplicate train, so it is **not** extended to reach the 5% target. A smaller clean
  religious test beats a larger contaminated one. **val = tail of GEN** (from ch. 50 backwards, about 2%). LEV goes
  back to train. Report the cross-split near-dup drops for the DEU slice separately.
- **Non-religious (conversational, dialogues, songs, poems, phrasebooks):** about 20% of *documents* go to test and about 10% to val,
  stratified by domain. This test slice is the one we care most about, because the corpus is Bible-heavy.
- **Multi-parallel sources are split jointly across languages.** The poems, proverbs and `data/raw/parallel/*` share
  French sides across languages: `group_id` must be language-independent (e.g. `poem/chien_puces`, not
  `fan/poem/...`). One group goes to one split for **all** languages. The same applies to Bible books: one global
  held-out book list for every language that has the book, since the French side (LS1910) is shared.
  Choose held-out books from those actually present (Punu raw has GEN/EXO/LEV/DEU; RUT/JON may not exist, so pick
  from what's there and report it).
- **Lexical and grammar documents are never held out as whole files** (`never_heldout_domains: [lexical, grammar]`).
  Test is an MT test, and holding out a lexicon would gut the RAG lexicon. Caveat: example sentences inside
  lexicon/grammar records go through the same cross-split leak removal, so any example that matches a val/test pair is
  dropped from train **and** from the RAG index.
- **Cross-split dedup:** after splitting, remove from train any record whose normalized `text` OR `fr` matches a val/test
  record exactly or by near-dup (MinHash Jaccard ≥ 0.8 on char 5-grams). Test wins. Check the French side **across languages** too.
  **Short-segment guard:** a match on **one side only** counts as a leak only if that side has ≥ 3 words and ≥ 15 chars
  (after `text_fold`). Shorter segments, such as greetings and formulae ("Bonjour" / "Mbolo"), are removed only when **both**
  sides match the same val/test pair. MinHash near-dup applies only to segments ≥ 15 chars. Report xsplit drops per
  language and per rule.
- **Test is excluded from the RAG index too.** Otherwise RAG eval is contaminated.
- Freeze: write `data/v1/splits/<lang>.{train,val,test}.ids` + sha256 in the manifest. Test never changes after v1.
- **Freeze gate, per slice:** a religious (Bible-aligned) test or val slice is frozen only after the data scientist's
  alignment check passes, with **proper-noun agreement ≥ 0.8 per book** (and per tail range, e.g. Punu DEU 31–34 and the
  GEN tail). Non-religious slices can freeze independently. The manifest records the freeze state per slice
  (`frozen_slices`). We never freeze references nobody has checked.
- **Manifest format:** `frozen_slices` is a **flat map `"<lang>.<slice>": bool`**. Absent or pending slices are explicitly
  `false` (e.g. `fan.religious` while the NT is realigned), never omitted. Details live in `slices["<lang>.<slice>"]`
  (`pairs_per_direction`, `descriptive_only`, `ids_file`, `sha256`, `strata`, `gate`). `religious_gate_passed`
  (`{"<lang>": bool}`) sits at the manifest top level. Consumers select eval files **only** through these fields, never
  by globbing `eval/`.
- **Freeze guard:** a build reads the previous manifest. For every slice frozen there, it never deletes its files and
  **aborts** if any of their sha256 would change, unless run with an explicit `--unfreeze <slice> --reason "…"`, which is
  logged in the manifest. A frozen test set changes only by a deliberate, recorded decision.
  Gate command (exit 0 = may freeze, 1 = a slice fails, 2 = unverifiable):
  `python -m retrievve.analysis.align_check data/v1/corpus/<lang>.jsonl --domain religious --split test,val --by group_id --strict`
  (pass = agree ≥ 0.8 **and** margin over the shuffled baseline ≥ 0.4, with ≥ 10 name occurrences per group).
  The guard also aborts on a **silent de-freeze** (a previously frozen slice marked false or omitted) and on a **vacuous pass**
  (a frozen slice with 0 compared files), and refuses to run while a `.promoting` marker from an interrupted build exists.
  **Schema transitions** of frozen files are allowed only via `--unfreeze-allow-added-fields` / `--unfreeze-allow-removed-fields`,
  which verify that every other field is byte-identical and no ids file changes. Two were done pre-E0 (2026-10-10:
  `orthography` added to the probe; `release` removed from eval/probe rows). **After the first reportable E0 result, frozen
  files never change schema again:** new per-row metadata goes into sidecar files keyed by `id`.
  The train side is checked report-only with `--split train [--scan-offsets]`; failing chapters get `low_align`
  (`filters.low_align_chapters`). **Chapter-level `low_align` decisions only for chapters with ≥ 10 name occurrences**
  (below that, scores are noise: translated names such as Lévite/Araméen/Jourdain; DEU 34 has a baseline of 1.0), plus
  `--scan-offsets` for shifts. Punu as of 2026-10-10: no chapter flagged (DEU 7 passes with the ethnonym-aware matcher, 0.917).
- Eval-only languages (`nzb`, `mye`, `lev`): split documents 50/50 into `rag_pool` / `test`.

## 4. Quality filters (parallel pairs)
Reject (keep in corpus with a `flags` entry; exclude from SFT and eval):
1. Empty side, or either side < 2 chars.
2. **French in the target field** (two detectors; either one flags `fr_in_target`):
   (a) stopword ratio in `text` > 0.30, with ≥ 1 hit and ≥ 3 tokens (removes 0.11%, all real errors on the manual sample);
   (b) **vocabulary detector**: the share of ≥ 3-letter tokens of `text` found in the LS1910 French vocabulary (freq ≥ 2)
   is ≥ 0.67 (removes about 0.86%, e.g. "Avoir honte ‖ être dans la gêne"; catches 2-token rows that (a) misses).
   **Precondition (v0.8): `text` has ≥ 2 tokens of ≥ 3 letters.** On one-word rows the detector only fires on real
   target words that look French (été, mange, pure…): precision rises from 19/62 = 31% to 17/17 with the precondition
   (`data/v1/reports/fr_detector_precision.md`).
   Blind spot, reported every build: the count of one-token target rows whose token is in the LS1910 French vocabulary and
   is **not** a lexicon headword (`build_report.fr_detector_blind_spot`), with up to 20 examples.
   The same per-language exemption applies to both: tokens that are lexicon headwords or target-side lexicon example
   tokens never count as French (e.g. Fang "mot" = person).
   The stopword list is **per language**: the French list (in config, not code) minus any word that occurs as a
   **lexicon headword or a token in lexicon example sentences (target side)** of that language (e.g. Fang "de").
   Never count frequency over train text: the French copy-paste rows would exempt themselves. A language without a
   lexicon gets no exemptions. Log the removed words.
3. Char length ratio `len(text)/len(fr)` outside **[0.4, 2.5]**, applied only when both sides are ≥ 15 chars →
   `len_ratio` (removes 3.6%: religious 4.3%, other 2.9%; fang 5.3%, punu 1.1%. Without the guard it would remove 16.2%,
   mostly lexical items).
4. `low_align`: the percentile rule is **dropped** (meaningless on a misaligned NT). After realignment, an absolute floor
   is re-swept on the realigned scores and set here.
5. Exact dup on (lang, text, fr) → keep one. Near-dup within train → keep one, unless dialogue/lexicon.
6. OCR records: `ocr: true` → `ocr_review` flag: blocked from SFT, LM and eval, allowed in RAG with a visible marker.
   OCR records never contribute lexicon exemptions (§4.2). The block is lifted per source (`ocr_review_lifted`) only after a
   spot check against the scan reaches **≥ 0.95 headword+gloss correct on a FRESH stratified sample** (never the sample the
   parser fixes were derived from). Galley 1964, 2026-10-10: 62/100 strict (94/100 after dehyphenation), so the gate is not
   met (`reports/galley_ocr_spotcheck.md`). Its linked examples (`fang_galley_examples_v0`) are **excluded from SFT, LM AND RAG**
   (only 46/63 are valid pairs, 29/63 attached to the right entry) until re-parsed and rechecked.

## 5. Training artifacts
### 5a. LM / continued pretraining: `data/v1/lm/<lang>.{train,val}.jsonl`
`{"text": "..."}`, target language only, train split only. Concatenate consecutive segments of the same `group_id`
into chunks counted in **training-tokenizer tokens** (not characters):
`chunk_tokens_max = floor(0.9 × max_seq_length)` (derived, not hard-coded), `chunk_tokens_min = 256` (provisional).
Pack whole segments greedily. Never split a segment unless it alone exceeds `chunk_tokens_max`; if it does, split it at
sentence punctuation. Don't shuffle inside a chunk.
**Tail chunk:** if the last chunk of a `group_id` has fewer than `chunk_tokens_min` tokens, merge it into the previous
chunk when the result is ≤ `chunk_tokens_max`. Otherwise keep it as a short chunk. Tail text is **never dropped**. A group
whose whole text is under `chunk_tokens_min` is one short chunk. Report the number of short chunks per language.
max_seq_length (512 vs 1024) is a v1 architecture decision. The builder takes `tokenizer` and `max_seq_length` as
parameters. Characters per token for fan/puu (first estimate is about 1.9, from n=1) goes in the stats report.
**SFT:** compute the token length of every rendered example, chat template included. Examples longer than max_seq_length
are excluded from SFT and listed in the report. Never truncate the assistant turn.
Report the exclusion rate **per task × language** at both max_seq_length 512 and 1024. If any task loses **> 2%** at
512, that triggers a 1024 run in the v1 architecture comparison. The rate alone does not switch the setting.

### 5b. SFT: `data/v1/sft/<lang>.{train,val}.jsonl` (mlx-lm chat format)
```json
{"messages": [
  {"role": "system", "content": "Tu es un assistant spécialiste du fang."},
  {"role": "user", "content": "Traduis en fang : <fr>"},
  {"role": "assistant", "content": "<text>"}],
 "meta": {"src_id": "fan:…", "task": "mt_fr2x", "domain": "conversational"}}
```
Tasks (from train-split records only):
| task | from | notes |
|---|---|---|
| `mt_fr2x` | parallel | |
| `mt_x2fr` | parallel | same pairs, reverse: **50/50 direction balance** |

**Both directions of a record live in the same split**, which holds by construction, since both are derived from one
record after splitting. v0 violated this: 27–40% of v0 valid pairs were the reverse of a train pair, so **v0 valid losses
are not generalisation estimates.** The same holds for lex_x2fr / lex_fr2x.
| `lex_x2fr` / `lex_fr2x` | lexicon | include pos and noun class in the answer when known |
| `grammar_qa` | grammar | rule chunk → question/answer; hand-checked sample |
| `dialogue` | dialogue_turn | previous turn(s) → next turn |
- 6–10 paraphrased French instruction templates per task, chosen by `hash(id) % n` (deterministic).
- **Domain mix cap:** religious ≤ 50% of SFT examples. Reach it by upsampling non-religious (max ×3) and then
  downsampling religious. Report the final mix. **`lexical` and `grammar` are never upsampled (×1).** Only
  sentence-level non-religious domains (conversational, literary, oral) are. (Punu lexical had reached ~37% of SFT.)
- `meta` is kept for analysis. Strip it in a final `mlx/` export if mlx-lm complains.

## 6. RAG index inputs: `data/v1/rag/<lang>.jsonl`
Train + val + (`rag_pool` for eval-only langs). **Never test.**
```json
{"id": "...", "lang": "fan", "type": "lexicon|grammar|example_pair",
 "embed_text": "...",      // what gets embedded (French-first: gloss_fr + headword, or fr sentence + text)
 "display": "...",         // what gets shown to the LLM in context
 "src_ids": ["..."], "source_id": "..."}
```
- Lexicon: one chunk per headword (headword, pos, class, plural, gloss_fr, ≤ 2 examples).
- Grammar: chunks of 200–400 tokens split on headings; prefix with the heading path.
- Example pairs: one chunk per parallel sentence.
- I'll handle the embedding model choice and vector building. You deliver the JSONL.

## 7. Eval artifacts: `data/v1/eval/<lang>.<task>.test.jsonl`
`{"id", "src", "ref", "direction", "domain"}` for `mt_fr2x`, `mt_x2fr`. Separate files for religious and non-religious
domains (reported separately: religious BLEU will be inflated).
Target: ≥ 500 test pairs for fan/puu if available. Report the actual number either way, **per slice**.
Eval-only languages with < 50 test pairs (currently mye 18, nzb 15, lev 0) are reported **descriptively only**:
scores and examples, with no significance claims and no decisions based on them.
Only `origin: human` records are ever eval sources or references (§10b).
Metrics: chrF++ (primary) and BLEU, using sacrebleu with its signature logged. System comparisons use a **paired
bootstrap** (1000 resamples, p < 0.05). Every test record carries a stable `id`, so systems are compared on identical
segments.

## 7b. Lexicon probe: `data/v1/eval/<lang>.lexprobe.jsonl`
Measures word knowledge without retrieval (e.g. for E1's loss-weighting ablation).
- **Sample:** puu **400** and fan **200** headwords (Fang only once the Galley OCR gate is lifted), deterministic (hash,
  seed 1729), stratified by POS. Homographs are merged (union of glosses); entries with an empty `gloss_fr` are dropped.
- **Isolation:** every lexicon record of a sampled headword is removed from **all SFT lex tasks in both directions**. It is
  **kept in RAG**, so RAG systems are **never** scored on the probe. Flag `in_sft_sentences` if the headword occurs
  (by `text_key`) in any SFT MT sentence, and report results split by that flag ("never seen" vs "seen in context").
- **Decoding:** no RAG, greedy, a fixed template, at most 32 new tokens. Answer = the first line of the output, with
  leading French determiners stripped (le, la, les, l', un, une, des, du, se, s').
- **x→fr:** gold = all glosses (split on ";" and on numbered senses; parenthetical remarks dropped). Primary: **EM_any**
  (`text_key` match against any gloss). Secondary: **chrF_max** over the glosses (chrF++ settings). Report the share
  of **definition-style glosses** (a gloss over 4 words, or a description rather than an equivalent word). EM_any is
  harsh on them, so EM_any is also reported on the equivalent-word subset.
- **fr→x:** gold = the headword plus variant spellings listed in the entry (not plurals). Primary: **EM_key**, which
  keeps tones and diacritics. Diagnostic: **EM_fold**; **EM_fold − EM_key = the "right word, wrong tones" rate**, the key
  number for preservation. Fang is reported per orthography tag and never mixed.
- **Frozen:** `frozen_slices["<lang>.lexprobe"]` is true, and the ids file and sha256 are in `slices["<lang>.lexprobe"]`. Once
  frozen, the probe never changes (freeze guard, §3).
- **Export schema** (`<lang>.lexprobe.jsonl`, the data side is the source of truth): `id, lang, headword, text_key,
  variants, gold_glosses, pos, orthography, fold_neighbours, in_sft_sentences, definition_gloss, src_ids, release`.
  `variants` = merged entries (same `text_key`) + spellings the entry lists. **`fold_neighbours` (tone-different words)
  are diagnostics only and never count as correct answers.**
- **Statistics:** CI half-width about ±6.4–6.9 points at n=200 and about ±4.9 at n=400. System comparisons: **exact
  McNemar** on discordant pairs for EM, and paired bootstrap for chrF_max. Differences under about 8–10 points (n=200)
  are not claimed.

## 8. What I need back first (stats report, before tuning thresholds)
Per language and per source: #records, #parallel, #monolingual, #lexicon; domain breakdown; histograms of
length ratio, alignment score, French-stopword ratio in `text`; character inventory (all codepoints with counts,
to catch look-alikes); top-20 near-dup clusters.
Token statistics (chars/token, tokens/word) are reported **per language × `orthography` tag**, not only per language:
in Qwen2.5's byte-level BPE, IPA letters (ɛ, ɔ, ŋ) are not merged and cost 2+ tokens each, so a mixed pool hides
the gap. Never respell IPA-orthography text into French-style letters to save tokens: it erases phonemic contrasts
(ɛ/e, ɔ/o). Fixing the tokenizer (extending the vocabulary) is a parked architecture option.
A JSON or Markdown file in `data/v1/reports/` is fine.

## 9. Layout and versioning
```
data/v1/{sources.yaml, manifest.json, corpus/, splits/, lm/, sft/, rag/, eval/, reports/}
```
`manifest.json`: spec version, build date, git-less code hash (sha256 of `src/retrievve/data/*.py`), config sha256,
`frozen_slices`, `slices`, `religious_gate_passed`, per-file line counts and sha256. Build timestamps live in
`manifest.json` only, so two builds of the same inputs give byte-identical data files. Parameters: `configs/v1_pipeline.yaml`.
Existing `data/corpus` and `data/train` stay untouched as v0 (for comparison with the adapters already trained on them).
**Canonical output** is the shared Dropbox `data/v1` (`make data` defaults `DATA_OUT` to it). A worktree build never silently
writes a second "live" copy. Builds stage into `data/v1.staging` and promote only after the freeze guard passes.
**Environment:** one venv per worktree (`uv sync`). Never run project code through another checkout's editable install.
Every run records the resolved `retrievve.__file__`, the git sha and a dirty flag, and dirty runs are not reportable.

## 10. Provenance / release flag
**Default: private until confirmed** (user decision, 2026-10-10). Every source is `release: false` unless the user has
confirmed its license. As of 2026-10-10 only `ls1910_nt` (public domain) is `release: true`. Copyrighted sources (Galley 1964,
the L'Harmattan preview, the Fang NT PDF, the Punu Bible 1992, the "Parlons yipunu" book and its derivatives, Raponda-Walker)
carry their license line. Unknown-license web sources say "unknown, pending user confirmation", and hand-made transcriptions
wait for individual confirmation. `release: false` data may be used for internal training and eval, and may be stored in a
**private** HF dataset. It must never appear in a public dataset, a released adapter or paper examples. Consequently,
**all target-language adapters are internal-only** until sources are confirmed.
`release` propagates via `source_id` to SFT/LM/RAG meta and to `mlx/RELEASE.<lang>.json`. **Frozen eval and probe rows do not
carry `release`:** provenance for them is `eval/RELEASE.<lang>.json` (id → source_ids, release, license), regenerated on every
build, never frozen, with its sha in `manifest.release_sidecars`. A license decision therefore never changes a frozen test file.
Every build prints and records the `release: true` source list (`build_report.release_true_sources`).

## 10b. Origin: synthetic and user-interaction data (signed off in principle by the lead, 2026-10-10)
PLAN.md foresees Claude-generated augmentation and a RAG "data flywheel". For a **preservation** corpus,
model-generated text must never pass as attested language.
- Every record carries `origin`: `human` (attested, from a source), `synthetic` (generated by any model, even if
  post-edited), or `user_interaction` (from app logs).
- **Test, val and every eval reference: `origin: human` only.** Synthetic data may go into train only, and is
  reported as its own share of the SFT mix. It is capped at 30% of the SFT examples until E1 shows its effect.
- **Release** (the Hugging Face dataset, released adapters, paper examples): `synthetic` only if `verified_by` is set
  by a native speaker. Unverified synthetic data is `release: false` regardless of its source.
- **`user_interaction`:** collected only with the user's explicit consent mechanism in the app, and stripped of
  personal data. It is never used for training until that mechanism exists, and is `release: false` by default.
- Synthetic generation records its generator (model id), prompt template id and seed in `extra.gen`.

## 6b. RAG embeddings
Embed `embed_text` built from `text`/`fr`, which are normalized and **keep their tones**. Never embed from `text_key`,
`text_fold` or the v0 `text_norm`. If the v0 `*.vecs.npy` files turn out to come from `text_norm`, they are rebuilt on
v1 `rag/` records.

## 11. Baselines (from the v0 audit, 2026-10-10)
- All v0 runs: `mlx-community/Qwen2.5-7B-Instruct-4bit`, LoRA r=16, scale 20, 16 layers, lr 1e-5, bs 4, max_seq 512.
- `models/fang-mlx-lora` was overwritten. Its current `adapters.safetensors` is a 200-iteration SFT from the base model.
  The long Fang run survives only as checkpoint `0002200` (Jun 27). Treat it as a separate artifact; do not delete it.
- v0 had no test split, so **no v0 number counts as a test result**. All baselines (base model zero-shot, base + RAG,
  v0 adapters) are re-evaluated on the v1 frozen test set.
- v0 LM was **not** truncated: `punu-mlx-lm` has one verse per line (n=8,927, p50 44 tokens, max 232). The
  truncation risk applied only to the v1 `chunk_chars` rule, which token-based packing now fixes. Unpacked v0 LM
  batches were mostly very short sequences; v1 packing fixes that too. (Measured 2026-10-10 by applied maths.)
- Passes over the data in v0 (iters × bs / n_train): fang-mlx 0.38 (stopped at 2,200 of 12,000), fang-mlx-sft 0.03,
  punu-mlx-lm 0.09, punu-mlx-sft 0.17. v0 scores therefore measure **undertraining**, not the method.
- From v1 on, every training run writes to its own `models/<run_id>/` directory. No shared adapter paths.

## Changelog
- v0.9 (2026-10-10): §3 Fang NT test=REV / val=HEB, NT book acceptance (official --strict run plus an independent second check for
  name-poor books; bar never lowered); freeze guard extensions (silent de-freeze, vacuous pass, `.promoting`, verified schema
  transitions; sidecars only after the first E0 result); chapter-level low_align only with ≥ 10 names, DEU 7 unflagged; §9
  canonical output and per-worktree environments; §10 private-until-confirmed default, the release sidecar for frozen eval files,
  adapters internal-only. Sources: the lead's and data scientist's reviews, the user's licensing decisions.
- v0.8 (2026-10-10, signed off): §2 `schwa_untoned` tag and case-preserving Ǝ→Ə; §3 manifest format (flat bool `frozen_slices` +
  `slices` + top-level `religious_gate_passed`; consumers never glob), **freeze guard**, gate command on the new package
  path, DEU 1–2 kept / DEU 7 flagged; §4.2b 2-token precondition for the vocabulary detector; §4.6 OCR gate on a fresh
  sample, Galley linked examples excluded everywhere; §7b probe export schema (fold_neighbours diagnostics only).
- v0.7 (2026-10-10, signed off, superseded by v0.8): §7b lexicon probe (frozen ids in the manifest; definition-gloss share reported) (puu 400 / fan 200, excluded from SFT lex in both directions, kept in RAG, no-RAG
  systems only; EM with tones kept + tone-blind diagnostic; McNemar). Proposed by applied maths.
- v0.6 (2026-10-10, signed off): §3 Fang NT excluded until realigned (acceptance: proper-noun agreement ≥ 0.8 per book; no
  stopgap; Fang religious added later as its own frozen slice); §2 orthography tag `nt_ipa` with a separate `variety`
  field, and apostrophe rules (elision only between letters; Punu ʼ, Fang '); §4 thresholds fixed from the
  data-scientist sweep (fr_in_target stopword + vocabulary detectors, len_ratio [0.4, 2.5] with a 15-char guard,
  percentile low_align dropped); §3 per-slice freeze gate (alignment check before any religious slice freezes); §5b both directions in one split, v0 valid leak documented.
- v0.5 (2026-10-10, signed off): §1 `origin` and `verified_by`; §2 mandatory per-source orthography tags (Fang `galley1964` vs
  `nt_modern`; no unification without a signed mapping) and the all-caps headword rule; §3 parallel-only book
  choice and tail-chapter fallback (Punu test = DEU tail, val = GEN tail); §5b no upsampling of lexical/grammar;
  §7 per-slice n and descriptive-only reporting for tiny eval sets; §10b origin policy. Sources: the data engineer's
  first full build, the applied-maths orthography and casing analysis, and the lead's sign-off on origin.
- v0.4 (2026-10-10, approved): §11 corrected (v0 LM not truncated; v0 passes per run); §1 `fr_alt`/`fr_note` and the per-source `fr_parens` rule for proverbs; §3 short-segment guard
  on cross-split leak removal; §4 per-language stopword list in config; §5a tail-chunk rule (merge if it fits, else keep; never drop); SFT over-length rate per
  task × language at 512/1024, with > 2% at 512 triggering a 1024 comparison. From the lead's review of PR #2 and the data engineer's dry run (PR #4).
- v0.3 (2026-10-10, signed off): §5a greedy whole-segment packing with derived `chunk_tokens_max`, SFT over-length exclusion
  (never truncate the assistant turn); §8 token stats per language × orthography, no lossy respelling; §11 v0 LM
  truncation estimate. Source: applied-maths tokenizer measurements (n=1, to confirm).
- v0.2 (2026-10-10, signed off): §5a chunking in model tokens; §7 paired bootstrap; §6b embedding source; `text_key` vs `text_fold` and which dedup uses which; lexical/grammar never held out, with example
  leak removal. Raised by the data engineer.
- v0.1 (2026-10-10): joint cross-language splits for multi-parallel sources; `text_fold` replaces the deprecated
  `text_norm`; ISO codes; §11 baselines and adapter-path rule. Credit: data scientist audit.
- v0 (2026-10-09): initial.
