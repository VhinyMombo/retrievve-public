"""Derived artifacts from canonical records only (DATA_SPEC §5–7). Nothing is built ad hoc."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .common import stable_hash
from .normalize import text_fold, text_key
from .quality import usable, usable_rag

PARALLEL_KINDS = {"verse", "sentence", "dialogue_turn", "example", "poem_line", "song_line"}

TEMPLATES = {
    "mt_fr2x": [
        "Traduis en {L} : {s}", "Traduis cette phrase en {L} :\n{s}", "Comment dit-on en {L} : « {s} » ?",
        "Donne la traduction en {L} de : {s}", "Phrase en français : {s}\nTraduction en {L} :",
        "Peux-tu traduire en {L} ?\n{s}", "Rends ce texte en {L} : {s}", "Français → {L} : {s}",
    ],
    "mt_x2fr": [
        "Traduis en français : {s}", "Que signifie cette phrase en {L} ?\n{s}",
        "Donne la traduction française de : {s}", "Phrase en {L} : {s}\nTraduction en français :",
        "Peux-tu traduire du {L} vers le français ?\n{s}", "Que veut dire « {s} » ?",
        "Rends ce texte {L} en français : {s}", "{L} → français : {s}",
    ],
    "lex_x2fr": [
        "Que signifie le mot {L} « {s} » ?", "Donne le sens en français de « {s} » ({L}).",
        "Définis le mot {L} : {s}", "Quel est le sens de « {s} » en {L} ?", "Mot {L} : {s}\nSens :",
        "Traduis ce mot {L} en français : {s}",
    ],
    "lex_fr2x": [
        "Comment dit-on « {s} » en {L} ?", "Quel est le mot {L} pour « {s} » ?",
        "Donne le mot {L} correspondant à : {s}", "Traduis ce mot en {L} : {s}", "Français : {s}\nMot {L} :",
        "En {L}, comment dit-on {s} ?",
    ],
    "dialogue": [
        "Voici le début d'une conversation en {L}. Donne la réplique suivante.\n{s}",
        "Continue ce dialogue en {L} :\n{s}", "Réponds en {L} à cette réplique :\n{s}",
        "Conversation en {L} :\n{s}\nRéplique suivante :", "Quelle pourrait être la réponse en {L} ?\n{s}",
        "Poursuis la conversation en {L}.\n{s}",
    ],
    "proverb_meaning": [
        "Que veut dire ce proverbe {L} ?\n{s}", "Explique le sens de ce proverbe {L} : {s}",
        "Quel est l'équivalent en français de ce proverbe {L} : {s}", "Proverbe {L} : {s}\nSens :",
        "Donne la signification de : {s}", "Interprète ce proverbe {L} : {s}",
    ],
    "grammar_qa": [
        "Explique cette règle de grammaire {L} : {s}", "En grammaire {L}, que faut-il savoir sur : {s} ?",
        "Décris le point de grammaire {L} suivant : {s}", "Grammaire {L} — {s} : explique.",
        "Peux-tu m'expliquer « {s} » en {L} ?", "Qu'est-ce que « {s} » dans la grammaire {L} ?",
    ],
}


def _pick(task: str, rid: str) -> str:
    t = TEMPLATES[task]
    return t[stable_hash(task, rid) % len(t)]


def _lex_answer(r: dict) -> str:
    lex = r.get("lex") or {}
    parts = [r["fr"]]
    if lex.get("pos"):
        parts.append(f"({lex['pos']})")
    if lex.get("noun_class"):
        parts.append(f"classe {lex['noun_class']}")
    if lex.get("plural"):
        parts.append(f"pluriel : {lex['plural']}")
    return " ".join(p for p in parts if p)


def sft_examples(records: list[dict], lang: str, split: str, cfg: dict,
                 exclude_lex_keys: set[str] | None = None) -> list[dict]:
    sc = cfg["sft"]
    L = sc["lang_names"].get(lang, lang)
    system = sc["system_prompt"].format(lang_name=L)
    rs = [r for r in records if r["lang"] == lang and r["split"] == split and usable(r)
          and (split == "train" or r.get("origin", "human") == "human")]                # v0.5: val human only
    out = []

    def ex(r, task, user, answer):
        return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                             {"role": "assistant", "content": answer}],
                "meta": {"src_id": r["id"], "task": task, "domain": r["domain"], "release": r["release"]}}

    fr2x_by_gloss = defaultdict(list)
    for r in rs:
        if not r["fr"] or not r["text"]:
            continue
        if sc.get("proverb_meaning") and (r.get("fr_alt") or r.get("fr_note")):
            # optional task (v0.4): x -> idiom/explanation; never used as an MT reference
            out.append(ex(r, "proverb_meaning", _pick("proverb_meaning", r["id"]).format(L=L, s=r["text"]),
                          r.get("fr_alt") or r["fr_note"]))
        if r["kind"] in PARALLEL_KINDS:
            # both directions from the same pair => exact 50/50 balance (§5b)
            out.append(ex(r, "mt_fr2x", _pick("mt_fr2x", r["id"]).format(L=L, s=r["fr"]), r["text"]))
            out.append(ex(r, "mt_x2fr", _pick("mt_x2fr", r["id"]).format(L=L, s=r["text"]), r["fr"]))
        elif r["kind"] == "lexicon":
            if exclude_lex_keys and r["text_key"] in exclude_lex_keys:
                continue                                          # lexicon probe headword (v0.7 §7b)
            out.append(ex(r, "lex_x2fr", _pick("lex_x2fr", r["id"]).format(L=L, s=r["text"]), _lex_answer(r)))
            fr2x_by_gloss[text_key(r["fr"])].append(r)
        elif r["kind"] == "grammar":
            # grammar rows: (example/rule in target, French explanation); QA generation needs the rule
            # chunk structure — TODO once fang/grammar/*.csv is readable. Hand-check sample required (§5b).
            out.append(ex(r, "grammar_qa", _pick("grammar_qa", r["id"]).format(L=L, s=r["text"]), r["fr"]))

    # lex_fr2x: headwords sharing a gloss are merged into one example listing all of them (N5), so the same
    # prompt never has conflicting targets.
    for _gloss, ents in sorted(fr2x_by_gloss.items()):
        ents = sorted(ents, key=lambda r: r["id"])
        heads = list(dict.fromkeys(e["text"] for e in ents))
        e0 = ents[0]
        item = ex(e0, "lex_fr2x", _pick("lex_fr2x", e0["id"]).format(L=L, s=e0["fr"]), " ; ".join(heads))
        if len(heads) > 1:
            item["meta"]["merged_headwords"] = len(heads)
            item["meta"]["release"] = all(e["release"] for e in ents)
        out.append(item)

    # dialogue: previous turn -> next turn, within a dialogue group, in order. The context turn may carry quality
    # flags, but never xsplit_dup (held-out text must not appear in train prompts); the target turn must be usable.
    by_group = defaultdict(list)
    for r in records:
        if r["lang"] == lang and r["split"] == split and r["kind"] == "dialogue_turn" and r["text"]:
            by_group[r["group_id"]].append(r)
    for turns in by_group.values():
        turns.sort(key=lambda r: (r["source_id"], r["order"]))
        for prev, nxt in zip(turns, turns[1:], strict=False):
            # context may carry quality flags (len_ratio, ...) but never held-out text (B3)
            if usable(nxt) and prev["source_id"] == nxt["source_id"] and "xsplit_dup" not in prev["flags"]:
                out.append(ex(nxt, "dialogue", _pick("dialogue", nxt["id"]).format(L=L, s=prev["text"]),
                              nxt["text"]))
    return out


def apply_domain_cap(examples: list[dict], cap: float, max_up: int,
                     upsample_domains: set[str] | None = None) -> tuple[list[dict], dict]:
    """religious <= cap: upsample ONLY `upsample_domains` (v0.5: conversational, literary, oral; lexical and
    grammar stay x1) up to x max_up, then downsample religious. Deterministic."""
    up_doms = upsample_domains if upsample_domains is not None else {"conversational", "literary", "oral"}
    rel = [e for e in examples if e["meta"]["domain"] == "religious"]
    non = [e for e in examples if e["meta"]["domain"] != "religious"]
    elig = [e for e in non if e["meta"]["domain"] in up_doms]
    fixed = [e for e in non if e["meta"]["domain"] not in up_doms]
    before = {"religious": len(rel), "other": len(non), "upsample_eligible": len(elig)}
    if not rel or len(rel) <= cap * len(examples):
        return examples, {"before": before, "after": before, "upsample": 1.0, "religious_kept": len(rel)}
    ratio = cap / (1 - cap)                                   # religious allowed = ratio * other
    need_other = len(rel) / ratio
    up = min(max_up, max(1.0, (need_other - len(fixed)) / max(len(elig), 1))) if elig else 1.0
    whole, frac = int(up), up - int(up)
    elig_up = elig * whole + [e for e in elig if stable_hash("up", e["meta"]["src_id"], e["meta"]["task"]) % 1000
                              < frac * 1000]
    non_up = fixed + elig_up
    max_rel = int(ratio * len(non_up))
    # keep/drop per source record, so both directions of a verse stay together (N1)
    by_src = defaultdict(list)
    for e in rel:
        by_src[e["meta"]["src_id"]].append(e)
    rel_kept = []
    for sid in sorted(by_src, key=lambda sid: stable_hash("rel", sid)):
        if len(rel_kept) + len(by_src[sid]) > max_rel:
            break
        rel_kept.extend(by_src[sid])
    out = sorted(non_up + rel_kept, key=lambda e: stable_hash("order", e["meta"]["src_id"], e["meta"]["task"]))
    after = {"religious": len(rel_kept), "other": len(non_up), "upsampled_domains": sorted(up_doms)}
    return out, {"before": before, "after": after, "upsample": round(up, 3), "religious_kept": len(rel_kept)}


# ------------------------------------------------------------------------------------------------ LM
def _hf_snapshot(name: str):
    from pathlib import Path
    base = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{name.replace('/', '--')}" / "snapshots"
    snaps = sorted(base.glob("*/tokenizer.json")) if base.exists() else []
    return snaps[-1].parent if snaps else None


class TokenCounter:
    """Pluggable (spec v0.2): base-model tokenizer from the local HF cache (`tokenizers` + jinja2 chat
    template, no transformers needed), else chars / chars_per_token."""

    def __init__(self, cfg: dict):
        lc = cfg["lm"]
        self.cpt = float(lc["chars_per_token"])
        self.tok, self.template, self.desc = None, None, f"chars/{self.cpt}"
        self.approx = True                                    # True until a real tokenizer is loaded (B4)
        name = lc.get("tokenizer")
        snap = _hf_snapshot(name) if name else None
        if snap is None:
            if name:
                self.desc += f" (tokenizer {name} not in local HF cache)"
            return
        try:
            import json

            from tokenizers import Tokenizer  # noqa: PLC0415
            self.tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
            self.desc = f"tokenizer:{name}"
            self.approx = False
            tc = json.loads((snap / "tokenizer_config.json").read_text())
            if tc.get("chat_template"):
                import jinja2  # noqa: PLC0415
                env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
                self.template = env.from_string(tc["chat_template"])
                self.desc += " + chat_template"
        except Exception as e:                                # pragma: no cover - environment dependent
            self.tok, self.approx = None, True
            self.desc = f"chars/{self.cpt} (tokenizer {name} failed: {type(e).__name__})"

    def __call__(self, s: str) -> int:
        if self.tok is not None:
            return len(self.tok.encode(s, add_special_tokens=False).ids)
        return int(len(s) / self.cpt + 0.999)

    def chat(self, messages: list[dict]) -> int:
        """Token count of a fully rendered chat example."""
        if self.template is not None:
            return self(self.template.render(messages=messages, add_generation_prompt=False))
        # fallback: ~ChatML overhead of 5 tokens per message
        return sum(self(m["content"]) + 5 for m in messages)


def lm_budget(cfg: dict) -> tuple[int, int]:
    lc = cfg["lm"]
    return int(lc["max_seq_length"] * lc["max_fill"]), int(lc["chunk_tokens_min"])


_SENT = re.compile(r"(?<=[.!?;:…»])\s+")


def _split_long(text: str, count, budget: int) -> list[str]:
    """Split a single over-budget segment at sentence punctuation (then at spaces as a last resort)."""
    parts, cur = [], ""
    for piece in _SENT.split(text):
        cand = f"{cur} {piece}".strip()
        if cur and count(cand) > budget:
            parts.append(cur)
            cur = piece
        else:
            cur = cand
    if cur:
        parts.append(cur)
    out = []
    for p in parts:
        if count(p) <= budget:
            out.append(p)
            continue
        words, buf = p.split(), []
        for w in words:
            if buf and count(" ".join(buf + [w])) > budget:
                out.append(" ".join(buf))
                buf = []
            buf.append(w)
        if buf:
            out.append(" ".join(buf))
    return out


def lm_chunks(records: list[dict], lang: str, split: str, count, budget: int, min_tok: int = 0) -> list[dict]:
    """Target-language text only. Greedy packing of whole consecutive segments of one group (never across
    groups, never shuffled); a segment is split only if it alone exceeds the budget (spec v0.2 §5a)."""
    rs = [r for r in records if r["lang"] == lang and r["split"] == split and usable(r) and r["text"]
          and r["kind"] not in ("lexicon", "grammar")]
    by_group = defaultdict(list)
    for r in rs:
        by_group[r["group_id"]].append(r)
    out = []
    for g in sorted(by_group):
        segs = sorted(by_group[g], key=lambda r: (r["source_id"], r["order"]))
        buf, n, rel = [], 0, True
        for r in segs:
            pieces = [r["text"]] if count(r["text"]) <= budget else _split_long(r["text"], count, budget)
            for piece in pieces:
                t = count(piece) + 1                          # +1 for the newline joiner
                if buf and n + t > budget:
                    out.append({"text": "\n".join(buf), "n_tokens": n, "group_id": g, "release": rel})
                    buf, n, rel = [], 0, True
                buf.append(piece)
                n += t
                rel = rel and r["release"]
        if buf:
            # tail rule (v0.4): a short final chunk merges into the previous one of the same group if it fits;
            # otherwise it stays short. Never dropped.
            prev = out[-1] if out and out[-1]["group_id"] == g else None
            if n < min_tok and prev and prev["n_tokens"] + n <= budget:
                prev["text"] += "\n" + "\n".join(buf)
                prev["n_tokens"] += n
                prev["release"] = prev["release"] and rel
            else:
                out.append({"text": "\n".join(buf), "n_tokens": n, "group_id": g, "release": rel})
    return out


# ------------------------------------------------------------------------------------------------ RAG
def rag_chunks(records: list[dict], lang: str, cfg: dict) -> list[dict]:
    """Train + val (+ rag_pool). Never test (§6). embed_text keeps tones; never built from keys (v0.2)."""
    rs = [r for r in records if r["lang"] == lang and r["split"] in ("train", "val", "rag_pool") and usable_rag(r)]
    out = []
    # lexicon: one chunk per headword (merge entries sharing text_key)
    heads = defaultdict(list)
    for r in rs:
        if r["kind"] == "lexicon" and r["text"]:
            heads[r["text_key"]].append(r)
    pairs = [r for r in rs if r["kind"] in PARALLEL_KINDS and r["text"] and r["fr"]]
    by_id = {r["id"]: r for r in pairs}
    # index example sentences by text_key tokens, then text_fold tokens as fallback (v0.2 lookup order)
    tok_key, tok_fold = defaultdict(list), defaultdict(list)
    for r in pairs:
        for t in set(r["text_key"].split()):
            tok_key[t].append(r)
        for t in set(r["text_fold"].split()):
            tok_fold[t].append(r)
    k = cfg["rag"]["max_examples_per_headword"]
    for key in sorted(heads):
        ents = heads[key]
        head = ents[0]["text"]
        glosses = list(dict.fromkeys(e["fr"] for e in ents if e["fr"]))
        lex = next((e["lex"] for e in ents if e.get("lex")), {}) or {}
        exs = []
        linked = [by_id[x] for e in ents for x in (e.get("lex") or {}).get("examples", []) if x in by_id]
        if linked:
            exs = sorted(linked, key=lambda r: (len(r["text"]), r["id"]))[:k]
        elif len(key.split()) == 1:
            exs = tok_key.get(key) or tok_fold.get(text_fold(head)) or []
            exs = sorted(exs, key=lambda r: (len(r["text"]), r["id"]))[:k]
        info = [f"{head}"]
        if lex.get("pos"):
            info.append(f"({lex['pos']})")
        if lex.get("noun_class"):
            info.append(f"classe {lex['noun_class']}")
        if lex.get("plural"):
            info.append(f"pl. {lex['plural']}")
        display = " ".join(info) + " : " + " ; ".join(glosses)
        marker = next((e["rag_marker"] for e in ents if e.get("rag_marker")), None)
        if marker:
            display = f"{marker} {display}"
        if exs:
            display += "\nExemples : " + " | ".join(f"{e['text']} — {e['fr']}" for e in exs)
        out.append({"id": f"rag:{ents[0]['id']}", "lang": lang, "type": "lexicon",
                    "embed_text": " ; ".join(glosses) + " — " + head, "display": display,
                    "src_ids": [e["id"] for e in ents] + [e["id"] for e in exs],
                    "source_id": ents[0]["source_id"], "release": all(e["release"] for e in ents + exs),
                    "ocr": any(e["ocr"] for e in ents + exs)})
    # grammar: one chunk per group, split by size, heading = group title
    lo, hi = cfg["rag"]["grammar_chunk_chars"]
    gram = defaultdict(list)
    for r in rs:
        if r["kind"] == "grammar":
            gram[r["group_id"]].append(r)
    for g in sorted(gram):
        rows = sorted(gram[g], key=lambda r: r["order"])
        heading = g.replace("_", " ")
        blocks, buf, ids = [], [], []
        for r in rows:
            line = f"{r['text']} — {r['fr']}" if r["fr"] else r["text"]
            if buf and sum(len(b) for b in buf) + len(line) > hi:
                blocks.append((buf, ids))
                buf, ids = [], []
            buf.append(line)
            ids.append(r["id"])
        if buf:
            if blocks and sum(len(b) for b in buf) < lo:      # short tail joins the previous block (N6)
                blocks[-1][0].extend(buf)
                blocks[-1][1].extend(ids)
            else:
                blocks.append((buf, ids))
        for lines, bids in blocks:
            body = "\n".join(lines)
            out.append({"id": f"rag:{bids[0]}", "lang": lang, "type": "grammar",
                        "embed_text": f"{heading}\n{body}", "display": f"[{heading}]\n{body}",
                        "src_ids": list(bids), "source_id": rows[0]["source_id"],
                        "release": all(x["release"] for x in rows)})
    for r in pairs:
        extra = "".join(f" ({x})" for x in (r.get("fr_alt"), r.get("fr_note")) if x)
        if r.get("rag_marker"):
            extra += f" {r['rag_marker']}"
        out.append({"id": f"rag:{r['id']}", "lang": lang, "type": "example_pair",
                    "embed_text": f"{r['fr']}\n{r['text']}", "display": f"{r['text']} — {r['fr']}{extra}",
                    "src_ids": [r["id"]], "source_id": r["source_id"], "release": r["release"], "ocr": r["ocr"]})
    return out


# ----------------------------------------------------------------------------------------------- eval
def eval_sets(records: list[dict], lang: str) -> dict[str, list[dict]]:
    rs = [r for r in records if r["lang"] == lang and r["split"] == "test" and usable(r)
          and r.get("origin", "human") == "human"
          and r["kind"] in PARALLEL_KINDS and r["text"] and r["fr"]]
    out = defaultdict(list)
    for r in sorted(rs, key=lambda r: r["id"]):
        dom = "religious" if r["domain"] == "religious" else "general"
        base = {"domain": r["domain"], "group_id": r["group_id"]}   # release lives in eval/RELEASE.<lang>.json (v0.9)
        out[f"mt_fr2x.{dom}"].append({"id": r["id"], "src": r["fr"], "ref": r["text"], "direction": "fr2x", **base})
        out[f"mt_x2fr.{dom}"].append({"id": r["id"], "src": r["text"], "ref": r["fr"], "direction": "x2fr", **base})
    return dict(out)


def mix_report(examples: list[dict]) -> dict:
    return {"by_task": dict(Counter(e["meta"]["task"] for e in examples)),
            "by_domain": dict(Counter(e["meta"]["domain"] for e in examples)),
            "release_false": sum(not e["meta"]["release"] for e in examples)}
