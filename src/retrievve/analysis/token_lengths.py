"""Token-length report for the training tokenizer (DATA_SPEC v0.2 §5a, §11).

Needs `tokenizers` (pip install tokenizers) and the tokenizer.json of the base model, by default from the
Hugging Face cache. Run from the repo root:
    retrievve token-lengths corpus data/corpus/fang*.jsonl   # chars/token p5/p50/p95, text and fr
    retrievve token-lengths lm  data/train/punu-lm.train.jsonl  # share of LM tokens seen per max_seq
    retrievve token-lengths sft data/train/fang-sft.train.jsonl # rendered SFT lengths, exclusions

Counts are on NFC text, without special tokens, except `sft` which renders the Qwen2.5 ChatML template.
`corpus` also reports the NFD token count: NFD adds codepoints, so chars/token must be read on NFC.
Files that are still 0-byte Dropbox placeholders are listed as `unhydrated` and skipped.
"""
from __future__ import annotations

import argparse
import collections
import json
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
HF_HUB = Path.home() / ".cache/huggingface/hub"


def load_tokenizer(model: str) -> Tokenizer:
    snaps = sorted((HF_HUB / f"models--{model.replace('/', '--')}" / "snapshots").glob("*/tokenizer.json"))
    if not snaps:
        raise SystemExit(f"tokenizer.json for {model} not found under {HF_HUB}")
    return Tokenizer.from_file(str(snaps[-1]))


def pct(xs: list[float], q: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


def read_rows(paths: list[Path], unhydrated: list[str]):
    for p in paths:
        if p.stat().st_size == 0:
            unhydrated.append(str(p))
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def chatml(messages: list[dict]) -> str:
    # Qwen2.5 chat template for plain-text turns (what mlx-lm renders via apply_chat_template)
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


class Counter:
    def __init__(self, tok: Tokenizer):
        self.tok = tok

    def __call__(self, s: str, form: str = "NFC") -> int:
        return len(self.tok.encode(unicodedata.normalize(form, s), add_special_tokens=False).ids)


def report_corpus(rows, n, group_by: str = "orthography") -> dict:
    # per field, and per `group_by` value (IPA letters ɛ ɔ ŋ cost 2 tokens each in Qwen2.5)
    acc = collections.defaultdict(lambda: {"ratios": [], "chars": 0, "toks": 0, "toks_nfd": 0})
    for r in rows:
        for field in ("text", "fr"):
            s = r.get(field)
            if not (isinstance(s, str) and s.strip()):
                continue
            s = unicodedata.normalize("NFC", s)
            k = n(s)
            for key in (field, f"{field}|{group_by}={r.get(group_by) or 'none'}"):
                a = acc[key]
                a["ratios"].append(len(s) / k)
                a["chars"] += len(s)
                a["toks"] += k
                a["toks_nfd"] += n(s, "NFD")
    return {
        key: {
            "n": len(a["ratios"]),
            "chars_per_token": round(a["chars"] / a["toks"], 3),
            **{f"p{int(q * 100)}": round(pct(a["ratios"], q), 3) for q in (0.05, 0.5, 0.95)},
            "nfd_extra_tokens_frac": round(a["toks_nfd"] / a["toks"] - 1, 4),
        }
        for key, a in sorted(acc.items())
    }


def report_lm(rows, n, seqs) -> dict:
    lens = [n(r["text"]) for r in rows if r.get("text")]
    tot = sum(lens)
    if not tot:
        return {"n_chunks": 0}
    return {
        "n_chunks": len(lens), "tokens": tot, "p50": pct(lens, 0.5), "p95": pct(lens, 0.95), "max": max(lens),
        **{f"share_seen@{L}": round(sum(min(k, L) for k in lens) / tot, 3) for L in seqs},
        **{f"chunks_truncated@{L}": sum(k > L for k in lens) for L in seqs},
    }


def report_sft(rows, n, seqs) -> dict:
    full, asst = [], []
    for r in rows:
        msgs = r["messages"]
        full.append(n(chatml(msgs)))
        asst.append(sum(n(m["content"]) + 2 for m in msgs if m["role"] == "assistant"))  # + <|im_end|>\n
    if not full:
        return {"n": 0}
    return {
        "n": len(full), "p50": pct(full, 0.5), "p95": pct(full, 0.95), "p99": pct(full, 0.99), "max": max(full),
        **{f"excluded@{L}": sum(x > L for x in full) for L in seqs},
        **{f"excluded_frac@{L}": round(sum(x > L for x in full) / len(full), 4) for L in seqs},
        "assistant_token_share": round(sum(asst) / sum(full), 3),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["corpus", "lm", "sft"])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-seq", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--group-by", default="orthography", help="corpus mode: record field to break down by")
    args = ap.parse_args(argv)

    n = Counter(load_tokenizer(args.model))
    unhydrated: list[str] = []
    rows = read_rows(args.paths, unhydrated)
    if args.mode == "corpus":
        out = report_corpus(rows, n, args.group_by)
    elif args.mode == "lm":
        out = report_lm(rows, n, args.max_seq)
    else:
        out = report_sft(rows, n, args.max_seq)
    print(json.dumps({"model": args.model, "mode": args.mode, "report": out, "unhydrated": unhydrated},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
