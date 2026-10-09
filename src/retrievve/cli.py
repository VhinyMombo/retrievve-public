"""`retrievve` command line entry point.

    retrievve build-data [--config configs/v1_pipeline.yaml] [--out data/v1]
    retrievve stats      [--out data/v1/reports]
    retrievve train      --config configs/train/<name>.yaml      (v1 training: not implemented yet)
    retrievve eval       ...                                      (v1 eval: not implemented yet)
    retrievve serve      ...                                      (v1 API: not implemented yet)
"""
from __future__ import annotations

import sys

COMMANDS = {
    "build-data": ("retrievve.data.build", "Build data/v1 from raw sources (DATA_SPEC)."),
    "stats": ("retrievve.analysis.corpus_stats", "DATA_SPEC §8 stats report and threshold sweeps."),
    "align-check": ("retrievve.analysis.align_check", "Proper-noun alignment check (Bible acceptance gate)."),
    "token-lengths": ("retrievve.analysis.token_lengths", "Token-length report for the training tokenizer."),
    "train": (None, "Fine-tune from configs/train/ (v1: not implemented yet)."),
    "eval": (None, "Evaluate on data/v1/eval (v1: not implemented yet)."),
    "serve": (None, "RAG assistant API (v1: not implemented yet)."),
}


def _usage() -> str:
    lines = ["usage: retrievve <command> [args]", "", "commands:"]
    lines += [f"  {name:<14} {help_}" for name, (_, help_) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}\n\n{_usage()}", file=sys.stderr)
        return 2
    module, _ = COMMANDS[cmd]
    if module is None:
        print(f"`retrievve {cmd}` is not implemented yet for v1.",
              file=sys.stderr)
        return 1
    import importlib

    mod = importlib.import_module(module)
    sys.argv = [f"retrievve {cmd}", *rest]
    result = mod.main(rest)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
