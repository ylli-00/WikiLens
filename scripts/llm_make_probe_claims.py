"""Write the probe claims file: 3 claims per label from the 75-claim corpus, a fixed seed.

Run from the repository root::

    .venv/bin/python scripts/llm_make_probe_claims.py --out /path/to/llm_runs/probe_claims.jsonl

Writes ``--out`` (never overwrites) and ``<out>.meta.json`` with the seed, the source files'
sha256 and the picked ids. No database is needed. See ``wikilense/llm/probe.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wikilense.llm.claims import file_sha256
from wikilense.llm.probe import DEFAULT_SEED, build_probe_claims, write_jsonl

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Build the probe file and its description."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--claims", type=Path, default=REPO / "data/corpus/claims.jsonl")
    p.add_argument("--pages", type=Path, default=REPO / "data/corpus/pages.jsonl")
    p.add_argument("--per-label", type=int, default=3)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args(argv)
    records, info = build_probe_claims(args.claims, args.pages, args.per_label, args.seed)
    write_jsonl(args.out, records)
    info["out"] = str(args.out)
    info["out_sha256"] = file_sha256(args.out)
    meta = args.out.with_suffix(args.out.suffix + ".meta.json")
    with meta.open("x", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
        fh.write("\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
