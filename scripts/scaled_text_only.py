#!/usr/bin/env python3
"""How many FEVEROUS train + dev claims have no text-only evidence set (no database).

A set is text-only when it holds only sentences and list items, as ``describe()`` in
``scripts/scaled_build.py`` decides for the claim pick; WikiLense does not index tables, so only
claims with such a set can be measured. Reads ``data/feverous/feverous_train_challenges.jsonl``
and ``feverous_dev_challenges.jsonl`` (the first line of each is a header and is skipped) and
prints, or writes with ``--out``, the counts as JSON.

Usage: .venv/bin/python scripts/scaled_text_only.py [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import scaled_build

FILES = ("feverous_train_challenges.jsonl", "feverous_dev_challenges.jsonl")


def count(folder: Path) -> dict[str, float | int]:
    counts = {"claims": 0, "no_text_only_set": 0}
    for name in FILES:
        with open(folder / name, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i == 0:  # the header line the official reader skips
                    continue
                _, any_text, _, _ = scaled_build.describe(json.loads(line))
                counts["claims"] += 1
                counts["no_text_only_set"] += not any_text
    return {**counts, "share": round(counts["no_text_only_set"] / counts["claims"], 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--feverous-dir", type=Path, default=REPO_ROOT / "data" / "feverous")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = {"files": list(FILES), **count(args.feverous_dir)}
    text = json.dumps(result, indent=1) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
