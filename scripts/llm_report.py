"""Print the measured summary of one LLM run folder (Markdown).

    .venv/bin/python scripts/llm_report.py /path/to/llm_runs/<run_id> [--examples 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wikilense.llm.report import summarize


def main(argv: list[str] | None = None) -> int:
    """Print the summary of the given run folder."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--examples", type=int, default=2, help="claims whose chunks are listed")
    args = p.parse_args(argv)
    print(summarize(args.run_dir, args.examples, cache_root=args.run_dir.parent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
