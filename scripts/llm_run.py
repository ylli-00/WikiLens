"""Run the LLM fact-checker on a claims file through the Claude subscription (``claude -p``).

Run from the repository root, with the pinned CLI copy in ``CLAUDE_BIN``. A probe::

    CLAUDE_BIN=/path/to/claude CUDA_VISIBLE_DEVICES= .venv/bin/python \
        scripts/llm_run.py --runs-root /path/to/llm_runs --run-id probe_smoke \
        --claims /path/to/llm_runs/probe_claims.jsonl --prompt generic_probe_v0 \
        --models haiku --effort low --limit-claims 3

``--dry-run`` plans and prints without any call or database query. See
``wikilense/llm/runner.py`` for the run folder, the resume rules and the exit codes, and
``wikilense/llm/NOTES.md`` for the design.
"""

from __future__ import annotations

import sys

from wikilense.llm.runner import main

if __name__ == "__main__":
    sys.exit(main())
