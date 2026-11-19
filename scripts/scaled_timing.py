"""The light timing of the scaled run: fast vs slow searches.

``scripts/run_experiments.py``'s ``scale`` protocol (``measure_scale``: the HNSW search, the exact
search, and the History-filtered search inline and with overfetch 10; per claim and statement one
untimed run, then 3 timed ones; ``mhnsw_ef_search`` = the chosen 100) on every claim of an
already loaded database, at k = 5, the headline k of the scaled run, instead of its built-in 10.
The full timing of the fast search at every k is ``wikilense eval`` (the main harness).

Run every timing under the same laptop setting (Turbo, CPU 50%, GPU 2,100 MHz, thermal guard),
alone, and compare sizes only within the scaled run.

Usage: .venv/bin/python scripts/scaled_timing.py --database wikilense_s1 --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from wikilense.config import REPO_ROOT

ALLOWED_DATABASES = ("wikilense_s0", "wikilense_s1")
K = 5


def experiments_module():
    """Return scripts/run_experiments.py as a module, its scale k set to :data:`K`."""
    spec = importlib.util.spec_from_file_location("run_experiments",
                                                  REPO_ROOT / "scripts" / "run_experiments.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_experiments"] = module   # its dataclasses look the module up when defined
    spec.loader.exec_module(module)
    module.SCALE_K = K
    return module


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--database", required=True, choices=ALLOWED_DATABASES)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; give each run its own file")
    from wikilense import db
    from wikilense.config import load_settings
    from wikilense.embedding import Embedder
    from wikilense.evaluate import load_ground_truth

    rx = experiments_module()
    settings = load_settings()
    conn = db.connect(settings, database=args.database)
    try:
        truth = load_ground_truth(conn)
        embedder = Embedder(settings.embedding_model)
        queries = embedder.embed_queries([t.text for t in truth], batch_size=256)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunk")
            n_chunks = int(cur.fetchone()[0])
        started = time.monotonic()
        measured = rx.measure_scale(conn, queries, [t.claim_id for t in truth],
                                    {t.claim_id: set(t.gold_titles) for t in truth})
        seconds = time.monotonic() - started
    finally:
        conn.rollback()
        conn.close()
    with open("/sys/devices/system/cpu/intel_pstate/max_perf_pct", encoding="ascii") as fh:
        cpu_cap = int(fh.read().strip())
    result = {"database": args.database, "k": K, "repeats": rx.SCALE_REPEATS,
              "ef_search": rx.EF_CHOSEN, "heading_filter": rx.HEADING_FILTER,
              "n_claims": len(truth), "n_chunks": n_chunks, "cpu_max_perf_pct": cpu_cap,
              "seconds": round(seconds, 1),
              "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "statements": measured}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
