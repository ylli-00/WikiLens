#!/usr/bin/env python3
"""Recall of the vector index over a transaction's own uncommitted rows, on a scaled database.

The ``uncommitted`` group of ``scripts/run_experiments.py`` (``measure_uncommitted``), with the
chunk vectors and claims of an already loaded scaled database instead of the 100-page corpus:
a scratch table with the chunk table's vector column and index is filled inside one
transaction and searched before COMMIT, rolled back, filled again, committed and searched again,
at ``mhnsw_ef_search`` 20, 100 and 400, against NumPy's exact top k. Each dataset is built three
times. The second dataset is the group's random vectors (1,000 rows, seed 1), queried with as
many random vectors as there are claims.

The scratch table lives in the test database (``WIKILENSE_TEST_DB_NAME``) and is dropped at the
end; the scaled database is only read.

Usage: .venv/bin/python scripts/scaled_uncommitted.py --database wikilense_s0 --out <folder>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import run_experiments as rx

from wikilense import db
from wikilense.config import load_settings
from wikilense.embedding import Embedder

ALLOWED_DATABASES = ("wikilense_s0", "wikilense_s1")
K = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database", required=True, choices=ALLOWED_DATABASES)
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("--k", type=int, default=K, help=f"hits per query (default {K})")
    args = parser.parse_args(argv)
    target = args.out / f"{rx.UNCOMMITTED}.json"
    if target.exists():
        raise SystemExit(f"{target} exists; give each run its own folder")
    settings = load_settings()
    if settings.test_db_name in ALLOWED_DATABASES:
        raise SystemExit("the test database must not be a scaled corpus")

    source = db.connect(settings, database=args.database)
    try:
        with source.cursor() as cur:
            cur.execute(rx.CHUNK_VECTORS_SQL)
            chunk_vectors = np.stack([db.vec_from_bytes(row[0]) for row in cur.fetchall()])
            cur.execute(rx.CLAIM_TEXTS_SQL)
            claim_texts = [str(row[0]) for row in cur.fetchall()]
        meta = rx.read_ingest_meta(source)
        index = rx.index_definition(source)
    finally:
        source.close()
    embedder = Embedder(settings.embedding_model)
    claim_vectors = np.asarray(embedder.embed_queries(claim_texts, batch_size=256),
                               dtype=np.float32)
    rng = np.random.default_rng(rx.UNCOMMITTED_RANDOM_SEED)
    random_vectors = rng.standard_normal(
        (rx.UNCOMMITTED_RANDOM_ROWS, db.VECTOR_DIM)).astype(np.float32)
    random_queries = rng.standard_normal((len(claim_texts), db.VECTOR_DIM)).astype(np.float32)
    chunk_label = (f"the {len(chunk_vectors)} chunk embeddings of {args.database}, queried "
                   f"with its {len(claim_texts)} claim embeddings")
    random_label = (f"{rx.UNCOMMITTED_RANDOM_ROWS} random normal vectors (seed "
                    f"{rx.UNCOMMITTED_RANDOM_SEED}), queried with {len(claim_texts)} random vectors")
    datasets = {
        "chunks": (chunk_vectors, claim_vectors, chunk_label),
        "random": (random_vectors, random_queries, random_label),
    }

    started = time.monotonic()
    conn = db.connect(settings, database=settings.test_db_name)
    results = {}
    try:
        for name, (vectors, queries, label) in datasets.items():
            builds = []
            for build in range(1, rx.UNCOMMITTED_BUILDS + 1):
                print(f"{datetime.now(timezone.utc):%H:%M:%S} UTC {name}: build {build} of "
                      f"{rx.UNCOMMITTED_BUILDS}", flush=True)
                builds.append(rx.measure_uncommitted(conn, vectors, queries, k=args.k))
            results[name] = {"label": label, "builds": builds}
    finally:
        try:
            if conn.open:
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute(rx.UNCOMMITTED_DROP_SQL)
                conn.commit()
        finally:
            conn.close()
    data = {
        "name": rx.UNCOMMITTED, "group": "uncommitted", "database": args.database,
        "scratch_database": settings.test_db_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - started, 1),
        "k": args.k, "efs": list(rx.UNCOMMITTED_EFS), "builds": rx.UNCOMMITTED_BUILDS,
        "mariadb_version": meta.get("mariadb_version"), "chunk_index": index,
        "datasets": results,
    }  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / f"{rx.UNCOMMITTED}.md").write_text(
        rx.uncommitted_markdown(data, source="`scripts/scaled_uncommitted.py`"), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
