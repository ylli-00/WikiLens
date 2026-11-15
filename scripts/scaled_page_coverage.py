"""Page coverage of the scaled run, as FEVEROUS's Table 4 reports document retrieval (decision #4c).

The paper (arXiv 2106.05707, Table 4) gives "document coverage" at the top 1, 2, 3 and 5 pages
(0.39, 0.49, 0.58, 0.69 on the dev set) without a formula. The only definition is the baseline
code, ``page_coverage()`` in ``src/feverous/baseline/retriever/eval_doc_retriever.py`` of
github.com/Raldir/FEVEROUS (read 2026-11-15): per claim, the share of the claim's distinct gold
pages (the union over all its evidence sets and evidence types) among the first predicted pages,
then the mean over claims; it prints the mean over SUPPORTS and REFUTES claims and the mean over
all claims. (That code scores only the first 3 pages whatever k was asked for; here every p is
scored.)

Our retriever returns chunks, so the predicted pages of a claim are the distinct pages of its
top ``N_CHUNKS`` chunks in rank order (``search.search``, strategy ``none``, ef_search 100).
Reported for p = 1, 2, 3, 5, 10: ``coverage`` (the FEVEROUS definition) and ``complete`` (every
gold page among the first p, the share of claims), each over all claims and over SUPPORTS +
REFUTES; plus how many claims had fewer than p distinct pages in their chunks.

Usage: .venv/bin/python scripts/scaled_page_coverage.py --database wikilense_s1 --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ALLOWED_DATABASES = ("wikilense_s0", "wikilense_s1")
N_CHUNKS = 50
EF_SEARCH = 100
PAGES = (1, 2, 3, 5, 10)


def distinct_pages(page_ids: Sequence[int]) -> list[int]:
    """Return the page ids in order of first appearance."""
    return list(dict.fromkeys(page_ids))


def coverage(gold: set[int], predicted: Sequence[int], p: int) -> float:
    """Return the share of ``gold`` among the first ``p`` predicted pages (FEVEROUS baseline)."""
    if not gold:
        raise ValueError("a claim without gold pages has no coverage")
    return len(gold & set(predicted[:p])) / len(gold)


def summarize(rows: list[tuple[str, set[int], list[int]]]) -> dict[str, Any]:
    """Return coverage and complete rates per p, over all claims and over SUPPORTS + REFUTES.
    ``rows`` are (label, gold pages, predicted distinct pages); claims without gold pages are
    left out (and counted)."""
    usable = [r for r in rows if r[1]]
    out: dict[str, Any] = {"claims": len(rows), "without_gold_pages": len(rows) - len(usable)}
    for name, subset in (("all", usable),
                         ("supports_refutes", [r for r in usable if r[0] != "NOT ENOUGH INFO"])):
        block: dict[str, Any] = {"n": len(subset)}
        for p in PAGES:
            covs = [coverage(gold, pred, p) for _, gold, pred in subset]
            block[f"p{p}"] = {
                "coverage": round(sum(covs) / len(covs), 4) if covs else None,
                "complete": round(sum(1 for c in covs if c == 1.0) / len(covs), 4) if covs else None,
                "fewer_than_p_pages": sum(1 for _, _, pred in subset if len(pred) < p),
            }
        out[name] = block
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--database", required=True, choices=ALLOWED_DATABASES)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    from wikilense import db, search
    from wikilense.config import load_settings
    from wikilense.embedding import Embedder
    from wikilense.evaluate import load_ground_truth

    settings = load_settings()
    conn = db.connect(settings, database=args.database)
    try:
        truth = load_ground_truth(conn)
        qvecs = Embedder(settings.embedding_model).embed_queries([t.text for t in truth],
                                                                  batch_size=256)
        rows = []
        for t, qvec in zip(truth, qvecs, strict=True):
            hits = search.search(conn, qvec, k=N_CHUNKS, strategy="none", ef_search=EF_SEARCH,
                                 query_text=t.text)
            rows.append((t.label, set(t.gold_pages), distinct_pages([h.page_id for h in hits])))
    finally:
        conn.rollback()
        conn.close()
    result = {"database": args.database, "n_chunks": N_CHUNKS, "ef_search": EF_SEARCH,
              "strategy": "none", "pages": list(PAGES),
              "definition": "FEVEROUS baseline eval_doc_retriever.page_coverage: per claim "
              "|gold pages in the first p predicted pages| / |gold pages|, gold pages pooled over "
              "all evidence sets; mean over claims",
              "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              **summarize(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
