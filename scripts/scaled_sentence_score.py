"""The FEVEROUS-style evidence score of the scaled run (owner's decisions #4a and #4b, 2026-11-11).

FEVEROUS lets a system return at most 5 sentences and 25 table cells or list items; a claim's
evidence counts only when a complete gold evidence set is among them. Our own score counts a
claim when a complete set lies inside the top-k chunks (about 7 units per chunk), which is far
easier. Decided: report both; pick the units by embedding each one and ranking it against the
claim, only the units of each claim's top 10 chunks, each unit embedded once.

Per claim of the database (its ``claim`` and ``claim_evidence`` tables, as the evaluation harness
reads them with ``evaluate.load_ground_truth``):

1. the claim embedded with ``Embedder.embed_queries`` (the BGE query instruction), then
   ``search.search(conn, qvec, k=10, strategy="none", ef_search=100)``: the plain vector-index
   search of the main table;
2. the candidates: every text unit (``sentence`` row: a sentence ``sentence_N`` or a list item
   ``item_N_M``) of those 10 chunks, once each;
3. each distinct unit embedded once with ``Embedder.embed_passages`` on the text the chunks are
   embedded on (``chunking.embedding_text``: page title and section path, then the unit's text);
4. selection ``embed``: the 5 sentences and the 25 list items with the highest dot product with the
   claim (vectors are L2-normalised, so this is the cosine); ties go to the earlier unit in hit
   order. Selection ``chunk_order`` (the free alternative of #4b, for comparison): the first 5
   sentences and 25 items in hit order;
5. a claim is found when one of its sentence-only evidence sets is a subset of the selection.
   Eligible claims are those with at least one sentence-only set (the harness's denominator);
   ``in_top_chunks`` (a complete set inside the 10 chunks) is the upper bound of both selections
   and equals our evidence recall@10.

Tables stay out (decision #8), so cells never count and a set with a cell is not sentence-only.

Writes ``<out_dir>/<database>/``: ``claims.jsonl`` (one record per claim), ``units.npz`` (the unit
ids and their vectors), ``summary.json`` and ``SENTENCE_SCORE.md`` (overall and the six groups,
label x one-article / needs-two, from the claims file's ``kind``).

Usage (from the repository root, on the GPU, under the laptop setting and the thermal guard):
    .venv/bin/python scripts/scaled_sentence_score.py --database wikilense_s1 \
        --claims <scaled_data>/claims_3000.jsonl --out-dir <scaled_data>/sentence_score
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ALLOWED_DATABASES = ("wikilense_s0", "wikilense_s1")
K = 10
EF_SEARCH = 100
STRATEGY = "none"
N_SENTENCES = 5
N_ITEMS = 25
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def log(message: str) -> None:
    print(f"{datetime.now().astimezone():%H:%M:%S} {message}", file=sys.stderr, flush=True)


def unit_kind(element_key: str) -> str:
    """Return ``sentence`` or ``item`` for a ``sentence`` row's element key."""
    if element_key.startswith("sentence_"):
        return "sentence"
    if element_key.startswith("item_"):
        return "item"
    raise ValueError(f"unknown text unit {element_key!r}")


def select(units: Sequence[tuple[int, str]], scores: Sequence[float] | None,
           n_sentences: int = N_SENTENCES, n_items: int = N_ITEMS) -> list[int]:
    """Return the selected unit ids: the best ``n_sentences`` sentences and ``n_items`` items.

    ``units`` are (unit id, kind) in hit order, without duplicates. With ``scores`` (one per unit)
    the highest scores win, ties to the earlier unit; without, hit order decides.
    """
    order = list(range(len(units)))
    if scores is not None:
        order.sort(key=lambda i: (-float(scores[i]), i))
    taken = {"sentence": 0, "item": 0}
    limit = {"sentence": n_sentences, "item": n_items}
    picked = []
    for i in order:
        uid, kind = units[i]
        if taken[kind] < limit[kind]:
            taken[kind] += 1
            picked.append(uid)
    return picked


def found(sets: Sequence[set[int]], selected: Sequence[int]) -> bool:
    """Return True when one evidence set lies entirely inside ``selected``."""
    chosen = set(selected)
    return any(s and s <= chosen for s in sets)


def rate(hits: int, n: int) -> float | None:
    return round(hits / n, 4) if n else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return overall and per-group counts and rates of the three scores."""
    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [r for r in rows if r["eligible"]]
        n = len(eligible)
        out: dict[str, Any] = {"claims": len(rows), "eligible": n}
        for name in ("in_top_chunks", "found_embed", "found_chunk_order"):
            hits = sum(1 for r in eligible if r[name])
            out[name] = {"hits": hits, "rate": rate(hits, n)}
        return out
    groups = {f"{label} / {kind}": block([r for r in records
                                          if r["label"] == label and r["kind"] == kind])
              for label in LABELS for kind in ("single", "needs2")}
    return {"overall": block(records), "groups": groups}


def markdown(database: str, summary: dict[str, Any], params: dict[str, Any]) -> str:
    """Return SENTENCE_SCORE.md."""
    def cell(b: dict[str, Any], name: str) -> str:
        v = b[name]
        return "-" if v["rate"] is None else f"{v['rate']:.3f} ({v['hits']}/{b['eligible']})"
    lines = [f"# FEVEROUS-style evidence score, `{database}`", "",
             (f"Written by `scripts/scaled_sentence_score.py` on {params['finished_at']}. "
             f"Top {K} chunks per claim (`search.search`, strategy `{STRATEGY}`, ef_search "
             f"{EF_SEARCH}); the units of those chunks ranked by embedding against the claim; "
             f"best {N_SENTENCES} sentences + {N_ITEMS} list items; found = one complete "
             "sentence-only gold set among them. Eligible = claims with a sentence-only set."), "",
             (f"- distinct units embedded: {params['n_units']} in {params['embed_seconds']:.1f} s "
             f"({params['device']}); candidates per claim: mean "
             f"{params['mean_candidates']:.1f} sentences + items"), "",
             ("| claims | eligible | complete set in top 10 chunks (ours) | 5 + 25 by embedding "
             "(FEVEROUS-style) | 5 + 25 in chunk order |"), "|---|---|---|---|---|"]
    rows = [("all", summary["overall"])] + list(summary["groups"].items())
    for name, b in rows:
        lines.append(f"| {name} ({b['claims']}) | {b['eligible']} | {cell(b, 'in_top_chunks')} "
                     f"| {cell(b, 'found_embed')} | {cell(b, 'found_chunk_order')} |")
    lines += ["", ("Rates are over the eligible claims of each row. The overall row is the raw "
              "count over the 3,000 claims (not reweighted to the FEVEROUS population).")]
    return "\n".join(lines) + "\n"


def run(database: str, claims_path: Path, out_dir: Path, batch_size: int) -> int:
    from wikilense import db, search
    from wikilense.chunking import embedding_text
    from wikilense.config import load_settings
    from wikilense.embedding import Embedder
    from wikilense.evaluate import load_ground_truth

    if database not in ALLOWED_DATABASES:
        raise SystemExit(f"refusing database {database!r}: only {ALLOWED_DATABASES}")
    started = time.monotonic()
    kinds = {}
    with claims_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            kinds[int(r["id"])] = r["kind"]
    settings = load_settings()
    conn = db.connect(settings, database=database)
    try:
        truth = load_ground_truth(conn)
        log(f"{database}: {len(truth)} claims")
        embedder = Embedder(settings.embedding_model)
        qvecs = embedder.embed_queries([t.text for t in truth], batch_size=batch_size)
        hits_of: dict[int, list[Any]] = {}
        for t, qvec in zip(truth, qvecs, strict=True):
            hits_of[t.claim_id] = search.search(conn, qvec, k=K, strategy=STRATEGY,
                                                ef_search=EF_SEARCH, query_text=t.text)
        log(f"searched {len(hits_of)} claims in {time.monotonic() - started:.0f} s")
        chunk_ids = sorted({h.chunk_id for hits in hits_of.values() for h in hits})
        units_of_chunk: dict[int, list[int]] = defaultdict(list)
        unit_info: dict[int, tuple[str, str]] = {}
        with conn.cursor() as cur:
            for start in range(0, len(chunk_ids), 1000):
                part = chunk_ids[start:start + 1000]
                marks = ",".join(["%s"] * len(part))
                cur.execute(
                    "SELECT cs.chunk_id, s.sentence_id, s.element_key, s.text "
                    "FROM chunk_sentence cs JOIN sentence s ON s.sentence_id = cs.sentence_id "
                    f"WHERE cs.chunk_id IN ({marks}) ORDER BY cs.chunk_id, s.page_id, s.ordinal",
                    part)
                for chunk_id, sid, key, text in cur.fetchall():
                    units_of_chunk[int(chunk_id)].append(int(sid))
                    unit_info[int(sid)] = (unit_kind(str(key)), str(text))
    finally:
        conn.rollback()
        conn.close()
    # the text to embed: as the chunk embeds it (title > section path: text)
    unit_text: dict[int, str] = {}
    for hits in hits_of.values():
        for h in hits:
            for sid in units_of_chunk[h.chunk_id]:
                unit_text.setdefault(sid, embedding_text(h.title, h.section_path, unit_info[sid][1]))
    unit_ids = sorted(unit_text)
    log(f"{len(unit_ids)} distinct units in {len(chunk_ids)} chunks; embedding")
    t0 = time.monotonic()
    vectors = embedder.embed_passages([unit_text[u] for u in unit_ids], batch_size=batch_size)
    embed_seconds = time.monotonic() - t0
    row_of = {u: i for i, u in enumerate(unit_ids)}
    target = out_dir / database
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "units.npz", unit_ids=np.array(unit_ids, dtype=np.int64), vectors=vectors)
    records = []
    n_candidates = []
    for t, qvec in zip(truth, qvecs, strict=True):
        units: list[tuple[int, str]] = []
        seen: set[int] = set()
        for h in hits_of[t.claim_id]:
            for sid in units_of_chunk[h.chunk_id]:
                if sid not in seen:
                    seen.add(sid)
                    units.append((sid, unit_info[sid][0]))
        scores = vectors[[row_of[u] for u, _ in units]] @ qvec if units else []
        by_embed = select(units, list(scores))
        by_order = select(units, None)
        n_candidates.append(len(units))
        records.append({
            "claim_id": t.claim_id, "label": t.label, "kind": kinds.get(t.claim_id),
            "eligible": bool(t.sentence_only_sets), "n_sets": len(t.sentence_only_sets),
            "n_candidates": len(units),
            "n_candidate_sentences": sum(1 for _, k in units if k == "sentence"),
            "in_top_chunks": found(t.sentence_only_sets, [u for u, _ in units]),
            "found_embed": found(t.sentence_only_sets, by_embed),
            "found_chunk_order": found(t.sentence_only_sets, by_order),
            "selected_embed": by_embed,
            "hit_chunk_ids": [h.chunk_id for h in hits_of[t.claim_id]],
        })
    with (target / "claims.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    import torch

    params = {"database": database, "k": K, "strategy": STRATEGY, "ef_search": EF_SEARCH,
              "n_sentences": N_SENTENCES, "n_items": N_ITEMS, "batch_size": batch_size,
              "embedding_model": settings.embedding_model,
              "embedding_revision": embedder.revision_label,
              "device": "cuda" if torch.cuda.is_available() else "cpu",
              "claims_file": str(claims_path), "n_units": len(unit_ids),
              "n_chunks": len(chunk_ids), "embed_seconds": round(embed_seconds, 1),
              "mean_candidates": float(np.mean(n_candidates)) if n_candidates else 0.0,
              "candidates_hist": dict(sorted(Counter(n_candidates).items())),
              "total_seconds": round(time.monotonic() - started, 1),
              "finished_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    summary = {"params": params, **summarize(records)}
    (target / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (target / "SENTENCE_SCORE.md").write_text(markdown(database, summary, params),
                                              encoding="utf-8")
    o = summary["overall"]
    log(f"done {database}: eligible {o['eligible']}, in top chunks {o['in_top_chunks']}, "
        f"embed {o['found_embed']}, chunk order {o['found_chunk_order']}, "
        f"{params['total_seconds']} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--database", required=True, choices=ALLOWED_DATABASES)
    p.add_argument("--claims", required=True, type=Path, help="claims_3000.jsonl (for 'kind')")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args(argv)
    return run(args.database, args.claims, args.out_dir, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
