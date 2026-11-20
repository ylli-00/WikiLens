#!/usr/bin/env python3
"""Values for the four metadata filters on a scaled database, picked by one rule.

The filters of ``wikilense query`` (page length, edit date, category, linked from) need values
that fit the corpus they run on: the values used on the 100-page corpus keep almost every chunk
of a scaled database, or almost none. This script picks the values on one loaded database by a
fixed rule, counts the chunks each value keeps, and writes both to a JSON file (and a Markdown
table) that ``scripts/measure_index_behaviour.py --filters`` and
``scripts/compare_separate_index.py --filters`` read.

The rule
    Four target shares of the chunk table: 50%, 5%, 0.5% and 0.05% (loose to very selective).
    For each filter and each target, the value whose number of passing chunks is nearest to
    the target on a log scale (``|log(passing / target)|``); ties go to the larger count, then
    to the first value in sort order.

    - length: ``min_words``, a page length in words; a chunk passes when its page has at least
      that many words.
    - edit date: ``edited_since``, a moment to the second (UTC); a chunk passes when its page
      was last edited at or after it. Pages without an edit date pass no date filter. Whole
      days are too coarse: the pages carry their revision of late December 2020, and the last
      day alone holds several percent of the chunks.
    - category: one category name; a chunk passes when its page is in that category.
    - linked from: one page title; a chunk passes when that page links to the chunk's page.

    No category and no linking page reaches 50% of the chunks, so for those two filters the
    loosest level is the largest one there is. When two targets pick the same value, the case
    is kept once, under the looser target.

    ``four_filters`` sets all four at once: length and edit date at their 50% values, and the
    category and the linking page whose pair keeps the most chunks together with those two
    (searched over every category and every linking page). ``none`` is the statement without a
    filter.

The counts are computed in Python from the page, chunk, category and link tables, with the
semantics of the SQL filters in ``wikilense.search``; ``measure_index_behaviour.py join_order``
counts the passing chunks again through the statement itself and stops when the two differ.

Usage: .venv/bin/python scripts/scaled_filters.py --database wikilense_s1 --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wikilense.search import Filters

TARGETS: tuple[float, ...] = (0.5, 0.05, 0.005, 0.0005)
FILTER_FIELDS: dict[str, str] = {
    "length": "min_words",
    "edited": "edited_since",
    "category": "category",
    "linked": "linked_from",
}
COMBINED = "four_filters"

PAGES_SQL = "SELECT page_id, title, n_words, last_edited FROM page"
CHUNKS_PER_PAGE_SQL = "SELECT page_id, COUNT(*) FROM chunk GROUP BY page_id"
CATEGORY_PAGES_SQL = (
    "SELECT category.name, page_category.page_id FROM page_category "
    "JOIN category ON category.category_id = page_category.category_id"
)
LINKS_SQL = (
    "SELECT DISTINCT link.from_page_id, link.to_page_id FROM link "
    "WHERE link.to_page_id IS NOT NULL"
)


def nearest(candidates: Iterable[tuple[Any, int]], target: float) -> tuple[Any, int]:
    """Return the ``(value, passing)`` whose ``passing`` is nearest ``target`` on a log scale.

    Candidates with no passing chunk are skipped. Ties go to the larger count, then to the
    first candidate in the order given.
    """
    best: tuple[float, int, int] | None = None
    chosen: tuple[Any, int] | None = None
    for position, (value, passing) in enumerate(candidates):
        if passing <= 0:
            continue
        key = (abs(math.log(passing / target)), -passing, position)
        if best is None or key < best:
            best, chosen = key, (value, passing)
    if chosen is None:
        raise ValueError("no candidate keeps any chunk")
    return chosen


def threshold_candidates(values: np.ndarray, weights: np.ndarray) -> list[tuple[int, int]]:
    """For every distinct integer value ``v``, the summed weight of the rows with a value >= ``v``.

    Returned in ascending order of ``v``: the loosest threshold first.
    """
    order = np.argsort(values, kind="stable")
    sorted_values, sorted_weights = values[order], weights[order]
    distinct, first = np.unique(sorted_values, return_index=True)
    at_or_above = np.cumsum(sorted_weights[::-1])[::-1]
    return [(int(distinct[i]), int(at_or_above[first[i]])) for i in range(len(distinct))]


def utc_seconds(value: datetime) -> int:
    """A stored ``last_edited`` value (naive, UTC) as whole seconds since the epoch."""
    return int(value.replace(tzinfo=timezone.utc).timestamp())


def from_utc_seconds(seconds: int) -> datetime:
    """The naive UTC ``datetime`` of :func:`utc_seconds`."""
    return datetime.fromtimestamp(seconds, timezone.utc).replace(tzinfo=None)


def load(conn: Any) -> dict[str, Any]:
    """Read the tables the four filters use."""
    with conn.cursor() as cur:
        cur.execute(PAGES_SQL)
        pages = {int(r[0]): {"title": str(r[1]), "n_words": int(r[2]), "last_edited": r[3]}
                 for r in cur.fetchall()}
        cur.execute(CHUNKS_PER_PAGE_SQL)
        chunks = {int(r[0]): int(r[1]) for r in cur.fetchall()}
        cur.execute(CATEGORY_PAGES_SQL)
        categories: dict[str, set[int]] = defaultdict(set)
        for name, page_id in cur.fetchall():
            categories[str(name)].add(int(page_id))
        cur.execute(LINKS_SQL)
        links: dict[int, set[int]] = defaultdict(set)
        for source, target in cur.fetchall():
            links[int(source)].add(int(target))
    return {"pages": pages, "chunks": chunks, "categories": dict(categories),
            "links": dict(links)}


def chunks_of(pages: Iterable[int], chunks: dict[int, int]) -> int:
    return sum(chunks.get(page_id, 0) for page_id in pages)


def pick(data: dict[str, Any]) -> dict[str, Any]:
    """Apply the rule to the loaded tables; return the cases with their passing chunks."""
    pages, chunks = data["pages"], data["chunks"]
    n_chunks = sum(chunks.values())
    page_ids = np.asarray(sorted(pages), dtype=np.int64)
    weights = np.asarray([chunks.get(int(p), 0) for p in page_ids], dtype=np.int64)
    words = np.asarray([pages[int(p)]["n_words"] for p in page_ids], dtype=np.int64)
    dated = [int(p) for p in page_ids if pages[int(p)]["last_edited"] is not None]
    moments = np.asarray([utc_seconds(pages[p]["last_edited"]) for p in dated], dtype=np.int64)
    moment_weights = np.asarray([chunks.get(p, 0) for p in dated], dtype=np.int64)
    title_of = {p: info["title"] for p, info in pages.items()}

    candidates: dict[str, list[tuple[Any, int]]] = {
        "length": threshold_candidates(words, weights),
        "edited": [(from_utc_seconds(moment), passing)
                   for moment, passing in threshold_candidates(moments, moment_weights)],
        "category": sorted(
            ((name, chunks_of(members, chunks)) for name, members in data["categories"].items()),
            key=lambda item: item[0],
        ),
        "linked": sorted(
            ((title_of[source], chunks_of(targets, chunks))
             for source, targets in data["links"].items() if source in title_of),
            key=lambda item: item[0],
        ),
    }
    cases: list[dict[str, Any]] = [{
        "name": "none", "filter": None, "target_share": None, "filters": {},
        "passing_chunks": n_chunks, "passing_share": 1.0,
    }]
    chosen: dict[str, dict[float, Any]] = {name: {} for name in FILTER_FIELDS}
    for name, field_name in FILTER_FIELDS.items():
        seen: set[Any] = set()
        for target in TARGETS:
            value, passing = nearest(candidates[name], target * n_chunks)
            chosen[name][target] = value
            if value in seen:
                continue
            seen.add(value)
            cases.append({
                "name": f"{name}_{target * 100:g}pct", "filter": name, "target_share": target,
                "filters": {field_name: value.isoformat() if isinstance(value, date) else value},
                "passing_chunks": passing, "passing_share": round(passing / n_chunks, 6),
            })

    # four_filters: length and date at their 50% values, then the best category / source pair.
    min_words = chosen["length"][TARGETS[0]]
    since = chosen["edited"][TARGETS[0]]
    base = {p for p, info in pages.items()
            if info["n_words"] >= min_words and info["last_edited"] is not None
            and info["last_edited"] >= since}
    categories_of: dict[int, list[str]] = defaultdict(list)
    for category, members in data["categories"].items():
        for page_id in members:
            categories_of[page_id].append(category)
    best: tuple[int, str, str] | None = None
    for source in sorted((s for s in data["links"] if s in title_of), key=title_of.__getitem__):
        reachable = data["links"][source] & base
        if not reachable or (best is not None and chunks_of(reachable, chunks) <= best[0]):
            continue
        per_category: dict[str, int] = defaultdict(int)
        for page_id in reachable:
            for category in categories_of.get(page_id, ()):
                per_category[category] += chunks.get(page_id, 0)
        for category in sorted(per_category):
            passing = per_category[category]
            if passing > 0 and (best is None or passing > best[0]):
                best = (passing, category, title_of[source])
    if best is None:
        raise ValueError("no category and linking page keep a chunk together")
    passing, category, linked_from = best
    cases.append({
        "name": COMBINED, "filter": COMBINED, "target_share": None,
        "filters": {"category": category, "min_words": min_words,
                    "edited_since": since.isoformat(), "linked_from": linked_from},
        "passing_chunks": passing, "passing_share": round(passing / n_chunks, 6),
    })
    return {"n_chunks": n_chunks, "n_pages": len(pages), "targets": list(TARGETS),
            "cases": cases}


def filters_of(spec: dict[str, Any]) -> Filters:
    """The ``Filters`` of one case's ``filters`` mapping (dates as ISO strings)."""
    values = dict(spec)
    for key in ("edited_since", "edited_before"):
        if key in values:
            text = values[key]
            values[key] = date.fromisoformat(text) if len(text) == 10 else datetime.fromisoformat(text)
    return Filters(**values)


def load_cases(path: Path) -> tuple[tuple[str, Filters, int], ...]:
    """Read a file written by this script: ``(name, Filters, passing_chunks)`` per case."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple((case["name"], filters_of(case["filters"]), int(case["passing_chunks"]))
                 for case in data["cases"])


def describe(filters: Filters) -> dict[str, Any]:
    """The fields a ``Filters`` sets, dates as ISO strings (for reports)."""
    return {key: (value.isoformat() if isinstance(value, date) else value)
            for key, value in asdict(filters).items() if value is not None}


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Filter values on {report['database']}",
        "",
        (f"Written {report['created_at']} by `scripts/scaled_filters.py`: {report['n_chunks']:,} "
         f"chunks on {report['n_pages']:,} pages. Rule (the script's docstring has it in full): "
         "per filter, the value whose passing chunks are nearest, on a log scale, to "
         + ", ".join(f"{t * 100:g}%" for t in report["targets"])
         + " of the chunk table; `four_filters` sets length and edit date at their 50% values "
         "and the category and linking page whose pair keeps the most chunks with them."),
        "",
        "| Case | Target share | Filter value | Chunks passing | Share |",
        "|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        target = "-" if case["target_share"] is None else f"{case['target_share'] * 100:g}%"
        value = "; ".join(f"{k} = {v}" for k, v in case["filters"].items()) or "no filter"
        lines.append(f"| {case['name']} | {target} | {value.replace('|', '/')} | "
                     f"{case['passing_chunks']:,} | {case['passing_share'] * 100:.3f}% |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database", required=True, help="the loaded database to read")
    parser.add_argument("--out", required=True, type=Path, help="JSON file; a .md is written next to it")
    args = parser.parse_args(argv)
    from wikilense import db
    from wikilense.config import load_settings

    settings = load_settings()
    if args.database == settings.test_db_name:
        raise SystemExit("refusing to read the test database")
    conn = db.connect(settings, database=args.database)
    try:
        report = {"database": args.database,
                  "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  **pick(load(conn))}
    finally:
        conn.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
