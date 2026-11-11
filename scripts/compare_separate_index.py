#!/usr/bin/env python3
"""Where the database-native search loses: MariaDB against a separate index over the same vectors.

The README argues for keeping the vectors inside MariaDB. This script measures the other side
with the simplest separate index there is: a copy of the chunk vectors held in the memory of
the application, searched without a database round trip.

It reads the ingested ``chunk`` table once (the "export" a separate index needs), embeds the
corpus claims as queries, and answers every query with each method, ``k`` = 10:

- MariaDB, the bare k-NN statement (strategy ``none``): the HNSW index, ``mhnsw_ef_search`` 100.
- MariaDB, the joined statement (strategy ``inline``), with and without a filter.
- MariaDB, strategy ``overfetch`` 10, under a filter.
- NumPy: an exact scan of the copied vectors (one matrix-vector product), and the same scan
  restricted to the rows a filter keeps (a pre-filter). Measured alone, and followed by the
  ``SELECT`` that fetches the hit rows from MariaDB by primary key, which is what an application
  with a separate index still has to do to show a result.
- hnswlib (optional): an in-process HNSW graph with ``M`` 16, the library's default
  ``ef_construction`` 200 and ``ef`` 100, and its filter callback.

Every hit list is compared with the exact ranking (NumPy, float64, ties by ``chunk_id``). The
filters are evaluated on the copy in Python; before anything is timed the script checks that
the copy keeps exactly the chunks the SQL filter keeps, and stops if it does not.

hnswlib is not a dependency of WikiLense and is in no requirements or lock file. Without it
the hnswlib rows are left out:

    pip install hnswlib            # optional, needs a C++ compiler
    python scripts/compare_separate_index.py

The script only reads the database. Output: ``results/separate_index.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wikilense import db, evaluate
from wikilense import search as searchmod
from wikilense.config import load_settings
from wikilense.search import Filters

DEFAULT_K = 10
DEFAULT_REPEATS = 5
EF_SEARCH = 100
OVERFETCH = 10
HNSWLIB_M = 16
HNSWLIB_EF_CONSTRUCTION = 200  # the library's default
HNSWLIB_EF = 100
HNSWLIB_SEED = 100
DEFAULT_NAME = "separate_index"

# The export: everything the copy needs. Fixed statements, no parameters.
EXPORT_CHUNKS_SQL = (
    "SELECT chunk.chunk_id, chunk.page_id, section.heading, chunk.embedding "
    "FROM chunk JOIN section ON section.section_id = chunk.section_id "
    "ORDER BY chunk.chunk_id"
)
EXPORT_PAGES_SQL = "SELECT page_id, title, n_words, last_edited FROM page"
EXPORT_CATEGORIES_SQL = (
    "SELECT category.name, page_category.page_id FROM page_category "
    "JOIN category ON category.category_id = page_category.category_id"
)
EXPORT_LINKS_SQL = (
    "SELECT src.title, link.to_page_id FROM link "
    "JOIN page AS src ON src.page_id = link.from_page_id "
    "WHERE link.to_page_id IS NOT NULL"
)
#: The hit rows by primary key, as an application with a separate index fetches them. The only
#: text that varies is the number of ``%s`` placeholders in the IN list.
ROWS_SQL_PREFIX = (
    "SELECT chunk.chunk_id, chunk.page_id, page.title, section.path, chunk.ordinal, "
    "chunk.n_words, chunk.text "
    "FROM chunk JOIN page ON page.page_id = chunk.page_id "
    "JOIN section ON section.section_id = chunk.section_id "
    "WHERE chunk.chunk_id IN ("
)
ROWS_SQL_SUFFIX = ")"


@dataclass(frozen=True)
class FilterCase:
    """One filter, as the SQL ``Filters`` and as a description for the report."""

    key: str
    label: str
    filters: Filters


FILTER_CASES: tuple[FilterCase, ...] = (
    FilterCase(
        "history",
        "section heading LIKE '%History%'",
        Filters(heading_like="%History%"),
    ),
    FilterCase(
        "four_filters",
        "the brief's four filters: category 'Member states of the United Nations', at least "
        "1,000 words, edited since 2020-12-01, linked from 'Asia'",
        Filters(
            category="Member states of the United Nations",
            min_words=1000,
            edited_since=date(2020, 12, 1),
            linked_from="Asia",
        ),
    ),
)


@dataclass
class Copy:
    """The data a separate index holds: the vectors, and the columns the filters need.

    ``vectors`` has one unit-length float32 row per chunk, in ``chunk_ids`` order. The page and
    link dictionaries are the relational data the two filter cases read.
    """

    chunk_ids: np.ndarray
    page_ids: np.ndarray
    headings: list[str]
    vectors: np.ndarray
    page_words: dict[int, int] = field(default_factory=dict)
    page_edited: dict[int, datetime | None] = field(default_factory=dict)
    category_pages: dict[str, set[int]] = field(default_factory=dict)
    linked_from: dict[str, set[int]] = field(default_factory=dict)
    export_seconds: float = 0.0
    rows_exported: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.chunk_ids.shape[0])


# ---------------------------------------------------------------------------------------------
# pure functions (tested in tests/test_compare_separate_index.py)
# ---------------------------------------------------------------------------------------------


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with every row scaled to length 1 (rows of zeros stay zeros)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def top_k(
    vectors: np.ndarray, query: np.ndarray, ids: np.ndarray, k: int, rows: np.ndarray | None = None
) -> list[int]:
    """Return the ids of the ``k`` rows nearest to ``query`` by cosine distance, nearest first.

    ``vectors`` and ``query`` are unit length, so the distance is ``1 - dot``. ``rows``
    restricts the scan to those row numbers (a pre-filter); None scans every row. Ties are
    broken by the smaller id. Fewer than ``k`` candidate rows returns all of them.
    """
    candidates = vectors if rows is None else vectors[rows]
    candidate_ids = ids if rows is None else ids[rows]
    if candidate_ids.shape[0] == 0:
        return []
    distances = 1.0 - candidates @ query
    if candidate_ids.shape[0] > k:
        # Keep everything tied with the k-th distance, so the tie-break below is exact.
        kth = np.partition(distances, k - 1)[k - 1]
        keep = np.flatnonzero(distances <= kth)
    else:
        keep = np.arange(candidate_ids.shape[0])
    order = np.lexsort((candidate_ids[keep], distances[keep]))[:k]
    return [int(value) for value in candidate_ids[keep][order]]


def heading_mask(headings: Sequence[str], needle: str) -> np.ndarray:
    """Return, per heading, whether it contains ``needle``, ignoring case.

    This is the copy's version of ``section.heading LIKE '%needle%'``. MariaDB's collation
    also ignores accents; the script checks that both keep the same chunks before it trusts it.
    """
    folded = needle.casefold()
    return np.fromiter((folded in heading.casefold() for heading in headings), dtype=bool,
                       count=len(headings))


def page_mask(page_ids: np.ndarray, passing_pages: set[int]) -> np.ndarray:
    """Return, per chunk, whether its page is in ``passing_pages``."""
    if not passing_pages:
        return np.zeros(page_ids.shape[0], dtype=bool)
    return np.isin(page_ids, np.fromiter(passing_pages, dtype=page_ids.dtype))


def four_filter_pages(copy: Copy, filters: Filters) -> set[int]:
    """Return the pages that pass category, minimum length, edit date and "linked from".

    Only the fields the brief's four-filter case sets are read; any other set field is an
    error, so a new case cannot silently be evaluated wrongly.
    """
    unsupported = [
        name
        for name in ("max_words", "heading_like", "path_like", "links_to", "titles", "edited_before")
        if getattr(filters, name) is not None
    ]
    if unsupported:
        raise ValueError(f"the copy does not evaluate these filter fields: {unsupported}")
    pages = set(copy.page_words)
    if filters.category is not None:
        pages &= copy.category_pages.get(filters.category, set())
    if filters.linked_from is not None:
        pages &= copy.linked_from.get(filters.linked_from, set())
    if filters.min_words is not None:
        pages = {page for page in pages if copy.page_words[page] >= filters.min_words}
    if filters.edited_since is not None:
        # page.last_edited holds naive UTC datetimes; a date means its midnight UTC.
        day = filters.edited_since
        since = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).replace(tzinfo=None)
        pages = {
            page for page in pages
            if copy.page_edited.get(page) is not None and copy.page_edited[page] >= since
        }
    return pages


def copy_mask(copy: Copy, case: FilterCase) -> np.ndarray:
    """Return the boolean row mask of ``case`` evaluated on the copy."""
    if case.key == "history":
        return heading_mask(copy.headings, "History")
    return page_mask(copy.page_ids, four_filter_pages(copy, case.filters))


def compare_lists(result: Sequence[int], reference: Sequence[int]) -> tuple[float, bool, bool]:
    """Return (share of ``reference`` found in ``result``, same set, same order).

    An empty reference (no chunk passes) counts as fully found by an empty result.
    """
    if not reference:
        return (1.0 if not result else 0.0), not result, not result
    found = len(set(result) & set(reference)) / len(reference)
    return found, set(result) == set(reference), list(result) == list(reference)


def summarise(
    name: str,
    where: str,
    kind: str,
    samples_ms: Sequence[float],
    results: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Return one report row: latency of ``samples_ms`` and agreement with ``references``.

    ``short`` counts the queries that returned fewer ids than the exact ranking has
    (``min(k, passing chunks)``); ``no_row`` those of them that returned nothing at all.
    """
    stats = evaluate.latency_stats(samples_ms)
    pairs = list(zip(results, references, strict=True))
    comparisons = [compare_lists(result, reference) for result, reference in pairs]
    return {
        "method": name,
        "where": where,
        "kind": kind,
        "p50_ms": round(stats.p50_ms, 4),
        "p95_ms": round(stats.p95_ms, 4),
        "mean_ms": round(stats.mean_ms, 4),
        "samples": stats.n,
        "queries": len(results),
        "mean_overlap_with_exact": round(sum(c[0] for c in comparisons) / len(comparisons), 4),
        "same_set_as_exact": sum(1 for c in comparisons if c[1]),
        "same_order_as_exact": sum(1 for c in comparisons if c[2]),
        "short": sum(1 for result, reference in pairs if len(result) < len(reference)),
        "no_row": sum(1 for result, reference in pairs if reference and not result),
    }


def time_methods(
    answers: Sequence[Callable[[int], list[int]]], n_queries: int, repeats: int
) -> list[tuple[list[float], list[list[int]]]]:
    """Answer every query with every method: one untimed pass, then ``repeats`` timed passes.

    Inside a pass each query goes to all the methods before the next query starts, and the
    method that goes first rotates. A drift of the machine during the run then reaches every
    method alike. (Measured on the laptop this was written on: the p50 of one statement moved
    from 0.54 to 1.03 ms within a minute, so timing one method after the other favours
    whichever runs in the calmer minute.) Returns, per method, the per-call times in ms and
    the results of the first timed pass.
    """
    count = len(answers)
    samples: list[list[float]] = [[] for _ in answers]
    first: list[list[list[int]]] = [[[] for _ in range(n_queries)] for _ in answers]
    for pass_index in range(-1, repeats):  # -1 is the untimed pass
        for index in range(n_queries):
            for step in range(count):
                method = (index + pass_index + 1 + step) % count
                start = time.perf_counter()
                result = answers[method](index)
                elapsed = (time.perf_counter() - start) * 1000.0
                if pass_index < 0:
                    continue
                samples[method].append(elapsed)
                if pass_index == 0:
                    first[method][index] = result
    return [(samples[method], first[method]) for method in range(count)]


# ---------------------------------------------------------------------------------------------
# database access (read only)
# ---------------------------------------------------------------------------------------------


def export_copy(conn: Any) -> Copy:
    """Read the vectors and the filter columns out of MariaDB; the time is ``export_seconds``."""
    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(EXPORT_CHUNKS_SQL)
        chunk_rows = cur.fetchall()
        cur.execute(EXPORT_PAGES_SQL)
        page_rows = cur.fetchall()
        cur.execute(EXPORT_CATEGORIES_SQL)
        category_rows = cur.fetchall()
        cur.execute(EXPORT_LINKS_SQL)
        link_rows = cur.fetchall()
    if not chunk_rows:
        raise SystemExit("the chunk table is empty: run 'wikilense ingest' first")
    vectors = np.vstack([db.vec_from_bytes(row[3]) for row in chunk_rows]).astype(np.float32)
    copy = Copy(
        chunk_ids=np.asarray([int(row[0]) for row in chunk_rows], dtype=np.int64),
        page_ids=np.asarray([int(row[1]) for row in chunk_rows], dtype=np.int64),
        headings=[str(row[2]) for row in chunk_rows],
        vectors=unit_rows(vectors).astype(np.float32),
    )
    for page_id, _title, n_words, last_edited in page_rows:
        copy.page_words[int(page_id)] = int(n_words)
        copy.page_edited[int(page_id)] = last_edited
    for name, page_id in category_rows:
        copy.category_pages.setdefault(str(name), set()).add(int(page_id))
    for title, to_page_id in link_rows:
        copy.linked_from.setdefault(str(title), set()).add(int(to_page_id))
    copy.export_seconds = time.perf_counter() - start
    copy.rows_exported = {
        "chunk": len(chunk_rows),
        "page": len(page_rows),
        "page_category": len(category_rows),
        "link (resolved)": len(link_rows),
    }
    return copy


def fetch_rows(conn: Any, chunk_ids: Sequence[int]) -> int:
    """Fetch the hit rows by primary key, as the CLI shows them; returns the number of rows."""
    if not chunk_ids:
        return 0
    sql = ROWS_SQL_PREFIX + ", ".join(["%s"] * len(chunk_ids)) + ROWS_SQL_SUFFIX
    with conn.cursor() as cur:
        cur.execute(sql, tuple(int(value) for value in chunk_ids))
        return len(cur.fetchall())


def sql_passing_chunks(conn: Any, case: FilterCase, n_chunks: int, query: np.ndarray) -> set[int]:
    """Return the chunk ids the SQL filter of ``case`` keeps (the inline statement, LIMIT all)."""
    hits = searchmod.search(conn, query, k=n_chunks, filters=case.filters, strategy="inline")
    return {hit.chunk_id for hit in hits}


# ---------------------------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------------------------


def _load_hnswlib() -> Any | None:
    try:
        import hnswlib
    except ImportError:
        return None
    return hnswlib


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _quiet_torch() -> None:
    """Leave PyTorch one thread once the queries are embedded.

    Nothing after this point uses the model, but its idle worker threads keep spinning for a
    while and made the MariaDB timings jump between runs (measured: p50 of the bare k-NN
    statement 0.6 to 0.9 ms with them, 0.5 ms without).
    """
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
    time.sleep(1.0)


def _single_blas_thread() -> int | None:
    """Limit NumPy's BLAS to one thread for the rest of the process; return the old count.

    A scan of a few thousand vectors is too small for a thread pool: with 20 threads the same
    product took 0.03 ms while the pool was spinning and up to 2 ms when it had to wake up
    (measured here). One thread is steady, and it is what one query gets from MariaDB too.
    Returns None when threadpoolctl is not installed (then nothing is limited).
    """
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        return None
    counts = [int(info["num_threads"]) for info in threadpool_info() if "num_threads" in info]
    threadpool_limits(limits=1)
    return max(counts) if counts else None


def run(conn: Any, embedder: Any, k: int, repeats: int, use_hnswlib: bool) -> dict[str, Any]:
    """Run the whole comparison on the connected database and return the report as a dict."""
    truths = evaluate.load_ground_truth(conn)
    if not truths:
        raise SystemExit("the claim table is empty: run 'wikilense ingest' first")
    texts = [truth.text for truth in truths]
    queries32 = unit_rows(np.asarray(embedder.embed_queries(texts), dtype=np.float32)).astype(
        np.float32
    )
    queries64 = queries32.astype(np.float64)
    n_queries = len(texts)
    _quiet_torch()
    blas_threads_before = _single_blas_thread()

    copy = export_copy(conn)
    vectors64 = copy.vectors.astype(np.float64)
    row_numbers = np.arange(copy.n)

    hnswlib = _load_hnswlib() if use_hnswlib else None
    index = None
    hnsw_info: dict[str, Any] | None = None
    if hnswlib is not None:
        index = hnswlib.Index(space="cosine", dim=copy.vectors.shape[1])
        index.init_index(
            max_elements=copy.n,
            ef_construction=HNSWLIB_EF_CONSTRUCTION,
            M=HNSWLIB_M,
            random_seed=HNSWLIB_SEED,
        )
        build_start = time.perf_counter()
        index.add_items(copy.vectors, row_numbers)  # labels are row numbers of the copy
        build_seconds = time.perf_counter() - build_start
        index.set_ef(HNSWLIB_EF)
        index.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.bin")
            index.save_index(path)
            index_bytes = os.path.getsize(path)
        hnsw_info = {
            "version": _distribution_version("hnswlib"),
            "M": HNSWLIB_M,
            "ef_construction": HNSWLIB_EF_CONSTRUCTION,
            "ef": HNSWLIB_EF,
            "build_seconds": round(build_seconds, 3),
            "index_file_bytes": index_bytes,
        }

    def ids_of(labels: Sequence[int]) -> list[int]:
        return [int(copy.chunk_ids[int(label)]) for label in labels]

    # --- the methods ---------------------------------------------------------------------
    Method = tuple[str, str, str, Callable[[int], list[int]]]  # name, where, kind, answer

    def mariadb(strategy: str, filters: Filters | None = None) -> Callable[[int], list[int]]:
        def answer(i: int) -> list[int]:
            hits = searchmod.search(
                conn, queries32[i], k=k, filters=filters, strategy=strategy, overfetch=OVERFETCH
            )
            return [hit.chunk_id for hit in hits]

        return answer

    def numpy_scan(case: FilterCase | None, fetch: bool) -> Callable[[int], list[int]]:
        def answer(i: int) -> list[int]:
            # The filter is evaluated on the copy inside the timed call: it is part of the query.
            rows = None if case is None else np.flatnonzero(copy_mask(copy, case))
            ids = top_k(copy.vectors, queries32[i], copy.chunk_ids, k, rows)
            if fetch:
                fetch_rows(conn, ids)
            return ids

        return answer

    def hnsw_graph(case: FilterCase | None) -> Callable[[int], list[int]]:
        def answer(i: int) -> list[int]:
            if case is None:
                labels, _ = index.knn_query(queries32[i], k=k)
                return ids_of(labels[0])
            allowed = copy_mask(copy, case)
            try:
                labels, _ = index.knn_query(
                    queries32[i], k=min(k, int(allowed.sum())),
                    filter=lambda label: bool(allowed[label]),
                )
            except RuntimeError:  # hnswlib found fewer passing nodes than asked for
                return []
            return ids_of(labels[0])

        return answer

    def measure(methods: Sequence[Method], references: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
        timed = time_methods([method[3] for method in methods], n_queries, repeats)
        return [
            summarise(name, where, kind, samples, results, references)
            for (name, where, kind, _), (samples, results) in zip(methods, timed, strict=True)
        ]

    with searchmod.ef_search_session(conn, EF_SEARCH):
        # --- no filter -------------------------------------------------------------------
        exact = [top_k(vectors64, queries64[i], copy.chunk_ids, k) for i in range(n_queries)]
        methods: list[Method] = [
            ("MariaDB, bare k-NN statement (HNSW, M=16, ef_search 100)", "MariaDB",
             "approximate", mariadb("none")),
            ("MariaDB, joined statement (strategy inline)", "MariaDB", "exact", mariadb("inline")),
            ("NumPy scan of the copy, ids only", "application memory", "exact",
             numpy_scan(None, fetch=False)),
            ("NumPy scan of the copy, then the hit rows from MariaDB by primary key",
             "application memory + MariaDB", "exact", numpy_scan(None, fetch=True)),
        ]
        if index is not None:
            methods.append(("hnswlib graph over the copy (M=16, ef 100), ids only",
                            "application memory", "approximate", hnsw_graph(None)))
        unfiltered = measure(methods, exact)

        # --- with a filter ---------------------------------------------------------------
        filtered: dict[str, Any] = {}
        for case in FILTER_CASES:
            mask = copy_mask(copy, case)
            passing_copy = {int(value) for value in copy.chunk_ids[mask]}
            passing_sql = sql_passing_chunks(conn, case, copy.n, queries32[0])
            if passing_copy != passing_sql:
                raise SystemExit(
                    f"filter {case.key!r}: the copy keeps {len(passing_copy)} chunks and the SQL "
                    f"filter {len(passing_sql)}; they must be the same chunks"
                )
            rows = np.flatnonzero(mask)
            exact_f = [
                top_k(vectors64, queries64[i], copy.chunk_ids, k, rows) for i in range(n_queries)
            ]
            methods = [
                ("MariaDB, strategy inline (the default)", "MariaDB", "exact",
                 mariadb("inline", case.filters)),
                (f"MariaDB, strategy overfetch {OVERFETCH}", "MariaDB", "approximate",
                 mariadb("overfetch", case.filters)),
                ("NumPy pre-filter and scan of the copy, ids only", "application memory",
                 "exact", numpy_scan(case, fetch=False)),
                ("NumPy pre-filter and scan, then the hit rows from MariaDB by primary key",
                 "application memory + MariaDB", "exact", numpy_scan(case, fetch=True)),
            ]
            if index is not None:
                methods.append(("hnswlib graph with its filter callback (ef 100), ids only",
                                "application memory", "approximate", hnsw_graph(case)))
            filtered[case.key] = {
                "label": case.label,
                "filters": evaluate.filters_parameter(case.filters),
                "passing_chunks": len(passing_sql),
                "rows": measure(methods, exact_f),
            }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": {
            "k": k,
            "repeats": repeats,
            "n_queries": n_queries,
            "n_chunks": copy.n,
            "dimensions": int(copy.vectors.shape[1]),
            "mhnsw_ef_search": EF_SEARCH,
            "overfetch": OVERFETCH,
            **db.vector_index_info(conn),
            "hnswlib": hnsw_info,
            "numpy_blas_threads": 1 if blas_threads_before is not None else None,
            "numpy_blas_threads_available": blas_threads_before,
            "embedding_model": getattr(embedder, "model_name", None),
            "embedding_device": getattr(embedder, "device", None),
        },
        "copy": {
            "export_seconds": round(copy.export_seconds, 3),
            "rows_exported": copy.rows_exported,
            "vector_bytes": int(copy.vectors.nbytes),
        },
        "unfiltered": unfiltered,
        "filtered": filtered,
        "ingest_meta": evaluate.read_ingest_meta(conn),
        "machine": evaluate.machine_info(embedder),
        "versions": evaluate.versions_info(conn),
        "notes": NOTES,
    }


NOTES = [
    ("Every method answers the same queries (the corpus claims, embedded once) with k hits, on "
    "one machine, from one client: one untimed pass over the queries, then `repeats` timed "
    "passes. Within a pass each query goes to every method of a table in turn, and the method "
    "that goes first rotates, so a drift of the machine reaches all of them alike. p50 and p95 "
    "use linear interpolation."),
    ("MariaDB times are the wall-clock time of one search() call as the Python client sees it "
    "(the statement returns the hit rows with title, section path and text). mhnsw_ef_search is "
    "set once for the run, not per call. Because the methods take turns, every statement runs "
    "with colder caches than in the evaluation harness, which repeats one statement in a block "
    "(results/final_ef_100.md has 0.51 ms p50 for the bare k-NN statement). The times also move "
    "between runs on this laptop (the repeat file shows by how much); read the ratios, not the "
    "last digit."),
    ("NumPy and hnswlib each use one thread per query, as one MariaDB query does."),
    ("The copy holds the vectors and the columns the filters read, not the text. 'ids only' rows "
    "return chunk ids; the rows 'then the hit rows from MariaDB' add the SELECT by primary key "
    "an application needs to show a result."),
    ("'exact' is NumPy over float64 copies of the vectors, cosine distance, ties by chunk_id. "
    "overlap = the share of the exact top k a method returned, averaged over the queries; "
    "short = queries that returned fewer ids than min(k, passing chunks); no row = those of "
    "them that returned nothing."),
    ("The filters run on the copy in Python, inside the timed call. Before timing, the script "
    "checks that the copy and the SQL filter keep exactly the same chunks."),
    ("This measures an index in the application's memory, not a vector database server: there is "
    "no network hop, no persistence and no concurrent writer. No dedicated vector store was "
    "measured."),
]


# ---------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------


def _table(rows: Sequence[dict[str, Any]], n_queries: int) -> list[str]:
    lines = [
        "| Method | Kind | p50 ms | p95 ms | Overlap with exact | Same order as exact | Short | No row |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['kind']} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} | "
            f"{row['mean_overlap_with_exact']:.3f} | {row['same_order_as_exact']}/{n_queries} | "
            f"{row['short']}/{n_queries} | {row['no_row']}/{n_queries} |"
        )
    return lines


def to_markdown(report: dict[str, Any], name: str = DEFAULT_NAME) -> str:
    """Return the Markdown form of ``report`` (the dict :func:`run` returns)."""
    params = report["parameters"]
    copy = report["copy"]
    n_queries = params["n_queries"]
    hnsw = params["hnswlib"]
    lines = [
        f"# MariaDB against a separate in-process index: {name}",
        "",
        (f"Generated {report['created_at']} by `scripts/compare_separate_index.py`. "
        f"{params['n_chunks']} chunks of {params['dimensions']} dimensions, {n_queries} queries "
        f"(the corpus claims), k = {params['k']}, {params['repeats']} timed passes after one "
        "untimed pass."),
        "",
        "The copy: "
        f"{copy['vector_bytes']:,} bytes of vectors read out of MariaDB in "
        f"{copy['export_seconds']:.2f} s, together with the columns the filters need ("
        + ", ".join(f"{count:,} `{table}` rows" for table, count in copy["rows_exported"].items())
        + "). A separate index has to repeat this, or apply every change twice, to stay current.",
        "",
    ]
    if hnsw is None:
        lines += ["hnswlib was not installed for this run, so its rows are missing.", ""]
    else:
        lines += [
            (f"hnswlib {hnsw.get('version', '')}: M = {hnsw['M']}, ef_construction = "
            f"{hnsw['ef_construction']} (the library's default), ef = {hnsw['ef']}; built in "
            f"{hnsw['build_seconds']:.2f} s on all cores, {hnsw['index_file_bytes']:,} bytes when "
            "saved. Its parallel build makes a slightly different graph every run."),
            "",
        ]
    lines += ["## No filter", "", *_table(report["unfiltered"], n_queries), ""]
    for case in report["filtered"].values():
        lines += [
            f"## Filter: {case['label']}",
            "",
            f"{case['passing_chunks']:,} of {params['n_chunks']:,} chunks pass.",
            "",
            *_table(case["rows"], n_queries),
            "",
        ]
    lines += ["## How to read this", ""]
    lines += [f"- {note}" for note in report["notes"]]
    machine, versions = report["machine"], report["versions"]
    lines += [
        "",
        (f"Machine: {machine['cpu']}, {machine['cpu_count']} logical cores, "
        f"{machine['memory_gb']} GB; NumPy BLAS threads used: {params['numpy_blas_threads']}. "
        f"MariaDB {versions['mariadb']}, index M = {params['index_m']}, "
        f"mhnsw_ef_search = {params['mhnsw_ef_search']}. Python {versions['python']}, "
        f"NumPy {versions['numpy']}."),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="hits per query (default %(default)s)")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="timed passes per method after one untimed pass (default %(default)s)")
    parser.add_argument("--out", default=str(REPO_ROOT / "results"),
                        help="output directory (default: results/ in the repository)")
    parser.add_argument("--name", default=DEFAULT_NAME, help="file stem (default %(default)s)")
    parser.add_argument("--no-hnswlib", action="store_true",
                        help="leave the hnswlib rows out even when hnswlib is installed")
    args = parser.parse_args(argv)
    if args.k < 1 or args.repeats < 1:
        parser.error("--k and --repeats must be at least 1")

    from wikilense.cli import make_embedder

    settings = load_settings()
    embedder = make_embedder(settings)
    conn = db.connect(settings)
    try:
        report = run(conn, embedder, args.k, args.repeats, use_hnswlib=not args.no_hnswlib)
    finally:
        conn.close()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.name}.json"
    md_path = out_dir / f"{args.name}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
                         encoding="utf-8")
    md_path.write_text(to_markdown(report, args.name), encoding="utf-8")
    for row in report["unfiltered"]:
        print(f"{row['p50_ms']:8.3f} ms p50  overlap {row['mean_overlap_with_exact']:.3f}  {row['method']}")
    for case in report["filtered"].values():
        print(f"-- {case['label']} ({case['passing_chunks']} chunks pass)")
        for row in case["rows"]:
            print(f"{row['p50_ms']:8.3f} ms p50  overlap {row['mean_overlap_with_exact']:.3f}  "
                  f"short {row['short']:2d}  {row['method']}")
    print(f"written: {json_path}, {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
