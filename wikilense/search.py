"""Query layer: k-nearest chunks by ``VEC_DISTANCE_COSINE`` combined with SQL filters and joins.

Every statement is a module constant or is assembled from the named fragments below, so the
README can quote them verbatim. Values are always bound as ``%s`` parameters; the only text that
ever varies in a statement is the number of placeholders in an ``IN (...)`` list and which
fixed fragments are present.

Three strategies, measured on MariaDB 11.8.9 (see docs/DESIGN.md and tests/test_search.py):

``inline``
    One statement: ``chunk STRAIGHT_JOIN page STRAIGHT_JOIN section [...] WHERE <filters>
    ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT k``. ``STRAIGHT_JOIN`` keeps
    ``chunk`` first in the join order; with a plain ``JOIN`` the optimizer starts from ``page``
    and sorts every chunk instead of using the vector index. With the join present MariaDB
    walks the vector index in distance order until ``k`` rows have passed the filters, so the
    result is never short when enough chunks match, at the price of a walk whose length grows
    with the selectivity of the filter and with the table (``mhnsw_ef_search`` does not change
    this walk).
``overfetch``
    The bare index query ``SELECT chunk_id, distance FROM chunk ORDER BY ... LIMIT k *
    overfetch`` as a derived table (this is the HNSW search proper, bounded by
    ``mhnsw_ef_search``), joined to ``page`` and ``section`` and filtered in the outer query,
    re-limited to ``k``. Bounded work; can return fewer than ``k`` rows when the filter is
    selective, because chunks outside the inner ``k * overfetch`` are never seen.
``none``
    The ``overfetch`` statement with factor 1 and no filters: plain k-nearest-neighbour search
    joined back to ``page`` and ``section`` for the hit metadata. Filters are ignored.

Results are ordered by distance ascending, ties by ``chunk_id``. The tie-break is applied in
Python: a second ``ORDER BY`` key after ``VEC_DISTANCE_COSINE(...)`` makes MariaDB drop the
vector index and sort the whole table (measured).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pymysql
import pymysql.cursors

from wikilense import db

#: Filtering strategies accepted by :func:`search` and :func:`explain_search`.
STRATEGIES: tuple[str, ...] = ("inline", "overfetch", "none", "rrf")

EF_SEARCH_VARIABLE = "mhnsw_ef_search"

#: The range MariaDB 11.8 accepts for ``mhnsw_ef_search`` (``information_schema.SYSTEM_VARIABLES``).
#: The server clamps a value outside it with only a warning, so :func:`ef_search_session`
#: refuses one instead: a caller would otherwise report a value that did not run.
MIN_EF_SEARCH = 1
MAX_EF_SEARCH = 10000

#: The ``ef_search`` value that means "leave the server's session value alone": ``--ef-search 0``
#: on the command line, ``ef_search=0`` in the web API and ``WIKILENSE_EF_SEARCH=0``.
EF_SEARCH_SERVER = 0

#: Largest ``overfetch`` factor :func:`search` accepts; ``k * overfetch`` is the inner LIMIT.
MAX_OVERFETCH = 1000

# ---------------------------------------------------------------------------------------------
# SQL: the k-nearest-neighbour core
# ---------------------------------------------------------------------------------------------

#: The index-driven query. Parameters: query vector (bytes), query vector, limit.
#: This exact shape (``ORDER BY VEC_DISTANCE_COSINE(column, constant) LIMIT n``) is what the
#: HNSW index serves; ``mhnsw_ef_search`` bounds the candidates it looks at.
KNN_SQL = (
    "SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, %s) AS distance "
    "FROM chunk "
    "ORDER BY VEC_DISTANCE_COSINE(embedding, %s) "
    "LIMIT %s"
)

# ---------------------------------------------------------------------------------------------
# SQL: the ``inline`` statement (filters and joins around the index-driven ORDER BY)
# ---------------------------------------------------------------------------------------------

#: Parameters: query vector (bytes).
INLINE_SELECT = (
    "SELECT chunk.chunk_id, chunk.page_id, page.title, section.path AS section_path, "
    "chunk.ordinal AS chunk_ordinal, chunk.n_words, "
    "VEC_DISTANCE_COSINE(chunk.embedding, %s) AS distance, chunk.text"
)
#: ``STRAIGHT_JOIN``: chunk stays the first table, so the vector index drives the scan and
#: page and section are primary-key lookups (EXPLAIN type ``eq_ref``).
INLINE_FROM = (
    "FROM chunk "
    "STRAIGHT_JOIN page ON page.page_id = chunk.page_id "
    "STRAIGHT_JOIN section ON section.section_id = chunk.section_id"
)
#: Parameters: query vector (bytes), k.
INLINE_ORDER_LIMIT = "ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s"
INLINE_JOIN_KEYWORD = "STRAIGHT_JOIN"

# ---------------------------------------------------------------------------------------------
# SQL: the ``overfetch`` / ``none`` statement (index-driven derived table, filters outside)
# ---------------------------------------------------------------------------------------------

OVERFETCH_SELECT = (
    "SELECT chunk.chunk_id, chunk.page_id, page.title, section.path AS section_path, "
    "chunk.ordinal AS chunk_ordinal, chunk.n_words, knn.distance, chunk.text"
)
#: Parameters: query vector (bytes), query vector, k * overfetch (from KNN_SQL).
OVERFETCH_FROM = (
    "FROM (" + KNN_SQL + ") AS knn "
    "JOIN chunk ON chunk.chunk_id = knn.chunk_id "
    "JOIN page ON page.page_id = chunk.page_id "
    "JOIN section ON section.section_id = chunk.section_id"
)
#: Parameters: k.
OVERFETCH_ORDER_LIMIT = "ORDER BY knn.distance, chunk.chunk_id LIMIT %s"
OVERFETCH_JOIN_KEYWORD = "JOIN"

# ---------------------------------------------------------------------------------------------
# SQL: the ``rrf`` statement (vector top-N and full-text top-N fused by reciprocal rank fusion)
# ---------------------------------------------------------------------------------------------

#: The full-text top-N over the FULLTEXT index ``ft_chunk_text`` on ``chunk.text`` (InnoDB,
#: natural-language mode: the words of the text, stopwords and tokens shorter than
#: ``innodb_ft_min_token_size`` = 3 dropped, ranked by InnoDB's relevance; a row that contains
#: none of the words is not returned). Equal relevance is common, so ``chunk_id`` breaks ties
#: before the LIMIT: which tied chunks make the top-N is then fixed, not left to the server
#: (the plan, a filesort after the full-text lookup, is the same). Parameters: query text,
#: query text, limit.
FT_SQL = (
    "SELECT chunk_id, MATCH(text) AGAINST (%s IN NATURAL LANGUAGE MODE) AS relevance "
    "FROM chunk "
    "WHERE MATCH(text) AGAINST (%s IN NATURAL LANGUAGE MODE) "
    "ORDER BY relevance DESC, chunk_id "
    "LIMIT %s"
)
#: The RRF smoothing constant (Cormack, Clarke and Buettcher, SIGIR 2009), the value the
#: MariaDB docs use; it is written literally in ``RRF_WITH`` (tests/test_search.py checks the
#: two agree).
RRF_K = 60
#: The two ranked lists as CTEs and their fusion. ``vec`` is :data:`KNN_SQL` (the HNSW search,
#: LIMIT N = k * overfetch) and ``ft`` is :data:`FT_SQL` (LIMIT N), each ranked by
#: ``ROW_NUMBER()`` over the LIMITed derived table: a window function written directly on the
#: index query makes MariaDB compute it over the whole table (measured: r_rows 8,868 instead of
#: N, 9.6 ms instead of 2.0 ms). ``fused`` sums ``1 / (60 + rank)`` over both lists per chunk;
#: the division is done in DOUBLE because an integer division would give a DECIMAL with
#: ``div_precision_increment`` (4) digits, and ranks past 60 would tie.
#: Parameters: query vector (bytes), query vector, N, query text, query text, N.
RRF_WITH = (
    "WITH vec AS ("
    "SELECT knn.chunk_id, ROW_NUMBER() OVER (ORDER BY knn.distance, knn.chunk_id) AS rnk "
    "FROM (" + KNN_SQL + ") AS knn), "
    "ft AS ("
    "SELECT matched.chunk_id, "
    "ROW_NUMBER() OVER (ORDER BY matched.relevance DESC, matched.chunk_id) AS rnk "
    "FROM (" + FT_SQL + ") AS matched), "
    "fused AS ("
    "SELECT ranked.chunk_id, SUM(CAST(1 AS DOUBLE) / (60 + ranked.rnk)) AS score "
    "FROM (SELECT chunk_id, rnk FROM vec UNION ALL SELECT chunk_id, rnk FROM ft) AS ranked "
    "GROUP BY ranked.chunk_id)"
)
#: Parameters: query vector (bytes): the cosine distance is computed here for every fused
#: chunk, so it is known also for a chunk that only the full-text list found.
RRF_SELECT = (
    "SELECT chunk.chunk_id, chunk.page_id, page.title, section.path AS section_path, "
    "chunk.ordinal AS chunk_ordinal, chunk.n_words, "
    "VEC_DISTANCE_COSINE(chunk.embedding, %s) AS distance, chunk.text, fused.score"
)
#: ``fused`` holds at most 2 N rows, so the joins are primary-key lookups from it; the filter
#: joins and predicates are added here, in the outer query, like in ``overfetch``.
RRF_FROM = (
    "FROM fused "
    "JOIN chunk ON chunk.chunk_id = fused.chunk_id "
    "JOIN page ON page.page_id = chunk.page_id "
    "JOIN section ON section.section_id = chunk.section_id"
)
#: Parameters: k. The tie-break can be in the statement here: this ORDER BY does not have to
#: be the vector-index shape.
RRF_ORDER_LIMIT = "ORDER BY fused.score DESC, chunk.chunk_id LIMIT %s"
RRF_JOIN_KEYWORD = "JOIN"

# ---------------------------------------------------------------------------------------------
# SQL: filter fragments (each added only when its Filters field is set; one parameter each,
# except FILTER_TITLES which takes one placeholder per title)
# ---------------------------------------------------------------------------------------------

FILTER_MIN_WORDS = "page.n_words >= %s"
FILTER_MAX_WORDS = "page.n_words <= %s"
FILTER_HEADING_LIKE = "section.heading LIKE %s"
FILTER_PATH_LIKE = "section.path LIKE %s"
#: ``page.title IN (%s, ...)``; the placeholder list is built by :func:`_placeholders`.
FILTER_TITLES_PREFIX = "page.title IN ("
FILTER_TITLES_SUFFIX = ")"

#: Pages linked from the page with the given title, as a join to a distinct list of target
#: pages. A ``chunk.page_id IN (SELECT ...)`` predicate is rewritten by the optimizer into a
#: semi-join whose table order ignores STRAIGHT_JOIN; measured on 11.8.9 it then starts from
#: ``link`` and drops the vector index. The derived table with DISTINCT is materialised once
#: and joined after the index scan. Parameter: the source page title.
LINKED_FROM_JOIN = (
    "(SELECT DISTINCT link.to_page_id AS page_id FROM link "
    "JOIN page AS src ON src.page_id = link.from_page_id "
    "WHERE src.title = %s AND link.to_page_id IS NOT NULL) AS linked "
    "ON linked.page_id = chunk.page_id"
)
#: Pages that link to the page with the given title. Parameter: the target page title.
LINKS_TO_JOIN = (
    "(SELECT DISTINCT link.from_page_id AS page_id FROM link "
    "JOIN page AS dst ON dst.page_id = link.to_page_id "
    "WHERE dst.title = %s) AS linking "
    "ON linking.page_id = chunk.page_id"
)

# ---------------------------------------------------------------------------------------------
# SQL: joining hits back to sentences, and page summaries
# ---------------------------------------------------------------------------------------------

#: ``WHERE cs.chunk_id IN (%s, ...)``; the placeholder list is built by :func:`_placeholders`.
HIT_SENTENCES_PREFIX = (
    "SELECT cs.chunk_id, s.element_key, s.text "
    "FROM chunk_sentence AS cs "
    "JOIN sentence AS s ON s.sentence_id = cs.sentence_id "
    "WHERE cs.chunk_id IN ("
)
HIT_SENTENCES_SUFFIX = ") ORDER BY cs.chunk_id, s.ordinal"

PAGE_SUMMARY_SQL = (
    "SELECT page.page_id, page.title, page.n_sentences, page.n_items, page.n_words, "
    "page.n_chars, page.n_sections, page.n_tables, page.n_lists, "
    "(SELECT COUNT(*) FROM chunk WHERE chunk.page_id = page.page_id) AS n_chunks, "
    "(SELECT COUNT(*) FROM link WHERE link.from_page_id = page.page_id) AS n_links_out, "
    "(SELECT COUNT(*) FROM link WHERE link.from_page_id = page.page_id "
    "AND link.to_page_id IS NOT NULL) AS n_links_out_resolved, "
    "(SELECT COUNT(*) FROM link WHERE link.to_page_id = page.page_id) AS n_links_in "
    "FROM page WHERE page.page_id = %s"
)
PAGE_SECTIONS_SQL = (
    "SELECT ordinal, heading, level, path FROM section WHERE page_id = %s ORDER BY ordinal"
)

# ---------------------------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with its page and section, in the column order of the SELECTs.

    ``distance`` is always the cosine distance to the query vector. ``score`` is the
    reciprocal-rank-fusion score of the ``rrf`` strategy and ``None`` for the others.
    """

    chunk_id: int
    page_id: int
    title: str
    section_path: str
    chunk_ordinal: int
    n_words: int
    distance: float
    text: str
    score: float | None = None


@dataclass(frozen=True)
class Filters:
    """SQL restrictions on the pages and sections a hit may come from; every field optional.

    ``heading_like`` and ``path_like`` are SQL ``LIKE`` patterns written by the caller (``%``
    and ``_`` are wildcards) and are always bound as parameters. ``section.heading`` uses the
    server's case- and accent-insensitive collation, ``section.path`` likewise; ``page.title``
    is ``utf8mb4_bin``, so ``linked_from``, ``links_to`` and ``titles`` match exactly.
    ``titles`` restricts hits to the listed page titles; an empty list matches no page.
    Raises TypeError or ValueError, naming the field, for a value of the wrong type or range.
    """

    min_words: int | None = None
    max_words: int | None = None
    heading_like: str | None = None
    path_like: str | None = None
    linked_from: str | None = None
    links_to: str | None = None
    titles: Sequence[str] | None = None

    def __post_init__(self) -> None:
        for name in ("min_words", "max_words"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Filters.{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"Filters.{name} must be >= 0, got {value}")
        if (
            self.min_words is not None
            and self.max_words is not None
            and self.min_words > self.max_words
        ):
            raise ValueError(
                f"Filters.min_words ({self.min_words}) is greater than max_words ({self.max_words})"
            )
        for name in ("heading_like", "path_like", "linked_from", "links_to"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"Filters.{name} must be a str, got {type(value).__name__}")
        if self.titles is not None:
            if isinstance(self.titles, (str, bytes)):
                raise TypeError("Filters.titles must be a sequence of titles, not a single str")
            titles = tuple(self.titles)
            if not all(isinstance(t, str) for t in titles):
                raise TypeError("Filters.titles must contain only str values")
            object.__setattr__(self, "titles", titles)

    def is_empty(self) -> bool:
        """Return True when no field is set, i.e. the filter is equivalent to ``None``."""
        return all(getattr(self, f.name) is None for f in fields(self))


@dataclass(frozen=True)
class SectionRow:
    """One section of a page as stored in the ``section`` table."""

    ordinal: int
    heading: str
    level: int
    path: str


@dataclass(frozen=True)
class PageSummary:
    """A page's stored statistics plus its chunk, link and section counts."""

    page_id: int
    title: str
    n_sentences: int
    n_items: int
    n_words: int
    n_chars: int
    n_sections: int
    n_tables: int
    n_lists: int
    n_chunks: int
    n_links_out: int
    n_links_out_resolved: int
    n_links_in: int
    sections: tuple[SectionRow, ...]


# ---------------------------------------------------------------------------------------------
# statement assembly
# ---------------------------------------------------------------------------------------------


def _placeholders(n: int) -> str:
    """Return ``n`` comma-separated ``%s`` placeholders (``n`` >= 1)."""
    if n < 1:
        raise ValueError("an IN list needs at least one placeholder")
    return ", ".join(["%s"] * n)


def _filter_fragments(
    filters: Filters, join_keyword: str
) -> tuple[list[str], list[Any], list[str], list[Any]]:
    """Return (join fragments, their params, WHERE fragments, their params) for ``filters``.

    Only set fields contribute. ``join_keyword`` is ``STRAIGHT_JOIN`` for the inline statement
    and ``JOIN`` otherwise; both are fixed literals. An empty ``titles`` list contributes
    nothing here: the callers short-circuit it (it matches no page).
    """
    joins: list[str] = []
    join_params: list[Any] = []
    wheres: list[str] = []
    where_params: list[Any] = []
    if filters.linked_from is not None:
        joins.append(join_keyword + " " + LINKED_FROM_JOIN)
        join_params.append(filters.linked_from)
    if filters.links_to is not None:
        joins.append(join_keyword + " " + LINKS_TO_JOIN)
        join_params.append(filters.links_to)
    if filters.min_words is not None:
        wheres.append(FILTER_MIN_WORDS)
        where_params.append(filters.min_words)
    if filters.max_words is not None:
        wheres.append(FILTER_MAX_WORDS)
        where_params.append(filters.max_words)
    if filters.heading_like is not None:
        wheres.append(FILTER_HEADING_LIKE)
        where_params.append(filters.heading_like)
    if filters.path_like is not None:
        wheres.append(FILTER_PATH_LIKE)
        where_params.append(filters.path_like)
    if filters.titles:
        wheres.append(FILTER_TITLES_PREFIX + _placeholders(len(filters.titles)) + FILTER_TITLES_SUFFIX)
        where_params.extend(filters.titles)
    return joins, join_params, wheres, where_params


def _check_search_args(
    qvec: np.ndarray,
    k: int,
    filters: Filters | None,
    strategy: str,
    overfetch: int,
    query_text: str | None,
) -> tuple[bytes, Filters]:
    """Validate the search arguments and return (vector bytes, effective Filters).

    Raises ValueError for an unknown strategy, ``k`` or ``overfetch`` below 1, ``overfetch``
    above :data:`MAX_OVERFETCH`, a query vector
    whose dimension is not ``db.VECTOR_DIM`` (MariaDB would not fail but return NULL distances
    and an arbitrary order), or a missing or blank ``query_text`` with strategy ``rrf``, and
    TypeError for non-int ``k`` / ``overfetch``, a filters value that is not a Filters or a
    ``query_text`` that is not a str.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose one of {STRATEGIES}")
    if strategy == "rrf":
        if query_text is None:
            raise ValueError("strategy 'rrf' needs query_text, the words for the full-text search")
        if not isinstance(query_text, str):
            raise TypeError(f"query_text must be a str, got {type(query_text).__name__}")
        if not query_text.strip():
            raise ValueError("query_text must not be blank with strategy 'rrf'")
    for name, value in (("k", k), ("overfetch", overfetch)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if overfetch > MAX_OVERFETCH:
        raise ValueError(f"overfetch must be <= {MAX_OVERFETCH}, got {overfetch}")
    if filters is None:
        filters = Filters()
    elif not isinstance(filters, Filters):
        raise TypeError(f"filters must be a Filters or None, got {type(filters).__name__}")
    qbytes = db.vec_param(qvec)
    dim = len(qbytes) // 4
    if dim != db.VECTOR_DIM:
        raise ValueError(
            f"query vector has {dim} dimensions but chunk.embedding is VECTOR({db.VECTOR_DIM})"
        )
    return qbytes, filters


def search_statement(
    qvec: np.ndarray,
    k: int = 10,
    filters: Filters | None = None,
    strategy: str = "inline",
    overfetch: int = 10,
    query_text: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """Return the SQL text and its parameter tuple that :func:`search` runs for these arguments.

    The text depends only on the strategy, on which filter fields are set and on the number
    of titles; every value (the vector bytes, the query text, the limits, the filter values)
    is in the parameter tuple. Raises as :func:`search` does for invalid arguments. For
    ``titles=[]`` (a filter that matches no page) the SQL is empty and the params are ``()``:
    no statement is run. ``query_text`` is used by strategy ``rrf`` only.
    """
    qbytes, filters = _check_search_args(qvec, k, filters, strategy, overfetch, query_text)
    if strategy == "none":
        filters = Filters()
        overfetch = 1
    elif filters.titles is not None and len(filters.titles) == 0:
        return "", ()
    if strategy == "rrf":
        joins, join_params, wheres, where_params = _filter_fragments(filters, RRF_JOIN_KEYWORD)
        parts = [RRF_WITH, RRF_SELECT, RRF_FROM, *joins]
        if wheres:
            parts.append("WHERE " + " AND ".join(wheres))
        parts.append(RRF_ORDER_LIMIT)
        n = k * overfetch
        params: tuple[Any, ...] = (
            qbytes, qbytes, n, query_text, query_text, n, qbytes, *join_params, *where_params, k
        )
        return " ".join(parts), params
    if strategy == "inline":
        joins, join_params, wheres, where_params = _filter_fragments(filters, INLINE_JOIN_KEYWORD)
        parts = [INLINE_SELECT, INLINE_FROM, *joins]
        if wheres:
            parts.append("WHERE " + " AND ".join(wheres))
        parts.append(INLINE_ORDER_LIMIT)
        params = (qbytes, *join_params, *where_params, qbytes, k)
        return " ".join(parts), params
    joins, join_params, wheres, where_params = _filter_fragments(filters, OVERFETCH_JOIN_KEYWORD)
    parts = [OVERFETCH_SELECT, OVERFETCH_FROM, *joins]
    if wheres:
        parts.append("WHERE " + " AND ".join(wheres))
    parts.append(OVERFETCH_ORDER_LIMIT)
    params = (qbytes, qbytes, k * overfetch, *join_params, *where_params, k)
    return " ".join(parts), params


# ---------------------------------------------------------------------------------------------
# session variable scope
# ---------------------------------------------------------------------------------------------


def resolve_ef_search(requested: int | None, default: int) -> int | None:
    """Return the ``mhnsw_ef_search`` to set for a query, or None to keep the session's value.

    ``requested`` is the caller's value (``--ef-search``, the ``ef_search`` query parameter), or
    None when it was not given; then ``default`` applies (``Settings.ef_search``, from
    ``WIKILENSE_EF_SEARCH``). :data:`EF_SEARCH_SERVER` (0) from either source means None.
    """
    value = default if requested is None else requested
    return None if value == EF_SEARCH_SERVER else value


@contextmanager
def ef_search_session(conn: pymysql.Connection, ef_search: int | None) -> Iterator[None]:
    """Set ``mhnsw_ef_search`` for the session inside the ``with`` block and restore it after.

    ``None`` leaves the session untouched. The previous value is read first and written back
    in a ``finally`` clause, so it is restored even when the block raises. Yields None.
    Raises ValueError, before anything is set, unless ``ef_search`` is None or an int from
    :data:`MIN_EF_SEARCH` to :data:`MAX_EF_SEARCH`.
    """
    if ef_search is None:
        yield
        return
    if (
        isinstance(ef_search, bool)
        or not isinstance(ef_search, int)
        or not MIN_EF_SEARCH <= ef_search <= MAX_EF_SEARCH
    ):
        raise ValueError(
            f"ef_search must be an int from {MIN_EF_SEARCH} to {MAX_EF_SEARCH} (the range of "
            f"mhnsw_ef_search), got {ef_search!r}"
        )
    previous = db.get_session_var(conn, EF_SEARCH_VARIABLE)
    db.set_session_var(conn, EF_SEARCH_VARIABLE, ef_search)
    try:
        yield
    finally:
        db.set_session_var(conn, EF_SEARCH_VARIABLE, previous)


# ---------------------------------------------------------------------------------------------
# public query functions
# ---------------------------------------------------------------------------------------------


def search(
    conn: pymysql.Connection,
    qvec: np.ndarray,
    k: int = 10,
    filters: Filters | None = None,
    strategy: str = "inline",
    overfetch: int = 10,
    ef_search: int | None = None,
    query_text: str | None = None,
) -> list[Hit]:
    """Return up to ``k`` hits for ``qvec``, ordered by distance then chunk_id (``rrf``: by
    score descending, then chunk_id).

    ``filters`` restricts the pages and sections (see :class:`Filters`); ``strategy`` is one
    of :data:`STRATEGIES` (module docstring); ``overfetch`` is the factor of the inner limit
    for the ``overfetch`` and ``rrf`` strategies and is ignored otherwise. ``query_text`` is
    the text for the full-text half of ``rrf`` (required there, ignored otherwise).
    ``ef_search`` sets the session's ``mhnsw_ef_search`` for this call only and restores the
    previous value afterwards, even when the statement fails. The ``inline`` and ``none``
    strategies return exactly ``k`` hits when at least ``k`` chunks match; ``overfetch`` and
    ``rrf`` may return fewer. Nothing is committed. Raises ValueError / TypeError for invalid
    arguments and ``pymysql.err.Error`` from the server.
    """
    sql, params = search_statement(qvec, k, filters, strategy, overfetch, query_text)
    if not sql:
        return []
    with ef_search_session(conn, ef_search), conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    hits = [
        Hit(
            chunk_id=int(row[0]),
            page_id=int(row[1]),
            title=str(row[2]),
            section_path=str(row[3]),
            chunk_ordinal=int(row[4]),
            n_words=int(row[5]),
            distance=float(row[6]),
            text=str(row[7]),
            score=float(row[8]) if len(row) > 8 else None,
        )
        for row in rows
    ]
    if strategy == "rrf":
        hits.sort(key=lambda h: (-(h.score or 0.0), h.chunk_id))
    else:
        hits.sort(key=lambda h: (h.distance, h.chunk_id))
    return hits


def explain_search(
    conn: pymysql.Connection,
    qvec: np.ndarray,
    k: int = 10,
    filters: Filters | None = None,
    strategy: str = "inline",
    overfetch: int = 10,
    analyze: bool = False,
    query_text: str | None = None,
) -> list[dict[str, Any]]:
    """Return the ``EXPLAIN`` rows (dicts) of the statement :func:`search` would run.

    With ``analyze=True`` the statement is executed under ``ANALYZE`` instead, which adds the
    measured ``r_rows`` (rows actually read per table) and ``r_filtered`` columns. The vector
    index is in use when a row for table ``chunk`` has ``key`` ``embedding`` and ``type``
    ``index``; for ``rrf`` a second ``chunk`` row has ``type`` ``fulltext`` and ``key``
    ``ft_chunk_text``. ``query_text`` is required for ``rrf`` as in :func:`search`. Returns
    ``[]`` for ``titles=[]`` (no statement runs).
    """
    sql, params = search_statement(qvec, k, filters, strategy, overfetch, query_text)
    if not sql:
        return []
    if not analyze:
        return db.explain(conn, sql, params)
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("ANALYZE " + sql, params)
        return [dict(row) for row in cur.fetchall()]


def hit_sentences(
    conn: pymysql.Connection, chunk_ids: Sequence[int]
) -> dict[int, list[tuple[str, str]]]:
    """Return ``{chunk_id: [(element_key, text), ...]}`` for every requested chunk, in page order.

    The map goes through ``chunk_sentence``; sentences are ordered by ``sentence.ordinal``.
    Every requested id is a key (an unknown chunk maps to an empty list); duplicates in
    ``chunk_ids`` are collapsed. An empty request returns ``{}`` without a query.
    Raises TypeError when an id is not an int.
    """
    ids: list[int] = []
    for cid in chunk_ids:
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise TypeError(f"chunk ids must be ints, got {type(cid).__name__}")
        if cid not in ids:
            ids.append(cid)
    result: dict[int, list[tuple[str, str]]] = {cid: [] for cid in ids}
    if not ids:
        return result
    sql = HIT_SENTENCES_PREFIX + _placeholders(len(ids)) + HIT_SENTENCES_SUFFIX
    with conn.cursor() as cur:
        cur.execute(sql, tuple(ids))
        for chunk_id, element_key, text in cur.fetchall():
            result[int(chunk_id)].append((str(element_key), str(text)))
    return result


def page_summary(conn: pymysql.Connection, page_id: int) -> PageSummary | None:
    """Return the :class:`PageSummary` of ``page_id`` (stats, counts, sections), or None.

    None means no page has that id. Raises TypeError when ``page_id`` is not an int.
    """
    if isinstance(page_id, bool) or not isinstance(page_id, int):
        raise TypeError(f"page_id must be an int, got {type(page_id).__name__}")
    with conn.cursor() as cur:
        cur.execute(PAGE_SUMMARY_SQL, (page_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute(PAGE_SECTIONS_SQL, (page_id,))
        sections = tuple(
            SectionRow(ordinal=int(o), heading=str(h), level=int(lv), path=str(p))
            for o, h, lv, p in cur.fetchall()
        )
    return PageSummary(
        page_id=int(row[0]),
        title=str(row[1]),
        n_sentences=int(row[2]),
        n_items=int(row[3]),
        n_words=int(row[4]),
        n_chars=int(row[5]),
        n_sections=int(row[6]),
        n_tables=int(row[7]),
        n_lists=int(row[8]),
        n_chunks=int(row[9]),
        n_links_out=int(row[10]),
        n_links_out_resolved=int(row[11]),
        n_links_in=int(row[12]),
        sections=sections,
    )
