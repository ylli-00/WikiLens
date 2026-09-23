"""Tests for wikilense.search on a synthetic corpus in the MariaDB test database.

The fixture: 3 pages ("Alpha" 100 words, "Beta" 5000, "Gamma" 250), 2 sections each, 12 chunks
(two per section) whose vectors are ``cos(a) e_0 + sin(a) e_i`` with ``a = 0.1 * chunk_id``, so
the distance to the query ``e_0`` is ``1 - cos(0.1 * chunk_id)`` and the exact order is chunk
1, 2, ..., 12. Chunks 1-4 belong to Alpha, 5-8 to Beta, 9-12 to Gamma. Links: Alpha -> Beta,
Beta -> Gamma, Gamma -> Alpha, and Alpha -> "Outside" (not a corpus page, to_page_id NULL).
Chunk texts are "chunk N"; chunk 2 also holds the word "glacier" twice and chunk 12 once, so a
full-text search for "glacier" (the ``rrf`` strategy) ranks chunk 2 first, chunk 12 second and
finds nothing else. The pure tests (statement text, argument validation) run without a server.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import Any

import numpy as np
import pymysql
import pytest

from wikilense import db as dbmod
from wikilense import search as searchmod
from wikilense.search import (
    STRATEGIES,
    Filters,
    Hit,
    PageSummary,
    SectionRow,
    explain_search,
    hit_sentences,
    page_summary,
    search,
    search_statement,
)

DIM = dbmod.VECTOR_DIM
N_CHUNKS = 12
ALPHA, BETA, GAMMA = 1, 2, 3
CHUNKS_OF = {ALPHA: [1, 2, 3, 4], BETA: [5, 6, 7, 8], GAMMA: [9, 10, 11, 12]}
KEYWORDS = {2: " glacier glacier", 12: " glacier"}  # appended to the chunk text
NO_MATCH_TEXT = "zzzznomatch"  # a word no chunk contains: rrf then reduces to the vector list


def _angled(angle: float, axis: int) -> np.ndarray:
    """Return the unit vector cos(angle) * e_0 + sin(angle) * e_axis."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = np.cos(angle)
    v[axis] = np.sin(angle)
    return v


QUERY = _angled(0.0, 1)  # e_0


def _expected_distance(chunk_id: int) -> float:
    return float(1 - np.cos(0.1 * chunk_id))


def _ids(hits: list[Hit]) -> list[int]:
    return [h.chunk_id for h in hits]


def _pages(hits: list[Hit]) -> set[int]:
    return {h.page_id for h in hits}


def _kw(strategy: str) -> dict[str, Any]:
    """Return the extra search() arguments a strategy needs: rrf requires ``query_text``."""
    return {"query_text": NO_MATCH_TEXT} if strategy == "rrf" else {}


@pytest.fixture
def corpus(db_conn: pymysql.Connection) -> pymysql.Connection:
    """The synthetic corpus described in the module docstring, committed; returns the conn."""
    dbmod.insert_rows(
        db_conn,
        "page",
        ("page_id", "title", "n_sentences", "n_items", "n_words", "n_chars", "n_sections",
         "n_tables", "n_lists"),
        [
            (ALPHA, "Alpha", 4, 1, 100, 600, 1, 0, 1),
            (BETA, "Beta", 4, 0, 5000, 30000, 1, 0, 0),
            (GAMMA, "Gamma", 4, 0, 250, 1500, 1, 0, 0),
        ],
    )
    dbmod.insert_rows(
        db_conn,
        "section",
        ("section_id", "page_id", "ordinal", "heading", "level", "path"),
        [
            (1, ALPHA, 0, "", 1, ""),
            (2, ALPHA, 1, "History", 2, "History"),
            (3, BETA, 0, "", 1, ""),
            (4, BETA, 1, "Geography", 2, "Geography"),
            (5, GAMMA, 0, "", 1, ""),
            (6, GAMMA, 1, "Early history", 2, "Early history"),
        ],
    )
    chunks = []
    for chunk_id in range(1, N_CHUNKS + 1):
        section_id = (chunk_id - 1) // 2 + 1
        page_id = (section_id - 1) // 2 + 1
        text = f"chunk {chunk_id}" + KEYWORDS.get(chunk_id, "")
        chunks.append(
            (chunk_id, page_id, section_id, chunk_id, text, 3,
             dbmod.vec_param(_angled(0.1 * chunk_id, chunk_id)))
        )
    dbmod.insert_rows(
        db_conn,
        "chunk",
        ("chunk_id", "page_id", "section_id", "ordinal", "text", "n_words", "embedding"),
        chunks,
    )
    dbmod.insert_rows(
        db_conn,
        "sentence",
        ("sentence_id", "page_id", "section_id", "element_key", "ordinal", "text"),
        [
            (1, ALPHA, 1, "sentence_0", 0, "Alpha lead one."),
            (2, ALPHA, 1, "sentence_1", 1, "Alpha lead two."),
            (3, ALPHA, 1, "sentence_2", 2, "Alpha lead three."),
            (4, ALPHA, 2, "sentence_3", 3, "Alpha history one."),
            (5, ALPHA, 2, "item_0_0", 4, "Alpha history item."),
        ],
    )
    dbmod.insert_rows(
        db_conn,
        "chunk_sentence",
        ("chunk_id", "sentence_id"),
        [(1, 1), (1, 2), (2, 2), (2, 3), (3, 4), (4, 4), (4, 5)],
    )
    dbmod.insert_rows(
        db_conn,
        "link",
        ("from_page_id", "to_title", "to_page_id", "source_element"),
        [
            (ALPHA, "Beta", BETA, "sentence_0"),
            (BETA, "Gamma", GAMMA, "sentence_0"),
            (GAMMA, "Alpha", ALPHA, "sentence_0"),
            (ALPHA, "Outside", None, "sentence_1"),
        ],
    )
    db_conn.commit()
    return db_conn


# ---------------------------------------------------------------------------------------------
# pure tests: statements and validation
# ---------------------------------------------------------------------------------------------


def test_filters_with_no_fields_is_empty_and_values_are_validated() -> None:
    assert Filters().is_empty()
    assert not Filters(titles=[]).is_empty()
    assert not Filters(min_words=0).is_empty()
    assert Filters(titles=["a", "b"]).titles == ("a", "b")
    with pytest.raises(TypeError, match="min_words"):
        Filters(min_words="5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_words"):
        Filters(max_words=-1)
    with pytest.raises(ValueError, match="greater than max_words"):
        Filters(min_words=10, max_words=5)
    with pytest.raises(TypeError, match="heading_like"):
        Filters(heading_like=5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="titles"):
        Filters(titles="Alpha")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="titles"):
        Filters(titles=["Alpha", 3])  # type: ignore[list-item]


def test_search_statement_shapes_and_parameters() -> None:
    qbytes = dbmod.vec_param(QUERY)
    sql, params = search_statement(QUERY, k=7)
    assert sql.startswith(searchmod.INLINE_SELECT)
    assert "STRAIGHT_JOIN page ON page.page_id = chunk.page_id" in sql
    assert sql.endswith("ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s")
    assert "WHERE" not in sql
    assert params == (qbytes, qbytes, 7)

    filters = Filters(min_words=5, heading_like="%History%", titles=["Ab", "Cd", "Ef"],
                      linked_from="Aare")
    sql, params = search_statement(QUERY, k=5, filters=filters)
    assert "STRAIGHT_JOIN (SELECT DISTINCT link.to_page_id" in sql
    assert "WHERE page.n_words >= %s AND section.heading LIKE %s AND page.title IN (%s, %s, %s)" in sql
    assert params == (qbytes, "Aare", 5, "%History%", "Ab", "Cd", "Ef", qbytes, 5)
    assert sql.count("%s") == len(params)
    for value in ("Aare", "%History%", "Ab", "History"):
        assert value not in sql  # values travel only as parameters

    sql, params = search_statement(QUERY, k=5, filters=filters, strategy="overfetch", overfetch=4)
    assert sql.startswith(searchmod.OVERFETCH_SELECT)
    assert "FROM (" + searchmod.KNN_SQL + ") AS knn" in sql
    assert "STRAIGHT_JOIN" not in sql and " JOIN (SELECT DISTINCT link.to_page_id" in sql
    assert sql.endswith("ORDER BY knn.distance, chunk.chunk_id LIMIT %s")
    assert params == (qbytes, qbytes, 20, "Aare", 5, "%History%", "Ab", "Cd", "Ef", 5)
    assert sql.count("%s") == len(params)

    sql_none, params_none = search_statement(QUERY, k=5, filters=filters, strategy="none")
    sql_plain, params_plain = search_statement(QUERY, k=5, strategy="overfetch", overfetch=1)
    assert (sql_none, params_none) == (sql_plain, params_plain)
    assert "WHERE" not in sql_none and params_none == (qbytes, qbytes, 5, 5)

    assert search_statement(QUERY, filters=Filters(titles=[])) == ("", ())
    sql_one, params_one = search_statement(QUERY, filters=Filters(titles=["Only"]))
    assert "page.title IN (%s)" in sql_one and params_one[1] == "Only"


def test_search_statement_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="strategy"):
        search_statement(QUERY, strategy="exact")
    with pytest.raises(ValueError, match="k must be >= 1"):
        search_statement(QUERY, k=0)
    with pytest.raises(TypeError, match="k must be an int"):
        search_statement(QUERY, k=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overfetch"):
        search_statement(QUERY, strategy="overfetch", overfetch=0)
    with pytest.raises(ValueError, match="dimensions"):
        search_statement(np.ones(5, dtype=np.float32))
    with pytest.raises(ValueError):
        search_statement(np.full(DIM, np.nan, dtype=np.float32))
    with pytest.raises(TypeError, match="filters"):
        search_statement(QUERY, filters={"min_words": 5})  # type: ignore[arg-type]


def test_rrf_statement_shape_and_parameters() -> None:
    qbytes = dbmod.vec_param(QUERY)
    sql, params = search_statement(QUERY, k=5, strategy="rrf", overfetch=4, query_text="Aare river")
    assert sql.startswith(searchmod.RRF_WITH) and searchmod.RRF_WITH.startswith("WITH vec AS (")
    assert "FROM (" + searchmod.KNN_SQL + ") AS knn" in sql  # the HNSW search, LIMIT N
    assert "FROM (" + searchmod.FT_SQL + ") AS matched" in sql  # the full-text search, LIMIT N
    assert "MATCH(text) AGAINST (%s IN NATURAL LANGUAGE MODE)" in searchmod.FT_SQL
    # equal relevance is common: chunk_id fixes which tied chunks make the full-text top-N
    assert searchmod.FT_SQL.endswith("ORDER BY relevance DESC, chunk_id LIMIT %s")
    assert sql.count("ROW_NUMBER() OVER (ORDER BY") == 2
    assert f"SUM(CAST(1 AS DOUBLE) / ({searchmod.RRF_K} + ranked.rnk)) AS score" in sql
    assert "VEC_DISTANCE_COSINE(chunk.embedding, %s) AS distance, chunk.text, fused.score" in sql
    assert sql.endswith("ORDER BY fused.score DESC, chunk.chunk_id LIMIT %s")
    assert "WHERE" not in sql.split("FROM fused")[1]
    assert params == (qbytes, qbytes, 20, "Aare river", "Aare river", 20, qbytes, 5)
    assert sql.count("%s") == len(params)
    assert "Aare" not in sql

    filters = Filters(min_words=5, heading_like="%History%", titles=["Ab", "Cd"], links_to="Aare")
    sql, params = search_statement(
        QUERY, k=3, filters=filters, strategy="rrf", overfetch=2, query_text="glacier"
    )
    outer = sql.split("FROM fused")[1]  # filters are outer-query joins and predicates
    assert " JOIN (SELECT DISTINCT link.from_page_id" in outer and "STRAIGHT_JOIN" not in sql
    assert "WHERE page.n_words >= %s AND section.heading LIKE %s AND page.title IN (%s, %s)" in outer
    assert params == (
        qbytes, qbytes, 6, "glacier", "glacier", 6, qbytes, "Aare", 5, "%History%", "Ab", "Cd", 3
    )
    assert sql.count("%s") == len(params)
    for value in ("glacier", "Aare", "%History%", "Ab"):
        assert value not in sql
    assert search_statement(QUERY, strategy="rrf", query_text="x", filters=Filters(titles=[])) \
        == ("", ())


def test_rrf_requires_query_text_and_the_other_strategies_ignore_it() -> None:
    with pytest.raises(ValueError, match="query_text"):
        search_statement(QUERY, strategy="rrf")
    with pytest.raises(ValueError, match="blank"):
        search_statement(QUERY, strategy="rrf", query_text="   ")
    with pytest.raises(TypeError, match="query_text"):
        search_statement(QUERY, strategy="rrf", query_text=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overfetch"):
        search_statement(QUERY, strategy="rrf", overfetch=0, query_text="x")
    for strategy in ("inline", "overfetch", "none"):
        assert search_statement(QUERY, k=4, strategy=strategy, query_text="ignored") == \
            search_statement(QUERY, k=4, strategy=strategy)


class SessionConn:
    """A fake connection that only knows the session variable ``mhnsw_ef_search``."""

    def __init__(self, value: int = 20) -> None:
        self.value = value
        self.sets: list[int] = []

    def cursor(self) -> contextlib.nullcontext[SessionConn]:
        return contextlib.nullcontext(self)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        if sql.startswith("SET SESSION mhnsw_ef_search"):
            self.value = params[0]
            self.sets.append(params[0])
        elif sql != "SELECT @@SESSION.mhnsw_ef_search":
            raise AssertionError(f"unexpected statement {sql!r}")

    def fetchone(self) -> tuple[int]:
        return (self.value,)


def test_ef_search_session_restores_the_value_when_the_block_raises() -> None:
    conn = SessionConn(value=33)
    with pytest.raises(RuntimeError, match="inside"), searchmod.ef_search_session(conn, 77):
        assert conn.value == 77
        raise RuntimeError("inside the block")
    assert conn.value == 33 and conn.sets == [77, 33]
    with searchmod.ef_search_session(conn, None):  # None leaves the session alone
        pass
    assert conn.sets == [77, 33]


def test_ef_search_outside_the_server_range_is_refused_before_anything_is_set() -> None:
    """MariaDB clamps mhnsw_ef_search to 1..10000 with a warning only, so 0 would run at 1 and
    20000 at 10000 while the caller reports its own value."""
    conn = SessionConn()
    for bad in (0, -1, searchmod.MAX_EF_SEARCH + 1, 2.5, True, "100"):
        with (
            pytest.raises(ValueError, match="ef_search"),
            searchmod.ef_search_session(conn, bad),  # type: ignore[arg-type]
        ):
            raise AssertionError("the block must not run")
    assert conn.sets == []
    for good in (searchmod.MIN_EF_SEARCH, searchmod.MAX_EF_SEARCH):
        with searchmod.ef_search_session(conn, good):
            assert conn.value == good


def test_resolve_ef_search_takes_the_request_then_the_default_and_zero_is_the_server() -> None:
    assert searchmod.resolve_ef_search(None, 100) == 100
    assert searchmod.resolve_ef_search(40, 100) == 40
    assert searchmod.resolve_ef_search(searchmod.EF_SEARCH_SERVER, 100) is None
    assert searchmod.resolve_ef_search(None, searchmod.EF_SEARCH_SERVER) is None


def test_overfetch_above_the_maximum_is_refused() -> None:
    with pytest.raises(ValueError, match="overfetch"):
        search_statement(QUERY, strategy="overfetch", overfetch=searchmod.MAX_OVERFETCH + 1)
    _sql, params = search_statement(QUERY, k=2, strategy="overfetch",
                                    overfetch=searchmod.MAX_OVERFETCH)
    assert params[2] == 2 * searchmod.MAX_OVERFETCH


class RowsConn:
    """A fake connection whose one statement returns the given rows (search() without a server)."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.executed: list[str] = []

    def cursor(self) -> contextlib.nullcontext[RowsConn]:
        return contextlib.nullcontext(self)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append(sql)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


def _row(chunk_id: int, distance: float, score: float | None = None) -> tuple[Any, ...]:
    row = (chunk_id, 1, "Alpha", "", 0, 3, distance, f"chunk {chunk_id}")
    return row if score is None else (*row, score)


def test_ties_are_broken_by_chunk_id_in_python() -> None:
    """The ORDER BY carries only the distance (a second key loses the vector index), so equal
    distances arrive in any order; search() puts them in chunk_id order."""
    rows = [_row(9, 0.2), _row(4, 0.1), _row(7, 0.1), _row(2, 0.1)]
    for strategy in ("inline", "overfetch", "none"):
        assert _ids(search(RowsConn(rows), QUERY, k=4, strategy=strategy)) == [2, 4, 7, 9]
    fused = [_row(9, 0.5, 0.02), _row(5, 0.1, 0.03), _row(3, 0.9, 0.02)]
    hits = search(RowsConn(fused), QUERY, k=3, strategy="rrf", query_text="x")
    assert _ids(hits) == [5, 3, 9]  # rrf: score descending, then chunk_id


def test_schema_declares_the_fulltext_index_on_chunk_text() -> None:
    script = dbmod.SCHEMA_PATH.read_text(encoding="utf-8")
    chunk_ddl = script.split("CREATE TABLE IF NOT EXISTS chunk (")[1].split(") ENGINE=InnoDB")[0]
    assert "FULLTEXT KEY ft_chunk_text (text)," in chunk_ddl
    assert "VECTOR INDEX (embedding)" in chunk_ddl


# ---------------------------------------------------------------------------------------------
# database tests
# ---------------------------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_knn_order_without_filters_is_the_expected_order(corpus, strategy: str) -> None:
    hits = search(corpus, QUERY, k=N_CHUNKS, strategy=strategy, **_kw(strategy))
    assert _ids(hits) == list(range(1, N_CHUNKS + 1))
    np.testing.assert_allclose(
        [h.distance for h in hits], [_expected_distance(i) for i in range(1, N_CHUNKS + 1)],
        atol=1e-6,
    )
    assert replace(hits[0], score=None) == Hit(
        chunk_id=1, page_id=ALPHA, title="Alpha", section_path="", chunk_ordinal=1, n_words=3,
        distance=hits[0].distance, text="chunk 1",
    )
    if strategy == "rrf":  # no full-text match: the score is the vector rank alone
        np.testing.assert_allclose([h.score for h in hits], [1 / (60 + r) for r in range(1, 13)])
    else:
        assert all(h.score is None for h in hits)
    assert hits[6].title == "Beta" and hits[6].section_path == "Geography"
    assert _ids(search(corpus, QUERY, k=3, strategy=strategy, **_kw(strategy))) == [1, 2, 3]


@pytest.mark.db
@pytest.mark.parametrize("strategy", ["inline", "overfetch"])
def test_word_count_filters_restrict_to_the_expected_pages(corpus, strategy: str) -> None:
    kw = {"strategy": strategy, "overfetch": N_CHUNKS}
    assert _ids(search(corpus, QUERY, k=N_CHUNKS, filters=Filters(min_words=1000), **kw)) == \
        CHUNKS_OF[BETA]
    assert _ids(search(corpus, QUERY, k=N_CHUNKS, filters=Filters(max_words=200), **kw)) == \
        CHUNKS_OF[ALPHA]
    both = Filters(min_words=150, max_words=300)
    assert _ids(search(corpus, QUERY, k=N_CHUNKS, filters=both, **kw)) == CHUNKS_OF[GAMMA]
    assert search(corpus, QUERY, k=N_CHUNKS, filters=Filters(min_words=10_000), **kw) == []


@pytest.mark.db
@pytest.mark.parametrize("strategy", ["inline", "overfetch"])
def test_heading_and_path_filters(corpus, strategy: str) -> None:
    kw = {"strategy": strategy, "overfetch": N_CHUNKS}
    # section.heading is case-insensitive: "Early history" matches "%History%" too
    hits = search(corpus, QUERY, k=N_CHUNKS, filters=Filters(heading_like="%History%"), **kw)
    assert _ids(hits) == [3, 4, 11, 12]
    assert [h.section_path for h in hits] == ["History"] * 2 + ["Early history"] * 2
    hits = search(corpus, QUERY, k=N_CHUNKS, filters=Filters(path_like="Geography%"), **kw)
    assert _ids(hits) == [7, 8]
    hits = search(corpus, QUERY, k=N_CHUNKS, filters=Filters(heading_like="History"), **kw)
    assert _ids(hits) == [3, 4]  # no wildcard: exact (case-insensitive) match only
    assert search(corpus, QUERY, k=N_CHUNKS, filters=Filters(path_like="Nowhere%"), **kw) == []


@pytest.mark.db
@pytest.mark.parametrize("strategy", ["inline", "overfetch"])
def test_link_filters(corpus, strategy: str) -> None:
    kw = {"strategy": strategy, "overfetch": N_CHUNKS}
    linked_from = lambda t: _ids(search(corpus, QUERY, k=N_CHUNKS, filters=Filters(linked_from=t), **kw))
    links_to = lambda t: _ids(search(corpus, QUERY, k=N_CHUNKS, filters=Filters(links_to=t), **kw))
    assert linked_from("Alpha") == CHUNKS_OF[BETA]  # the unresolved "Outside" link adds nothing
    assert linked_from("Beta") == CHUNKS_OF[GAMMA]
    assert linked_from("Gamma") == CHUNKS_OF[ALPHA]
    assert linked_from("Nope") == []
    assert links_to("Alpha") == CHUNKS_OF[GAMMA]
    assert links_to("Beta") == CHUNKS_OF[ALPHA]
    assert links_to("Outside") == []  # not a corpus page: no page row, so no match
    assert linked_from("alpha") == []  # titles are utf8mb4_bin: exact case


@pytest.mark.db
@pytest.mark.parametrize("strategy", ["inline", "overfetch"])
def test_titles_filter_with_zero_one_and_three_titles(corpus, strategy: str) -> None:
    kw = {"strategy": strategy, "overfetch": N_CHUNKS}
    assert search(corpus, QUERY, k=N_CHUNKS, filters=Filters(titles=[]), **kw) == []
    assert _ids(search(corpus, QUERY, k=N_CHUNKS, filters=Filters(titles=["Beta"]), **kw)) == \
        CHUNKS_OF[BETA]
    three = Filters(titles=["Alpha", "Gamma", "Nope"])
    assert _ids(search(corpus, QUERY, k=N_CHUNKS, filters=three, **kw)) == \
        CHUNKS_OF[ALPHA] + CHUNKS_OF[GAMMA]


@pytest.mark.db
def test_combined_filters_and_strategy_none_ignores_them(corpus) -> None:
    filters = Filters(min_words=150, heading_like="%history%", links_to="Alpha")
    assert _ids(search(corpus, QUERY, k=N_CHUNKS, filters=filters)) == [11, 12]
    assert _ids(search(corpus, QUERY, k=N_CHUNKS, filters=filters, strategy="overfetch",
                       overfetch=N_CHUNKS)) == [11, 12]
    assert _ids(search(corpus, QUERY, k=3, filters=Filters(titles=["Gamma"]), strategy="none")) \
        == [1, 2, 3]
    assert _ids(search(corpus, QUERY, k=3, filters=Filters(titles=[]), strategy="none")) == [1, 2, 3]


@pytest.mark.db
def test_empty_filters_equal_no_filter(corpus) -> None:
    for strategy in STRATEGIES:
        kw = _kw(strategy)
        assert search_statement(QUERY, k=5, filters=Filters(), strategy=strategy, **kw) == \
            search_statement(QUERY, k=5, filters=None, strategy=strategy, **kw)
        assert search(corpus, QUERY, k=5, filters=Filters(), strategy=strategy, **kw) == \
            search(corpus, QUERY, k=5, filters=None, strategy=strategy, **kw)


@pytest.mark.db
def test_inline_and_overfetch_agree_when_the_filter_is_not_selective(corpus) -> None:
    filters = Filters(min_words=1)  # every page passes
    inline = search(corpus, QUERY, k=5, filters=filters, strategy="inline")
    over = search(corpus, QUERY, k=5, filters=filters, strategy="overfetch", overfetch=2)
    assert inline == over
    assert _ids(inline) == [1, 2, 3, 4, 5]


@pytest.mark.db
def test_selective_filter_inline_walks_on_while_overfetch_depends_on_its_factor(corpus) -> None:
    """Gamma's chunks (9-12) are the farthest from the query. With k=2:

    - inline: MariaDB keeps walking the vector index past the first k rows until k rows pass
      the title predicate, so exactly k hits come back (measured on 11.8.9);
    - overfetch: only the inner ``k * overfetch`` nearest chunks are ever seen, so a factor
      of 4 (inner LIMIT 8, chunks 1-8) finds nothing and a factor of 5 (inner LIMIT 10)
      recovers the two nearest Gamma chunks.
    """
    filters = Filters(titles=["Gamma"])
    assert _ids(search(corpus, QUERY, k=2, filters=filters, strategy="inline")) == [9, 10]
    assert search(corpus, QUERY, k=2, filters=filters, strategy="overfetch", overfetch=1) == []
    assert search(corpus, QUERY, k=2, filters=filters, strategy="overfetch", overfetch=4) == []
    assert _ids(search(corpus, QUERY, k=2, filters=filters, strategy="overfetch", overfetch=5)) \
        == [9, 10]
    assert _ids(search(corpus, QUERY, k=2, filters=filters, strategy="overfetch", overfetch=6)) \
        == [9, 10]


@pytest.mark.db
def test_explain_search_reports_the_vector_index(corpus) -> None:
    plan = explain_search(corpus, QUERY, k=3)
    assert [row["table"] for row in plan] == ["chunk", "page", "section"], plan
    assert plan[0]["key"] == "embedding" and plan[0]["type"] == "index", plan[0]
    assert plan[1]["type"] == "eq_ref" and plan[2]["type"] == "eq_ref", plan
    for strategy in ("overfetch", "none"):
        plan = explain_search(corpus, QUERY, k=3, strategy=strategy)
        derived = [row for row in plan if row["select_type"] == "DERIVED"]
        assert len(derived) == 1 and derived[0]["table"] == "chunk", plan
        assert derived[0]["key"] == "embedding" and derived[0]["type"] == "index", derived
    analyzed = explain_search(corpus, QUERY, k=3, analyze=True)
    assert analyzed[0]["table"] == "chunk" and analyzed[0]["key"] == "embedding", analyzed
    assert "r_rows" in analyzed[0] and float(analyzed[0]["r_rows"]) == 3.0, analyzed[0]
    assert explain_search(corpus, QUERY, filters=Filters(titles=[])) == []


@pytest.mark.db
def test_rrf_fuses_the_vector_and_full_text_ranks(corpus) -> None:
    """With N = k * overfetch = 6 the vector list is chunks 1-6 and the full-text list for
    "glacier" is chunk 2 (two occurrences), then chunk 12 (one). Chunk 2 is in both lists and
    comes first; chunk 1 (vector rank 1 only, 1/61) beats chunk 12 (full-text rank 2 only,
    1/62); the rest follow the vector ranks. Chunk 12 was never seen by the vector search, and
    its distance is still the cosine distance."""
    hits = search(corpus, QUERY, k=6, strategy="rrf", overfetch=1, query_text="glacier")
    assert _ids(hits) == [2, 1, 12, 3, 4, 5]
    np.testing.assert_allclose(
        [h.score for h in hits], [1 / 62 + 1 / 61, 1 / 61, 1 / 62, 1 / 63, 1 / 64, 1 / 65],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        [h.distance for h in hits], [_expected_distance(i) for i in _ids(hits)], atol=1e-6
    )
    assert hits[2].title == "Gamma" and hits[2].text == "chunk 12 glacier"
    assert hits[0].text == "chunk 2 glacier glacier"
    # N = 10: chunk 10 (1/70, vector only) falls behind chunk 12 (1/62) and is cut by LIMIT k
    hits = search(corpus, QUERY, k=10, strategy="rrf", overfetch=1, query_text="glacier")
    assert _ids(hits) == [2, 1, 12, 3, 4, 5, 6, 7, 8, 9]
    # the query text is bound as data and natural-language mode has no operators: "-meltwater"
    # is just a word that no chunk contains
    hits = search(corpus, QUERY, k=3, strategy="rrf", overfetch=2,
                  query_text="+glacier -meltwater")
    assert _ids(hits) == [2, 1, 12]
    assert all(h.score is None for h in search(corpus, QUERY, k=3, strategy="inline"))


@pytest.mark.db
def test_rrf_without_a_full_text_match_is_the_vector_list_with_rank_scores(corpus) -> None:
    hits = search(corpus, QUERY, k=N_CHUNKS, strategy="rrf", query_text=NO_MATCH_TEXT)
    assert _ids(hits) == list(range(1, N_CHUNKS + 1))
    np.testing.assert_allclose(
        [h.score for h in hits], [1 / (60 + r) for r in range(1, N_CHUNKS + 1)], rtol=1e-9
    )
    # InnoDB stopwords ("the") and tokens under innodb_ft_min_token_size (3) match nothing
    assert _ids(search(corpus, QUERY, k=3, strategy="rrf", query_text="the of a")) == [1, 2, 3]


@pytest.mark.db
def test_rrf_filters_apply_to_the_fused_list_only(corpus) -> None:
    """Filters are outer-query predicates: a chunk outside both top-N lists is never seen.
    With N = 6 the fused list is chunks 1-6 (Alpha, Beta) plus chunk 12 (Gamma, full text)."""
    kw = {"strategy": "rrf", "overfetch": 2, "query_text": "glacier"}
    assert _ids(search(corpus, QUERY, k=3, filters=Filters(titles=["Gamma"]), **kw)) == [12]
    assert _ids(search(corpus, QUERY, k=3, filters=Filters(links_to="Alpha"), **kw)) == [12]
    assert _ids(search(corpus, QUERY, k=3, filters=Filters(linked_from="Alpha"), **kw)) == [5, 6]
    assert _ids(search(corpus, QUERY, k=3, filters=Filters(max_words=200), **kw)) == [2, 1, 3]
    assert _ids(search(corpus, QUERY, k=3, filters=Filters(heading_like="%history%"), **kw)) \
        == [12, 3, 4]
    # Beta's Geography chunks (7, 8) are outside both top-6 lists: nothing, until N reaches them
    assert search(corpus, QUERY, k=3, filters=Filters(path_like="Geography%"), **kw) == []
    assert _ids(search(corpus, QUERY, k=3, filters=Filters(path_like="Geography%"), strategy="rrf",
                       overfetch=3, query_text="glacier")) == [7, 8]
    assert search(corpus, QUERY, k=3, filters=Filters(min_words=10_000), **kw) == []
    assert search(corpus, QUERY, k=3, filters=Filters(titles=[]), **kw) == []
    # with N covering the whole table the filter sees every chunk, like inline
    assert _ids(search(corpus, QUERY, k=4, filters=Filters(titles=["Gamma"]), strategy="rrf",
                       overfetch=3, query_text="glacier")) == [12, 9, 10, 11]


@pytest.mark.db
def test_rrf_explain_shows_both_indexes_and_ef_search_is_restored(corpus) -> None:
    kw = {"strategy": "rrf", "overfetch": 2, "query_text": "glacier"}
    plan = explain_search(corpus, QUERY, k=3, **kw)
    assert plan[0]["select_type"] == "PRIMARY", plan
    chunk_rows = {(row["type"], row["key"]) for row in plan if row["table"] == "chunk"}
    assert {("index", "embedding"), ("fulltext", "ft_chunk_text")} <= chunk_rows, plan
    analyzed = explain_search(corpus, QUERY, k=3, analyze=True, **kw)
    by_key = {row["key"]: row for row in analyzed if row["table"] == "chunk"}
    assert float(by_key["embedding"]["r_rows"]) == 6.0, analyzed  # the vector top-N, N = 6
    assert float(by_key["ft_chunk_text"]["r_rows"]) == 2.0, analyzed  # the two matches
    with pytest.raises(ValueError, match="query_text"):
        explain_search(corpus, QUERY, k=3, strategy="rrf")
    assert explain_search(corpus, QUERY, filters=Filters(titles=[]), **kw) == []
    dbmod.set_session_var(corpus, "mhnsw_ef_search", 33)
    assert _ids(search(corpus, QUERY, k=2, ef_search=77, **kw)) == [2, 1]
    assert dbmod.get_session_var(corpus, "mhnsw_ef_search") == 33


@pytest.mark.db
def test_fulltext_index_exists_in_the_database(db_conn) -> None:
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT INDEX_NAME, INDEX_TYPE FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
            ("chunk", "text"),
        )
        assert cur.fetchall() == (("ft_chunk_text", "FULLTEXT"),)


@pytest.mark.db
def test_ef_search_is_set_for_the_call_and_restored_even_when_the_query_raises(corpus) -> None:
    dbmod.set_session_var(corpus, "mhnsw_ef_search", 33)
    hits = search(corpus, QUERY, k=2, ef_search=77)
    assert _ids(hits) == [1, 2]
    assert dbmod.get_session_var(corpus, "mhnsw_ef_search") == 33
    with corpus.cursor() as cur:  # make the statement itself fail on the server
        cur.execute("DROP TABLE chunk_sentence")
        cur.execute("DROP TABLE chunk")
    with pytest.raises(pymysql.err.ProgrammingError):
        search(corpus, QUERY, k=2, ef_search=77)
    assert dbmod.get_session_var(corpus, "mhnsw_ef_search") == 33


@pytest.mark.db
def test_hit_sentences_maps_chunks_to_their_sentences_in_order(corpus) -> None:
    result = hit_sentences(corpus, [4, 1, 2, 999, 1])
    assert list(result) == [4, 1, 2, 999]  # requested order, duplicates collapsed
    assert result[1] == [("sentence_0", "Alpha lead one."), ("sentence_1", "Alpha lead two.")]
    assert result[2] == [("sentence_1", "Alpha lead two."), ("sentence_2", "Alpha lead three.")]
    assert result[4] == [("sentence_3", "Alpha history one."), ("item_0_0", "Alpha history item.")]
    assert result[999] == []
    assert hit_sentences(corpus, []) == {}
    assert hit_sentences(corpus, [5]) == {5: []}  # a chunk without mapped sentences
    with pytest.raises(TypeError):
        hit_sentences(corpus, ["1"])  # type: ignore[list-item]


@pytest.mark.db
def test_page_summary(corpus) -> None:
    summary = page_summary(corpus, ALPHA)
    assert summary == PageSummary(
        page_id=ALPHA, title="Alpha", n_sentences=4, n_items=1, n_words=100, n_chars=600,
        n_sections=1, n_tables=0, n_lists=1, n_chunks=4, n_links_out=2,
        n_links_out_resolved=1, n_links_in=1,
        sections=(SectionRow(0, "", 1, ""), SectionRow(1, "History", 2, "History")),
    )
    assert replace(page_summary(corpus, GAMMA), sections=()).n_links_in == 1
    assert page_summary(corpus, 999) is None
    with pytest.raises(TypeError):
        page_summary(corpus, "1")  # type: ignore[arg-type]


@pytest.mark.db
def test_injection_shaped_inputs_are_data(corpus) -> None:
    hostile = ["Alpha' OR '1'='1", "x'); DROP TABLE page; -- ", "%' OR 1=1 -- "]
    assert search(corpus, QUERY, k=N_CHUNKS, filters=Filters(heading_like=hostile[2])) == []
    assert search(corpus, QUERY, k=N_CHUNKS, filters=Filters(titles=hostile)) == []
    assert search(corpus, QUERY, k=N_CHUNKS, filters=Filters(linked_from=hostile[0])) == []
    assert search(corpus, QUERY, k=N_CHUNKS, filters=Filters(path_like=hostile[1]),
                  strategy="overfetch") == []
    assert _ids(search(corpus, QUERY, k=3, strategy="rrf", query_text=hostile[1])) == [1, 2, 3]
    sql, params = search_statement(QUERY, filters=Filters(titles=hostile, heading_like=hostile[2]))
    assert all(value not in sql for value in hostile) and hostile[0] in params
    with corpus.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM page")
        assert cur.fetchone()[0] == 3
