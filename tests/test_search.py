"""Tests for wikilense.search on a synthetic corpus in the MariaDB test database.

The fixture: 3 pages ("Alpha" 100 words, "Beta" 5000, "Gamma" 250), 2 sections each, 12 chunks
(two per section) whose vectors are ``cos(a) e_0 + sin(a) e_i`` with ``a = 0.1 * chunk_id``, so
the distance to the query ``e_0`` is ``1 - cos(0.1 * chunk_id)`` and the exact order is chunk
1, 2, ..., 12. Chunks 1-4 belong to Alpha, 5-8 to Beta, 9-12 to Gamma. Links: Alpha -> Beta,
Beta -> Gamma, Gamma -> Alpha, and Alpha -> "Outside" (not a corpus page, to_page_id NULL).
The pure tests (statement text, argument validation) run without a server.
"""

from __future__ import annotations

from dataclasses import replace

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
        chunks.append(
            (chunk_id, page_id, section_id, chunk_id, f"chunk {chunk_id}", 3,
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


# ---------------------------------------------------------------------------------------------
# database tests
# ---------------------------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_knn_order_without_filters_is_the_expected_order(corpus, strategy: str) -> None:
    hits = search(corpus, QUERY, k=N_CHUNKS, strategy=strategy)
    assert _ids(hits) == list(range(1, N_CHUNKS + 1))
    np.testing.assert_allclose(
        [h.distance for h in hits], [_expected_distance(i) for i in range(1, N_CHUNKS + 1)],
        atol=1e-6,
    )
    assert hits[0] == Hit(chunk_id=1, page_id=ALPHA, title="Alpha", section_path="",
                          chunk_ordinal=1, n_words=3, distance=hits[0].distance, text="chunk 1")
    assert hits[6].title == "Beta" and hits[6].section_path == "Geography"
    assert _ids(search(corpus, QUERY, k=3, strategy=strategy)) == [1, 2, 3]


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
        assert search_statement(QUERY, k=5, filters=Filters(), strategy=strategy) == \
            search_statement(QUERY, k=5, filters=None, strategy=strategy)
        assert search(corpus, QUERY, k=5, filters=Filters(), strategy=strategy) == \
            search(corpus, QUERY, k=5, filters=None, strategy=strategy)


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
    with pytest.raises(ValueError):  # a Python-side failure restores it too
        search(corpus, QUERY, k=0, ef_search=77)
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
    sql, params = search_statement(QUERY, filters=Filters(titles=hostile, heading_like=hostile[2]))
    assert all(value not in sql for value in hostile) and hostile[0] in params
    with corpus.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM page")
        assert cur.fetchone()[0] == 3
