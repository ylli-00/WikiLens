"""Tests for wikilense.evaluate.

The pure metric functions run on hand-made hit lists without a server. The database tests use
the synthetic corpus of tests/test_search.py (12 chunks whose vector is ``cos(a) e_0 +
sin(a) e_i`` with ``a = 0.1 * chunk_id``, so a query ``e_0`` ranks them 1, 2, ..., 12) plus
sentences, a chunk-to-sentence map and six claims with evidence sets that cover every case of
``load_ground_truth``. The fake embedder returns a vector chosen per claim text, so the exact
ranking, and with it every recall value, is known by hand.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pymysql
import pytest

from wikilense import db as dbmod
from wikilense import search as searchmod
from wikilense.config import Settings
from wikilense.evaluate import (
    SERVER_DEFAULT_EF_SEARCH,
    ClaimOutcome,
    ClaimTruth,
    EvalResult,
    KMetrics,
    LatencyStats,
    article_recall_at_k,
    chunk_sentence_ids,
    covered_units,
    evaluate,
    evidence_rank,
    evidence_recall_at_k,
    gold_page_rank,
    latency_stats,
    load_ground_truth,
    oracle_title_filters,
    parse_vector_index,
    results_markdown,
    unit_coverage_at_k,
    worst_claims,
    write_results,
)
from wikilense.search import Filters, Hit

DIM = dbmod.VECTOR_DIM
N_CHUNKS = 12
ALPHA, BETA, GAMMA = 1, 2, 3


def _angled(angle: float, axis: int) -> np.ndarray:
    """Return the unit vector cos(angle) * e_0 + sin(angle) * e_axis."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = np.cos(angle)
    v[axis] = np.sin(angle)
    return v


E0 = _angled(0.0, 1)
#: Nearest to chunk 9 (distance 0), then 1, 2, 3, ... (distance 1 - cos(0.9) cos(0.1 j)).
NEAR_NINE = _angled(0.9, 9)

#: claim_id -> (text, query vector). Claims 1, 3, 4, 5, 6 rank the chunks 1..12; claim 2 ranks
#: them 9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12.
CLAIM_QUERIES: dict[int, tuple[str, np.ndarray]] = {
    1: ("alpha claim", E0),
    2: ("gamma claim", NEAR_NINE),
    3: ("beta claim", E0),
    4: ("alpha two", E0),
    5: ("alpha unresolved", E0),
    6: ("nowhere claim", E0),
}


class VectorEmbedder:
    """embed_queries returns the vector registered for each text; ``calls`` records the inputs."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []
        self.model_name = "fake-query-embedder"
        self.device = "cpu"

    def embed_queries(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([self.vectors[text] for text in texts]).astype(np.float32)


@pytest.fixture
def embedder() -> VectorEmbedder:
    return VectorEmbedder({text: vec for text, vec in CLAIM_QUERIES.values()})


@pytest.fixture
def corpus(db_conn: pymysql.Connection) -> pymysql.Connection:
    """The synthetic corpus and claims described in the module docstring, committed."""
    dbmod.insert_rows(
        db_conn,
        "page",
        ("page_id", "title", "n_sentences", "n_items", "n_words", "n_chars", "n_sections",
         "n_tables", "n_lists"),
        [
            (ALPHA, "Alpha", 4, 1, 100, 600, 1, 0, 1),
            (BETA, "Beta", 4, 0, 5000, 30000, 1, 0, 0),
            (GAMMA, "Gamma", 1, 1, 250, 1500, 1, 0, 1),
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
            (6, GAMMA, 5, "sentence_0", 0, "Gamma lead."),
            (7, GAMMA, 5, "item_0_0", 1, "Gamma item."),
        ],
    )
    # chunk 1 {1, 2}, chunk 2 {2, 3}, chunk 3 {4}, chunk 4 {4, 5}, chunk 9 {6}, chunk 10 {7};
    # the Beta chunks 5-8 carry no sentence.
    dbmod.insert_rows(
        db_conn,
        "chunk_sentence",
        ("chunk_id", "sentence_id"),
        [(1, 1), (1, 2), (2, 2), (2, 3), (3, 4), (4, 4), (4, 5), (9, 6), (10, 7)],
    )
    dbmod.insert_rows(
        db_conn,
        "claim",
        ("claim_id", "split", "text", "label", "challenge"),
        [
            (1, "train", "alpha claim", "SUPPORTS", "Other"),
            (2, "dev", "gamma claim", "REFUTES", "Numerical Reasoning"),
            (3, "train", "beta claim", "REFUTES", "Other"),
            (4, "train", "alpha two", "NOT ENOUGH INFO", "Other"),
            (5, "dev", "alpha unresolved", "SUPPORTS", "Other"),
            (6, "train", "nowhere claim", "REFUTES", "Other"),
        ],
    )
    dbmod.insert_rows(
        db_conn,
        "claim_evidence",
        ("claim_id", "evidence_set", "position", "element_id", "page_title", "element_type",
         "page_id", "sentence_id"),
        [
            # claim 1: one sentence-only set {1, 2}
            (1, 0, 0, "Alpha_sentence_0", "Alpha", "sentence", ALPHA, 1),
            (1, 0, 1, "Alpha_sentence_1", "Alpha", "sentence", ALPHA, 2),
            # claim 2: a sentence and a list item, both resolved -> sentence-only {6, 7}
            (2, 0, 0, "Gamma_sentence_0", "Gamma", "sentence", GAMMA, 6),
            (2, 0, 1, "Gamma_item_0_0", "Gamma", "item", GAMMA, 7),
            # claim 3: a table cell (page resolved, no sentence id) -> not sentence-only
            (3, 0, 0, "Beta_cell_0_0_0", "Beta", "cell", BETA, None),
            # claim 4: two alternative sentence-only sets {3} and {4, 5}
            (4, 0, 0, "Alpha_sentence_2", "Alpha", "sentence", ALPHA, 3),
            (4, 1, 0, "Alpha_sentence_3", "Alpha", "sentence", ALPHA, 4),
            (4, 1, 1, "Alpha_item_0_0", "Alpha", "item", ALPHA, 5),
            # claim 5: an unresolved page, and a sentence of a corpus page whose id did not
            # resolve -> gold page Alpha, no sentence-only set
            (5, 0, 0, "Outside_sentence_0", "Outside", "sentence", None, None),
            (5, 1, 0, "Alpha_sentence_99", "Alpha", "sentence", ALPHA, None),
            # claim 6: nothing resolved
            (6, 0, 0, "Nowhere_sentence_0", "Nowhere", "sentence", None, None),
        ],
    )
    db_conn.commit()
    return db_conn


# ---------------------------------------------------------------------------------------------
# pure metric functions
# ---------------------------------------------------------------------------------------------

HIT_PAGES = [3, 3, 7, 9]
HIT_UNITS = [[1, 2], [2, 3], [9], [4]]


def test_article_recall_at_k() -> None:
    assert article_recall_at_k(HIT_PAGES, {9}, 1) is False
    assert article_recall_at_k(HIT_PAGES, {9}, 3) is False
    assert article_recall_at_k(HIT_PAGES, {9}, 4) is True
    assert article_recall_at_k(HIT_PAGES, {9}, 10) is True  # k larger than the hit list
    assert article_recall_at_k(HIT_PAGES, {3}, 1) is True
    assert article_recall_at_k(HIT_PAGES, {7, 100}, 3) is True
    # duplicate pages across chunks do not help a page that is not gold
    assert article_recall_at_k([3, 3, 3], {7}, 3) is False
    assert article_recall_at_k([3, 3, 3], {3}, 1) is True
    for k in (1, 3, 10):
        assert article_recall_at_k(HIT_PAGES, set(), k) is False  # empty gold
        assert article_recall_at_k([], {3}, k) is False  # no hits
    with pytest.raises(ValueError):
        article_recall_at_k(HIT_PAGES, {3}, 0)
    with pytest.raises(ValueError):
        article_recall_at_k(HIT_PAGES, {3}, True)


def test_gold_page_rank() -> None:
    assert gold_page_rank(HIT_PAGES, {3}) == 1
    assert gold_page_rank(HIT_PAGES, {7}) == 3
    assert gold_page_rank(HIT_PAGES, {9, 7}) == 3
    assert gold_page_rank(HIT_PAGES, {1}) is None
    assert gold_page_rank(HIT_PAGES, set()) is None
    assert gold_page_rank([], {3}) is None


def test_covered_units_and_evidence_recall() -> None:
    assert covered_units(HIT_UNITS, 1) == {1, 2}
    assert covered_units(HIT_UNITS, 2) == {1, 2, 3}
    assert covered_units(HIT_UNITS, 99) == {1, 2, 3, 4, 9}
    assert covered_units([], 5) == set()
    # two alternative sets, of which only {3} is covered at k = 2
    sets = [{1, 4}, {3}]
    assert evidence_recall_at_k(HIT_UNITS, sets, 1) is False
    assert evidence_recall_at_k(HIT_UNITS, sets, 2) is True
    assert evidence_recall_at_k(HIT_UNITS, sets, 3) is True
    # a single set that needs chunks 1 and 4
    assert evidence_recall_at_k(HIT_UNITS, [{1, 4}], 3) is False
    assert evidence_recall_at_k(HIT_UNITS, [{1, 4}], 4) is True
    assert evidence_recall_at_k(HIT_UNITS, [{1, 4}], 10) is True  # k larger than hits
    # a partially covered set never counts, whatever the k
    assert evidence_recall_at_k(HIT_UNITS, [{1, 42}], 10) is False
    # no sets, or only an empty set: never recalled
    assert evidence_recall_at_k(HIT_UNITS, [], 4) is False
    assert evidence_recall_at_k(HIT_UNITS, [set()], 4) is False
    assert evidence_recall_at_k([], [{1}], 4) is False
    with pytest.raises(ValueError):
        evidence_recall_at_k(HIT_UNITS, sets, 0)


def test_unit_coverage_at_k_is_the_best_share_over_the_sets() -> None:
    sets = [{1, 4}, {3}]
    assert unit_coverage_at_k(HIT_UNITS, sets, 1) == 0.5  # {1, 4}: 1 of 2; {3}: 0 of 1
    assert unit_coverage_at_k(HIT_UNITS, sets, 2) == 1.0  # {3} fully covered
    assert unit_coverage_at_k(HIT_UNITS, [{1, 4}], 3) == 0.5
    assert unit_coverage_at_k(HIT_UNITS, [{1, 4}], 4) == 1.0
    assert unit_coverage_at_k(HIT_UNITS, [{1, 2, 3, 42}], 10) == 0.75
    assert unit_coverage_at_k(HIT_UNITS, [{40, 41}], 10) == 0.0
    assert unit_coverage_at_k(HIT_UNITS, [], 4) == 0.0
    assert unit_coverage_at_k(HIT_UNITS, [set(), {1}], 1) == 1.0  # empty sets are ignored
    assert unit_coverage_at_k([], [{1}], 4) == 0.0
    # coverage 1.0 exactly when evidence recall holds
    for k in (1, 2, 3, 4, 8):
        assert (unit_coverage_at_k(HIT_UNITS, sets, k) == 1.0) is evidence_recall_at_k(
            HIT_UNITS, sets, k
        )


def test_evidence_rank() -> None:
    assert evidence_rank(HIT_UNITS, [{1, 4}, {3}]) == 2
    assert evidence_rank(HIT_UNITS, [{1, 4}]) == 4
    assert evidence_rank(HIT_UNITS, [{1, 2}]) == 1
    assert evidence_rank(HIT_UNITS, [{1, 42}]) is None
    assert evidence_rank(HIT_UNITS, []) is None
    assert evidence_rank([], [{1}]) is None


def test_latency_stats() -> None:
    stats = latency_stats([float(i) for i in range(1, 11)])
    assert (stats.n, stats.p50_ms, stats.mean_ms, stats.min_ms, stats.max_ms) == (
        10, 5.5, 5.5, 1.0, 10.0
    )
    assert stats.p95_ms == pytest.approx(9.55)  # linear interpolation between 9 and 10
    single = latency_stats([2.5])
    assert (single.p50_ms, single.p95_ms, single.mean_ms) == (2.5, 2.5, 2.5)
    with pytest.raises(ValueError):
        latency_stats([])


def test_evaluate_rejects_bad_arguments_before_touching_the_database() -> None:
    class NoConnection:
        def cursor(self):  # pragma: no cover - reaching it is the failure
            raise AssertionError("the database must not be touched")

    conn = NoConnection()
    fake = VectorEmbedder({})
    with pytest.raises(ValueError, match="ks"):
        evaluate(conn, fake, ks=())
    with pytest.raises(ValueError, match="k must be"):
        evaluate(conn, fake, ks=(1, 0))
    with pytest.raises(ValueError, match="repeats"):
        evaluate(conn, fake, repeats=0)
    with pytest.raises(ValueError, match="strategy"):
        evaluate(conn, fake, strategy="fast")
    for bad in (0, -5, searchmod.MAX_EF_SEARCH + 1, "fast", 2.5, True):
        with pytest.raises(ValueError, match="ef_search"):
            evaluate(conn, fake, ef_search=bad)
    with pytest.raises(ValueError, match="ignores filters"):
        evaluate(conn, fake, strategy="none", filters=Filters(min_words=1))
    with pytest.raises(ValueError, match="ignores filters"):
        evaluate(conn, fake, strategy="none", claim_filters=oracle_title_filters)
    with pytest.raises(ValueError, match="not both"):
        evaluate(
            conn, fake, strategy="inline", filters=Filters(min_words=1),
            claim_filters=oracle_title_filters,
        )
    with pytest.raises(TypeError, match="filters"):
        evaluate(conn, fake, strategy="inline", filters={"min_words": 1})


def test_strategy_validation_follows_search_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every strategy that search.STRATEGIES lists passes validation, read at call time."""

    class NoConnection:
        def cursor(self):
            raise AssertionError("validation passed: the database was reached")

    fake = VectorEmbedder({})
    assert "rrf" in searchmod.STRATEGIES
    for strategy in searchmod.STRATEGIES:
        with pytest.raises(AssertionError, match="validation passed"):
            evaluate(NoConnection(), fake, strategy=strategy, ef_search=SERVER_DEFAULT_EF_SEARCH)
    monkeypatch.setattr(searchmod, "STRATEGIES", ("inline", "none"))
    with pytest.raises(ValueError, match="'inline', 'none'"):
        evaluate(NoConnection(), fake, strategy="rrf", ef_search=SERVER_DEFAULT_EF_SEARCH)
    with pytest.raises(ValueError, match="use one of 'inline' with them"):
        evaluate(NoConnection(), fake, strategy="none", filters=Filters(min_words=1))


def _pure_evaluate(
    monkeypatch: pytest.MonkeyPatch, truths: list[ClaimTruth], search_fn: Any, **kwargs: Any
) -> EvalResult:
    """Run ``evaluate`` with every database helper replaced, so only ``search_fn`` decides.

    Chunk ``c`` carries the single sentence id ``c``; the query vectors are all ``E0``.
    """
    import contextlib

    from wikilense import evaluate as evalmod

    monkeypatch.setattr(evalmod, "load_ground_truth", lambda conn: truths)
    monkeypatch.setattr(evalmod, "_corpus_counts",
                        lambda conn: {"n_pages": 2, "n_chunks": 10, "n_sentences": 10})
    monkeypatch.setattr(evalmod, "read_ingest_meta", lambda conn: {})
    monkeypatch.setattr(evalmod, "vector_index_info",
                        lambda conn: {"index_m": 16, "index_distance": "cosine"})
    monkeypatch.setattr(evalmod, "global_cache_size", lambda conn: 0)
    monkeypatch.setattr(evalmod, "versions_info", lambda conn: {})
    monkeypatch.setattr(evalmod, "ef_search_session",
                        lambda conn, ef: contextlib.nullcontext())
    monkeypatch.setattr(evalmod.db, "get_session_var", lambda conn, name: 20)
    monkeypatch.setattr(evalmod, "chunk_sentence_ids",
                        lambda conn, ids: {int(c): {int(c)} for c in ids})
    monkeypatch.setattr(searchmod, "search", search_fn)
    embedder = VectorEmbedder({t.text: E0 for t in truths})
    return evaluate(object(), embedder, ef_search=SERVER_DEFAULT_EF_SEARCH, **kwargs)


def _hit(chunk_id: int, page_id: int) -> Hit:
    return Hit(chunk_id, page_id, f"P{page_id}", "", 0, 1, 0.01 * chunk_id, f"chunk {chunk_id}")


def test_recall_at_k_comes_from_the_limit_k_query_even_when_it_is_not_a_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overfetch-like search: the candidates are the first ``k * 2`` chunks of the ranking
    1..10, and the filter keeps page 2 (chunks 6..10). The LIMIT 1 query sees chunks 1-2 and
    returns nothing; the LIMIT 3 query sees 1-6 and returns chunk 6. Recall@1 must describe the
    LIMIT 1 query (0 claims), not the first hit of the LIMIT 3 query."""

    def overfetch_like(conn, qvec, k=10, filters=None, strategy="inline", overfetch=10,
                       ef_search=None, query_text=None):
        pool = range(1, min(k * 2, 10) + 1)
        return [_hit(c, 2) for c in pool if c >= 6][:k]

    truths = [ClaimTruth(1, "train", "SUPPORTS", "claim one", {2}, [{6}], {"P2"})]
    result = _pure_evaluate(monkeypatch, truths, overfetch_like, ks=(1, 3), repeats=1,
                            strategy="overfetch", overfetch=2)
    assert {m.k: m.article_hits for m in result.per_k} == {1: 0, 3: 1}
    assert {m.k: m.evidence_hits for m in result.per_k} == {1: 0, 3: 1}
    assert {m.k: m.unit_coverage for m in result.per_k} == {1: 0.0, 3: 1.0}
    assert result.prefix_mismatches == {1: 1}  # the LIMIT 1 hits are not the LIMIT 3 prefix
    assert result.claims[0].hit_chunk_ids == (6,)  # the diagnostics keep the LIMIT max_k list
    assert result.claims[0].gold_page_rank == 1


def test_unstable_claims_are_the_ones_whose_max_k_hits_change_between_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def drifting(conn, qvec, k=10, filters=None, strategy="inline", overfetch=10,
                 ef_search=None, query_text=None):
        calls["n"] += 1
        if query_text == "claim two" and calls["n"] > 4:  # changes after the first timed pass
            return [_hit(c, 1) for c in (2, 1)][:k]
        return [_hit(c, 1) for c in (1, 2)][:k]

    truths = [
        ClaimTruth(1, "train", "SUPPORTS", "claim one", {1}, [{1}], {"P1"}),
        ClaimTruth(2, "train", "SUPPORTS", "claim two", {1}, [{2}], {"P1"}),
    ]
    result = _pure_evaluate(monkeypatch, truths, drifting, ks=(2,), repeats=3, strategy="none")
    assert result.unstable_claims == [2]
    assert "Claims whose top-2 hits changed between repeats: 2." in results_markdown(result)


def test_parse_vector_index_reads_m_and_distance() -> None:
    ddl = (
        "CREATE TABLE `chunk` (\n  `chunk_id` int(10) unsigned NOT NULL,\n"
        "  `embedding` vector(384) NOT NULL,\n  PRIMARY KEY (`chunk_id`),\n"
        "  VECTOR KEY `embedding` (`embedding`) `M`='16' `DISTANCE`='cosine'\n) ENGINE=InnoDB"
    )
    assert parse_vector_index(ddl) == {"index_m": 16, "index_distance": "cosine"}
    assert parse_vector_index("VECTOR KEY `embedding` (`embedding`) `M`=6 `DISTANCE`=euclidean") == {
        "index_m": 6, "index_distance": "euclidean"
    }
    assert parse_vector_index("VECTOR KEY `embedding` (`embedding`)") == {
        "index_m": None, "index_distance": None
    }
    assert parse_vector_index("") == {"index_m": None, "index_distance": None}


def _sample_result() -> EvalResult:
    """A small hand-made EvalResult for the output tests."""
    lat = LatencyStats(n=4, p50_ms=1.5, p95_ms=2.85, mean_ms=1.75, min_ms=1.0, max_ms=3.0)
    return EvalResult(
        parameters={"ks": [1, 3], "max_k": 3, "repeats": 2, "strategy": "none",
                    "filters": None, "ef_search": None, "ef_search_source": "server",
                    "ef_search_effective": 20, "mhnsw_max_cache_size": 16777216,
                    "index_m": 16, "index_distance": "cosine"},
        ingest_meta={"chunk_max_words": "120", "embedding_model": "m"},
        machine={"platform": "test", "cpu_count": 2, "gpu": None},
        versions={"python": "3.12", "mariadb": "11.8.9"},
        n_claims=2,
        n_evidence_claims=1,
        per_k=[
            KMetrics(1, 1, 0.5, 0, 0.0, 0.5, lat),
            KMetrics(3, 2, 1.0, 1, 1.0, 1.0, lat),
        ],
        embedding_latency=lat,
        claims=[
            ClaimOutcome(7, "train", "SUPPORTS", "a | claim\nwith pipe", ("Alpha",), True, 1, 3,
                         1.0, (1, 2, 3)),
            ClaimOutcome(8, "dev", "REFUTES", "b", ("Beta",), False, None, None, 0.0, (4, 5, 6)),
        ],
        unstable_claims=[],
        prefix_mismatches={1: 0},
        notes=["a note"],
        created_at="2026-09-17T00:00:00+00:00",
    )


def test_write_results_produces_valid_json_and_a_markdown_table_per_metric(tmp_path: Path) -> None:
    result = _sample_result()
    json_path, md_path = write_results(result, out_dir=tmp_path / "out", name="sample")
    assert json_path == tmp_path / "out" / "sample.json"
    assert md_path == tmp_path / "out" / "sample.md"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data == json.loads(json.dumps(result.to_dict()))
    assert data["per_k"][1] == {
        "k": 3, "article_hits": 2, "article_recall": 1.0, "evidence_hits": 1,
        "evidence_recall": 1.0, "unit_coverage": 1.0,
        "sql_latency": {"n": 4, "p50_ms": 1.5, "p95_ms": 2.85, "mean_ms": 1.75,
                        "min_ms": 1.0, "max_ms": 3.0},
    }
    assert data["prefix_mismatches"] == {"1": 0}
    assert data["claims"][0]["gold_page_rank"] == 1 and data["claims"][1]["gold_page_rank"] is None

    md = md_path.read_text(encoding="utf-8")
    assert md == results_markdown(result, "sample")
    headings = [line for line in md.splitlines() if line.startswith("## ")]
    assert headings[0] == "## Parameters"
    for heading in (
        "## Article recall@k", "## Evidence recall@k", "## Unit coverage@k",
        "## SQL latency per k (ms)", "## Query embedding latency (ms)",
    ):
        assert heading in headings
    assert md.index("## Parameters") < md.index("## Article recall@k")
    assert "| 3 | 2 | 2 | 1.000 |" in md  # article recall row at k = 3
    assert "| 1 | 0 | 1 | 0.000 |" in md  # evidence recall row at k = 1
    assert "| 4 | 1.50 | 2.85 | 1.75 | 1.00 | 3.00 |" in md  # latency cells
    assert "| strategy | none |" in md
    assert "a \\| claim with pipe" in md  # pipes escaped, newline collapsed
    assert "not in top k" in md and "n/a" in md
    # every table row has as many cells as its header (an escaped pipe is not a separator)
    for block in md.split("\n\n"):
        lines = [line for line in block.splitlines() if line.startswith("|")]
        if lines:
            widths = {len(re.findall(r"(?<!\\)\|", line)) for line in lines}
            assert len(widths) == 1, block


def test_write_results_rejects_a_name_with_a_path(tmp_path: Path) -> None:
    for bad in ("", "../x", "sub/name", "."):
        with pytest.raises(ValueError):
            write_results(_sample_result(), out_dir=tmp_path, name=bad)


def test_worst_claims_orders_missing_first_then_largest_rank() -> None:
    def outcome(claim_id: int, rank: int | None) -> ClaimOutcome:
        return ClaimOutcome(claim_id, "train", "SUPPORTS", "t", (), False, rank, None, 0.0, ())

    result = _sample_result()
    result.claims = [outcome(1, 2), outcome(2, None), outcome(3, 20), outcome(4, None), outcome(5, 1)]
    assert [c.claim_id for c in worst_claims(result)] == [2, 4, 3, 1, 5]
    assert [c.claim_id for c in worst_claims(result, n=2)] == [2, 4]


# ---------------------------------------------------------------------------------------------
# database tests
# ---------------------------------------------------------------------------------------------


@pytest.mark.db
def test_load_ground_truth_on_an_empty_database(db_conn: pymysql.Connection) -> None:
    assert load_ground_truth(db_conn) == []


@pytest.mark.db
def test_load_ground_truth(corpus: pymysql.Connection) -> None:
    truths = load_ground_truth(corpus)
    assert [t.claim_id for t in truths] == [1, 2, 3, 4, 5, 6]
    by_id = {t.claim_id: t for t in truths}
    assert by_id[1] == ClaimTruth(
        claim_id=1, split="train", label="SUPPORTS", text="alpha claim",
        gold_pages={ALPHA}, sentence_only_sets=[{1, 2}], gold_titles={"Alpha"},
    )
    assert by_id[1].eligible is True
    # a list item counts as a text unit
    assert by_id[2].gold_pages == {GAMMA} and by_id[2].sentence_only_sets == [{6, 7}]
    assert by_id[2].split == "dev" and by_id[2].label == "REFUTES"
    # a cell resolves the page but makes the set not sentence-only
    assert by_id[3].gold_pages == {BETA} and by_id[3].gold_titles == {"Beta"}
    assert by_id[3].sentence_only_sets == [] and by_id[3].eligible is False
    # alternative sets keep the evidence_set order
    assert by_id[4].sentence_only_sets == [{3}, {4, 5}]
    # an unresolved sentence id excludes the set; the resolved page still counts as gold
    assert by_id[5].gold_pages == {ALPHA} and by_id[5].sentence_only_sets == []
    # nothing resolved
    assert by_id[6].gold_pages == set() and by_id[6].gold_titles == set()
    assert by_id[6].sentence_only_sets == []
    assert oracle_title_filters(by_id[3]) == Filters(titles=["Beta"])
    assert oracle_title_filters(by_id[6]) == Filters(titles=[])


@pytest.mark.db
def test_chunk_sentence_ids(corpus: pymysql.Connection) -> None:
    assert chunk_sentence_ids(corpus, [1, 4, 5, 4, 999]) == {1: {1, 2}, 4: {4, 5}, 5: set(),
                                                             999: set()}
    assert chunk_sentence_ids(corpus, []) == {}


@pytest.mark.db
def test_evaluate_end_to_end_with_known_rankings(
    corpus: pymysql.Connection, embedder: VectorEmbedder, tmp_path: Path
) -> None:
    ef_before = dbmod.get_session_var(corpus, "mhnsw_ef_search")
    result = evaluate(corpus, embedder, ks=(10, 1, 5, 3), repeats=2, strategy="none",
                      ef_search=40)
    assert dbmod.get_session_var(corpus, "mhnsw_ef_search") == ef_before
    assert result.parameters["ks"] == [1, 3, 5, 10]
    assert result.parameters["ef_search"] == 40
    assert result.parameters["ef_search_effective"] == 40
    assert result.parameters["ef_search_source"] == "argument"
    assert result.parameters["strategy"] == "none" and result.parameters["filters"] is None
    assert result.parameters["overfetch"] is None
    assert result.parameters["n_chunks"] == N_CHUNKS and result.parameters["n_pages"] == 3
    # the index and server state the run was measured on, read from the server itself
    assert result.parameters["index_m"] == 16 and result.parameters["index_distance"] == "cosine"
    with corpus.cursor() as cur:
        cur.execute("SELECT @@GLOBAL.mhnsw_max_cache_size")
        cache_size = int(cur.fetchone()[0])
    assert result.parameters["mhnsw_max_cache_size"] == cache_size > 0
    assert result.n_claims == 6 and result.n_evidence_claims == 3

    # the claims were embedded once as a batch (after one warm-up call), then one by one for
    # the latency, repeats times
    texts = [CLAIM_QUERIES[i][0] for i in range(1, 7)]
    assert embedder.calls[0] == [texts[0]]
    assert embedder.calls[1] == texts
    assert embedder.calls[2:] == [[t] for _ in range(2) for t in texts]
    assert result.embedding_latency.n == 12

    # article recall over 6 claims: claims 1, 2, 4, 5 at rank 1; claim 3 (Beta) at rank 5;
    # claim 6 never (no gold page)
    article = {m.k: (m.article_hits, m.article_recall) for m in result.per_k}
    assert article == {1: (4, 4 / 6), 3: (4, 4 / 6), 5: (5, 5 / 6), 10: (5, 5 / 6)}
    # evidence recall over the 3 eligible claims: 1 at rank 1, 4 at rank 2 (chunk 2 covers
    # sentence 3), 2 at rank 10 (chunk 10 carries sentence 7)
    evidence = {m.k: (m.evidence_hits, m.evidence_recall) for m in result.per_k}
    assert evidence == {1: (1, 1 / 3), 3: (2, 2 / 3), 5: (2, 2 / 3), 10: (3, 1.0)}
    coverage = {m.k: m.unit_coverage for m in result.per_k}
    assert coverage[1] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert coverage[3] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert coverage[5] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert coverage[10] == pytest.approx(1.0)

    by_id = {c.claim_id: c for c in result.claims}
    assert by_id[1].gold_page_rank == 1 and by_id[1].evidence_rank == 1
    assert by_id[2].gold_page_rank == 1 and by_id[2].evidence_rank == 10
    assert by_id[2].hit_chunk_ids == (9, 1, 2, 3, 4, 5, 6, 7, 8, 10)
    assert by_id[3].gold_page_rank == 5 and by_id[3].evidence_rank is None
    assert by_id[3].eligible is False
    assert by_id[4].gold_page_rank == 1 and by_id[4].evidence_rank == 2
    assert by_id[5].gold_page_rank == 1 and by_id[5].evidence_rank is None
    assert by_id[6].gold_page_rank is None and by_id[6].gold_titles == ()
    assert by_id[1].hit_chunk_ids == tuple(range(1, 11))
    assert [c.claim_id for c in worst_claims(result, n=3)] == [6, 3, 1]

    # latency: repeats x claims samples per k, all positive; the 12 chunks are exact, so the
    # hits are stable and the LIMIT k hits are prefixes of LIMIT 10
    for m in result.per_k:
        assert m.sql_latency.n == 12
        assert 0 < m.sql_latency.min_ms <= m.sql_latency.p50_ms <= m.sql_latency.p95_ms
        assert m.sql_latency.p95_ms <= m.sql_latency.max_ms
    assert result.unstable_claims == []
    assert result.prefix_mismatches == {1: 0, 3: 0, 5: 0}
    assert result.ingest_meta == {}  # nothing ingested into the test database
    assert result.versions["mariadb"].startswith("11.")
    assert result.machine["embedding_device"] == "cpu" and result.machine["gpu"] is None
    assert result.parameters["embedding_model"] == "fake-query-embedder"

    json_path, md_path = write_results(result, out_dir=tmp_path, name="e2e")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["n_claims"] == 6 and data["per_k"][3]["evidence_recall"] == 1.0
    assert "| 5 | 5 | 6 | 0.833 |" in md_path.read_text(encoding="utf-8")


@pytest.mark.db
def test_ef_search_defaults_to_the_settings_and_can_keep_the_server_value(
    corpus: pymysql.Connection, embedder: VectorEmbedder, settings: Settings
) -> None:
    server_value = dbmod.get_session_var(corpus, "mhnsw_ef_search")
    chosen = server_value + 13
    result = evaluate(corpus, embedder, ks=(1,), repeats=1, strategy="none",
                      settings=replace(settings, ef_search=chosen))
    assert result.parameters["ef_search"] == chosen
    assert result.parameters["ef_search_effective"] == chosen
    assert result.parameters["ef_search_source"] == "settings"
    assert dbmod.get_session_var(corpus, "mhnsw_ef_search") == server_value  # restored
    kept = evaluate(corpus, embedder, ks=(1,), repeats=1, strategy="none",
                    ef_search=SERVER_DEFAULT_EF_SEARCH, settings=replace(settings, ef_search=chosen))
    assert kept.parameters["ef_search"] is None
    assert kept.parameters["ef_search_effective"] == server_value
    assert kept.parameters["ef_search_source"] == "server"
    # an explicit int wins over the settings
    explicit = evaluate(corpus, embedder, ks=(1,), repeats=1, strategy="none", ef_search=7,
                        settings=replace(settings, ef_search=chosen))
    assert explicit.parameters["ef_search"] == 7 and explicit.parameters["ef_search_effective"] == 7
    assert explicit.parameters["ef_search_source"] == "argument"
    # settings=None loads them from the environment / .env: the same source label
    loaded = evaluate(corpus, embedder, ks=(1,), repeats=1, strategy="none")
    assert loaded.parameters["ef_search_source"] == "settings"
    assert loaded.parameters["ef_search"] == loaded.parameters["ef_search_effective"] >= 1


@pytest.mark.db
def test_every_search_call_gets_the_claim_text_as_query_text(
    corpus: pymysql.Connection, embedder: VectorEmbedder, settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_search = searchmod.search
    calls: list[dict[str, Any]] = []

    def with_query_text(conn, qvec, k=10, filters=None, strategy="inline", overfetch=10,
                        ef_search=None, query_text=None):
        calls.append({"strategy": strategy, "k": k, "query_text": query_text})
        return real_search(conn, qvec, k=k, filters=filters, strategy="none")

    monkeypatch.setattr(searchmod, "search", with_query_text)
    monkeypatch.setattr(searchmod, "STRATEGIES", ("none", "rrf"))
    result = evaluate(corpus, embedder, ks=(2, 1), repeats=1, strategy="rrf", overfetch=4,
                      settings=settings)
    texts = [CLAIM_QUERIES[i][0] for i in range(1, 7)]
    assert result.parameters["strategy"] == "rrf" and result.parameters["overfetch"] == 4
    assert len(calls) == 2 * 2 * 6  # (warm-up + 1 timed pass) x 2 ks x 6 claims
    assert all(call["strategy"] == "rrf" for call in calls)
    assert [call["query_text"] for call in calls[:4]] == [texts[0], texts[0], texts[1], texts[1]]
    assert {call["query_text"] for call in calls} == set(texts)
    assert result.metrics_at(1).article_hits == 4  # the searches still ran (strategy none)

    # a search function without the parameter is called without it
    calls.clear()

    def without_query_text(conn, qvec, k=10, filters=None, strategy="inline", overfetch=10,
                           ef_search=None):
        calls.append({"k": k})
        return real_search(conn, qvec, k=k, filters=filters, strategy="none")

    monkeypatch.setattr(searchmod, "search", without_query_text)
    result = evaluate(corpus, embedder, ks=(1,), repeats=1, strategy="none", settings=settings)
    assert len(calls) == 12 and result.metrics_at(1).article_hits == 4


@pytest.mark.db
def test_evaluate_with_oracle_title_filters(
    corpus: pymysql.Connection, embedder: VectorEmbedder
) -> None:
    result = evaluate(corpus, embedder, ks=(1, 3), repeats=1, strategy="inline",
                      claim_filters=oracle_title_filters)
    assert result.parameters["claim_filters"] == "oracle_title_filters"
    assert result.parameters["filters"] is None
    by_id = {c.claim_id: c for c in result.claims}
    # Beta is the only page allowed for claim 3, so its chunks 5-8 come first
    assert by_id[3].gold_page_rank == 1 and by_id[3].hit_chunk_ids == (5, 6, 7)
    assert by_id[1].hit_chunk_ids == (1, 2, 3)
    assert by_id[2].hit_chunk_ids == (9, 10, 11)  # Gamma only: chunk 10 now at rank 2
    assert by_id[2].evidence_rank == 2
    assert by_id[6].hit_chunk_ids == () and by_id[6].gold_page_rank is None  # titles=[]
    article = {m.k: m.article_hits for m in result.per_k}
    assert article == {1: 5, 3: 5}
    evidence = {m.k: m.evidence_hits for m in result.per_k}
    assert evidence == {1: 1, 3: 3}


@pytest.mark.db
def test_evaluate_with_a_global_filter_and_overfetch(
    corpus: pymysql.Connection, embedder: VectorEmbedder
) -> None:
    # only Beta has 5000 words; every claim then sees the Beta chunks 5, 6, 7, 8 only
    result = evaluate(corpus, embedder, ks=(1, 3), repeats=1, strategy="overfetch",
                      filters=Filters(min_words=1000), overfetch=3)
    assert result.parameters["filters"] == {
        "min_words": 1000, "max_words": None, "heading_like": None, "path_like": None,
        "linked_from": None, "links_to": None, "titles": None,
    }
    assert result.parameters["overfetch"] == 3
    by_id = {c.claim_id: c for c in result.claims}
    assert by_id[3].gold_page_rank == 1  # in the LIMIT 3 query (inner LIMIT 9: chunks 1-9)
    assert all(by_id[i].gold_page_rank is None for i in (1, 2, 4, 5, 6))
    # recall@1 is the LIMIT 1 query: its inner LIMIT 3 sees chunks 1-3 only, no Beta chunk
    assert {m.k: m.article_hits for m in result.per_k} == {1: 0, 3: 1}
    # every claim's LIMIT 1 query is empty while its LIMIT 3 query finds Beta chunks
    assert result.prefix_mismatches == {1: 6}


@pytest.mark.db
def test_evaluate_refuses_an_empty_database(
    db_conn: pymysql.Connection, embedder: VectorEmbedder
) -> None:
    with pytest.raises(ValueError, match="claim table is empty"):
        evaluate(db_conn, embedder, ks=(1,), repeats=1)
