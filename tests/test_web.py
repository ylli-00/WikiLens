"""Tests for wikilense.web with the FastAPI TestClient against the MariaDB test database.

The synthetic corpus mirrors tests/test_search.py in miniature: 3 pages ("Alpha" 100 words,
"Beta" 5000, "Gamma" 250) with a lead and one titled section each, 12 chunks (two per section)
whose vectors are ``cos(a) e_0 + sin(a) e_i`` with ``a = 0.1 * chunk_id``, and links
Alpha -> Beta, Beta -> Gamma. The fake embedder answers every query with ``e_0``, so the exact
ranking is chunk 1, 2, ..., 12 with distance ``1 - cos(0.1 * chunk_id)``. Chunks 1-4 belong to
Alpha, 5-8 to Beta, 9-12 to Gamma; the odd section of each page carries a heading. Chunk 12's
text also holds the word "glacier" (the only full-text match for it, for ``rrf``), and Alpha's
lead has two sentences: chunk 1 covers both, chunk 2 the second one (the overlap), so
``sentences=1`` has rows to return.

The claim picker's tests run on the claim corpus (the section "the claim picker" below): the
same corpus with claim rows and two claim files written to ``tmp_path``. No test opens a
connection to another database than the test database: S1 is either not offered, stood in for by
the test database through a stubbed probe and ``db.connect``, or refused before connecting.

Tests that reach the server are marked ``db``; validation, the page and the unreachable-server
path run without it.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pymysql
import pytest
from fastapi.testclient import TestClient

from wikilense import db as dbmod
from wikilense import picker
from wikilense import search as searchmod
from wikilense import web as webmod
from wikilense.config import Settings
from wikilense.evaluate import (
    ClaimTruth,
    article_recall_at_k,
    chunk_sentence_ids,
    evidence_recall_at_k,
    gold_page_rank,
    load_ground_truth,
    unit_coverage_at_k,
)
from wikilense.search import EF_SEARCH_VARIABLE, STRATEGIES, Hit

DIM = dbmod.VECTOR_DIM
N_CHUNKS = 12
ALPHA, BETA, GAMMA = 1, 2, 3
HIT_FIELDS = [f.name for f in fields(Hit)]
FT_WORD = "glacier"  # in chunk 12's text only
ALPHA_SENTENCES = [("sentence_0", "Alpha is a page."), ("sentence_1", "It has two sentences.")]

#: Settings for tests that must never reach a server (an invalid host fails at once).
OFFLINE_SETTINGS = Settings(db_password="not-a-real-password", db_host="db.invalid", db_port=3307)


class FakeEmbedder:
    """Returns the unit vector e_0 for every query and records the texts."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_queries(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        self.queries.extend(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def _angled(angle: float, axis: int) -> np.ndarray:
    """Return the unit vector cos(angle) * e_0 + sin(angle) * e_axis."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = np.cos(angle)
    v[axis] = np.sin(angle)
    return v


def _db_time(*parts: int) -> datetime:
    """Return the naive datetime a ``DATETIME`` column holds for this UTC moment."""
    return datetime(*parts, tzinfo=timezone.utc).replace(tzinfo=None)


def _expected_distance(chunk_id: int) -> float:
    return float(1 - np.cos(0.1 * chunk_id))


def _ids(body: dict[str, Any]) -> list[int]:
    return [hit["chunk_id"] for hit in body["hits"]]


@pytest.fixture
def corpus(db_conn: pymysql.Connection) -> pymysql.Connection:
    """The synthetic corpus of the module docstring, committed; returns the connection."""
    dbmod.insert_rows(
        db_conn,
        "page",
        dbmod.SCHEMA_COLUMNS["page"],
        [
            (ALPHA, "Alpha", 4, 0, 100, 600, 1, 0, 0, None, None),
            (BETA, "Beta", 4, 0, 5000, 30000, 1, 0, 0, 102, _db_time(2020, 12, 15, 12)),
            (GAMMA, "Gamma", 4, 0, 250, 1500, 1, 0, 0, None, None),
        ],
    )
    dbmod.insert_rows(
        db_conn,
        "section",
        dbmod.SCHEMA_COLUMNS["section"],
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
        text = f"chunk {chunk_id} text" + (f" {FT_WORD}" if chunk_id == N_CHUNKS else "")
        chunks.append(
            (
                chunk_id,
                page_id,
                section_id,
                chunk_id,
                text,
                3,
                dbmod.vec_param(_angled(0.1 * chunk_id, chunk_id)),
            )
        )
    dbmod.insert_rows(db_conn, "chunk", dbmod.SCHEMA_COLUMNS["chunk"], chunks)
    dbmod.insert_rows(
        db_conn,
        "sentence",
        dbmod.SCHEMA_COLUMNS["sentence"],
        [(i + 1, ALPHA, 1, key, i, text) for i, (key, text) in enumerate(ALPHA_SENTENCES)],
    )
    dbmod.insert_rows(
        db_conn, "chunk_sentence", dbmod.SCHEMA_COLUMNS["chunk_sentence"], [(1, 1), (1, 2), (2, 2)]
    )
    dbmod.insert_rows(
        db_conn,
        "link",
        ("from_page_id", "to_title", "to_page_id", "source_element"),
        [(ALPHA, "Beta", BETA, "sentence_0"), (BETA, "Gamma", GAMMA, "sentence_0")],
    )
    dbmod.insert_rows(db_conn, "category", ("category_id", "name"), [(1, "Rivers")])
    dbmod.insert_rows(db_conn, "page_category", ("page_id", "category_id"), [(BETA, 1)])
    db_conn.commit()
    return db_conn


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def client(
    settings: Settings, corpus: pymysql.Connection, fake_embedder: FakeEmbedder
) -> TestClient:
    """A TestClient of an app on the test database with the fake embedder injected."""
    app = webmod.create_app(settings, embedder=fake_embedder, database=settings.test_db_name)
    return TestClient(app)


# ---------------------------------------------------------------------------------------------
# without a server
# ---------------------------------------------------------------------------------------------


def test_strategy_parameter_accepts_exactly_the_search_strategies() -> None:
    assert get_args(webmod.Strategy) == STRATEGIES


def test_create_app_keeps_settings_embedder_and_database_on_state() -> None:
    embedder = FakeEmbedder()
    app = webmod.create_app(OFFLINE_SETTINGS, embedder=embedder, database="some_db")
    assert app.state.settings is OFFLINE_SETTINGS
    assert app.state.embedder is embedder
    assert app.state.database == "some_db"
    assert webmod.create_app(OFFLINE_SETTINGS, embedder=embedder).state.database is None


def test_index_page_is_self_contained_html_with_the_form() -> None:
    client = TestClient(webmod.create_app(OFFLINE_SETTINGS, embedder=FakeEmbedder()))
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    html = resp.text
    assert html == webmod.render_index(OFFLINE_SETTINGS)
    assert "__" not in html  # every template marker was filled in
    assert "<form" in html and "<script>" in html
    # every search.Filters field, the strategy, overfetch, ef_search and the sentence switch
    for name, _field in webmod.API_FILTER_FIELDS:
        assert f'name="{name}"' in html
    for name in ("q", "k", "strategy", "overfetch", "ef_search", "sentences"):
        assert f'name="{name}"' in html
    for strategy in STRATEGIES:
        assert f'value="{strategy}"' in html
    assert f'placeholder="{OFFLINE_SETTINGS.ef_search} (default)' in html
    assert "/api/search" in html
    # a link with the form's parameters fills the form and runs the search
    assert "new URLSearchParams(window.location.search)" in html and "form.requestSubmit()" in html
    assert f'href="{webmod.DOCS_URL}"' in html and f'href="{webmod.OPENAPI_URL}"' in html
    # no external assets, no framework
    assert "<script src" not in html and "<link" not in html and "@import" not in html
    assert "http://" not in html and "https://" not in html


def test_docs_and_openapi_schema_are_served() -> None:
    client = TestClient(webmod.create_app(OFFLINE_SETTINGS, embedder=FakeEmbedder()))
    assert client.get(webmod.DOCS_URL).status_code == 200
    schema = client.get(webmod.OPENAPI_URL).json()
    parameters = {p["name"] for p in schema["paths"]["/api/search"]["get"]["parameters"]}
    assert parameters == set(webmod.API_PARAMETER_NAMES)
    assert "/" not in schema["paths"]  # the page is not part of the API


def test_api_search_rejects_bad_parameters_with_422_or_400() -> None:
    client = TestClient(webmod.create_app(OFFLINE_SETTINGS, embedder=FakeEmbedder()))
    for params in (
        {"k": 3},  # q missing
        {"q": ""},
        {"q": "x", "k": 0},
        {"q": "x", "k": webmod.MAX_K + 1},
        {"q": "x", "k": "three"},
        {"q": "x", "min_words": -1},
        {"q": "x", "min_words": "many"},
        {"q": "x", "max_words": -1},
        {"q": "x", "strategy": "exact"},
        {"q": "x", "overfetch": 0},
        {"q": "x", "overfetch": searchmod.MAX_OVERFETCH + 1},
        {"q": "x", "ef_search": -1},
        {"q": "x", "ef_search": searchmod.MAX_EF_SEARCH + 1},  # the server would clamp it
    ):
        resp = client.get("/api/search", params=params)
        assert resp.status_code == 422, params
        assert isinstance(resp.json()["detail"], list)  # FastAPI's validation report
    resp = client.get("/api/search", params={"q": "x", "min_words": 10, "max_words": 5})
    assert resp.status_code == 422 and "greater than max_words" in resp.json()["detail"]
    resp = client.get("/api/search", params={"q": "   "})
    assert resp.status_code == 400
    assert "blank" in resp.json()["detail"]
    # a mistyped or unknown parameter is refused instead of being silently ignored
    resp = client.get("/api/search", params={"q": "x", "heading_like": "%History%"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "heading_like" in detail and "accepted: " + ", ".join(webmod.API_PARAMETER_NAMES) in detail
    resp = client.get("/api/search", params={"q": "x", "K": 3, "titel": "Aare"})
    assert resp.status_code == 400 and "K, titel" in resp.json()["detail"]


@pytest.mark.parametrize(
    "params",
    [
        {"min_words": 100},
        {"max_words": 100},
        {"heading": "%History%"},
        {"path": "Geo%"},
        {"linked_from": "Aare"},
        {"links_to": "Bern"},
        {"titles": ["Aare"]},
        {"min_words": 100, "heading": "%History%"},
    ],
)
def test_api_search_rejects_a_filter_with_strategy_none_as_422(params: dict[str, Any]) -> None:
    embedder = FakeEmbedder()
    client = TestClient(webmod.create_app(OFFLINE_SETTINGS, embedder=embedder))
    resp = client.get("/api/search", params={"q": "x", "strategy": "none", **params})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail.startswith("strategy 'none' runs without filters")
    for name in params:
        assert name in detail  # every offending parameter is named
    assert "inline" in detail and "overfetch" in detail and "rrf" in detail  # the alternatives
    assert embedder.queries == [webmod.WARM_UP_TEXT]  # refused before embedding the query


def test_api_search_reports_an_unreachable_server_as_503(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = FakeEmbedder()
    client = TestClient(webmod.create_app(OFFLINE_SETTINGS, embedder=embedder))
    assert embedder.queries == [webmod.WARM_UP_TEXT]  # the model is loaded when the app is built

    def refuse(settings: Settings | None = None, database: str | None = None) -> Any:
        raise pymysql.err.OperationalError(2003, "Can't connect to MySQL server on 'db.invalid'")

    monkeypatch.setattr(webmod.db, "connect", refuse)
    resp = client.get("/api/search", params={"q": "x"})
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "db.invalid:3307" in detail and "Can't connect" in detail
    assert "not-a-real-password" not in resp.text
    assert embedder.queries == [webmod.WARM_UP_TEXT, "x"]  # embedded before connecting


class ClosingConnection:
    """Stands in for a PyMySQL connection that the search never really uses."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # MariaDB 1191 has no entry in PyMySQL's error map, so it arrives as OperationalError
        (pymysql.err.OperationalError(1191, "Can't find FULLTEXT index matching the column list"),
         "no FULLTEXT index"),
        (pymysql.err.ProgrammingError(1146, "Table 'wikilense.chunk' doesn't exist"),
         "has no schema"),
        (pymysql.err.OperationalError(1054, "Unknown column 'page.last_edited' in 'WHERE'"),
         "older schema"),
    ],
)
def test_api_search_reports_a_missing_index_or_schema_as_503(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str
) -> None:
    conn = ClosingConnection()
    monkeypatch.setattr(webmod.db, "connect", lambda settings=None, database=None: conn)

    def failing_search(*args: Any, **kwargs: Any) -> list[Hit]:
        raise error

    monkeypatch.setattr(webmod.search, "search", failing_search)
    client = TestClient(webmod.create_app(OFFLINE_SETTINGS, embedder=FakeEmbedder()))
    resp = client.get("/api/search", params={"q": "x", "strategy": "rrf"})
    assert resp.status_code == 503, resp.text
    assert expected in resp.json()["detail"]
    assert conn.closed


def test_a_query_model_of_the_wrong_dimension_is_503_not_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WideEmbedder(FakeEmbedder):
        def embed_queries(self, texts, batch_size=64, show_progress=False):
            self.queries.extend(texts)
            return np.ones((len(texts), 768), dtype=np.float32)

    monkeypatch.setattr(webmod.db, "connect", NoDatabase())
    client = TestClient(webmod.create_app(OFFLINE_SETTINGS, embedder=WideEmbedder()))
    resp = client.get("/api/search", params={"q": "x"})
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "768 dimensions" in detail and "WIKILENSE_EMBEDDING_MODEL" in detail


class NoDatabase:
    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("the database must not be reached")


def test_the_page_script_never_writes_html_from_data() -> None:
    """Hits, sentences, SQL and errors reach the page through textContent only, so a title or a
    chunk text containing markup is shown as text; innerHTML would make it markup."""
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in webmod.INDEX_HTML
    assert "textContent" in webmod.INDEX_HTML


def test_create_app_without_warm_up_reports_no_model_load_time() -> None:
    embedder = FakeEmbedder()
    app = webmod.create_app(OFFLINE_SETTINGS, embedder=embedder, warm_up=False)
    assert embedder.queries == [] and app.state.model_load_ms is None
    warmed = webmod.create_app(OFFLINE_SETTINGS, embedder=embedder)
    assert embedder.queries == [webmod.WARM_UP_TEXT]
    assert isinstance(warmed.state.model_load_ms, float) and warmed.state.model_load_ms >= 0


# ---------------------------------------------------------------------------------------------
# with the test database
# ---------------------------------------------------------------------------------------------


@pytest.mark.db
def test_api_search_returns_hits_sql_explain_ef_search_and_timing(
    client: TestClient, fake_embedder: FakeEmbedder, settings: Settings, corpus: pymysql.Connection
) -> None:
    resp = client.get("/api/search", params={"q": "anything at all", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"hits", "sql", "explain", "ef_search", "timing_ms"}
    assert _ids(body) == [1, 2, 3]
    assert [list(hit) for hit in body["hits"]] == [HIT_FIELDS] * 3
    assert [hit["title"] for hit in body["hits"]] == ["Alpha"] * 3
    assert [hit["section_path"] for hit in body["hits"]] == ["", "", "History"]
    assert body["hits"][0]["text"] == "chunk 1 text"
    assert body["hits"][0]["n_words"] == 3 and body["hits"][0]["page_id"] == ALPHA
    assert all(hit["score"] is None for hit in body["hits"])  # rrf only
    for hit in body["hits"]:
        assert hit["distance"] == pytest.approx(_expected_distance(hit["chunk_id"]), abs=1e-6)
    assert body["sql"].startswith("SELECT ")
    assert "ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s" in body["sql"]
    assert "WHERE" not in body["sql"]
    chunk_rows = [row for row in body["explain"] if row["table"] == "chunk"]
    assert len(chunk_rows) == 1
    assert chunk_rows[0]["key"] == "embedding" and chunk_rows[0]["type"] == "index"
    assert {row["table"] for row in body["explain"]} == {"chunk", "page", "section"}
    # the effective mhnsw_ef_search: the settings' value, or the server's when that is 0
    expected_ef = settings.ef_search or dbmod.get_session_var(corpus, EF_SEARCH_VARIABLE)
    assert body["ef_search"] == expected_ef
    assert set(body["timing_ms"]) == {"model_load", "embed", "sql"}
    assert all(isinstance(v, float) and v >= 0 for v in body["timing_ms"].values())
    assert body["timing_ms"]["model_load"] == client.app.state.model_load_ms
    assert fake_embedder.queries == [webmod.WARM_UP_TEXT, "anything at all"]


@pytest.mark.db
def test_api_search_sentences_lists_the_chunk_sentence_rows_per_hit(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "x", "k": 3, "sentences": 1}).json()
    assert _ids(body) == [1, 2, 3]
    assert [list(hit) for hit in body["hits"]] == [[*HIT_FIELDS, "sentences"]] * 3
    expected = [{"element_key": key, "text": text} for key, text in ALPHA_SENTENCES]
    assert body["hits"][0]["sentences"] == expected  # chunk 1: both sentences, page order
    assert body["hits"][1]["sentences"] == expected[1:]  # chunk 2: the overlapping one
    assert body["hits"][2]["sentences"] == []  # chunk 3: no chunk_sentence rows
    body = client.get("/api/search", params={"q": "x", "k": 1, "sentences": 0}).json()
    assert "sentences" not in body["hits"][0]
    body = client.get("/api/search", params={"q": "x", "k": 1}).json()
    assert "sentences" not in body["hits"][0]


@pytest.mark.db
def test_api_search_rrf_hits_carry_a_score_and_use_the_full_text_index(
    client: TestClient,
) -> None:
    # vector list (N = 3): chunks 1, 2, 3; full-text list for "glacier": chunk 12 only.
    # Chunk 1 and 12 tie at 1/61 and the lower chunk_id comes first.
    body = client.get(
        "/api/search", params={"q": FT_WORD, "k": 3, "strategy": "rrf", "overfetch": 1}
    ).json()
    assert _ids(body) == [1, 12, 2]
    assert [hit["score"] for hit in body["hits"]] == pytest.approx([1 / 61, 1 / 61, 1 / 62])
    assert body["hits"][1]["text"] == f"chunk 12 text {FT_WORD}"
    for hit in body["hits"]:
        assert hit["distance"] == pytest.approx(_expected_distance(hit["chunk_id"]), abs=1e-6)
    assert "MATCH(text) AGAINST (%s IN NATURAL LANGUAGE MODE)" in body["sql"]
    assert FT_WORD not in body["sql"]  # the query text is a bound parameter
    assert any(row["type"] == "fulltext" for row in body["explain"])
    # a filter applies to the fused list: only Gamma's chunk 12 matches it
    body = client.get(
        "/api/search",
        params={"q": FT_WORD, "k": 3, "strategy": "rrf", "overfetch": 1, "titles": ["Gamma"]},
    ).json()
    assert _ids(body) == [12]


@pytest.mark.db
def test_api_search_default_k_is_five(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "x"}).json()
    assert _ids(body) == [1, 2, 3, 4, 5]


@pytest.mark.db
def test_api_search_category_and_edit_date_filters(client: TestClient) -> None:
    """Only Beta has a category (Rivers) and a revision date (2020-12-15 12:00 UTC)."""
    body = client.get("/api/search", params={"q": "x", "k": 3, "category": "Rivers"}).json()
    assert _ids(body) == [5, 6, 7] and "category.name = %s" in body["sql"]
    assert "Rivers" not in body["sql"]
    body = client.get("/api/search", params={"q": "x", "k": 3, "edited_since": "2020-12-15"}).json()
    assert _ids(body) == [5, 6, 7] and "page.last_edited >= %s" in body["sql"]
    body = client.get("/api/search", params={"q": "x", "k": 3, "edited_before": "2020-12-15"}).json()
    assert body["hits"] == [] and "page.last_edited < %s" in body["sql"]  # no other page is dated
    body = client.get("/api/search", params={"q": "x", "k": 3, "category": ""}).json()
    assert _ids(body) == [1, 2, 3]  # a blank form field is no filter
    resp = client.get("/api/search", params={"q": "x", "edited_since": "2020-13-01"})
    assert resp.status_code == 422
    resp = client.get(
        "/api/search",
        params={"q": "x", "edited_since": "2021-01-01", "edited_before": "2020-01-01"},
    )
    assert resp.status_code == 422 and "not earlier than" in resp.json()["detail"]


@pytest.mark.db
def test_api_search_filters_and_strategy_reach_the_sql(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "x", "k": 3, "min_words": 1000}).json()
    assert _ids(body) == [5, 6, 7]
    assert {hit["title"] for hit in body["hits"]} == {"Beta"}
    assert "page.n_words >= %s" in body["sql"] and "1000" not in body["sql"]

    body = client.get("/api/search", params={"q": "x", "k": 3, "heading": "%history%"}).json()
    assert _ids(body) == [3, 4, 11]  # History (Alpha) and Early history (Gamma), case-insensitive
    assert "section.heading LIKE %s" in body["sql"] and "history" not in body["sql"]

    body = client.get("/api/search", params={"q": "x", "k": 2, "linked_from": "Alpha"}).json()
    assert _ids(body) == [5, 6]  # Alpha links to Beta only
    assert "link.to_page_id" in body["sql"] and "Alpha" not in body["sql"]

    body = client.get("/api/search", params={"q": "x", "k": 3, "max_words": 200}).json()
    assert _ids(body) == [1, 2, 3] and "page.n_words <= %s" in body["sql"]  # Alpha only
    body = client.get("/api/search", params={"q": "x", "k": 3, "path": "Geo%"}).json()
    assert _ids(body) == [7, 8] and "section.path LIKE %s" in body["sql"]  # Beta > Geography
    body = client.get("/api/search", params={"q": "x", "k": 3, "links_to": "Gamma"}).json()
    assert _ids(body) == [5, 6, 7] and "link.from_page_id" in body["sql"]  # Beta links to Gamma
    body = client.get(
        "/api/search", params={"q": "x", "k": 3, "titles": ["Gamma", "Alpha", "Nope"]}
    ).json()
    assert _ids(body) == [1, 2, 3]
    assert "page.title IN (%s, %s, %s)" in body["sql"] and "Gamma" not in body["sql"]
    body = client.get("/api/search", params={"q": "x", "k": 3, "titles": ["Gamma", ""]}).json()
    assert _ids(body) == [9, 10, 11] and "page.title IN (%s)" in body["sql"]  # blank dropped

    body = client.get("/api/search", params={"q": "x", "k": 3, "strategy": "none"}).json()
    assert _ids(body) == [1, 2, 3]
    assert "WHERE" not in body["sql"]
    resp = client.get(
        "/api/search", params={"q": "x", "k": 3, "min_words": 1000, "strategy": "none"}
    )
    assert resp.status_code == 422  # none would drop the filter silently: refused
    assert "min_words" in resp.json()["detail"]

    body = client.get(
        "/api/search",
        params={"q": "x", "k": 3, "min_words": 1000, "strategy": "overfetch", "ef_search": 50},
    ).json()
    assert _ids(body) == [5, 6, 7]  # inner limit 30 covers all 12 chunks
    assert ") AS knn" in body["sql"] and "page.n_words >= %s" in body["sql"]
    assert body["ef_search"] == 50

    resp = client.get("/api/search", params={"q": "x", "k": 3, "heading": "", "linked_from": ""})
    assert resp.status_code == 200 and _ids(resp.json()) == [1, 2, 3]  # empty = no filter


@pytest.mark.db
def test_api_search_ef_search_zero_leaves_the_session_value_and_reports_it(
    client: TestClient, corpus: pymysql.Connection
) -> None:
    server_value = dbmod.get_session_var(corpus, EF_SEARCH_VARIABLE)
    body = client.get("/api/search", params={"q": "x", "k": 1, "ef_search": 0}).json()
    assert body["ef_search"] == server_value and _ids(body) == [1]
    body = client.get("/api/search", params={"q": "x", "k": 1, "ef_search": 7}).json()
    assert body["ef_search"] == 7


@pytest.mark.db
def test_api_search_on_an_empty_database_is_503(settings: Settings, db_conn: Any) -> None:
    app = webmod.create_app(settings, embedder=FakeEmbedder(), database=settings.test_db_name)
    resp = TestClient(app).get("/api/search", params={"q": "x"})
    assert resp.status_code == 503
    assert "empty" in resp.json()["detail"]


@pytest.mark.db
def test_connection_is_opened_per_request_and_closed_afterwards(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[pymysql.Connection] = []
    real_connect = dbmod.connect

    def recording_connect(
        settings: Settings | None = None, database: str | None = None
    ) -> pymysql.Connection:
        conn = real_connect(settings, database=database)
        opened.append(conn)
        return conn

    monkeypatch.setattr(webmod.db, "connect", recording_connect)
    assert client.get("/api/search", params={"q": "x", "k": 1}).status_code == 200
    assert client.get("/api/search", params={"q": "y", "k": 1}).status_code == 200
    assert len(opened) == 2
    assert all(not conn.open for conn in opened)

    # ... and on the error path, where the request fails after the connection was opened
    def boom(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("explain failed on purpose")

    monkeypatch.setattr(webmod.search, "explain_search", boom)
    failing = TestClient(client.app, raise_server_exceptions=False)
    assert failing.get("/api/search", params={"q": "z", "k": 1}).status_code == 500
    assert len(opened) == 3
    assert not opened[2].open


# ---------------------------------------------------------------------------------------------
# the claim picker, the database choice and the repeated timing
# ---------------------------------------------------------------------------------------------
#
# The claim corpus adds to the synthetic corpus a page without chunks ("Aare", first in title
# order), one sentence of Beta (chunk 5) and one of Gamma (chunk 9), and two claim files in
# tmp_path: the demo file in FEVEROUS's format, whose ids are exactly the claim table's, and a
# file in the scaled run's format that repeats claim 101 and adds two claims of its own.

AARE = 4
BETA_SENTENCE = (3, BETA, 3, "sentence_0", 0, "Beta is a long page.")
GAMMA_SENTENCE = (4, GAMMA, 5, "sentence_0", 0, "Gamma begins here.")
DEMO_FILE = "demo_claims.jsonl"
SCALED_FILE = "scaled_claims.jsonl"
S1 = webmod.SCALED_DATABASE

#: (claim_id, split, label, text, evidence sets) of the demo claim file; each is a claim row.
DEMO_CLAIMS: list[tuple[int, str, str, str, list[list[str]]]] = [
    # one sentence-only set on Alpha, held by chunk 1 (both) and chunk 2 (the second)
    (101, "train", "SUPPORTS", "Alpha has two sentences.",
     [["Alpha_sentence_0", "Alpha_sentence_1"]]),
    # Beta's sentence, held by chunk 5
    (102, "dev", "REFUTES", "Beta is a short page.", [["Beta_sentence_0"]]),
    # a table cell only: no evidence recall
    (103, "train", "NOT ENOUGH INFO", "Gamma's table says so.", [["Gamma_cell_0_0_0"]]),
    # a sentence-only set and a set with a table cell
    (104, "train", "SUPPORTS", "Alpha says it, and so does Beta's table.",
     [["Alpha_sentence_0"], ["Beta_cell_0_1_0", "Beta_sentence_0"]]),
    # Gamma is there, its sentence_7 is not a row of sentence: dropped
    (105, "train", "REFUTES", "Gamma's eighth sentence says so.", [["Gamma_sentence_7"]]),
    # Delta is not a page of the database: not offered
    (106, "train", "SUPPORTS", "Alpha and Delta agree.",
     [["Alpha_sentence_0", "Delta_sentence_0"]]),
    # its claim_evidence row does not resolve the sentence (see CLAIM_EVIDENCE): dropped
    (107, "dev", "SUPPORTS", "Beta is long.", [["Beta_sentence_0"]]),
    # one set on two pages
    (108, "train", "NOT ENOUGH INFO", "Alpha and Beta together.",
     [["Alpha_sentence_1", "Beta_sentence_0"]]),
]
#: The scaled-format file: claim 101 again, a claim on Gamma, and a claim whose table set cites
#: a page the database does not hold.
SCALED_CLAIMS = [
    DEMO_CLAIMS[0],
    (201, "train", "REFUTES", "Gamma begins here.", [["Gamma_sentence_0"]]),
    (202, "dev", "SUPPORTS", "Alpha, or Epsilon's table.",
     [["Alpha_sentence_0"], ["Epsilon_cell_0_0_0"]]),
]
CLAIM_TEXT = {cid: text for cid, _, _, text, _ in DEMO_CLAIMS + SCALED_CLAIMS}
#: claim_evidence as the ingest writes it for DEMO_CLAIMS, except claim 107, whose row leaves
#: Beta's sentence unresolved (written before that sentence was a row), so the picker's
#: resolution of the file differs from the table.
CLAIM_EVIDENCE = [
    (101, 0, 0, "Alpha_sentence_0", "Alpha", "sentence", ALPHA, 1),
    (101, 0, 1, "Alpha_sentence_1", "Alpha", "sentence", ALPHA, 2),
    (102, 0, 0, "Beta_sentence_0", "Beta", "sentence", BETA, 3),
    (103, 0, 0, "Gamma_cell_0_0_0", "Gamma", "cell", GAMMA, None),
    (104, 0, 0, "Alpha_sentence_0", "Alpha", "sentence", ALPHA, 1),
    (104, 1, 0, "Beta_cell_0_1_0", "Beta", "cell", BETA, None),
    (104, 1, 1, "Beta_sentence_0", "Beta", "sentence", BETA, 3),
    (105, 0, 0, "Gamma_sentence_7", "Gamma", "sentence", GAMMA, None),
    (106, 0, 0, "Alpha_sentence_0", "Alpha", "sentence", ALPHA, 1),
    (106, 0, 1, "Delta_sentence_0", "Delta", "sentence", None, None),
    (107, 0, 0, "Beta_sentence_0", "Beta", "sentence", BETA, None),
    (108, 0, 0, "Alpha_sentence_1", "Alpha", "sentence", ALPHA, 2),
    (108, 0, 1, "Beta_sentence_0", "Beta", "sentence", BETA, 3),
]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def claim_files(tmp_path: Path) -> tuple[Path, Path]:
    """The demo claim file (FEVEROUS format) and the scaled one (``evidence_sets`` format)."""
    demo = _write_jsonl(
        tmp_path / DEMO_FILE,
        [
            {"evidence": [{"content": s, "context": {}} for s in sets], "id": cid,
             "claim": text, "label": label, "challenge": "Other", "split": split}
            for cid, split, label, text, sets in DEMO_CLAIMS
        ],
    )
    scaled = _write_jsonl(
        tmp_path / SCALED_FILE,
        [
            {"id": cid, "split": split, "label": label, "claim": text,
             "kind": picker.claim_kind(sets), "challenge": "Other",
             "evidence_sets": [{"elements": s, "text_only": True, "findable": True}
                               for s in sets]}
            for cid, split, label, text, sets in SCALED_CLAIMS
        ],
    )
    return demo, scaled


@pytest.fixture
def claim_corpus(corpus: pymysql.Connection) -> pymysql.Connection:
    """The synthetic corpus plus the rows of the claim corpus (see above), committed."""
    dbmod.insert_rows(
        corpus, "page", dbmod.SCHEMA_COLUMNS["page"],
        [(AARE, "Aare", 0, 0, 0, 0, 0, 0, 0, None, None)],
    )
    dbmod.insert_rows(
        corpus, "sentence", dbmod.SCHEMA_COLUMNS["sentence"], [BETA_SENTENCE, GAMMA_SENTENCE]
    )
    dbmod.insert_rows(
        corpus, "chunk_sentence", dbmod.SCHEMA_COLUMNS["chunk_sentence"], [(5, 3), (9, 4)]
    )
    dbmod.insert_rows(
        corpus, "claim", dbmod.SCHEMA_COLUMNS["claim"],
        [(cid, split, text, label, "Other") for cid, split, label, text, _ in DEMO_CLAIMS],
    )
    dbmod.insert_rows(
        corpus, "claim_evidence", dbmod.SCHEMA_COLUMNS["claim_evidence"], CLAIM_EVIDENCE
    )
    corpus.commit()
    return corpus


def _claim_app(
    settings: Settings,
    claim_files: Sequence[Path],
    embedder: FakeEmbedder | None = None,
    offer_s1: bool = False,
) -> TestClient:
    """A TestClient of an app on the test database with the given claim files."""
    app = webmod.create_app(
        settings,
        embedder=embedder or FakeEmbedder(),
        database=settings.test_db_name,
        offer_s1=offer_s1,
        claim_files=claim_files,
    )
    return TestClient(app)


@pytest.fixture
def claim_client(
    settings: Settings,
    claim_corpus: pymysql.Connection,
    fake_embedder: FakeEmbedder,
    claim_files: tuple[Path, Path],
) -> TestClient:
    """The app on the claim corpus with both claim files, without S1."""
    return _claim_app(settings, claim_files, fake_embedder)


def _claim_search(client: TestClient, cid: int, **params: Any) -> dict[str, Any]:
    """Search claim ``cid`` by its text and return the JSON (asserting a 200)."""
    resp = client.get("/api/search", params={"q": CLAIM_TEXT[cid], "claim_id": cid, **params})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _evaluation_summary(
    conn: pymysql.Connection, body: dict[str, Any], truth: ClaimTruth, k: int
) -> dict[str, Any]:
    """The summary's values computed with the evaluation's functions from the response's hits."""
    hit_pages = [hit["page_id"] for hit in body["hits"]]
    units = chunk_sentence_ids(conn, _ids(body))
    hit_units = [units[cid] for cid in _ids(body)]
    expected: dict[str, Any] = {
        "gold_page_found": article_recall_at_k(hit_pages, truth.gold_pages, k),
        "gold_page_rank": gold_page_rank(hit_pages[:k], truth.gold_pages),
        "evidence_recall_applies": truth.eligible,
    }
    if truth.eligible:
        expected["evidence_recall"] = evidence_recall_at_k(hit_units, truth.sentence_only_sets, k)
        expected["unit_coverage"] = unit_coverage_at_k(hit_units, truth.sentence_only_sets, k)
    return expected


class RecordingConnect:
    """Stands in for ``db.connect``: records every database asked for and opens only the test
    database; with ``redirect_s1`` a request for S1 opens the test database instead. Any other
    name fails the test, so no test reaches the corpus or the scaled databases."""

    def __init__(self, test_db: str, redirect_s1: bool = False) -> None:
        self.test_db = test_db
        self.redirect_s1 = redirect_s1
        self.requested: list[str | None] = []
        self.real = dbmod.connect

    def __call__(
        self, settings: Settings | None = None, database: str | None = None
    ) -> pymysql.Connection:
        self.requested.append(database)
        if self.redirect_s1 and database == S1:
            database = self.test_db
        if database != self.test_db:
            raise AssertionError(f"a connection to database {database!r} was requested")
        return self.real(settings, database=database)


def _no_probe(settings: Settings) -> None:
    raise AssertionError("S1 must not be probed")


# --- the picker list ---------------------------------------------------------------------------


@pytest.mark.db
def test_api_claims_lists_the_claims_whose_evidence_the_database_holds(
    claim_client: TestClient, settings: Settings
) -> None:
    body = claim_client.get("/api/claims").json()
    assert body["database"] == settings.test_db_name
    # the demo file holds exactly the claim table's ids; the scaled file does not
    assert body["own_files"] == [DEMO_FILE]
    # the demo file's claims first, then the other file's; 106 (Delta) and 202 (Epsilon's
    # table) cite a page the database lacks and are neither offered nor listed as left out
    assert [(c["id"], c["kind"], c["sources"]) for c in body["claims"]] == [
        (101, "single", [DEMO_FILE, SCALED_FILE]),
        (102, "single", [DEMO_FILE]),
        (103, "single", [DEMO_FILE]),
        (104, "either", [DEMO_FILE]),
        (108, "needs2", [DEMO_FILE]),
        (201, "single", [SCALED_FILE]),
    ]
    assert body["claims"][0] == {
        "id": 101, "label": "SUPPORTS", "text": CLAIM_TEXT[101], "kind": "single",
        "sources": [DEMO_FILE, SCALED_FILE],
    }
    assert body["dropped"] == [
        {"claim_id": 105, "reason": "Gamma_sentence_7 is not a row of sentence"},
        {"claim_id": 107, "reason": "its resolved evidence differs from the claim_evidence rows"},
    ]
    named = claim_client.get("/api/claims", params={"database": settings.test_db_name}).json()
    assert named == body


@pytest.mark.db
def test_api_claims_is_empty_without_a_claim_file_of_the_database(
    settings: Settings, claim_corpus: pymysql.Connection, claim_files: tuple[Path, Path]
) -> None:
    """No file holds exactly the claim table's ids, so the page hides the picker."""
    _, scaled = claim_files
    for files in ((scaled,), (), (scaled.parent / "absent.jsonl",)):
        body = _claim_app(settings, files).get("/api/claims").json()
        assert (body["own_files"], body["claims"], body["dropped"]) == ([], [], []), files
    client = _claim_app(settings, (scaled,))
    resp = client.get("/api/search", params={"q": CLAIM_TEXT[201], "claim_id": 201})
    assert resp.status_code == 404  # nothing is offered, not even the scaled file's claims
    assert "picker.hidden = claims.length === 0;" in webmod.INDEX_HTML


@pytest.mark.db
def test_build_catalog_truth_of_own_claims_is_load_ground_truth(
    claim_corpus: pymysql.Connection, claim_files: tuple[Path, Path]
) -> None:
    claims = picker.read_claim_files(claim_files)
    cat = picker.build_catalog(claim_corpus, "demo", claims)
    assert (cat.database, cat.n_pages, cat.n_chunks, cat.own_files) == ("demo", 4, 12, (DEMO_FILE,))
    stored = {t.claim_id: t for t in load_ground_truth(claim_corpus)}
    own = [cid for cid in cat.claims if cid in stored]
    assert own == [101, 102, 103, 104, 108]
    for cid in own:
        assert cat.claims[cid].truth == stored[cid], cid
    # a claim of the other file gets the truth the same rows would give in claim_evidence
    assert cat.claims[201].truth == ClaimTruth(
        201, "train", "REFUTES", CLAIM_TEXT[201], {GAMMA}, [{4}], {"Gamma"}
    )
    assert cat.claims[104].gold_units() == {1: "sentence_0", 3: "sentence_0"}


# --- the marks and the summary -----------------------------------------------------------------


@pytest.mark.db
def test_marks_flag_gold_pages_and_the_chunks_holding_gold_sentences(
    claim_client: TestClient,
) -> None:
    body = _claim_search(claim_client, 102, k=6, sentences=1)  # Beta's sentence, in chunk 5
    assert _ids(body) == [1, 2, 3, 4, 5, 6]
    nothing = {"gold_page": False, "gold_evidence": False, "gold_sentence_keys": []}
    assert body["marks"] == [
        *({"chunk_id": cid, **nothing} for cid in (1, 2, 3, 4)),
        {"chunk_id": 5, "gold_page": True, "gold_evidence": True,
         "gold_sentence_keys": ["sentence_0"]},
        {"chunk_id": 6, "gold_page": True, "gold_evidence": False, "gold_sentence_keys": []},
    ]
    # the marked key is one of the hit's sentences, which the page highlights
    assert body["hits"][4]["sentences"] == [
        {"element_key": "sentence_0", "text": BETA_SENTENCE[5]}
    ]
    # a set on two pages: chunk 1 holds Alpha's two sentences, only sentence_1 is gold
    body = _claim_search(claim_client, 108, k=6)
    assert [(m["chunk_id"], m["gold_page"], m["gold_sentence_keys"]) for m in body["marks"]] == [
        (1, True, ["sentence_1"]),
        (2, True, ["sentence_1"]),
        (3, True, []),
        (4, True, []),
        (5, True, ["sentence_0"]),
        (6, True, []),
    ]
    assert [m["gold_evidence"] for m in body["marks"]] == [True, True, False, False, True, False]


@pytest.mark.db
@pytest.mark.parametrize(
    ("cid", "params", "hits", "expected"),
    [
        # recalled: the gold page first, both gold sentences in chunk 1
        (101, {"k": 3}, [1, 2, 3],
         {"gold_page_found": True, "gold_page_rank": 1, "evidence_recall": True,
          "unit_coverage": 1.0, "units_covered": 2, "units_total": 2}),
        # not recalled: Beta's first chunk is at rank 5
        (102, {"k": 3}, [1, 2, 3],
         {"gold_page_found": False, "gold_page_rank": None, "evidence_recall": False,
          "unit_coverage": 0.0, "units_covered": 0, "units_total": 1}),
        (102, {"k": 6}, [1, 2, 3, 4, 5, 6],
         {"gold_page_found": True, "gold_page_rank": 5, "evidence_recall": True,
          "unit_coverage": 1.0, "units_covered": 1, "units_total": 1}),
        # a filter that excludes the gold page
        (101, {"k": 3, "min_words": 1000}, [5, 6, 7],
         {"gold_page_found": False, "gold_page_rank": None, "evidence_recall": False,
          "unit_coverage": 0.0, "units_covered": 0, "units_total": 2}),
        # the gold page without the gold sentences: Alpha's History chunks hold no sentence
        (101, {"k": 3, "heading": "%History%"}, [3, 4, 11],
         {"gold_page_found": True, "gold_page_rank": 1, "evidence_recall": False,
          "unit_coverage": 0.0, "units_covered": 0, "units_total": 2}),
        # half of a two-page set at k = 2, all of it at k = 5
        (108, {"k": 2}, [1, 2],
         {"gold_page_found": True, "gold_page_rank": 1, "evidence_recall": False,
          "unit_coverage": 0.5, "units_covered": 1, "units_total": 2}),
        (108, {"k": 5}, [1, 2, 3, 4, 5],
         {"gold_page_found": True, "gold_page_rank": 1, "evidence_recall": True,
          "unit_coverage": 1.0, "units_covered": 2, "units_total": 2}),
    ],
)
def test_summary_applies_the_evaluation_definitions_at_k(
    claim_client: TestClient,
    claim_corpus: pymysql.Connection,
    cid: int,
    params: dict[str, Any],
    hits: list[int],
    expected: dict[str, Any],
) -> None:
    body = _claim_search(claim_client, cid, **params)
    assert _ids(body) == hits
    k = params["k"]
    assert body["summary"] == {
        "k": k, "n_hits": len(hits), "evidence_recall_applies": True,
        "n_evidence_sets": 1, "table_sets": 0, **expected,
    }
    truth = {t.claim_id: t for t in load_ground_truth(claim_corpus)}[cid]
    computed = _evaluation_summary(claim_corpus, body, truth, k)
    assert {key: body["summary"][key] for key in computed} == computed


@pytest.mark.db
def test_a_table_only_claim_has_no_evidence_recall(claim_client: TestClient) -> None:
    body = _claim_search(claim_client, 103, k=10)
    assert body["claim"] == {
        "id": 103, "label": "NOT ENOUGH INFO", "text": CLAIM_TEXT[103], "kind": "single",
        "sources": [DEMO_FILE], "kind_text": picker.KIND_TEXT["single"], "split": "train",
        "evidence_sets": [
            {"sentence_only": False,
             "elements": [{"element_id": "Gamma_cell_0_0_0", "page": "Gamma", "type": "cell",
                           "element_key": "cell_0_0_0", "in_table": True, "text": None}]},
        ],
    }
    assert body["summary"] == {
        "k": 10, "n_hits": 10, "gold_page_found": True, "gold_page_rank": 9,
        "evidence_recall_applies": False, "evidence_recall": None, "unit_coverage": None,
        "units_covered": None, "units_total": None, "n_evidence_sets": 1, "table_sets": 1,
    }
    assert not any(m["gold_evidence"] for m in body["marks"])
    assert [m["gold_page"] for m in body["marks"]] == [False] * 8 + [True, True]


@pytest.mark.db
def test_a_claim_with_a_sentence_set_and_a_table_set(claim_client: TestClient) -> None:
    body = _claim_search(claim_client, 104, k=5)
    assert body["claim"]["kind"] == "either"
    assert body["claim"]["evidence_sets"] == [
        {"sentence_only": True,
         "elements": [{"element_id": "Alpha_sentence_0", "page": "Alpha", "type": "sentence",
                       "element_key": "sentence_0", "in_table": False,
                       "text": ALPHA_SENTENCES[0][1]}]},
        {"sentence_only": False,
         "elements": [{"element_id": "Beta_cell_0_1_0", "page": "Beta", "type": "cell",
                       "element_key": "cell_0_1_0", "in_table": True, "text": None},
                      {"element_id": "Beta_sentence_0", "page": "Beta", "type": "sentence",
                       "element_key": "sentence_0", "in_table": False,
                       "text": BETA_SENTENCE[5]}]},
    ]
    assert body["summary"] == {
        "k": 5, "n_hits": 5, "gold_page_found": True, "gold_page_rank": 1,
        "evidence_recall_applies": True, "evidence_recall": True, "unit_coverage": 1.0,
        "units_covered": 1, "units_total": 1, "n_evidence_sets": 2, "table_sets": 1,
    }
    # the table set's sentence is a gold text unit too, so Beta's chunk 5 is marked
    assert [m["gold_evidence"] for m in body["marks"]] == [True, False, False, False, True]
    assert all(m["gold_page"] for m in body["marks"])  # Alpha and Beta are both gold pages


# --- a claim that is dropped, edited or not offered ----------------------------------------------


@pytest.mark.db
def test_api_search_refuses_a_q_that_is_not_the_claims_text(
    claim_client: TestClient, fake_embedder: FakeEmbedder
) -> None:
    resp = claim_client.get(
        "/api/search", params={"q": "Alpha has two sentences", "claim_id": 101}
    )
    assert resp.status_code == 422
    assert resp.json()["detail"].startswith("q is not the text of claim 101")
    assert fake_embedder.queries == [webmod.WARM_UP_TEXT]  # refused before embedding
    # surrounding blanks are not an edit
    resp = claim_client.get(
        "/api/search", params={"q": f"  {CLAIM_TEXT[101]} ", "claim_id": 101, "k": 1}
    )
    assert resp.status_code == 200 and "summary" in resp.json()


@pytest.mark.db
@pytest.mark.parametrize("cid", [105, 106, 107, 202, 999_999])
def test_api_search_answers_a_claim_that_is_not_offered_with_404(
    claim_client: TestClient, cid: int
) -> None:
    resp = claim_client.get("/api/search", params={"q": CLAIM_TEXT.get(cid, "x"), "claim_id": cid})
    assert resp.status_code == 404
    assert resp.json()["detail"].startswith(f"claim {cid} is not offered on database")


def _script_function(name: str) -> str:
    """Return the source of one function of the page script, up to its closing brace."""
    start = webmod.INDEX_HTML.index(f"  function {name}(")
    return webmod.INDEX_HTML[start : webmod.INDEX_HTML.index("\n  }\n", start)]


def test_the_page_drops_the_claim_when_the_query_is_edited() -> None:
    """The marks belong to the claim's own text, so editing the query drops the claim, and the
    address bar loses claim_id: the hidden field is emptied and formParams skips empty fields."""
    html = webmod.INDEX_HTML
    start = html.index("query.addEventListener('input'")
    listener = html[start : html.index("});", start)]
    assert "query.value !== picked.text" in listener and "dropClaim()" in listener
    drop = _script_function("dropClaim")
    assert "claimIdField.value = '';" in drop and "updateUrl(formParams());" in drop
    assert "} else if (value !== '') {" in _script_function("formParams")
    assert '<input type="hidden" name="claim_id" id="claim-id">' in html


# --- the free-text note and the page list -------------------------------------------------------


def test_the_page_explains_free_text_search() -> None:
    page = webmod.render_index(OFFLINE_SETTINGS)
    text = " ".join(page.split())
    assert "Free-text search: there is no answer key, so the page cannot tell whether a hit is right." in text
    assert (
        'The hits are the closest passages among the <span id="page-count">pages</span> of this '
        "database, not all of Wikipedia." in text
    )
    # the count comes from /api/databases; the note is hidden while a claim is picked
    assert "pageCount.textContent = d ? fmt(d.n_pages) + ' pages' : 'pages';" in page
    assert "freeNote.hidden = picked !== null;" in page
    assert "getJSON(dbQuery('/api/pages'))" in page


@pytest.mark.db
def test_api_pages_and_api_databases_report_the_pages_of_the_database(
    claim_client: TestClient, settings: Settings
) -> None:
    body = claim_client.get("/api/pages").json()
    assert body == {
        "database": settings.test_db_name, "n_pages": 4,
        "titles": ["Aare", "Alpha", "Beta", "Gamma"],  # sorted, not in page_id order
    }
    named = claim_client.get("/api/pages", params={"database": settings.test_db_name}).json()
    assert named == body
    databases = claim_client.get("/api/databases").json()["databases"]
    assert databases == [
        {"name": settings.test_db_name, "label": "Demo: 4 pages, 12 chunks", "n_pages": 4,
         "n_chunks": 12},
    ]


# --- API errors --------------------------------------------------------------------------------


def _offline_client(offer_s1: bool = False) -> TestClient:
    """An app whose demo database is the settings' test database and which reaches no server."""
    app = webmod.create_app(
        OFFLINE_SETTINGS, embedder=FakeEmbedder(),
        database=OFFLINE_SETTINGS.test_db_name, offer_s1=offer_s1,
    )
    return TestClient(app)


def test_every_endpoint_rejects_an_unknown_parameter_with_400() -> None:
    client = _offline_client()
    for path, params, accepted in (
        ("/api/databases", {"database": "x"}, "accepted: none"),
        ("/api/claims", {"databse": "x"}, "accepted: database"),
        ("/api/pages", {"database": "x", "q": "Alpha"}, "accepted: database"),
        ("/api/search",
         {"q": "x", "claim_id": 1, "database": "x", "repeat": 2, "repeats": 2},
         "accepted: " + ", ".join(webmod.API_PARAMETER_NAMES)),
    ):
        resp = client.get(path, params=params)
        assert resp.status_code == 400, path
        detail = resp.json()["detail"]
        assert detail.startswith("unknown query parameter(s): ") and accepted in detail, path
    assert {"claim_id", "database", "repeat"} <= set(webmod.API_PARAMETER_NAMES)
    schema = client.get(webmod.OPENAPI_URL).json()["paths"]
    assert "parameters" not in schema["/api/databases"]["get"]
    for path in ("/api/claims", "/api/pages"):
        assert [p["name"] for p in schema[path]["get"]["parameters"]] == ["database"]


@pytest.mark.parametrize("name", ["wikilense_s0", "wikilense", "x", S1])
@pytest.mark.parametrize("offer_s1", [False, True])
def test_a_database_that_is_not_offered_is_422_without_a_connection(
    monkeypatch: pytest.MonkeyPatch, name: str, offer_s1: bool
) -> None:
    """S1 is offered only when probed loaded; here the probe finds it missing. No other name is
    ever offered, and none is refused by trying to connect to it."""
    requested: list[str | None] = []

    def recording(settings: Settings | None = None, database: str | None = None) -> Any:
        requested.append(database)
        raise pymysql.err.OperationalError(2003, "Can't connect to MySQL server")

    monkeypatch.setattr(webmod.db, "connect", recording)
    monkeypatch.setattr(webmod, "probe_s1", lambda settings: None)
    client = _offline_client(offer_s1)
    for path, params in (
        ("/api/claims", {"database": name}),
        ("/api/pages", {"database": name}),
        ("/api/search", {"q": "x", "database": name}),
        ("/api/search", {"q": "x", "database": name, "claim_id": 101, "repeat": 3}),
    ):
        resp = client.get(path, params=params)
        assert resp.status_code == 422, (path, params)
        assert resp.json()["detail"] == (
            f"database {name!r} is not offered by this server; offered: "
            f"{OFFLINE_SETTINGS.test_db_name}"
        )
    assert requested == []
    assert client.app.state.embedder.queries == [webmod.WARM_UP_TEXT]


def test_wikilense_s0_is_refused_while_s1_is_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webmod.db, "connect", NoDatabase())
    monkeypatch.setattr(webmod, "probe_s1", lambda settings: (7686, 132359))
    resp = _offline_client(offer_s1=True).get("/api/pages", params={"database": "wikilense_s0"})
    assert resp.status_code == 422
    assert resp.json()["detail"].endswith(f"offered: {OFFLINE_SETTINGS.test_db_name}, {S1}")


@pytest.mark.db
def test_the_response_keys_grow_only_with_the_new_parameters(
    claim_client: TestClient, settings: Settings
) -> None:
    old = {"hits", "sql", "explain", "ef_search", "timing_ms"}
    assert set(claim_client.get("/api/search", params={"q": "x"}).json()) == old
    body = claim_client.get("/api/search", params={"q": "x", "database": settings.test_db_name})
    assert set(body.json()) == old | {"database"}
    assert body.json()["database"] == settings.test_db_name
    assert set(claim_client.get("/api/search", params={"q": "x", "repeat": 1}).json()) == old | {
        "repeat"
    }
    assert set(_claim_search(claim_client, 101)) == old | {"claim", "marks", "summary"}


# --- repeated timing ---------------------------------------------------------------------------


def test_repeat_outside_its_range_is_422() -> None:
    client = _offline_client()
    for repeat in (0, -1, webmod.MAX_REPEAT + 1, "twice"):
        resp = client.get("/api/search", params={"q": "x", "repeat": repeat})
        assert resp.status_code == 422, repeat
        assert isinstance(resp.json()["detail"], list)


class StepClock:
    """Stands in for the ``time`` module of wikilense.web: ``perf_counter`` returns the given
    values in order (a call past the end fails the request)."""

    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)
        self.calls = 0

    def perf_counter(self) -> float:
        value = self.values[self.calls]
        self.calls += 1
        return value


@pytest.mark.db
@pytest.mark.parametrize("run_ms", [[4.0, 9.0, 1.0], [4.0, 9.0, 1.0, 6.0]])
def test_repeat_runs_the_statement_n_times_with_one_embedding(
    claim_client: TestClient,
    fake_embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    run_ms: list[float],
) -> None:
    real_search = webmod.search.search
    calls: list[int] = []

    def counting(*args: Any, **kwargs: Any) -> list[Hit]:
        calls.append(kwargs["k"])
        return real_search(*args, **kwargs)

    monkeypatch.setattr(webmod.search, "search", counting)
    # api_search reads the clock before and after the embedding, then before and after each run
    runs = [(float(n), n + ms / 1000.0) for n, ms in enumerate(run_ms, start=1)]
    clock = StepClock([0.0, 0.010, *(t for run in runs for t in run)])
    monkeypatch.setattr(webmod, "time", clock)
    body = claim_client.get(
        "/api/search", params={"q": "x", "k": 3, "repeat": len(run_ms)}
    ).json()
    assert clock.calls == 2 + 2 * len(run_ms)
    assert calls == [3] * len(run_ms)
    assert fake_embedder.queries == [webmod.WARM_UP_TEXT, "x"]
    raw = [(end - start) * 1000.0 for start, end in runs]
    assert body["timing_ms"]["embed"] == round(0.010 * 1000.0, 3)
    assert body["timing_ms"]["sql"] == round(raw[0], 3)  # the first run's time
    assert body["repeat"] == {
        "runs": len(run_ms),
        "sql_ms": [round(ms, 3) for ms in raw],
        "median_ms": round(statistics.median(raw), 3),
        "min_ms": round(min(raw), 3),
        "max_ms": round(max(raw), 3),
        "same_hits": True,
        "differing_runs": [],
    }
    assert body["repeat"]["sql_ms"] == run_ms
    assert body["repeat"]["median_ms"] == (4.0 if len(run_ms) == 3 else 5.0)
    assert _ids(body) == [1, 2, 3]


@pytest.mark.db
def test_repeat_reports_a_run_that_returned_other_hits(
    claim_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_search = webmod.search.search
    calls: list[int] = []

    def varying(*args: Any, **kwargs: Any) -> list[Hit]:
        calls.append(1)
        hits = real_search(*args, **kwargs)
        return hits[::-1] if len(calls) == 2 else hits

    monkeypatch.setattr(webmod.search, "search", varying)
    body = claim_client.get("/api/search", params={"q": "x", "k": 3, "repeat": 3}).json()
    assert len(calls) == 3
    assert body["repeat"]["same_hits"] is False and body["repeat"]["differing_runs"] == [2]
    assert _ids(body) == [1, 2, 3]  # the first run's hits are the ones shown
    page = webmod.INDEX_HTML
    assert "returned other hits than run 1, shown here" in page


# --- the database choice -----------------------------------------------------------------------


@pytest.mark.db
def test_api_databases_without_s1_lists_the_demo_only(
    settings: Settings, claim_corpus: pymysql.Connection, claim_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(webmod, "probe_s1", _no_probe)
    body = _claim_app(settings, claim_files).get("/api/databases").json()
    assert body == {
        "default": settings.test_db_name,
        "databases": [
            {"name": settings.test_db_name, "label": "Demo: 4 pages, 12 chunks", "n_pages": 4,
             "n_chunks": 12},
        ],
        "s1": S1,
        "s1_note": None,
        "examples_note": webmod.EXAMPLES_NOTE,
        "examples": [asdict(e) for e in picker.EXAMPLES],
    }


@pytest.mark.db
def test_api_databases_with_s1_missing_shows_the_note(
    settings: Settings, claim_corpus: pymysql.Connection, claim_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[Settings] = []
    monkeypatch.setattr(webmod, "probe_s1", lambda s: probes.append(s))  # returns None
    connect = RecordingConnect(settings.test_db_name)
    monkeypatch.setattr(webmod.db, "connect", connect)
    client = _claim_app(settings, claim_files, offer_s1=True)
    body = client.get("/api/databases").json()
    assert body["s1_note"] == webmod.S1_MISSING_NOTE
    assert [d["name"] for d in body["databases"]] == [settings.test_db_name]
    resp = client.get("/api/search", params={"q": "x", "database": S1})
    assert resp.status_code == 422
    assert connect.requested == [settings.test_db_name]  # the demo's size only
    assert len(probes) == 3  # a missing S1 is probed again at every use (the 422 lists the offer)
    lines = webmod.prepare(client.app)
    assert lines == [
        (
            f"Demo: 4 pages, 12 chunks ({settings.test_db_name}); claim file: {DEMO_FILE}; "
            "6 claims offered, 2 left out"
        ),
        f"{S1} is not loaded (make fetch-s1, then make load-s1)",
    ]


def test_database_label_names_the_database_and_its_size() -> None:
    assert webmod.database_label(S1, 7686, 132359) == "Scaled run (S1): 7,686 pages, 132,359 chunks"
    assert webmod.database_label("wikilense", 100, 4598) == "Demo: 100 pages, 4,598 chunks"


@pytest.mark.db
def test_a_loaded_s1_is_offered_and_searched_on_its_own_connection(
    settings: Settings, claim_corpus: pymysql.Connection, claim_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1 is stood in for by the test database: every connection asked for S1 opens it."""
    probes: list[Settings] = []

    def loaded(s: Settings) -> tuple[int, int]:
        probes.append(s)
        return (7686, 132359)

    monkeypatch.setattr(webmod, "probe_s1", loaded)
    connect = RecordingConnect(settings.test_db_name, redirect_s1=True)
    monkeypatch.setattr(webmod.db, "connect", connect)
    client = _claim_app(settings, claim_files, offer_s1=True)

    body = client.get("/api/databases").json()
    assert body["s1_note"] is None
    # the probe decides whether S1 is offered; the sizes listed are read from each database,
    # here the test database
    assert body["databases"] == [
        {"name": settings.test_db_name, "label": "Demo: 4 pages, 12 chunks", "n_pages": 4,
         "n_chunks": 12},
        {"name": S1, "label": "Scaled run (S1): 4 pages, 12 chunks", "n_pages": 4,
         "n_chunks": 12},
    ]
    assert connect.requested == [settings.test_db_name, S1]

    body = _claim_search(client, 101, k=3, database=S1)
    assert body["database"] == S1 and body["summary"]["gold_page_rank"] == 1
    # a short connection for the catalog, then the search's own connection
    assert connect.requested == [settings.test_db_name, S1, S1, S1]
    _claim_search(client, 101, k=3, database=S1)
    assert connect.requested == [settings.test_db_name, S1, S1, S1, S1, S1]
    claims = client.get("/api/claims", params={"database": S1}).json()
    assert claims["database"] == S1 and claims["own_files"] == [DEMO_FILE]
    pages = client.get("/api/pages", params={"database": S1}).json()
    assert pages["database"] == S1 and pages["n_pages"] == 4
    assert connect.requested[-1] == S1
    resp = client.get("/api/claims", params={"database": "wikilense_s0"})
    assert resp.status_code == 422
    assert set(connect.requested) == {settings.test_db_name, S1}
    assert len(probes) == 1  # a loaded S1 is remembered


# --- links -------------------------------------------------------------------------------------


def test_the_page_restores_a_link_and_reports_an_unknown_claim_or_database() -> None:
    page = webmod.INDEX_HTML
    assert "var wantedClaim = initial.get('claim_id');" in page
    assert "var wanted = initial.get('database');" in page
    assert "if (wanted) dbSelect.value = wanted;" in page
    # applyUrl leaves the claim and the database to the picker and the selector, and ticks the
    # repeat checkbox only for the value the page sends
    assert "if (name === 'database' || name === 'claim_id') return;" in _script_function("applyUrl")
    assert "else if (field.type === 'checkbox') field.checked = value === field.value;" in page
    assert f'<input name="repeat" type="checkbox" value="{webmod.PAGE_REPEAT}">' in (
        webmod.render_index(OFFLINE_SETTINGS)
    )
    run = _script_function("runFromUrl")
    assert "' is not offered on database '" in run and "nothing was searched." in run
    assert "'\" is not offered on this server (offered: '" in page
    # an example link carries the selected database
    assert "if (name) params.set('database', name);" in _script_function("renderExamples")


@pytest.mark.db
def test_api_search_accepts_the_parameters_a_link_sends(
    claim_client: TestClient, settings: Settings
) -> None:
    params = {
        "database": settings.test_db_name,
        "claim_id": "101",
        "k": "3",
        "sentences": "1",
        "repeat": str(webmod.PAGE_REPEAT),
        "max_words": "200",
        "q": CLAIM_TEXT[101],
    }
    resp = claim_client.get("/api/search", params=params)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert _ids(body) == [1, 2, 3] and "page.n_words <= %s" in body["sql"]
    assert body["database"] == settings.test_db_name
    assert body["repeat"]["runs"] == webmod.PAGE_REPEAT
    assert body["summary"]["evidence_recall"] is True
    assert all("sentences" in hit for hit in body["hits"])


@pytest.mark.db
def test_repeat_rolls_back_between_runs(
    claim_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each repeated search gets a transaction of its own: a REPEATABLE READ snapshot spanning
    several index searches can leave the index stale (docs/VECTOR_INDEX.md, MDEV-41321)."""
    events: list[str] = []
    real_search = webmod.search.search
    real_connect = webmod.db.connect

    def searching(*args: Any, **kwargs: Any) -> list[Hit]:
        events.append("search")
        return real_search(*args, **kwargs)

    def connecting(*args: Any, **kwargs: Any) -> pymysql.Connection:
        conn = real_connect(*args, **kwargs)
        real_rollback = conn.rollback

        def rollback() -> None:
            events.append("rollback")
            real_rollback()

        conn.rollback = rollback  # type: ignore[method-assign]
        return conn

    monkeypatch.setattr(webmod.search, "search", searching)
    monkeypatch.setattr(webmod.db, "connect", connecting)
    assert claim_client.get("/api/search", params={"q": "x", "repeat": 3}).status_code == 200
    assert events == ["search", "rollback", "search", "rollback", "search"]
    events.clear()
    assert claim_client.get("/api/search", params={"q": "x"}).status_code == 200
    assert events == ["search"]  # a single search is unchanged


@pytest.mark.db
def test_a_claim_file_that_cannot_be_read_leaves_the_databases_and_free_text_working(
    settings: Settings, claim_corpus: pymysql.Connection, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"id": 1, "claim": \n', encoding="utf-8")
    client = _claim_app(settings, [broken])
    resp = client.get("/api/databases")
    assert resp.status_code == 200
    assert resp.json()["databases"][0]["n_chunks"] == N_CHUNKS
    claims = client.get("/api/claims")
    assert claims.status_code == 503 and "claim files cannot be read" in claims.json()["detail"]
    assert client.get("/api/search", params={"q": "x", "database": settings.test_db_name}).status_code == 200


def test_the_page_follows_a_database_switch_and_a_link_repeat_count() -> None:
    page = webmod.render_index(OFFLINE_SETTINGS)
    # the hidden attribute wins over the form's display rules
    assert "[hidden] { display: none !important; }" in page
    # a database switch clears the old results and runs a shown search again on the new database
    assert "if (shown && query.value.trim() !== '') form.requestSubmit();" in page
    # a link's repeat count from 1 to MAX_REPEAT is kept; another value is reported
    assert f"var MAX_REPEAT = {webmod.MAX_REPEAT};" in page
    assert "else if (name === 'repeat') applyRepeat(field, value);" in page
    assert "in the link is not a whole number from 1 to" in page
    # without the database list a free-text link still runs, with its database for the API to check
    assert "if (wanted) dbSelect.replaceChildren(new Option(wanted, wanted));" in page


def _set_ingest_stamp(conn: pymysql.Connection, stamp: str) -> None:
    """Write ``ingest_meta.ingested_at`` as an ingest does at its end, and commit."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ingest_meta WHERE `key` = %s", (webmod.INGEST_STAMP_KEY,))
        cur.execute(
            "INSERT INTO ingest_meta (`key`, `value`) VALUES (%s, %s)",
            (webmod.INGEST_STAMP_KEY, stamp),
        )
    conn.commit()


@pytest.mark.db
def test_a_catalog_is_reused_until_the_database_is_ingested_again(
    claim_client: TestClient, claim_corpus: pymysql.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog is kept with the ingest stamp: the same stamp reuses it, a new stamp (a new
    ingest) rebuilds it and the page sees the change without a restart; without a stamp (an
    ingest still running) nothing is kept."""
    builds: list[str] = []
    real_build = webmod.picker.build_catalog

    def counting(conn: Any, name: str, claims: Any) -> Any:
        builds.append(name)
        return real_build(conn, name, claims)

    monkeypatch.setattr(webmod.picker, "build_catalog", counting)

    def offered() -> list[int]:
        return [c["id"] for c in claim_client.get("/api/claims").json()["claims"]]

    # no stamp: built at every use
    before = offered()
    offered()
    assert len(builds) == 2

    _set_ingest_stamp(claim_corpus, "2026-01-01T00:00:00+00:00")
    assert offered() == before
    assert offered() == before
    assert len(builds) == 3  # the same stamp: the cached catalog

    # a new ingest with one claim fewer: the claim table no longer holds exactly the demo
    # file's ids, so no claim file belongs to the database and the picker has nothing to offer
    with claim_corpus.cursor() as cur:
        cur.execute("DELETE FROM claim WHERE claim_id = %s", (before[0],))
    claim_corpus.commit()
    assert offered() == before  # the stamp has not changed yet: still the cached catalog
    assert len(builds) == 3
    _set_ingest_stamp(claim_corpus, "2026-01-02T00:00:00+00:00")
    assert offered() == []
    assert len(builds) == 4


@pytest.mark.db
def test_s1_loaded_while_the_server_runs_is_offered_at_the_next_request(
    settings: Settings, claim_corpus: pymysql.Connection, claim_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = False

    def probe(s: Settings) -> tuple[int, int] | None:
        return (7686, 132359) if loaded else None

    monkeypatch.setattr(webmod, "probe_s1", probe)
    monkeypatch.setattr(webmod.db, "connect", RecordingConnect(settings.test_db_name, redirect_s1=True))
    client = _claim_app(settings, claim_files, offer_s1=True)
    body = client.get("/api/databases").json()
    assert [d["name"] for d in body["databases"]] == [settings.test_db_name]
    assert body["s1_note"] == webmod.S1_MISSING_NOTE
    loaded = True  # make load-s1 finished in another terminal
    body = client.get("/api/databases").json()
    assert [d["name"] for d in body["databases"]] == [settings.test_db_name, S1]
    assert body["s1_note"] is None
    assert client.get("/api/claims", params={"database": S1}).status_code == 200


@pytest.mark.db
def test_the_database_sizes_are_read_at_every_use(
    claim_client: TestClient, claim_corpus: pymysql.Connection
) -> None:
    first = claim_client.get("/api/databases").json()["databases"][0]
    dbmod.insert_rows(
        claim_corpus, "page", dbmod.SCHEMA_COLUMNS["page"],
        [(99, "Late page", 0, 0, 0, 0, 0, 0, 0, None, None)],
    )
    claim_corpus.commit()
    second = claim_client.get("/api/databases").json()["databases"][0]
    assert second["n_pages"] == first["n_pages"] + 1
