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

Tests that reach the server are marked ``db``; validation, the page and the unreachable-server
path run without it.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, get_args

import numpy as np
import pymysql
import pytest
from fastapi.testclient import TestClient

from wikilense import db as dbmod
from wikilense import search as searchmod
from wikilense import web as webmod
from wikilense.config import Settings
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
            (ALPHA, "Alpha", 4, 0, 100, 600, 1, 0, 0),
            (BETA, "Beta", 4, 0, 5000, 30000, 1, 0, 0),
            (GAMMA, "Gamma", 4, 0, 250, 1500, 1, 0, 0),
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
