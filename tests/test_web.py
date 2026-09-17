"""Tests for wikilense.web with the FastAPI TestClient against the MariaDB test database.

The synthetic corpus mirrors tests/test_search.py in miniature: 3 pages ("Alpha" 100 words,
"Beta" 5000, "Gamma" 250) with a lead and one titled section each, 12 chunks (two per section)
whose vectors are ``cos(a) e_0 + sin(a) e_i`` with ``a = 0.1 * chunk_id``, and links
Alpha -> Beta, Beta -> Gamma. The fake embedder answers every query with ``e_0``, so the exact
ranking is chunk 1, 2, ..., 12 with distance ``1 - cos(0.1 * chunk_id)``. Chunks 1-4 belong to
Alpha, 5-8 to Beta, 9-12 to Gamma; the odd section of each page carries a heading.

Tests that reach the server are marked ``db``; validation, the page and the unreachable-server
path run without it.
"""

from __future__ import annotations

from typing import Any, get_args

import numpy as np
import pymysql
import pytest
from fastapi.testclient import TestClient

from wikilense import db as dbmod
from wikilense import web as webmod
from wikilense.config import Settings
from wikilense.search import STRATEGIES

DIM = dbmod.VECTOR_DIM
N_CHUNKS = 12
ALPHA, BETA, GAMMA = 1, 2, 3
HIT_FIELDS = [
    "chunk_id", "page_id", "title", "section_path", "chunk_ordinal", "n_words", "distance", "text"
]  # fmt: skip

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
        chunks.append(
            (
                chunk_id,
                page_id,
                section_id,
                chunk_id,
                f"chunk {chunk_id} text",
                3,
                dbmod.vec_param(_angled(0.1 * chunk_id, chunk_id)),
            )
        )
    dbmod.insert_rows(db_conn, "chunk", dbmod.SCHEMA_COLUMNS["chunk"], chunks)
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
    assert html == webmod.INDEX_HTML
    assert "<form" in html and "<script>" in html
    for name in ("q", "k", "min_words", "heading", "linked_from", "strategy", "ef_search"):
        assert f'name="{name}"' in html
    for strategy in STRATEGIES:
        assert f'value="{strategy}"' in html
    assert "/api/search" in html
    # no external assets, no framework
    assert "<script src" not in html and "<link" not in html and "@import" not in html
    assert "http://" not in html and "https://" not in html


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
        {"q": "x", "strategy": "exact"},
        {"q": "x", "ef_search": 0},
    ):
        resp = client.get("/api/search", params=params)
        assert resp.status_code == 422, params
        assert isinstance(resp.json()["detail"], list)
    resp = client.get("/api/search", params={"q": "   "})
    assert resp.status_code == 400
    assert "blank" in resp.json()["detail"]


def test_api_search_reports_an_unreachable_server_as_503(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = FakeEmbedder()
    client = TestClient(webmod.create_app(OFFLINE_SETTINGS, embedder=embedder))

    def refuse(settings: Settings | None = None, database: str | None = None) -> Any:
        raise pymysql.err.OperationalError(2003, "Can't connect to MySQL server on 'db.invalid'")

    monkeypatch.setattr(webmod.db, "connect", refuse)
    resp = client.get("/api/search", params={"q": "x"})
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "db.invalid:3307" in detail and "Can't connect" in detail
    assert "not-a-real-password" not in resp.text
    assert embedder.queries == ["x"]  # embedded before connecting, so the model is not blocked


# ---------------------------------------------------------------------------------------------
# with the test database
# ---------------------------------------------------------------------------------------------


@pytest.mark.db
def test_api_search_returns_hits_sql_explain_and_timing(
    client: TestClient, fake_embedder: FakeEmbedder
) -> None:
    resp = client.get("/api/search", params={"q": "anything at all", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"hits", "sql", "explain", "timing_ms"}
    assert _ids(body) == [1, 2, 3]
    assert [list(hit) for hit in body["hits"]] == [HIT_FIELDS] * 3
    assert [hit["title"] for hit in body["hits"]] == ["Alpha"] * 3
    assert [hit["section_path"] for hit in body["hits"]] == ["", "", "History"]
    assert body["hits"][0]["text"] == "chunk 1 text"
    assert body["hits"][0]["n_words"] == 3 and body["hits"][0]["page_id"] == ALPHA
    for hit in body["hits"]:
        assert hit["distance"] == pytest.approx(_expected_distance(hit["chunk_id"]), abs=1e-6)
    assert body["sql"].startswith("SELECT ")
    assert "ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s" in body["sql"]
    assert "WHERE" not in body["sql"]
    chunk_rows = [row for row in body["explain"] if row["table"] == "chunk"]
    assert len(chunk_rows) == 1
    assert chunk_rows[0]["key"] == "embedding" and chunk_rows[0]["type"] == "index"
    assert {row["table"] for row in body["explain"]} == {"chunk", "page", "section"}
    assert set(body["timing_ms"]) == {"embed", "sql"}
    assert all(isinstance(v, float) and v >= 0 for v in body["timing_ms"].values())
    assert fake_embedder.queries == ["anything at all"]


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

    body = client.get(
        "/api/search", params={"q": "x", "k": 3, "min_words": 1000, "strategy": "none"}
    ).json()
    assert _ids(body) == [1, 2, 3]  # none ignores the filters
    assert "WHERE" not in body["sql"]

    body = client.get(
        "/api/search",
        params={"q": "x", "k": 3, "min_words": 1000, "strategy": "overfetch", "ef_search": 50},
    ).json()
    assert _ids(body) == [5, 6, 7]  # inner limit 30 covers all 12 chunks
    assert ") AS knn" in body["sql"] and "page.n_words >= %s" in body["sql"]

    resp = client.get("/api/search", params={"q": "x", "k": 3, "heading": "", "linked_from": ""})
    assert resp.status_code == 200 and _ids(resp.json()) == [1, 2, 3]  # empty = no filter


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
