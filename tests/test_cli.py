"""Tests for wikilense.cli: argument parsing of every subcommand, the text and JSON output of
``query``, the error paths (server unreachable, empty database, missing schema, bad arguments)
and the exit codes.

The database, search, ingest, evaluate and embedding layers are replaced by fakes through
monkeypatch, so no server and no model are needed; nothing here is marked ``db`` or ``slow``.
The fake settings point at ``db.invalid`` so that an accidental real connection fails at once.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import pymysql
import pytest

import wikilense
from wikilense import cli
from wikilense import db as dbmod
from wikilense import evaluate as evalmod
from wikilense.config import Settings, SettingsError
from wikilense.corpus import DEFAULT_CORPUS_DIR
from wikilense.evaluate import EvalResult, KMetrics, LatencyStats
from wikilense.ingest import IngestError, IngestReport
from wikilense.search import Filters, Hit

DIM = dbmod.VECTOR_DIM
SETTINGS = Settings(db_password="not-a-real-password", db_host="db.invalid", db_port=3307)

HITS = [
    Hit(
        chunk_id=7,
        page_id=1,
        title="Aare",
        section_path="Geography > Course",
        chunk_ordinal=3,
        n_words=98,
        distance=0.123456,
        text=("The Aare is a tributary of the High Rhine and the longest river within Switzerland. " * 4)
        .strip(),
    ),
    Hit(
        chunk_id=2,
        page_id=1,
        title="Aare",
        section_path="",
        chunk_ordinal=0,
        n_words=12,
        distance=0.25,
        text="Lead chunk of the page.",
    ),
    Hit(
        chunk_id=9,
        page_id=2,
        title="Bern",
        section_path="History",
        chunk_ordinal=1,
        n_words=40,
        distance=0.5,
        text="Bern lies on the Aare.",
    ),
]
EXPLAIN_ROWS = [
    {
        "id": 1,
        "select_type": "SIMPLE",
        "table": "chunk",
        "type": "index",
        "possible_keys": None,
        "key": "embedding",
        "key_len": "1538",
        "ref": None,
        "rows": "3",
        "Extra": "",
    }
]


# ---------------------------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------------------------


class FakeCursor:
    """Answers ``SELECT COUNT(*) FROM chunk`` with a fixed count, or raises a given error."""

    def __init__(self, count: int, error: Exception | None) -> None:
        self.count = count
        self.error = error
        self.executed: list[tuple[str, Any]] = []

    def __enter__(self):  # returns self, like a real cursor
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        if self.error is not None:
            raise self.error

    def fetchone(self) -> tuple[int]:
        return (self.count,)


class FakeConnection:
    """A connection that only hands out FakeCursors and remembers whether it was closed."""

    def __init__(self, count: int, error: Exception | None) -> None:
        self.count = count
        self.error = error
        self.closed = False

    def cursor(self, *args: object, **kwargs: object) -> FakeCursor:
        return FakeCursor(self.count, self.error)

    def close(self) -> None:
        self.closed = True


class FakeDatabase:
    """Stands in for ``db.connect``: every call returns a new FakeConnection."""

    def __init__(self) -> None:
        self.count = 42
        self.error: Exception | None = None
        self.connections: list[FakeConnection] = []

    def connect(self, settings: Settings | None = None, database: str | None = None) -> Any:
        conn = FakeConnection(self.count, self.error)
        self.connections.append(conn)
        return conn


class FakeEmbedder:
    """Returns the unit vector e_0 for every query and records the texts."""

    model_name = "fake/model"
    device = "cpu"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.make_calls: list[Settings] = []

    def embed_queries(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        self.queries.extend(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def _refuse_connection(settings: Settings | None = None, database: str | None = None) -> Any:
    raise pymysql.err.OperationalError(
        2003, "Can't connect to MySQL server on 'db.invalid' ([Errno 111] Connection refused)"
    )


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fixed_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr(cli, "load_settings", lambda: SETTINGS)
    return SETTINGS


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> FakeDatabase:
    fake = FakeDatabase()
    monkeypatch.setattr(cli.db, "connect", fake.connect)
    return fake


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    embedder = FakeEmbedder()

    def make(settings: Settings) -> FakeEmbedder:
        embedder.make_calls.append(settings)
        return embedder

    monkeypatch.setattr(cli, "make_embedder", make)
    return embedder


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace search.search and explain_search; returns the arguments of the last calls."""
    calls: dict[str, Any] = {}

    def fake_search_fn(
        conn: Any,
        qvec: np.ndarray,
        k: int = 10,
        filters: Filters | None = None,
        strategy: str = "inline",
        overfetch: int = 10,
        ef_search: int | None = None,
    ) -> list[Hit]:
        calls.update(
            conn=conn,
            qvec=qvec,
            k=k,
            filters=filters,
            strategy=strategy,
            overfetch=overfetch,
            ef_search=ef_search,
        )
        return HITS[:k]

    def fake_explain(
        conn: Any,
        qvec: np.ndarray,
        k: int = 10,
        filters: Filters | None = None,
        strategy: str = "inline",
        overfetch: int = 10,
        analyze: bool = False,
    ) -> list[dict[str, Any]]:
        calls["explain"] = {"k": k, "filters": filters, "strategy": strategy, "overfetch": overfetch}
        return [dict(row) for row in EXPLAIN_ROWS]

    monkeypatch.setattr(cli.search, "search", fake_search_fn)
    monkeypatch.setattr(cli.search, "explain_search", fake_explain)
    return calls


def _latency(p50: float) -> LatencyStats:
    return LatencyStats(n=10, p50_ms=p50, p95_ms=p50 * 2, mean_ms=p50, min_ms=p50 / 2, max_ms=p50 * 3)


def _eval_result(ks: tuple[int, ...]) -> EvalResult:
    """A small but structurally real EvalResult for ``ks`` (recall grows with k)."""
    per_k = [
        KMetrics(
            k=k,
            article_hits=k,
            article_recall=min(1.0, 0.5 + 0.1 * i),
            evidence_hits=k,
            evidence_recall=min(1.0, 0.4 + 0.1 * i),
            unit_coverage=min(1.0, 0.3 + 0.1 * i),
            sql_latency=_latency(4.0 + i),
        )
        for i, k in enumerate(ks)
    ]
    return EvalResult(
        parameters={"ks": list(ks)},
        ingest_meta={},
        machine={},
        versions={},
        n_claims=75,
        n_evidence_claims=65,
        per_k=per_k,
        embedding_latency=_latency(9.5),
        claims=[],
        unstable_claims=[],
        prefix_mismatches={},
        notes=[],
        created_at="2026-09-17T00:00:00+00:00",
    )


@pytest.fixture
def fake_evaluate(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace evaluate.evaluate() and write_results() with recording fakes; returns the calls."""
    calls: dict[str, Any] = {}

    def evaluate(conn: Any, embedder: Any, **kwargs: Any) -> EvalResult:
        calls["evaluate"] = {"conn": conn, "embedder": embedder, **kwargs}
        return _eval_result(tuple(kwargs["ks"]))

    def write_results(
        result: EvalResult, out_dir: str | Path = "results", name: str = "baseline"
    ) -> tuple[Path, Path]:
        calls["write"] = {"result": result, "out_dir": out_dir, "name": name}
        return Path(out_dir) / f"{name}.json", Path(out_dir) / f"{name}.md"

    monkeypatch.setattr(cli.evaluate, "evaluate", evaluate)
    monkeypatch.setattr(cli.evaluate, "write_results", write_results)
    return calls


# ---------------------------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------------------------


def test_query_prints_rank_distance_title_path_and_wrapped_text(
    capsys: pytest.CaptureFixture[str],
    fake_db: FakeDatabase,
    fake_embedder: FakeEmbedder,
    fake_search: dict[str, Any],
) -> None:
    assert cli.main(["query", "Aare river", "--k", "2"]) == 0
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines[0] == " 1. 0.1235  Aare > Geography > Course  [chunk_id 7, 98 words]"
    second = lines.index(" 2. 0.2500  Aare  [chunk_id 2, 12 words]")
    body = [line for line in lines[1:second] if line.strip()]
    assert len(body) >= 3 and all(line.startswith("    ") for line in body)
    assert " ".join(line.strip() for line in body) == HITS[0].text
    assert all(len(line) <= cli.TEXT_WIDTH for line in lines)
    assert lines[second + 1] == "    Lead chunk of the page."
    assert "SQL:" not in out
    assert fake_search["k"] == 2
    assert fake_search["filters"] == Filters()
    assert fake_search["strategy"] == "inline"
    assert fake_search["overfetch"] == cli.DEFAULT_OVERFETCH
    assert fake_search["ef_search"] is None
    assert fake_search["conn"] is fake_db.connections[0]
    assert fake_embedder.queries == ["Aare river"]
    assert fake_embedder.make_calls == [SETTINGS]
    assert fake_db.connections[0].closed
    assert err.startswith("2 hits; embedding ") and "strategy inline" in err
    assert err.count("\n") == 1


def test_query_options_map_to_filters_and_search_arguments(
    fake_db: FakeDatabase, fake_embedder: FakeEmbedder, fake_search: dict[str, Any]
) -> None:
    argv = [
        "query", "x", "--k", "3", "--min-words", "5000", "--max-words", "9000",
        "--heading", "%History%", "--path", "Geography%", "--linked-from", "Aare",
        "--links-to", "Bern", "--strategy", "overfetch", "--overfetch", "4", "--ef-search", "40",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    assert fake_search["filters"] == Filters(
        min_words=5000,
        max_words=9000,
        heading_like="%History%",
        path_like="Geography%",
        linked_from="Aare",
        links_to="Bern",
    )
    assert fake_search["k"] == 3
    assert fake_search["strategy"] == "overfetch"
    assert fake_search["overfetch"] == 4
    assert fake_search["ef_search"] == 40
    assert fake_search["qvec"].shape == (DIM,)


def test_query_default_k_is_five(
    fake_db: FakeDatabase, fake_embedder: FakeEmbedder, fake_search: dict[str, Any]
) -> None:
    assert cli.main(["query", "x"]) == 0
    assert fake_search["k"] == cli.DEFAULT_K == 5


def test_query_json_is_an_array_of_hit_dicts(
    capsys: pytest.CaptureFixture[str],
    fake_db: FakeDatabase,
    fake_embedder: FakeEmbedder,
    fake_search: dict[str, Any],
) -> None:
    assert cli.main(["query", "x", "--k", "3", "--json"]) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert isinstance(payload, list) and len(payload) == 3
    assert list(payload[0]) == [f.name for f in fields(Hit)]
    assert payload == [asdict(hit) for hit in HITS]
    assert err.startswith("3 hits; ")  # the timing line never pollutes stdout


def test_query_json_with_explain_is_an_object_with_hits_sql_and_explain(
    capsys: pytest.CaptureFixture[str],
    fake_db: FakeDatabase,
    fake_embedder: FakeEmbedder,
    fake_search: dict[str, Any],
) -> None:
    assert cli.main(["query", "x", "--k", "1", "--json", "--explain"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"hits", "sql", "explain"}
    assert payload["hits"] == [asdict(HITS[0])]
    assert payload["sql"].startswith("SELECT ")
    assert "ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, %s) LIMIT %s" in payload["sql"]
    assert payload["explain"] == EXPLAIN_ROWS
    assert fake_search["explain"] == {
        "k": 1, "filters": Filters(), "strategy": "inline", "overfetch": cli.DEFAULT_OVERFETCH
    }  # fmt: skip


def test_query_explain_prints_sql_parameters_and_plan(
    capsys: pytest.CaptureFixture[str],
    fake_db: FakeDatabase,
    fake_embedder: FakeEmbedder,
    fake_search: dict[str, Any],
) -> None:
    assert cli.main(["query", "x", "--k", "1", "--min-words", "10", "--explain"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(" 1. 0.1235  Aare > Geography > Course")
    assert "\nSQL:\n" in out
    assert "page.n_words >= %s" in out
    assert f"parameters: <vector[{DIM}]>, 10, <vector[{DIM}]>, 1" in out
    plan = out.split("EXPLAIN:\n", 1)[1].splitlines()
    assert plan[0].split() == list(EXPLAIN_ROWS[0])
    assert plan[1].split() == ["1", "SIMPLE", "chunk", "index", "NULL", "embedding", "1538", "NULL", "3"]
    assert fake_search["explain"]["filters"] == Filters(min_words=10)


def test_format_hits_wraps_at_the_given_width_and_names_the_empty_case() -> None:
    assert cli.format_hits([]) == "no hits"
    text = cli.format_hits(HITS, width=60)
    body = [line for line in text.splitlines() if line.startswith("    ")]
    assert body and all(len(line) <= 60 for line in body)
    assert text.count("\n\n") == 2  # one blank line between hits


# ---------------------------------------------------------------------------------------------
# error paths and exit codes
# ---------------------------------------------------------------------------------------------


def test_query_reports_unreachable_server_on_one_line_without_loading_the_model(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, fake_embedder: FakeEmbedder
) -> None:
    monkeypatch.setattr(cli.db, "connect", _refuse_connection)
    assert cli.main(["query", "x"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1
    assert err.startswith("wikilense: cannot connect to MariaDB at db.invalid:3307 as 'wikilense'")
    assert "Connection refused" in err and "error 2003" in err
    assert "Traceback" not in err
    assert "not-a-real-password" not in err
    assert fake_embedder.make_calls == []


def test_query_reports_empty_database_before_loading_the_model(
    capsys: pytest.CaptureFixture[str], fake_db: FakeDatabase, fake_embedder: FakeEmbedder
) -> None:
    fake_db.count = 0
    assert cli.main(["query", "x"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "wikilense: database 'wikilense' is empty: run 'wikilense ingest' first\n"
    assert fake_embedder.make_calls == []
    assert fake_db.connections[0].closed


def test_query_reports_missing_schema(
    capsys: pytest.CaptureFixture[str], fake_db: FakeDatabase, fake_embedder: FakeEmbedder
) -> None:
    fake_db.error = pymysql.err.ProgrammingError(1146, "Table 'wikilense.chunk' doesn't exist")
    assert cli.main(["query", "x"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert err.startswith("wikilense: database 'wikilense' has no schema")
    assert "init-db" in err
    assert fake_embedder.make_calls == []
    assert fake_db.connections[0].closed


def test_query_reports_other_database_errors_on_one_line(
    capsys: pytest.CaptureFixture[str], fake_db: FakeDatabase, fake_embedder: FakeEmbedder
) -> None:
    fake_db.error = pymysql.err.ProgrammingError(1064, "You have an error in your SQL syntax")
    assert cli.main(["query", "x"]) == 1
    err = capsys.readouterr().err
    assert err == "wikilense: database error: You have an error in your SQL syntax (error 1064)\n"


def test_missing_settings_is_exit_1(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> Settings:
        raise SettingsError("WIKILENSE_DB_PASSWORD is not set: add it to .env")

    monkeypatch.setattr(cli, "load_settings", missing)
    assert cli.main(["query", "x"]) == 1
    assert capsys.readouterr().err == "wikilense: WIKILENSE_DB_PASSWORD is not set: add it to .env\n"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["bogus"],
        ["query"],
        ["query", "x", "--k", "0"],
        ["query", "x", "--k", "two"],
        ["query", "x", "--strategy", "exact"],
        ["query", "x", "--min-words", "-1"],
        ["query", "x", "--ef-search", "0"],
        ["query", "x", "--overfetch", "0"],
        ["init-db", "--bogus"],
        ["ingest", "--batch-size", "0"],
        ["eval", "--k", "1,x"],
        ["eval", "--k", "3,3"],
        ["eval", "--k", "0"],
        ["eval", "--repeats", "0"],
        ["eval", "--name", "sub/dir"],
        ["eval", "--name", ""],
        ["eval", "--overfetch", "0"],
        ["serve", "--port", "70000"],
        ["serve", "--port", "0"],
    ],
)
def test_bad_arguments_exit_2_with_one_stderr_line(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    assert cli.main(argv) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1
    assert err.startswith("wikilense: ") and "--help" in err
    assert "Traceback" not in err


def test_query_rejects_min_words_above_max_words_as_usage_error(
    capsys: pytest.CaptureFixture[str], fake_db: FakeDatabase
) -> None:
    assert cli.main(["query", "x", "--min-words", "10", "--max-words", "5"]) == 2
    err = capsys.readouterr().err
    assert "greater than max_words" in err and err.count("\n") == 1
    assert fake_db.connections == []  # rejected before any connection


def test_help_lists_the_subcommands_and_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    for name in ("init-db", "ingest", "query", "eval", "serve"):
        assert name in out
    assert cli.main(["query", "--help"]) == 0
    out = capsys.readouterr().out
    for option in ("--k", "--min-words", "--heading", "--linked-from", "--strategy",
                   "--ef-search", "--explain", "--json"):  # fmt: skip
        assert option in out


def test_version_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"wikilense {wikilense.__version__}"


def test_argument_type_helpers() -> None:
    assert cli.k_list("1,3,5") == (1, 3, 5)
    assert cli.positive_int("7") == 7
    assert cli.non_negative_int("0") == 0
    assert cli.port_number("65535") == 65535
    import argparse

    for func, value in (
        (cli.k_list, "1,,2"),
        (cli.k_list, "2,2"),
        (cli.positive_int, "0"),
        (cli.non_negative_int, "-1"),
        (cli.port_number, "65536"),
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            func(value)


# ---------------------------------------------------------------------------------------------
# init-db, ingest, eval, serve
# ---------------------------------------------------------------------------------------------


def test_init_db_applies_the_schema_and_closes_the_connection(
    capsys: pytest.CaptureFixture[str], fake_db: FakeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, bool]] = []

    def apply_schema(conn: Any, reset: bool = False) -> int:
        calls.append((conn, reset))
        return 9

    monkeypatch.setattr(cli.db, "apply_schema", apply_schema)
    assert cli.main(["init-db"]) == 0
    assert calls == [(fake_db.connections[0], False)]
    assert fake_db.connections[0].closed
    out = capsys.readouterr().out
    assert "9 statements" in out and "'wikilense'" in out
    assert cli.main(["init-db", "--reset"]) == 0
    assert calls[1] == (fake_db.connections[1], True)
    assert "reset" in capsys.readouterr().out


def test_init_db_reports_unreachable_server(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.db, "connect", _refuse_connection)
    assert cli.main(["init-db"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("wikilense: cannot connect to MariaDB") and err.count("\n") == 1


def test_ingest_maps_options_to_run_ingest_and_prints_the_report(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def run_ingest(settings: Settings, **kwargs: Any) -> IngestReport:
        calls.append({"settings": settings, **kwargs})
        return IngestReport(
            n_pages=3,
            n_sections=7,
            n_sentences=50,
            n_units_empty=1,
            n_chunks=12,
            n_links=20,
            n_links_resolved=5,
            n_links_skipped=0,
            n_claims=2,
            n_evidence=4,
            n_evidence_page_resolved=4,
            n_evidence_sentence_resolved=3,
            seconds={
                "schema": 0.1,
                "parse": 0.5,
                "embed": 2.0,
                "load": 1.0,
                "resolve": 0.2,
                "total": 3.8,
            },
        )

    monkeypatch.setattr(cli.ingest, "run_ingest", run_ingest)
    assert cli.main(["ingest"]) == 0
    assert calls[0]["settings"] is SETTINGS
    assert calls[0]["corpus_dir"] == DEFAULT_CORPUS_DIR
    assert calls[0]["reset"] is True
    assert calls[0]["batch_size"] == 64
    assert calls[0]["use_prefix"] is True
    assert isinstance(calls[0]["progress"], bool)
    out = capsys.readouterr().out
    assert "ingest finished in 3.8 s" in out
    assert "pages 3, sections 7, sentences 50 (empty 1), chunks 12" in out
    assert "links 20 (resolved 5, skipped 0)" in out
    assert "claims 2, evidence ids 4 (page resolved 4, sentence resolved 3)" in out

    argv = ["ingest", "--corpus-dir", str(tmp_path), "--no-reset", "--batch-size", "16", "--no-prefix"]
    assert cli.main(argv) == 0
    assert calls[1]["corpus_dir"] == tmp_path
    assert calls[1]["reset"] is False
    assert calls[1]["batch_size"] == 16
    assert calls[1]["use_prefix"] is False


def test_ingest_refusal_and_missing_corpus_are_exit_1(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(settings: Settings, **kwargs: Any) -> IngestReport:
        raise IngestError("refusing to ingest into 'wikilense_test': it is the test database")

    monkeypatch.setattr(cli.ingest, "run_ingest", refuse)
    assert cli.main(["ingest"]) == 1
    err = capsys.readouterr().err
    assert err == "wikilense: refusing to ingest into 'wikilense_test': it is the test database\n"

    def missing(settings: Settings, **kwargs: Any) -> IngestReport:
        raise FileNotFoundError("/nowhere/pages.jsonl not found; run scripts/build_corpus.py first")

    monkeypatch.setattr(cli.ingest, "run_ingest", missing)
    assert cli.main(["ingest", "--corpus-dir", "/nowhere"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("wikilense: /nowhere/pages.jsonl not found") and err.count("\n") == 1


def test_eval_maps_options_to_evaluate_and_write_results(
    capsys: pytest.CaptureFixture[str],
    fake_db: FakeDatabase,
    fake_embedder: FakeEmbedder,
    fake_evaluate: dict[str, Any],
    tmp_path: Path,
) -> None:
    argv = [
        "eval", "--k", "1,5", "--repeats", "2", "--strategy", "overfetch", "--overfetch", "3",
        "--ef-search", "30", "--out", str(tmp_path), "--name", "run1",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    ev = fake_evaluate["evaluate"]
    assert ev["ks"] == (1, 5)
    assert ev["repeats"] == 2
    assert ev["strategy"] == "overfetch"
    assert ev["overfetch"] == 3
    assert ev["ef_search"] == 30
    assert ev["filters"] is None
    assert ev["embedder"] is fake_embedder
    assert ev["conn"] is fake_db.connections[0]
    assert fake_db.connections[0].closed
    written = fake_evaluate["write"]
    assert isinstance(written["result"], EvalResult)
    assert written["out_dir"] == tmp_path and written["name"] == "run1"
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split() == list(cli.EVAL_SUMMARY_COLUMNS)
    assert lines[1].split() == ["1", "0.500", "0.400", "0.300", "4.00", "8.00"]
    assert lines[2].split() == ["5", "0.600", "0.500", "0.400", "5.00", "10.00"]
    assert lines[3].startswith("claims 75 (evidence-eligible 65); query embedding p50 9.50 ms")
    assert lines[4] == f"results written: {tmp_path / 'run1.json'}, {tmp_path / 'run1.md'}"


def test_eval_defaults_follow_the_evaluate_module(
    fake_db: FakeDatabase, fake_embedder: FakeEmbedder, fake_evaluate: dict[str, Any]
) -> None:
    assert cli.main(["eval"]) == 0
    ev = fake_evaluate["evaluate"]
    assert ev["ks"] == evalmod.DEFAULT_KS == (1, 3, 5, 10, 20)
    assert ev["repeats"] == evalmod.DEFAULT_REPEATS
    assert ev["strategy"] == evalmod.DEFAULT_STRATEGY
    assert ev["overfetch"] == evalmod.DEFAULT_OVERFETCH
    assert ev["ef_search"] is None
    assert fake_evaluate["write"]["out_dir"] == evalmod.DEFAULT_RESULTS_DIR
    assert fake_evaluate["write"]["name"] == evalmod.DEFAULT_NAME


def test_eval_reports_evaluate_argument_errors_on_one_line(
    capsys: pytest.CaptureFixture[str],
    fake_db: FakeDatabase,
    fake_embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_claims(conn: Any, embedder: Any, **kwargs: Any) -> EvalResult:
        raise ValueError("the database holds no claims: run 'wikilense ingest' first")

    monkeypatch.setattr(cli.evaluate, "evaluate", no_claims)
    assert cli.main(["eval"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "wikilense: the database holds no claims: run 'wikilense ingest' first\n"
    assert fake_db.connections[0].closed


def test_eval_refuses_an_empty_database_before_loading_the_model(
    capsys: pytest.CaptureFixture[str],
    fake_db: FakeDatabase,
    fake_embedder: FakeEmbedder,
    fake_evaluate: dict[str, Any],
) -> None:
    fake_db.count = 0
    assert cli.main(["eval"]) == 1
    assert "is empty" in capsys.readouterr().err
    assert "evaluate" not in fake_evaluate
    assert fake_embedder.make_calls == []


def test_serve_checks_the_database_warms_the_model_and_runs_uvicorn(
    capsys: pytest.CaptureFixture[str],
    fake_db: FakeDatabase,
    fake_embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from wikilense import web

    sentinel = object()
    created: list[dict[str, Any]] = []
    runs: list[tuple[Any, dict[str, Any]]] = []

    def create_app(settings: Settings, *, embedder: Any = None, database: str | None = None) -> Any:
        created.append({"settings": settings, "embedder": embedder, "database": database})
        return sentinel

    monkeypatch.setattr(web, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: runs.append((app, kwargs)))
    assert cli.main(["serve", "--port", "9999"]) == 0
    assert created == [{"settings": SETTINGS, "embedder": fake_embedder, "database": None}]
    assert runs == [(sentinel, {"host": "127.0.0.1", "port": 9999, "log_level": "info"})]
    assert fake_embedder.queries == ["warm-up"]
    assert fake_db.connections[0].closed
    err = capsys.readouterr().err
    assert "http://127.0.0.1:9999/" in err and "42 chunks" in err

    assert cli.main(["serve", "--host", "0.0.0.0"]) == 0
    assert runs[1][1] == {"host": "0.0.0.0", "port": 8000, "log_level": "info"}


def test_serve_fails_fast_when_the_server_is_unreachable(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, fake_embedder: FakeEmbedder
) -> None:
    monkeypatch.setattr(cli.db, "connect", _refuse_connection)
    assert cli.main(["serve"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("wikilense: cannot connect to MariaDB") and err.count("\n") == 1
    assert fake_embedder.make_calls == []
