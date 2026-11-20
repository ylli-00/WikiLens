"""Tests for the web page's live fact-check and its copyable prompt: ``wikilense.factcheck``,
``/api/factcheck`` and ``/api/prompt``.

Covered: the setting (off by default; an API key on its own turns nothing on; where the key is
read from), the experiment's record and its refusals (a changed prompt set, the wrong number of
runs, a missing file, a system prompt that does not judge as of December 2020), the request the
runner builds from the passages, the citations in the reason, the reliability figures and the
missing ones, the caveat when the reader's search differs from the experiment's, every way a call
can end without a verdict, one model call per click, the page script's free-text rule, the
question hint, the CLI check, and that the API key never leaves the backend. The copyable prompt:
the text is the experiment's request with its one changed sentence, it is built with the
fact-check off and without a key or a call, for free text and picked claims, it says why when
it cannot be built, and a search stopped at the time limit is a 504 on both routes.

Everything here is pure. The backends are fakes, or the Claude Code CLI is a shell script in
``tmp_path`` that prints a canned login status and version; the real ``claude`` is never found
(``shutil.which`` is stubbed) and every real backend's call method raises. The database is never
reached: ``db.connect``, ``db.get_session_var`` and the search functions are stubbed, as in
``tests/test_web.py``. The API key is a made-up value.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pymysql
import pytest
from fastapi.testclient import TestClient

from wikilense import factcheck
from wikilense import web as webmod
from wikilense.config import (
    DEFAULT_FACTCHECK,
    FACTCHECK_MODES,
    Settings,
    SettingsError,
    load_settings,
)
from wikilense.llm import api_backend, claims, cli_backend, retrieval
from wikilense.llm import prompts as prompts_module
from wikilense.llm.backend import CallRequest, CallResult
from wikilense.search import Hit

FAKE_KEY = "sk-ant-test-NOT-A-REAL-KEY-1234567890"
OTHER_FAKE_KEY = "sk-ant-test-NOT-A-REAL-KEY-from-the-environment"
DB_PASSWORD = "not-a-real-password"
#: Settings for an app that must never reach a server (an invalid host fails at once).
OFFLINE_SETTINGS = Settings(db_password=DB_PASSWORD, db_host="db.invalid", db_port=3307)
HEADERS = {factcheck.HEADER: factcheck.HEADER_VALUE}
STATEMENT = "The Aare flows through Bern."
#: What the stubbed server reports as its session value of mhnsw_ef_search.
SERVER_EF = 40
#: Environment variables that would change what is built, removed for every test.
CLEARED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_BIN")


# ---------------------------------------------------------------------------------------------
# fakes and helpers
# ---------------------------------------------------------------------------------------------


class FakeEmbedder:
    """Returns the unit vector e_0 for every query and records the texts."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_queries(
        self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        self.queries.extend(texts)
        out = np.zeros((len(texts), webmod.db.VECTOR_DIM), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class NoDatabase:
    """Stands in for ``db.connect`` where the database must not be reached."""

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("the database must not be reached")


class FakeCursor:
    """The cursor of :class:`FakeConnection`: answers the chunk count only."""

    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn

    def execute(self, statement: str, params: object = None) -> None:
        self.conn.statements.append(statement)

    def fetchone(self) -> tuple[int]:
        return (self.conn.n_chunks,)


class FakeConnection:
    """Stands in for a PyMySQL connection; the stubbed search never uses it."""

    def __init__(self, n_chunks: int = 12) -> None:
        self.n_chunks = n_chunks
        self.statements: list[str] = []
        self.closed = False

    def cursor(self) -> contextlib.nullcontext[FakeCursor]:
        return contextlib.nullcontext(FakeCursor(self))

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class SearchStub:
    """The stubbed search: returns the first ``k`` of ``hits`` and records every call."""

    def __init__(self, hits: Sequence[Hit], session_ef: int) -> None:
        self.hits = list(hits)
        self.session_ef = session_ef
        self.databases: list[str | None] = []
        self.searches: list[dict[str, Any]] = []


def install_search(
    monkeypatch: pytest.MonkeyPatch, hits: Sequence[Hit], session_ef: int = SERVER_EF
) -> SearchStub:
    """Stub ``db.connect``, ``db.get_session_var`` and the search functions; return the stub."""
    stub = SearchStub(hits, session_ef)

    def connect(settings: Settings | None = None, database: str | None = None) -> FakeConnection:
        stub.databases.append(database)
        return FakeConnection()

    def fake_search(conn: Any, qvec: Any, **kwargs: Any) -> list[Hit]:
        stub.searches.append(kwargs)
        return stub.hits[: kwargs["k"]]

    monkeypatch.setattr(webmod.db, "connect", connect)
    monkeypatch.setattr(webmod.db, "get_session_var", lambda conn, name: stub.session_ef)
    monkeypatch.setattr(webmod.search, "search", fake_search)
    monkeypatch.setattr(webmod.search, "search_statement", lambda *a, **k: ("SELECT 1", []))
    monkeypatch.setattr(webmod.search, "explain_search", lambda *a, **k: [])
    return stub


def make_hits(n: int) -> list[Hit]:
    """Return ``n`` hits; the odd ones are lead sections (no section path)."""
    return [
        Hit(
            chunk_id=100 + i,
            page_id=i,
            title=f"Page {i}",
            section_path="" if i % 2 else f"Section {i} > Part",
            chunk_ordinal=i,
            n_words=10 * i,
            distance=0.01 * i,
            text=f"Passage text number {i}.",
        )
        for i in range(1, n + 1)
    ]


def ok_result(label: str = "SUPPORTS", justification: str = "Passage [1] says so.") -> CallResult:
    """Return a schema-valid answer of the fact_check_v1 schema."""
    return CallResult(
        "ok",
        label=label,
        parse_status="ok",
        answer={"result": {"label": label, "justification": justification}},
    )


#: The base URL of the test clients: this machine, as the fact-check requires.
LOCAL_URL = "http://127.0.0.1:8000"


class FakeBackend:
    """A backend that records each request and returns ``result`` (or raises ``raises``)."""

    name = "fake_backend"

    def __init__(
        self, result: CallResult | None = None, raises: BaseException | None = None
    ) -> None:
        self.result = result if result is not None else ok_result()
        self.raises = raises
        self.requests: list[CallRequest] = []

    def call(self, request: CallRequest, stop: Any) -> CallResult:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        return self.result


def make_checker(
    backend: Any,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
    mode: str = factcheck.SUBSCRIPTION,
    secrets: tuple[str, ...] = (),
) -> factcheck.FactChecker:
    """Return an enabled checker with ``backend`` in place of a real one (already checked: a
    fake has no login to check)."""
    return factcheck.FactChecker(
        mode, backend=backend, experiment=experiment, reliability=reliability, secrets=secrets,
        checked=True,
    )


def make_client(
    checker: factcheck.FactChecker | None,
    settings: Settings = OFFLINE_SETTINGS,
    database: str | None = None,
    copy_prompt: factcheck.CopyPrompt | None = None,
) -> TestClient:
    """Return a TestClient of an app with the fake embedder, ``checker`` and ``copy_prompt``
    (None: what the app loads by itself), addressed to 127.0.0.1 like a browser on the server's
    own machine."""
    app = webmod.create_app(
        settings,
        embedder=FakeEmbedder(),
        database=database,
        warm_up=False,
        offer_s1=False,
        fact_checker=checker,
        copy_prompt=copy_prompt,
    )
    return TestClient(app, base_url=LOCAL_URL)


def post(client: TestClient, headers: dict[str, str] | None = None, **params: Any) -> Any:
    """POST /api/factcheck with ``params`` and the fact-check header (unless ``headers``)."""
    return client.post(
        "/api/factcheck", params=params, headers=HEADERS if headers is None else headers
    )


def matching_settings(experiment: factcheck.Experiment, **changes: Any) -> Settings:
    """Return offline settings with the experiment's ef_search and embedding model."""
    base = replace(
        OFFLINE_SETTINGS,
        ef_search=experiment.ef_search,
        embedding_model=experiment.embedding_model,
    )
    return replace(base, **changes)


def retrieved_run_record() -> dict[str, Any]:
    """Return the original ``retrieved`` run of results/llm/runs.json, read here directly."""
    record = json.loads(factcheck.RUNS_FILE.read_text(encoding="utf-8"))
    runs = [
        r for r in record["runs"]
        if r["run"] == "original" and r["conditions"] == ["retrieved"]
    ]
    assert len(runs) == 1
    return runs[0]


def write_env(path: Path, *lines: str) -> Path:
    """Write a dotenv file with the database password and ``lines``; return its path."""
    path.write_text("\n".join([f"WIKILENSE_DB_PASSWORD={DB_PASSWORD}", *lines]) + "\n",
                    encoding="utf-8")
    return path


def script_function(name: str) -> str:
    """Return the source of one function of the page script, up to its closing brace."""
    start = webmod.INDEX_HTML.index(f"  function {name}(")
    return webmod.INDEX_HTML[start : webmod.INDEX_HTML.index("\n  }\n", start)]


def click_handler() -> str:
    """Return the source of the Fact-check button's click handler."""
    start = webmod.INDEX_HTML.index("fcButton.addEventListener('click'")
    return webmod.INDEX_HTML[start : webmod.INDEX_HTML.index("\n  });\n", start)]


def half_up(value: float) -> str:
    """Return ``value`` (a fraction as written in the JSON) in percent, one decimal, half up."""
    return str((Decimal(repr(value)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def fake_cli(tmp_path: Path, status: dict[str, Any]) -> tuple[Path, Path]:
    """Write a fake ``claude`` that prints ``status`` for ``auth status --json`` and a version
    for ``--version``, logs its arguments, and fails for anything else; return it and the log."""
    log = tmp_path / "cli_calls.log"
    script = tmp_path / "fake-claude"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> {shlex.quote(str(log))}\n'
        'if [ "$1" = "--version" ]; then printf "%s\\n" "9.9.9 (Claude Code)"; exit 0; fi\n'
        'if [ "$1" = "auth" ] && [ "$2" = "status" ]; then\n'
        f"  printf '%s\\n' {shlex.quote(json.dumps(status))}\n"
        "  exit 0\n"
        "fi\n"
        'printf "%s\\n" "unexpected arguments: $*" >&2\n'
        "exit 3\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script, log


def _forbidden(what: str) -> Callable[..., Any]:
    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError(f"{what} must not run in these tests")

    return refuse


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No WIKILENSE_* or Anthropic variable from the shell, no ``claude`` on the PATH, no
    database, no real model call, and temporary folders inside ``tmp_path``."""
    for name in list(os.environ):
        if name.startswith("WIKILENSE_") or name in CLEARED_ENV:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: None)
    monkeypatch.setattr(webmod.db, "connect", NoDatabase())
    for cls, names in (
        (cli_backend.ClaudeCliBackend, ("call", "_run_once")),
        (api_backend.AnthropicApiBackend, ("call", "client", "preflight")),
    ):
        for name in names:
            monkeypatch.setattr(cls, name, _forbidden(f"{cls.__name__}.{name}"))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))


@pytest.fixture(scope="module")
def experiment() -> factcheck.Experiment:
    return factcheck.load_experiment()


@pytest.fixture(scope="module")
def reliability() -> factcheck.Reliability:
    return factcheck.load_reliability()


# ---------------------------------------------------------------------------------------------
# the setting: off by default, and a key alone turns nothing on
# ---------------------------------------------------------------------------------------------


def test_the_setting_is_off_by_default_and_refuses_an_unknown_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert FACTCHECK_MODES == ("off", "subscription", "api") and DEFAULT_FACTCHECK == "off"
    settings = load_settings(env_file=write_env(tmp_path / ".env"))
    assert settings.factcheck == "off" and settings.claude_bin == ""
    assert Settings(db_password=DB_PASSWORD).factcheck == "off"

    on = load_settings(env_file=write_env(tmp_path / "on.env", "WIKILENSE_FACTCHECK=Subscription",
                                          "WIKILENSE_CLAUDE_BIN=/opt/claude/bin/claude"))
    assert on.factcheck == "subscription" and on.claude_bin == "/opt/claude/bin/claude"

    bad_file = write_env(tmp_path / "bad.env", "WIKILENSE_FACTCHECK=maybe")
    with pytest.raises(SettingsError, match="WIKILENSE_FACTCHECK must be one of"):
        load_settings(env_file=bad_file)
    monkeypatch.setenv("WIKILENSE_FACTCHECK", "on")
    with pytest.raises(SettingsError, match="got 'on'"):
        load_settings(env_file=write_env(tmp_path / "plain.env"))


def test_an_app_with_default_settings_shows_the_off_line_and_refuses_a_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copy = factcheck.load_copy_prompt()  # the copy box's experiment, loaded whatever the setting
    for module, name in ((cli_backend, "ClaudeCliBackend"), (api_backend, "AnthropicApiBackend"),
                         (factcheck, "load_experiment"), (factcheck, "load_reliability")):
        monkeypatch.setattr(module, name, _forbidden(name))
    monkeypatch.setattr(api_backend, "load_api_key", _forbidden("load_api_key"))
    client = make_client(None, copy_prompt=copy)  # the live checker loads nothing
    checker = client.app.state.fact_checker
    assert checker.mode == "off" and not checker.enabled and checker.backend is None

    status = client.get("/api/factcheck")
    assert status.status_code == 200
    assert status.json() == {"mode": "off", "enabled": False, "note": factcheck.OFF_NOTE,
                             "docs_link": factcheck.DOCS_LINK}
    assert factcheck.DOCS_LINK.startswith("https://")
    assert factcheck.DOCS_LINK.endswith("docs/SETUP.md#turn-on-the-live-fact-check")
    assert 'docs/SETUP.md, "Turn on the live fact-check"' in factcheck.OFF_NOTE

    for headers in (HEADERS, {}):
        resp = post(client, headers=headers, q=STATEMENT)
        assert resp.status_code == 403
        assert resp.json() == {"detail": factcheck.OFF_NOTE}
    assert checker.backend is None
    assert checker.check().startswith("off")


def test_an_api_key_alone_turns_nothing_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = write_env(tmp_path / ".env", f"ANTHROPIC_API_KEY={FAKE_KEY}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setattr(api_backend, "load_api_key", _forbidden("load_api_key"))
    monkeypatch.setattr(factcheck, "resolve_api_key", _forbidden("resolve_api_key"))
    monkeypatch.setattr(api_backend, "AnthropicApiBackend", _forbidden("AnthropicApiBackend"))
    settings = load_settings(env_file=env_file)
    assert settings.factcheck == "off"

    checker = factcheck.build_factchecker(
        settings, environ={"ANTHROPIC_API_KEY": FAKE_KEY}, env_file=env_file
    )
    assert checker.mode == "off" and not checker.enabled
    assert checker.backend is None and checker.secrets == () and checker.experiment is None

    client = make_client(None, settings=settings)
    assert client.get("/api/factcheck").json()["enabled"] is False
    assert post(client, q=STATEMENT).status_code == 403


def test_the_api_mode_reads_the_key_from_the_environment_first_then_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = write_env(tmp_path / ".env", f"ANTHROPIC_API_KEY={FAKE_KEY}")
    real_load = api_backend.load_api_key
    reads: list[Path] = []

    def spy(path: Path) -> str:
        reads.append(path)
        return real_load(path)

    monkeypatch.setattr(api_backend, "load_api_key", spy)
    settings = replace(OFFLINE_SETTINGS, factcheck="api")

    from_env = factcheck.build_factchecker(
        settings, environ={"ANTHROPIC_API_KEY": OTHER_FAKE_KEY}, env_file=env_file
    )
    assert from_env.enabled and from_env.unavailable is None
    assert from_env.secrets == (OTHER_FAKE_KEY,) and reads == []
    assert isinstance(from_env.backend, api_backend.AnthropicApiBackend)

    from_file = factcheck.build_factchecker(settings, environ={}, env_file=env_file)
    assert from_file.enabled and from_file.secrets == (FAKE_KEY,)
    assert reads == [env_file]

    assert factcheck.resolve_api_key({"ANTHROPIC_API_KEY": f"  {OTHER_FAKE_KEY} "},
                                     env_file) == OTHER_FAKE_KEY
    for bad in ("sk-ant-REPLACE_ME", "not-a-key-0123456789"):
        with pytest.raises(api_backend.ApiBackendError) as info:
            factcheck.resolve_api_key({"ANTHROPIC_API_KEY": bad}, env_file)
        assert bad not in str(info.value)

    no_key = factcheck.build_factchecker(settings, environ={},
                                         env_file=write_env(tmp_path / "nokey.env"))
    assert not no_key.enabled and no_key.backend is None
    assert "ANTHROPIC_API_KEY" in no_key.unavailable

    redirected = factcheck.build_factchecker(
        settings,
        environ={"ANTHROPIC_API_KEY": FAKE_KEY, "ANTHROPIC_BASE_URL": "https://example.org"},
        env_file=env_file,
    )
    assert not redirected.enabled and "ANTHROPIC_BASE_URL" in redirected.unavailable


# ---------------------------------------------------------------------------------------------
# free text only
# ---------------------------------------------------------------------------------------------


def test_a_picked_claim_is_refused_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    stub = install_search(monkeypatch, make_hits(5))
    backend = FakeBackend()
    client = make_client(make_checker(backend, experiment, reliability))
    resp = post(client, q=STATEMENT, claim_id=3)
    assert resp.status_code == 422
    assert "free text only" in resp.json()["detail"]
    assert backend.requests == [] and stub.searches == [] and stub.databases == []


def test_the_endpoint_takes_the_search_parameters_without_sentences_and_repeat(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    assert webmod.FACTCHECK_PARAMETER_NAMES == tuple(
        n for n in webmod.API_PARAMETER_NAMES if n not in ("sentences", "repeat")
    )
    stub = install_search(monkeypatch, make_hits(5))
    backend = FakeBackend()
    client = make_client(make_checker(backend, experiment, reliability))
    for extra in ({"repeat": 2}, {"sentences": 1}):
        resp = post(client, q=STATEMENT, **extra)
        assert resp.status_code == 400
        assert next(iter(extra)) in resp.json()["detail"]
    assert backend.requests == [] and stub.searches == []


def test_the_page_script_offers_the_fact_check_for_free_text_only() -> None:
    update = script_function("updateFactcheck")
    assert "picked === null" in update
    assert "!lastParams.has('claim_id')" in update
    assert "fcPanel.hidden = !shown" in update

    handler = click_handler()
    deleted = re.search(
        r"\[([^\]]*)\]\.forEach\(function \(name\) \{ params\.delete\(name\); \}\)", handler
    )
    assert deleted is not None
    assert set(re.findall(r"'(\w+)'", deleted.group(1))) == {"claim_id", "sentences", "repeat"}
    assert handler.index("new URLSearchParams(lastParams)") < deleted.start()
    assert deleted.end() < handler.index("fetch('/api/factcheck?'")
    assert "method: 'POST'" in handler and "headers[fc.header] = fc.header_value" in handler
    # the only POST of the page is in the click handler: no call without a click
    assert webmod.INDEX_HTML.count("fetch('/api/factcheck?'") == 1

    assert "fc.question_words" in script_function("looksLikeQuestion")
    render = script_function("renderVerdict")
    assert "p.passages && p.passages.length && same" in render


# ---------------------------------------------------------------------------------------------
# the request: the runner's own rendering of the passages
# ---------------------------------------------------------------------------------------------


def test_the_user_turn_is_the_runners_rendering_of_the_passages(
    experiment: factcheck.Experiment,
) -> None:
    passages = [
        {"rank": 1, "chunk_id": 11, "page_id": 1, "title": "Aare", "section_path": "",
         "chunk_ordinal": 0, "n_words": 5, "distance": 0.1, "text": "The Aare is a river."},
        {"rank": 2, "chunk_id": 22, "page_id": 2, "title": "Bern",
         "section_path": "Geography > Rivers", "chunk_ordinal": 3, "n_words": 5,
         "distance": 0.2, "text": "Bern lies on the Aare."},
    ]
    request = factcheck.experiment_request(experiment, STATEMENT, passages)
    prompt = experiment.prompt
    assert request.user_text == prompt.render_user(
        "retrieved", {"claim": STATEMENT, "evidence": claims.render_retrieved(passages)}
    )
    assert request.user_text == (
        "EVIDENCE:\n"
        "[1] Aare\nThe Aare is a river.\n\n"
        "[2] Bern > Geography > Rivers\nBern lies on the Aare.\n\n"
        "CLAIM: The Aare flows through Bern.\n"
    )

    record = retrieved_run_record()
    loaded = prompts_module.load_prompt_set(record["prompt_id"])
    assert loaded.prompt_hash == record["prompt_hash"] == prompt.prompt_hash
    assert request.system == loaded.system
    assert request.schema == loaded.schema and request.schema_json == loaded.schema_json
    assert request.label_path == loaded.label_path
    assert request.model_id == record["model"][0]
    assert request.effort == record["effort"]
    assert request.condition == "retrieved" == record["conditions"][0]
    assert request.cache_prefix == ""

    retrieval_record = record["retrieval"]
    assert (experiment.database, experiment.k, experiment.strategy, experiment.ef_search,
            experiment.embedding_model) == (
        retrieval_record["database"], retrieval_record["k"], retrieval_record["strategy"],
        retrieval_record["ef_search"], retrieval_record["embedding_model"])
    assert experiment.cli_version == record["cli_version"]


def test_passages_of_gives_the_runners_hit_dicts() -> None:
    hits = make_hits(7)
    assert factcheck.passages_of(hits, 5) == [
        retrieval._hit_dict(rank, hit) for rank, hit in enumerate(hits[:5], 1)
    ]
    assert [p["rank"] for p in factcheck.passages_of(hits[:3], 5)] == [1, 2, 3]
    assert factcheck.passages_of([], 5) == []


@pytest.mark.parametrize(("k", "sent"), [(7, 5), (3, 3)])
def test_the_endpoint_sends_at_most_the_experiments_passages(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
    k: int,
    sent: int,
) -> None:
    assert experiment.k == 5
    hits = make_hits(7)
    stub = install_search(monkeypatch, hits)
    backend = FakeBackend()
    client = make_client(make_checker(backend, experiment, reliability),
                         settings=matching_settings(experiment), database=experiment.database)
    resp = post(client, q=f"  {STATEMENT}  ", k=k)
    assert resp.status_code == 200, resp.text
    assert [s["k"] for s in stub.searches] == [k]
    assert len(backend.requests) == 1
    request = backend.requests[0]
    expected = factcheck.passages_of(hits[:k], experiment.k)
    assert len(expected) == sent
    assert request.user_text == factcheck.experiment_request(
        experiment, STATEMENT, expected).user_text
    assert len(re.findall(r"^\[\d+\] Page ", request.user_text, flags=re.MULTILINE)) == sent
    body = resp.json()
    assert [p["rank"] for p in body["passages"]] == list(range(1, sent + 1))
    assert [p["chunk_id"] for p in body["passages"]] == [h.chunk_id for h in hits[:sent]]


# ---------------------------------------------------------------------------------------------
# the reason and its citations
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "parts", "cited", "unknown"),
    [
        ("As [2] says.", [{"text": "As "}, {"text": "[2]", "passages": [2]}, {"text": " says."}],
         [2], []),
        ("Both [1, 3] agree", [{"text": "Both "}, {"text": "[1, 3]", "passages": [1, 3]},
                               {"text": " agree"}], [1, 3], []),
        ("[2-3] give the year.", [{"text": "[2-3]", "passages": [2, 3]},
                                  {"text": " give the year."}], [2, 3], []),
        ("See [passage 2].", [{"text": "See "}, {"text": "[passage 2]", "passages": [2]},
                              {"text": "."}], [2], []),
        ("Only [7] says so.", [{"text": "Only "}, {"text": "[7]", "passages": []},
                               {"text": " says so."}], [], [7]),
        ("No passage mentions it.", [{"text": "No passage mentions it."}], [], []),
    ],
    ids=["one", "list", "range", "passage word", "out of range", "none"],
)
def test_reason_parts_finds_the_citations(
    reason: str, parts: list[dict[str, Any]], cited: list[int], unknown: list[int]
) -> None:
    got_parts, got_cited, got_unknown = factcheck.reason_parts(reason, 5)
    assert got_parts == parts
    assert got_cited == cited and got_unknown == unknown
    assert "".join(p["text"] for p in got_parts) == reason


@pytest.mark.parametrize(
    ("justification", "cited", "unknown"),
    [
        ("Passage [2] names the river, and [1, 3] agree.", [1, 2, 3], []),
        ("The passages do not mention it.", [], []),
        ("See [7], which names no passage.", [], [7]),
    ],
    ids=["cited", "no citation", "only out of range"],
)
def test_the_response_carries_the_citations(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
    justification: str,
    cited: list[int],
    unknown: list[int],
) -> None:
    install_search(monkeypatch, make_hits(5))
    backend = FakeBackend(ok_result("REFUTES", justification))
    client = make_client(make_checker(backend, experiment, reliability))
    body = post(client, q=STATEMENT).json()
    assert body["verdict"] == "REFUTES" and body["verdict_text"] == "Refuted"
    assert body["reason"] == justification
    assert body["reason_parts"] == factcheck.reason_parts(justification, 5)[0]
    assert body["cited"] == cited and body["cited_unknown"] == unknown
    assert body["no_citation"] == (factcheck.NO_CITATION_TEXT if not cited else None)
    assert body["reason_note"] == factcheck.REASON_NOTE
    assert body["reason_label"] == f"{experiment.model_name}'s reason"
    assert body["judged_as_of"] == factcheck.JUDGED_AS_OF_TEXT


# ---------------------------------------------------------------------------------------------
# the caveat when the search differs from the experiment's
# ---------------------------------------------------------------------------------------------


DIFFERENCES = [
    pytest.param({}, {}, False,
                 lambda e: f"database {OFFLINE_SETTINGS.db_name} instead of {e.database}",
                 id="database"),
    pytest.param({"k": 7}, {}, True, lambda e: f"k 7 instead of {e.k}", id="k"),
    pytest.param({"strategy": "overfetch"}, {}, True,
                 lambda e: f"strategy overfetch instead of {e.strategy}", id="strategy"),
    pytest.param({"heading": "%History%"}, {}, True,
                 lambda e: "filters heading (the experiment used none)", id="filter heading"),
    pytest.param({"titles": ["Page 1", "Page 2"]}, {}, True,
                 lambda e: "filters titles (the experiment used none)", id="filter titles"),
    pytest.param({"ef_search": 250}, {}, True,
                 lambda e: f"mhnsw_ef_search 250 instead of {e.ef_search}", id="ef_search"),
    pytest.param({"ef_search": 0}, {}, True,
                 lambda e: f"mhnsw_ef_search {SERVER_EF} instead of {e.ef_search}",
                 id="ef_search 0 (server value)"),
    pytest.param({}, {"ef_search": 0}, True,
                 lambda e: f"mhnsw_ef_search {SERVER_EF} instead of {e.ef_search}",
                 id="WIKILENSE_EF_SEARCH 0"),
    pytest.param({}, {"embedding_model": "intfloat/e5-small-v2"}, True,
                 lambda e: f"embedding model intfloat/e5-small-v2 instead of {e.embedding_model}",
                 id="embedding model"),
]


@pytest.mark.parametrize(("params", "settings_changes", "on_experiment_db", "phrase"),
                         DIFFERENCES)
def test_a_search_that_differs_from_the_experiments_gives_a_caveat(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
    params: dict[str, Any],
    settings_changes: dict[str, Any],
    on_experiment_db: bool,
    phrase: Callable[[factcheck.Experiment], str],
) -> None:
    assert SERVER_EF != experiment.ef_search
    install_search(monkeypatch, make_hits(7))
    client = make_client(
        make_checker(FakeBackend(), experiment, reliability),
        settings=matching_settings(experiment, **settings_changes),
        database=experiment.database if on_experiment_db else None,
    )
    resp = post(client, q=STATEMENT, **params)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    expected = phrase(experiment)
    assert body["settings_differ"] == [expected]
    assert len(body["caveats"]) == 1
    caveat = body["caveats"][0]
    assert caveat.startswith("Your search differs from the experiment's (" + expected + ")")
    assert (f"on {experiment.database} (k {experiment.k}, strategy {experiment.strategy}, "
            f"mhnsw_ef_search {experiment.ef_search}, no filters)") in caveat


def test_the_experiments_own_settings_give_no_caveat(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    stub = install_search(monkeypatch, make_hits(7), session_ef=experiment.ef_search)
    client = make_client(make_checker(FakeBackend(), experiment, reliability),
                         settings=matching_settings(experiment), database=experiment.database)
    for params in (
        {},
        {"k": experiment.k, "strategy": experiment.strategy, "ef_search": experiment.ef_search},
        {"heading": "", "titles": [" "]},
        {"ef_search": 0},  # the server's session value equals the experiment's here
    ):
        body = post(client, q=STATEMENT, **params).json()
        assert body["settings_differ"] == [], params
        assert body["caveats"] == [], params
        assert body["database"] == experiment.database
    assert set(stub.databases) == {experiment.database}

    state = SimpleNamespace(database=experiment.database, settings=matching_settings(experiment))
    same = {"k": experiment.k, "strategy": experiment.strategy}
    assert webmod.factcheck_differences(
        state, same, {"ef_search": experiment.ef_search}, experiment) == []


def test_factcheck_differences_lists_every_difference_in_order(
    experiment: factcheck.Experiment,
) -> None:
    state = SimpleNamespace(database="wikilense",
                            settings=matching_settings(experiment, embedding_model="other/model"))
    params = {"k": 9, "strategy": "rrf", "category": "Rivers", "min_words": 0}
    out = webmod.factcheck_differences(state, params, {"ef_search": 33}, experiment)
    assert out == [
        f"database wikilense instead of {experiment.database}",
        f"k 9 instead of {experiment.k}",
        f"strategy rrf instead of {experiment.strategy}",
        "filters min_words, category (the experiment used none)",
        f"mhnsw_ef_search 33 instead of {experiment.ef_search}",
        f"embedding model other/model instead of {experiment.embedding_model}",
    ]


def test_the_api_route_always_carries_its_caveat(
    experiment: factcheck.Experiment, reliability: factcheck.Reliability
) -> None:
    api = make_checker(FakeBackend(), experiment, reliability, mode=factcheck.API)
    caveats = api.caveats([])
    assert len(caveats) == 1 and "Anthropic API" in caveats[0]
    sub = make_checker(FakeBackend(), experiment, reliability)
    assert sub.caveats([]) == []
    sub.backend_version = experiment.cli_version
    assert sub.caveats([]) == []


# ---------------------------------------------------------------------------------------------
# the reliability figures
# ---------------------------------------------------------------------------------------------


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d[\d,]*(?:\.\d+)?", text)


def test_missing_figures_are_named_as_missing_and_never_filled_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
) -> None:
    path = tmp_path / "partial_results.json"
    path.write_text(json.dumps({"three_class": {"conditions": {"retrieved": {
        "per_label": {
            "SUPPORTS": {"precision": 0.625},
            "REFUTES": {"precision": True},
            "NOT ENOUGH INFO": {"recall": 0.5},
        },
        "accuracy": {"right": 1234, "n": 2000, "pct": "61.7", "wilson_pct": ["59.5", "63.8"]},
    }}}}), encoding="utf-8")
    rel = factcheck.load_reliability(path)
    src = str(path)
    assert rel.file_error is None
    assert rel.per_verdict["SUPPORTS"].pct == half_up(0.625) == "62.5"
    assert rel.per_verdict["REFUTES"].missing and rel.per_verdict["NOT ENOUGH INFO"].missing
    assert rel.overall.pct == "61.7" and (rel.overall.right, rel.overall.n) == (1234, 2000)
    assert rel.supported_refuted == factcheck.Figure(
        None, source=f"{src}: supported_refuted.conditions.retrieved.accuracy")
    assert rel.missed_gold_page.missing
    assert rel.missing() == [
        f"{src}: three_class.conditions.retrieved.per_label.REFUTES.precision",
        f"{src}: three_class.conditions.retrieved.per_label.NOT ENOUGH INFO.precision",
        f"{src}: supported_refuted.conditions.retrieved.accuracy",
        f"{src}: retrieval_link.all_claims.gold_page_in_top5.no",
    ]
    assert rel.as_json()["missing"] == rel.missing()
    assert all(f["missing"] is (f["pct"] is None) for f in rel.as_json()["per_verdict"].values())

    allowed = {"62.5", "61.7", "1,234", "2,000", "59.5", "63.8", str(experiment.k)}
    sources = [fig.source for fig in rel.figures().values()]
    for label in ("SUPPORTS", "REFUTES"):
        lines = factcheck.reliability_lines(rel, label, experiment.model_name, experiment.k)
        text = " ".join(lines.values())
        for source in sources:
            text = text.replace(source, "")
        assert set(_numbers(text)) <= allowed, (label, text)
        assert "not in the results file" in lines["lower_end"]
        assert "supported/refuted accuracy is not in the results file" in lines["overall"]
        assert lines["limit"] == factcheck.LIMIT_TEXT
    refuted = factcheck.reliability_lines(rel, "REFUTES", experiment.model_name, experiment.k)
    assert refuted["verdict"] == (
        'How often "Refuted" was right in the experiment: not in the results file '
        f"({src}: three_class.conditions.retrieved.per_label.REFUTES.precision).")

    install_search(monkeypatch, make_hits(5))
    client = make_client(make_checker(FakeBackend(ok_result("REFUTES", "As [1] says.")),
                                      experiment, rel))
    body = post(client, q=STATEMENT).json()
    assert body["reliability"]["missing"] == rel.missing()
    assert body["reliability_lines"] == refuted


def test_an_unreadable_results_file_sets_file_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    for path in (broken, tmp_path / "absent.json"):
        rel = factcheck.load_reliability(path)
        assert rel.file_error is not None and rel.file_error.startswith(f"cannot read {path}")
        assert all(fig.missing for fig in rel.figures().values())
        assert len(rel.missing()) == 6
        lines = factcheck.reliability_lines(rel, "SUPPORTS", "Haiku", 5)
        assert "not in the results file" in lines["verdict"]
        assert lines["overall"].startswith("Overall accuracy: not in the results file")


def test_the_real_results_file_gives_the_recorded_figures(
    reliability: factcheck.Reliability, experiment: factcheck.Experiment
) -> None:
    data = json.loads(factcheck.RESULTS_FILE.read_text(encoding="utf-8"))
    cond = data["three_class"]["conditions"]["retrieved"]
    assert reliability.file_error is None and reliability.missing() == []
    for label in prompts_module.LABELS:
        precision = cond["per_label"][label]["precision"]
        assert reliability.per_verdict[label].pct == half_up(precision), label
    overall = cond["accuracy"]
    sides = data["supported_refuted"]["conditions"]["retrieved"]["accuracy"]
    missed = data["retrieval_link"]["all_claims"]["gold_page_in_top5"]["no"]
    for fig, entry in ((reliability.overall, overall), (reliability.supported_refuted, sides),
                       (reliability.missed_gold_page, missed)):
        assert fig.pct == entry["pct"]
        assert (fig.right, fig.n) == (entry["right"], entry["n"])
        assert fig.interval_pct == tuple(entry["wilson_pct"])
    lines = factcheck.reliability_lines(reliability, "SUPPORTS", experiment.model_name,
                                        experiment.k)
    assert f"right {half_up(cond['per_label']['SUPPORTS']['precision'])}% of the time" in \
        lines["verdict"]
    assert f"Overall it was right on {overall['pct']}% of the claims" in lines["overall"]
    assert f"on {sides['pct']}% of the supported or refuted ones" in lines["overall"]
    assert f"({missed['pct']}%, interval {missed['wilson_pct'][0]}–" in lines["lower_end"]


def test_a_precision_on_a_half_is_rounded_half_up(tmp_path: Path) -> None:
    """247/2000 and 469/2000 are 12.35% and 23.45%: one decimal, half up, gives 12.4 and 23.5,
    as the results file's own rule (exact fractions, then half up) gives them."""
    path = tmp_path / "halves.json"
    precisions = {"SUPPORTS": 247 / 2000, "REFUTES": 469 / 2000, "NOT ENOUGH INFO": 0.375}
    path.write_text(json.dumps({"three_class": {"conditions": {"retrieved": {"per_label": {
        label: {"precision": value} for label, value in precisions.items()}}}}}),
        encoding="utf-8")
    rel = factcheck.load_reliability(path)
    got = {label: rel.per_verdict[label].pct for label in precisions}
    assert got == {"SUPPORTS": "12.4", "REFUTES": "23.5", "NOT ENOUGH INFO": "37.5"}
    assert got == {label: half_up(value) for label, value in precisions.items()}


# ---------------------------------------------------------------------------------------------
# calls that give no verdict, and the requests refused without a call
# ---------------------------------------------------------------------------------------------


FAILURES = [
    pytest.param(CallResult("error", error="no answer after 60 s", error_kind="timeout"),
                 factcheck.SUBSCRIPTION, 504, f"No answer within {factcheck.TIMEOUT_S:.0f} s",
                 id="timeout"),
    pytest.param(CallResult("window", error="5-hour limit reached", error_kind="exhausted"),
                 factcheck.SUBSCRIPTION, 503, "usage limit of this machine's Claude login",
                 id="window"),
    pytest.param(CallResult("fatal", error="Invalid API key, please run /login",
                            error_kind="auth"),
                 factcheck.SUBSCRIPTION, 503, "is not logged in", id="auth subscription"),
    pytest.param(CallResult("fatal", error="401 authentication_error", error_kind="auth"),
                 factcheck.API, 503, "The API key was refused", id="auth api"),
    pytest.param(CallResult("parse_failed", parse_status="failed",
                            error="no schema-valid answer"),
                 factcheck.SUBSCRIPTION, 502, "not in the expected form", id="parse_failed"),
    pytest.param(CallResult("refusal", error="usage policy"),
                 factcheck.SUBSCRIPTION, 502, "The model declined", id="refusal"),
    pytest.param(CallResult("money", error="spend cap reached", error_kind="cap"),
                 factcheck.API, 503, "spending cap", id="money"),
    pytest.param(CallResult("fatal", error="claude-sonnet answered", error_kind="model"),
                 factcheck.SUBSCRIPTION, 503, "cannot be reached", id="fatal other"),
    pytest.param(CallResult("error", error="connection reset", error_kind="transient"),
                 factcheck.SUBSCRIPTION, 502, "The model call failed", id="error other"),
    pytest.param(CallResult("ok", label="SUPPORTS", answer={"result": {"label": "SUPPORTS"}}),
                 factcheck.SUBSCRIPTION, 502, "The model call failed", id="ok without reason"),
]


@pytest.mark.parametrize(("result", "mode", "code", "phrase"), FAILURES)
def test_a_call_without_a_verdict_answers_with_a_plain_detail(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
    result: CallResult,
    mode: str,
    code: int,
    phrase: str,
) -> None:
    install_search(monkeypatch, make_hits(5))
    backend = FakeBackend(result)
    checker = make_checker(backend, experiment, reliability, mode=mode)
    resp = post(make_client(checker), q=STATEMENT)
    assert resp.status_code == code
    body = resp.json()
    assert len(backend.requests) == 1
    assert "verdict" not in body and "reason" not in body
    assert isinstance(body["detail"], str) and phrase in body["detail"]
    assert "no verdict" in body["detail"]
    assert result.error is None or result.error not in body["detail"]
    assert body["status"] == result.status and body["error_kind"] == result.error_kind
    assert body["error"] == result.error
    assert set(body["timing_ms"]) == {"embed", "sql", "model_call"}
    assert body["timing_ms"]["model_call"] >= 0
    assert not checker.lock.locked()


def test_a_backend_that_raises_is_a_502_and_frees_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    install_search(monkeypatch, make_hits(5))
    backend = FakeBackend(raises=RuntimeError("socket closed"))
    checker = make_checker(backend, experiment, reliability)
    client = make_client(checker)
    resp = post(client, q=STATEMENT)
    assert resp.status_code == 502
    body = resp.json()
    assert "verdict" not in body and "no verdict" in body["detail"]
    assert "socket closed" not in body["detail"]
    assert body["status"] == "error" and body["error_kind"] == "backend"
    assert body["error"] == "RuntimeError: socket closed"
    assert body["timing_ms"]["model_call"] >= 0
    assert not checker.lock.locked()
    backend.raises = None
    assert post(client, q=STATEMENT).status_code == 200
    assert len(backend.requests) == 2


def test_a_second_call_while_one_runs_is_429(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    install_search(monkeypatch, make_hits(5))
    backend = FakeBackend()
    checker = make_checker(backend, experiment, reliability)
    client = make_client(checker)
    assert checker.lock.acquire(blocking=False)
    try:
        resp = post(client, q=STATEMENT)
    finally:
        checker.lock.release()
    assert resp.status_code == 429
    assert resp.json() == {"detail": "a fact-check is already running on this server; wait for "
                                     "it to finish"}
    assert backend.requests == []
    assert post(client, q=STATEMENT).status_code == 200 and len(backend.requests) == 1


def test_a_statement_over_the_cap_is_422_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    stub = install_search(monkeypatch, make_hits(5))
    backend = FakeBackend()
    checker = make_checker(backend, experiment, reliability)
    client = make_client(checker)
    cap = factcheck.MAX_STATEMENT_CHARS
    resp = post(client, q="a" * (cap + 1))
    assert resp.status_code == 422
    assert f"{cap + 1:,} characters" in resp.json()["detail"]
    assert f"at most {cap:,}" in resp.json()["detail"]
    assert backend.requests == [] and stub.searches == []
    with pytest.raises(factcheck.FactCheckRefused) as info:
        checker.run("b" * (cap + 1), make_hits(5), [])
    assert info.value.status_code == 422 and backend.requests == []
    # the cap counts the statement without its surrounding blanks
    assert post(client, q="  " + "a" * cap + "  ").status_code == 200
    assert len(backend.requests) == 1


def test_a_search_without_hits_is_422_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    stub = install_search(monkeypatch, [])
    backend = FakeBackend()
    resp = post(make_client(make_checker(backend, experiment, reliability)), q=STATEMENT)
    assert resp.status_code == 422
    assert "no passages" in resp.json()["detail"]
    assert len(stub.searches) == 1 and backend.requests == []


def test_a_request_without_the_header_is_403_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    stub = install_search(monkeypatch, make_hits(5))
    backend = FakeBackend()
    client = make_client(make_checker(backend, experiment, reliability))
    for headers in ({}, {factcheck.HEADER: "0"}):
        resp = post(client, headers=headers, q=STATEMENT)
        assert resp.status_code == 403
        assert f"send the header {factcheck.HEADER}: {factcheck.HEADER_VALUE}" in \
            resp.json()["detail"]
    assert backend.requests == [] and stub.searches == []


# ---------------------------------------------------------------------------------------------
# one model call per click
# ---------------------------------------------------------------------------------------------


def test_one_click_is_one_model_call(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    stub = install_search(monkeypatch, make_hits(5))
    backend = FakeBackend()
    client = make_client(make_checker(backend, experiment, reliability))
    assert client.get("/").status_code == 200
    status = client.get("/api/factcheck").json()
    assert status["enabled"] is True and backend.requests == []
    assert status["model"] == experiment.model_id and status["passages"] == experiment.k
    assert status["header"] == factcheck.HEADER and status["header_value"] == "1"
    assert status["experiment"] == experiment.describe()
    for clicks in (1, 2, 3):
        resp = post(client, q=STATEMENT)
        assert resp.status_code == 200
        assert len(backend.requests) == clicks and len(stub.searches) == clicks
        body = resp.json()
        assert set(body["timing_ms"]) == {"embed", "sql", "model_call"}
        assert body["model"] == experiment.model_id and body["mode"] == "subscription"
    client.get("/")
    client.get("/api/factcheck")
    assert len(backend.requests) == 3


# ---------------------------------------------------------------------------------------------
# the API key never leaves the backend
# ---------------------------------------------------------------------------------------------


def test_the_api_key_never_leaves_the_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    env_file = write_env(tmp_path / ".env", f"ANTHROPIC_API_KEY={FAKE_KEY}")
    checker = factcheck.build_factchecker(replace(OFFLINE_SETTINGS, factcheck="api"),
                                          environ={}, env_file=env_file)
    assert checker.enabled and checker.secrets == (FAKE_KEY,)
    assert isinstance(checker.backend, api_backend.AnthropicApiBackend)
    assert FAKE_KEY not in json.dumps(checker.status())
    line = checker.check()
    assert "key read (not shown)" in line and FAKE_KEY not in line
    assert FAKE_KEY not in repr(checker)
    assert checker.scrub(f"a {FAKE_KEY} b") == "a [hidden] b"
    assert checker.scrub(None) is None

    install_search(monkeypatch, make_hits(5))
    backend = FakeBackend(ok_result("REFUTES", f"Passage [1] says otherwise; {FAKE_KEY}."))
    checker.backend = backend
    client = make_client(checker)
    ok = post(client, q=STATEMENT)
    assert ok.status_code == 200 and FAKE_KEY not in ok.text
    assert ok.json()["reason"] == "Passage [1] says otherwise; [hidden]."
    assert ok.json()["cited"] == [1]

    backend.result = CallResult("error", error=f"401 invalid x-api-key {FAKE_KEY}",
                                error_kind="auth")
    failed = post(client, q=STATEMENT)
    assert failed.status_code == 502 and FAKE_KEY not in failed.text
    assert failed.json()["error"] == "401 invalid x-api-key [hidden]"
    for path in ("/", "/api/factcheck"):
        resp = client.get(path)
        assert resp.status_code == 200 and FAKE_KEY not in resp.text
    assert len(backend.requests) == 2
    assert FAKE_KEY not in repr(checker)
    assert FAKE_KEY not in caplog.text
    assert all(FAKE_KEY not in str(r.args) for r in caplog.records)


# ---------------------------------------------------------------------------------------------
# the experiment's record and its refusals
# ---------------------------------------------------------------------------------------------


def _retrieved_original(record: dict[str, Any]) -> dict[str, Any]:
    return next(r for r in record["runs"]
                if r["run"] == "original" and r["conditions"] == ["retrieved"])


def _change_system(runs: Path, prompts_dir: Path) -> None:
    system = prompts_dir / "fact_check_v1" / "system.txt"
    system.write_text(system.read_text(encoding="utf-8") + "\nAnswer briefly.\n",
                      encoding="utf-8")


def _drop_run(runs: Path, prompts_dir: Path) -> None:
    record = json.loads(runs.read_text(encoding="utf-8"))
    record["runs"].remove(_retrieved_original(record))
    runs.write_text(json.dumps(record), encoding="utf-8")


def _double_run(runs: Path, prompts_dir: Path) -> None:
    record = json.loads(runs.read_text(encoding="utf-8"))
    record["runs"].append(dict(_retrieved_original(record)))
    runs.write_text(json.dumps(record), encoding="utf-8")


def _remove_record(runs: Path, prompts_dir: Path) -> None:
    runs.unlink()


def _drop_judging_date(runs: Path, prompts_dir: Path) -> None:
    system = prompts_dir / "fact_check_v1" / "system.txt"
    text = system.read_text(encoding="utf-8")
    assert "as of December 2020" in text
    system.write_text(text.replace("as of December 2020", "as of today"), encoding="utf-8")
    # the record names the changed set's hash, so the date check is what refuses it
    record = json.loads(runs.read_text(encoding="utf-8"))
    _retrieved_original(record)["prompt_hash"] = prompts_module.load_prompt_set(
        "fact_check_v1", prompts_dir).prompt_hash
    runs.write_text(json.dumps(record), encoding="utf-8")


def _remove_template(runs: Path, prompts_dir: Path) -> None:
    (prompts_dir / "fact_check_v1" / "user_retrieved.txt").unlink()


@pytest.fixture
def record_copy(tmp_path: Path) -> tuple[Path, Path]:
    """A copy of runs.json and of the fact_check_v1 prompt set in ``tmp_path``."""
    runs = tmp_path / "runs.json"
    shutil.copyfile(factcheck.RUNS_FILE, runs)
    prompts_dir = tmp_path / "prompts"
    shutil.copytree(prompts_module.PROMPTS_DIR / "fact_check_v1", prompts_dir / "fact_check_v1")
    return runs, prompts_dir


def test_an_unchanged_copy_of_the_record_loads(
    record_copy: tuple[Path, Path], experiment: factcheck.Experiment
) -> None:
    runs, prompts_dir = record_copy
    copy = factcheck.load_experiment(runs, prompts_dir=prompts_dir)
    assert copy.prompt.prompt_hash == experiment.prompt.prompt_hash
    assert {k: v for k, v in copy.describe().items() if k != "source"} == {
        k: v for k, v in experiment.describe().items() if k != "source"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_change_system, "differs from the one the experiment ran"),
        (_drop_run, "holds 0 original runs of the 'retrieved' condition, not one"),
        (_double_run, "holds 2 original runs of the 'retrieved' condition, not one"),
        (_remove_record, "cannot read the experiment's record"),
        (_drop_judging_date, "does not judge as of December 2020"),
        (_remove_template, "the experiment's prompt set cannot be used"),
    ],
    ids=["changed system prompt", "no run", "two runs", "missing record", "no judging date",
         "missing template"],
)
def test_load_experiment_refuses_and_the_fact_check_stays_off(
    monkeypatch: pytest.MonkeyPatch,
    record_copy: tuple[Path, Path],
    change: Callable[[Path, Path], None],
    message: str,
) -> None:
    runs, prompts_dir = record_copy
    change(runs, prompts_dir)
    with pytest.raises(factcheck.FactCheckError, match=re.escape(message)) as info:
        factcheck.load_experiment(runs, prompts_dir=prompts_dir)
    reason = str(info.value)

    monkeypatch.setattr(prompts_module, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(cli_backend, "ClaudeCliBackend", _forbidden("ClaudeCliBackend"))
    checker = factcheck.build_factchecker(
        replace(OFFLINE_SETTINGS, factcheck="subscription"), environ={}, runs_file=runs)
    assert not checker.enabled and checker.backend is None
    assert checker.unavailable == reason
    detail = checker.off_detail()
    assert detail == f"The live fact-check is set to subscription but cannot run: {reason}"

    client = make_client(checker)
    resp = post(client, q=STATEMENT)
    assert resp.status_code == 403 and resp.json() == {"detail": detail}
    status = client.get("/api/factcheck").json()
    assert status["enabled"] is False and status["mode"] == "subscription"
    assert status["note"] == detail


# ---------------------------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "question"),
    [
        ("The Aare flows through Bern?", True),
        ("Does the Aare flow through Bern", True),
        ("  who founded Bern  ", True),
        ("The Aare flows through Bern.", False),
        ("Isabel founded Bern.", False),
        ("", False),
    ],
)
def test_looks_like_question(text: str, question: bool) -> None:
    assert factcheck.looks_like_question(text) is question


def test_the_response_gives_the_question_hint_for_a_question_only(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    install_search(monkeypatch, make_hits(5))
    checker = make_checker(FakeBackend(), experiment, reliability)
    assert checker.status()["question_words"] == list(factcheck.QUESTION_WORDS)
    client = make_client(checker)
    assert post(client, q="Is the Aare in Bern?").json()["question_hint"] == \
        factcheck.QUESTION_HINT
    assert post(client, q=STATEMENT).json()["question_hint"] is None


# ---------------------------------------------------------------------------------------------
# check(): the CLI's login, without a model call
# ---------------------------------------------------------------------------------------------


def test_check_with_a_logged_in_cli_names_the_model_and_effort(
    tmp_path: Path, experiment: factcheck.Experiment
) -> None:
    script, log = fake_cli(tmp_path, {"loggedIn": True, "authMethod": "claude.ai",
                                      "apiProvider": "firstParty", "subscriptionType": "max"})
    checker = factcheck.build_factchecker(
        replace(OFFLINE_SETTINGS, factcheck="subscription", claude_bin=str(script)), environ={})
    assert checker.enabled and isinstance(checker.backend, cli_backend.ClaudeCliBackend)
    assert checker.backend.max_attempts == 1 and checker.backend.timeout_s == factcheck.TIMEOUT_S
    assert Path(checker.backend.cwd).is_relative_to(tmp_path)

    line = checker.check()
    assert line.startswith(f"subscription, {experiment.model_id} at effort {experiment.effort}, "
                           f"prompt {experiment.prompt.prompt_id} ")
    assert "login claude.ai, plan max" in line and line.endswith("CLI 9.9.9 (Claude Code)")
    assert checker.enabled and checker.backend_version == "9.9.9 (Claude Code)"
    assert log.read_text(encoding="utf-8").splitlines() == ["auth status --json", "--version"]
    assert checker.caveats([]) == [
        (f"This server's Claude Code CLI is 9.9.9 (Claude Code); the experiment ran "
         f"{experiment.cli_version}.")
    ]


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ({"loggedIn": False}, "the claude CLI is not logged in"),
        ({"loggedIn": True, "authMethod": "api_key"}, "that route bills per token"),
    ],
    ids=["not logged in", "API-key login"],
)
def test_check_with_a_cli_that_cannot_be_used_makes_the_checker_unavailable(
    tmp_path: Path, status: dict[str, Any], reason: str
) -> None:
    script, log = fake_cli(tmp_path, status)
    checker = factcheck.build_factchecker(
        replace(OFFLINE_SETTINGS, factcheck="subscription", claude_bin=str(script)), environ={})
    assert checker.enabled
    line = checker.check()
    assert not checker.enabled and reason in checker.unavailable
    assert line == checker.off_detail()
    assert line.startswith("The live fact-check is set to subscription but cannot run")
    assert checker.status()["enabled"] is False and checker.status()["note"] == line
    assert log.read_text(encoding="utf-8").splitlines() == ["auth status --json"]


def test_subscription_without_a_cli_binary_is_unavailable() -> None:
    checker = factcheck.build_factchecker(replace(OFFLINE_SETTINGS, factcheck="subscription"),
                                          environ={})
    assert not checker.enabled and checker.backend is None
    assert "no Claude Code CLI found" in checker.unavailable
    assert "WIKILENSE_CLAUDE_BIN" in checker.off_detail()


# ---------------------------------------------------------------------------------------------
# stop rules, the host rule, citation forms, lazy login check, rate limits
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "first_status"),
    [
        (replace(ok_result(), stop_signal="overage: isUsingOverage"), 200),
        (replace(ok_result(), stop_signal="overage: a 5-minute cache write"), 200),
        (CallResult("fatal", error_kind="api_key_source", error="init.apiKeySource is 'x'"), 503),
        (CallResult("fatal", error_kind="wrong_model", error="wrong model served"), 503),
        (CallResult("money", error_kind="max_usd", error="cap reached",
                    stop_signal="money: --max-usd reached"), 503),
    ],
)
def test_a_stop_signal_or_a_fatal_result_turns_the_fact_check_off(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
    result: CallResult,
    first_status: int,
) -> None:
    install_search(monkeypatch, make_hits(5))
    backend = FakeBackend(result)
    checker = make_checker(backend, experiment, reliability)
    client = make_client(checker)
    first = post(client, q=STATEMENT)
    assert first.status_code == first_status
    body = first.json()
    assert body["turned_off"] and "restart the server" in body["turned_off"]
    if first_status == 200:
        assert body["verdict"] == "SUPPORTS" and body["turned_off"] in body["caveats"]
    else:
        assert "verdict" not in body and body["turned_off"] in body["detail"]
    assert not checker.enabled
    second = post(client, q=STATEMENT)
    assert second.status_code == 403 and "restart the server" in second.json()["detail"]
    assert client.get("/api/factcheck").json()["enabled"] is False
    assert len(backend.requests) == 1


def test_an_ordinary_failure_leaves_the_fact_check_on(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    install_search(monkeypatch, make_hits(5))
    checker = make_checker(FakeBackend(CallResult("parse_failed", error_kind="invalid_answer")),
                           experiment, reliability)
    resp = post(make_client(checker), q=STATEMENT)
    assert resp.status_code == 502 and resp.json()["turned_off"] is None
    assert checker.enabled


@pytest.mark.parametrize(
    ("base_url", "origin", "allowed"),
    [
        ("http://127.0.0.1:8000", None, True),
        ("http://localhost:8000", None, True),
        ("http://localhost:8000", "http://127.0.0.1:8000", True),
        ("http://rebind.attacker.example:8000", None, False),
        ("http://rebind.attacker.example:8000", "http://rebind.attacker.example:8000", False),
        ("http://laptop.local:8000", None, False),
        ("http://127.0.0.1:8000", "http://rebind.attacker.example:8000", False),
    ],
)
def test_the_fact_check_answers_only_requests_addressed_to_this_machine(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
    base_url: str,
    origin: str | None,
    allowed: bool,
) -> None:
    install_search(monkeypatch, make_hits(5))
    backend = FakeBackend()
    app = webmod.create_app(OFFLINE_SETTINGS, embedder=FakeEmbedder(), warm_up=False,
                            offer_s1=False, fact_checker=make_checker(backend, experiment,
                                                                      reliability))
    client = TestClient(app, base_url=base_url)
    headers = dict(HEADERS) if origin is None else {**HEADERS, "Origin": origin}
    resp = post(client, headers=headers, q=STATEMENT)
    status = client.get("/api/factcheck", headers={} if origin is None else {"Origin": origin})
    if allowed:
        assert resp.status_code == 200 and len(backend.requests) == 1
        assert status.json()["enabled"] is True
    else:
        assert resp.status_code == 403 and resp.json()["detail"] == factcheck.LOOPBACK_NOTE
        assert backend.requests == []
        assert status.json()["enabled"] is False
        assert status.json()["note"] == factcheck.LOOPBACK_NOTE


@pytest.mark.parametrize(
    ("reason", "cited"),
    [
        ("Passages [1, 2, and 3] confirm it.", [1, 2, 3]),
        ("Passage 2 confirms it.", [2]),
        ("It flows through Bern (passage 3).", [3]),
        ("As passages 1 and 2 say.", [1, 2]),
        ("See [2 & 4].", [2, 4]),
        ("The passages say nothing about it.", []),
    ],
)
def test_reason_parts_reads_the_written_out_citation_forms(reason: str, cited: list[int]) -> None:
    parts, found, unknown = factcheck.reason_parts(reason, 5)
    assert found == cited and unknown == []
    assert "".join(p["text"] for p in parts) == reason
    assert sorted({n for p in parts for n in p.get("passages", [])}) == cited


@pytest.mark.parametrize(
    ("kind", "status", "wording"),
    [("time_limit", 504, "No answer within 60 s"), ("rate_limit", 503, "per-minute rate limit")],
)
def test_the_api_time_and_rate_limits_have_their_own_messages(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
    kind: str,
    status: int,
    wording: str,
) -> None:
    install_search(monkeypatch, make_hits(5))
    checker = make_checker(FakeBackend(CallResult("error", error_kind=kind, error="x")),
                           experiment, reliability, mode=factcheck.API)
    resp = post(make_client(checker), q=STATEMENT)
    assert resp.status_code == status and wording in resp.json()["detail"]
    assert checker.enabled


def test_a_runner_that_cannot_be_imported_leaves_the_fact_check_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = factcheck.importlib.import_module

    def no_fcntl(name: str, *args: Any) -> Any:
        if name == "wikilense.llm.runner":
            raise ImportError("No module named 'fcntl'")
        return real(name, *args)

    monkeypatch.setattr(factcheck.importlib, "import_module", no_fcntl)
    script, _log = fake_cli(tmp_path, {"loggedIn": True, "authMethod": "claude.ai"})
    settings = replace(OFFLINE_SETTINGS, factcheck="subscription", claude_bin=str(script))
    checker = factcheck.build_factchecker(settings, environ={})
    assert not checker.enabled and checker.backend is None
    assert "cannot be imported" in checker.off_detail() and "fcntl" in checker.off_detail()


class CheckedBackend(FakeBackend):
    """A fake backend whose login check answers ``login`` or raises ``refuse``."""

    def __init__(self, refuse: str | None = None) -> None:
        super().__init__()
        self.refuse = refuse
        self.preflights = 0

    def preflight(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        self.preflights += 1
        if self.refuse:
            raise cli_backend.CliBackendError(self.refuse)
        return {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"}

    def version(self) -> str:
        return "9.9.9 (Claude Code)"


def test_the_first_call_checks_the_login_when_the_server_did_not(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    install_search(monkeypatch, make_hits(5))
    good = CheckedBackend()
    checker = factcheck.FactChecker(factcheck.SUBSCRIPTION, backend=good, experiment=experiment,
                                    reliability=reliability)
    client = make_client(checker)
    assert post(client, q=STATEMENT).status_code == 200
    assert post(client, q=STATEMENT).status_code == 200
    assert good.preflights == 1 and len(good.requests) == 2 and checker.checked

    bad = CheckedBackend(refuse="the claude CLI is not logged in")
    checker = factcheck.FactChecker(factcheck.SUBSCRIPTION, backend=bad, experiment=experiment,
                                    reliability=reliability)
    resp = post(make_client(checker), q=STATEMENT)
    assert resp.status_code == 503 and "not logged in" in resp.json()["detail"]
    assert bad.requests == [] and not checker.enabled


def test_the_api_route_makes_one_try_also_on_a_rate_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    settings = replace(OFFLINE_SETTINGS, factcheck="api")
    checker = factcheck.build_factchecker(settings, env_file=tmp_path / "absent.env")
    assert checker.enabled
    assert checker.backend.max_attempts == 1 and checker.backend.rate_limit_attempts == 1
    runner_default = api_backend.AnthropicApiBackend(tmp_path / "absent.env", ("SUPPORTS",),
                                                     max_usd=1.0)
    assert runner_default.rate_limit_attempts == api_backend.RATE_LIMIT_MAX_ATTEMPTS


def test_a_whitespace_only_filter_is_a_filter_as_in_api_search(
    experiment: factcheck.Experiment,
) -> None:
    state = SimpleNamespace(database=experiment.database, settings=matching_settings(experiment))
    same = {"k": experiment.k, "strategy": experiment.strategy}
    body = {"ef_search": experiment.ef_search}
    assert webmod.factcheck_differences(state, {**same, "heading": "  "}, body, experiment) == [
        "filters heading (the experiment used none)"]
    assert webmod.factcheck_differences(state, {**same, "heading": ""}, body, experiment) == []
    assert webmod.factcheck_differences(
        state, {**same, "titles": ["  ", ""]}, body, experiment) == []


def test_the_page_drops_a_reply_that_belongs_to_an_older_search() -> None:
    assert "fcSeq++" in script_function("resetFactcheck")
    handler = click_handler()
    assert "var seq = fcSeq;" in handler
    assert handler.index("var seq = fcSeq;") > handler.index("resetFactcheck();")
    assert "if (seq !== fcSeq || lastBody === null) return;" in handler
    assert "Statement checked" in script_function("renderVerdict")


def test_the_verdict_names_the_statement_it_checked(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    install_search(monkeypatch, make_hits(5))
    client = make_client(make_checker(FakeBackend(), experiment, reliability))
    assert post(client, q=f"  {STATEMENT}  ").json()["statement"] == STATEMENT


# ---------------------------------------------------------------------------------------------
# the copyable prompt: GET /api/prompt and the page's copy box (no key, no model call)
# ---------------------------------------------------------------------------------------------

#: The system prompt's last sentence and its plain replacement, written out here.
TOOL_SENTENCE = (
    'Answer with one StructuredOutput call: "result" holds "label" (exactly one of the three) and '
    '"justification" (one sentence, at most 30 words: the deciding fact and the passages or '
    "sources used, such as [2]). No other text."
)
PLAIN_SENTENCE = (
    "Answer with the label (exactly one of SUPPORTS, REFUTES or NOT ENOUGH INFO) and one sentence "
    "of justification, at most 30 words: the deciding fact and the passages or sources used, such "
    "as [2]. No other text."
)
#: A made-up benchmark claim the stubbed catalog offers.
CLAIM_ID = 7
CLAIM_TEXT = "The Aare rises in the Bernese Alps."


def two_passages() -> list[dict[str, Any]]:
    """Return two passages in the runner's hit-dict form."""
    return [
        {"rank": 1, "chunk_id": 11, "page_id": 1, "title": "Aare", "section_path": "",
         "chunk_ordinal": 0, "n_words": 5, "distance": 0.1, "text": "The Aare is a river."},
        {"rank": 2, "chunk_id": 22, "page_id": 2, "title": "Bern",
         "section_path": "Geography > Rivers", "chunk_ordinal": 3, "n_words": 5,
         "distance": 0.2, "text": "Bern lies on the Aare."},
    ]


def install_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the claim catalog with :data:`CLAIM_ID` and the marks of a picked claim."""
    offered = SimpleNamespace(
        claim=SimpleNamespace(text=CLAIM_TEXT),
        details=lambda: {"id": CLAIM_ID, "text": CLAIM_TEXT, "label": "SUPPORTS"},
    )
    monkeypatch.setattr(
        webmod, "catalog", lambda state, name: SimpleNamespace(claims={CLAIM_ID: offered})
    )
    monkeypatch.setattr(webmod.picker, "search_marks",
                        lambda conn, claim, hits, k: ([{} for _ in hits], {"k": k}))


def get_prompt(client: TestClient, **params: Any) -> Any:
    """GET /api/prompt with ``params`` and no extra header."""
    return client.get("/api/prompt", params=params)


def test_the_two_sentences_are_the_ones_the_copy_route_names() -> None:
    assert factcheck.TOOL_ANSWER_SENTENCE == TOOL_SENTENCE
    assert factcheck.PLAIN_ANSWER_SENTENCE == PLAIN_SENTENCE


def test_the_copy_text_is_the_experiments_request_with_one_sentence_changed(
    experiment: factcheck.Experiment,
) -> None:
    passages = two_passages()
    request = factcheck.experiment_request(experiment, STATEMENT, passages)
    system_file = (prompts_module.PROMPTS_DIR / "fact_check_v1" / "system.txt").read_text(
        encoding="utf-8")
    assert request.system == system_file
    assert system_file.count(TOOL_SENTENCE) == 1 and system_file.endswith(TOOL_SENTENCE + "\n")

    text = factcheck.copy_prompt_text(experiment, STATEMENT, passages)
    assert text == system_file.replace(TOOL_SENTENCE, PLAIN_SENTENCE) + "\n" + request.user_text
    assert text.replace(PLAIN_SENTENCE, TOOL_SENTENCE) == system_file + "\n" + request.user_text
    original = set(system_file.splitlines()) | set(request.user_text.splitlines())
    assert [line for line in text.splitlines() if line not in original] == [PLAIN_SENTENCE]
    assert text.endswith(
        PLAIN_SENTENCE + "\n\n"
        "EVIDENCE:\n"
        "[1] Aare\nThe Aare is a river.\n\n"
        "[2] Bern > Geography > Rivers\nBern lies on the Aare.\n\n"
        "CLAIM: The Aare flows through Bern.\n"
    )
    assert "StructuredOutput" not in text
    # the prompt set is the one the experiment ran: its hash is still the recorded one
    assert prompts_module.load_prompt_set("fact_check_v1").prompt_hash == \
        retrieved_run_record()["prompt_hash"] == experiment.prompt.prompt_hash


def test_copyable_system_needs_the_tool_call_sentence_exactly_once() -> None:
    assert factcheck.copyable_system("Label it. " + TOOL_SENTENCE + "\n") == \
        "Label it. " + PLAIN_SENTENCE + "\n"
    for system, count in (("Label it.", 0), (f"{TOOL_SENTENCE}\n{TOOL_SENTENCE}", 2)):
        with pytest.raises(factcheck.FactCheckError, match=f"{count} times, not once"):
            factcheck.copyable_system(system)


def _system_with(runs: Path, prompts_dir: Path, new: str) -> None:
    """Replace the tool-call sentence in the copied system prompt by ``new`` and record the new
    hash, so that only the copy route's own check refuses the set."""
    system = prompts_dir / "fact_check_v1" / "system.txt"
    system.write_text(system.read_text(encoding="utf-8").replace(TOOL_SENTENCE, new),
                      encoding="utf-8")
    record = json.loads(runs.read_text(encoding="utf-8"))
    _retrieved_original(record)["prompt_hash"] = prompts_module.load_prompt_set(
        "fact_check_v1", prompts_dir).prompt_hash
    runs.write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda runs, d: _system_with(runs, d, "Answer in one word."), "0 times, not once"),
        (lambda runs, d: _system_with(runs, d, f"{TOOL_SENTENCE} {TOOL_SENTENCE}"),
         "2 times, not once"),
        (_change_system, "differs from the one the experiment ran"),
        (_remove_record, "cannot read the experiment's record"),
    ],
    ids=["no tool-call sentence", "two of them", "changed system prompt", "missing record"],
)
def test_a_prompt_that_cannot_be_built_says_why_and_the_endpoint_answers_503(
    monkeypatch: pytest.MonkeyPatch,
    record_copy: tuple[Path, Path],
    change: Callable[[Path, Path], None],
    message: str,
) -> None:
    runs, prompts_dir = record_copy
    change(runs, prompts_dir)
    copy = factcheck.load_copy_prompt(runs, prompts_dir=prompts_dir)
    assert copy.experiment is None and copy.unavailable is not None
    assert message in copy.unavailable
    stub = install_search(monkeypatch, make_hits(5))
    resp = get_prompt(make_client(None, copy_prompt=copy), q=STATEMENT)
    assert resp.status_code == 503
    assert resp.json() == {
        "detail": f"The prompt cannot be copied on this server: {copy.unavailable}"}
    assert stub.searches == [] and stub.databases == []


def test_a_runner_that_cannot_be_imported_leaves_the_copy_prompt_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = factcheck.importlib.import_module

    def no_fcntl(name: str, *args: Any) -> Any:
        if name == "wikilense.llm.runner":
            raise ImportError("No module named 'fcntl'")
        return real(name, *args)

    monkeypatch.setattr(factcheck.importlib, "import_module", no_fcntl)
    copy = factcheck.load_copy_prompt()
    assert copy.experiment is None
    assert "cannot be imported" in str(copy.unavailable) and "fcntl" in str(copy.unavailable)


def test_the_prompt_works_with_the_fact_check_off_and_without_a_key_or_header(
    monkeypatch: pytest.MonkeyPatch, experiment: factcheck.Experiment
) -> None:
    monkeypatch.setattr(api_backend, "load_api_key", _forbidden("load_api_key"))
    hits = make_hits(7)
    stub = install_search(monkeypatch, hits)
    client = make_client(None)  # default settings: the live fact-check is off
    assert client.app.state.fact_checker.mode == factcheck.OFF
    assert client.get("/api/factcheck").json()["enabled"] is False
    resp = get_prompt(client, q=f"  {STATEMENT}  ", k=7)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    passages = factcheck.passages_of(hits, experiment.k)
    assert body["text"] == factcheck.copy_prompt_text(experiment, STATEMENT, passages)
    assert body["statement"] == STATEMENT and body["claim_id"] is None
    assert body["n_passages"] == experiment.k == 5
    assert body["passages"] == [{k: p[k] for k in ("rank", "chunk_id", "title", "section_path")}
                                for p in passages]
    assert body["change_note"] == factcheck.COPY_CHANGE_NOTE
    assert "one change" in body["change_note"] and "tool call" in body["change_note"]
    assert body["figures_note"] == factcheck.copy_figures_note(experiment.model_name)
    assert "measured with Haiku" in body["figures_note"] and "%" not in body["figures_note"]
    assert body["answer_key_note"] is None
    assert "reliability" not in body and "reliability_lines" not in body
    assert body["database"] == OFFLINE_SETTINGS.db_name
    assert body["prompt_id"] == experiment.prompt.prompt_id
    assert body["prompt_hash"] == experiment.prompt.prompt_hash
    assert set(body["timing_ms"]) == {"embed", "sql"}
    assert [s["k"] for s in stub.searches] == [7]
    assert stub.searches[0]["statement_seconds"] == OFFLINE_SETTINGS.statement_seconds


def test_the_prompt_never_returns_a_key_and_makes_no_model_call(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    install_search(monkeypatch, make_hits(5))
    backend = FakeBackend()
    checker = make_checker(backend, experiment, reliability, mode=factcheck.API,
                           secrets=(FAKE_KEY,))
    client = make_client(checker, settings=replace(OFFLINE_SETTINGS, factcheck="api"))
    for params in ({"q": STATEMENT}, {"q": STATEMENT, "k": 3, "strategy": "overfetch"}):
        resp = get_prompt(client, **params)
        assert resp.status_code == 200, resp.text
        assert FAKE_KEY not in resp.text and "sk-ant-" not in resp.text
    assert backend.requests == []


def test_a_picked_claim_puts_its_own_text_after_claim_and_names_the_answer_key(
    monkeypatch: pytest.MonkeyPatch, experiment: factcheck.Experiment
) -> None:
    hits = make_hits(5)
    install_search(monkeypatch, hits)
    install_claim(monkeypatch)
    resp = get_prompt(make_client(None), q=CLAIM_TEXT, claim_id=CLAIM_ID)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["statement"] == CLAIM_TEXT and body["claim_id"] == CLAIM_ID
    assert body["text"] == factcheck.copy_prompt_text(
        experiment, CLAIM_TEXT, factcheck.passages_of(hits, experiment.k))
    assert body["text"].endswith(f"\n\nCLAIM: {CLAIM_TEXT}\n")
    assert body["answer_key_note"] == factcheck.COPY_ANSWER_KEY_NOTE
    assert "answer key is on this page" in body["answer_key_note"]


def test_the_prompt_takes_the_fact_check_parameters_and_is_in_the_api_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert webmod.PROMPT_PARAMETER_NAMES == webmod.FACTCHECK_PARAMETER_NAMES
    assert "claim_id" in webmod.PROMPT_PARAMETER_NAMES
    stub = install_search(monkeypatch, make_hits(5))
    client = make_client(None)
    for extra in ({"repeat": 2}, {"sentences": 1}, {"statment": "x"}):
        resp = get_prompt(client, q=STATEMENT, **extra)
        assert resp.status_code == 400
        assert next(iter(extra)) in resp.json()["detail"]
    assert stub.searches == []
    paths = client.get(webmod.OPENAPI_URL).json()["paths"]
    listed = [p["name"] for p in paths["/api/prompt"]["get"]["parameters"]]
    assert sorted(listed) == sorted(webmod.PROMPT_PARAMETER_NAMES)
    assert "post" not in paths["/api/prompt"]
    for path, method in (("/api/search", "get"), ("/api/prompt", "get"),
                         ("/api/factcheck", "post")):
        assert "max_statement_time" in paths[path][method]["responses"]["504"]["description"]


def test_a_search_without_hits_gives_no_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    install_search(monkeypatch, [])
    resp = get_prompt(make_client(None), q=STATEMENT)
    assert resp.status_code == 422 and "no passages" in resp.json()["detail"]


@pytest.mark.parametrize("route", ["prompt", "factcheck"])
def test_a_search_stopped_at_the_time_limit_is_504_on_the_routes_that_run_it_again(
    monkeypatch: pytest.MonkeyPatch,
    experiment: factcheck.Experiment,
    reliability: factcheck.Reliability,
    route: str,
) -> None:
    install_search(monkeypatch, make_hits(5))

    def stopped(conn: Any, qvec: Any, **kwargs: Any) -> list[Hit]:
        raise pymysql.err.OperationalError(
            webmod.search.ER_STATEMENT_TIMEOUT,
            "Query execution was interrupted (max_statement_time exceeded)",
        )

    monkeypatch.setattr(webmod.search, "search", stopped)
    backend = FakeBackend()
    client = make_client(make_checker(backend, experiment, reliability))
    resp = get_prompt(client, q=STATEMENT) if route == "prompt" else post(client, q=STATEMENT)
    assert resp.status_code == 504, resp.text
    assert resp.json() == {
        "detail": webmod.search.statement_stopped_message(OFFLINE_SETTINGS.statement_seconds)}
    assert backend.requests == []


def test_the_page_offers_the_copy_box_for_free_text_and_picked_claims_whatever_the_setting(
) -> None:
    page = webmod.render_index(OFFLINE_SETTINGS)
    assert '<section id="cp-panel" hidden>' in page and "<summary>Copy the prompt</summary>" in page
    assert '<textarea id="cp-text" readonly' in page
    update = script_function("updateCopy")
    for word in ("picked", "claim_id", "fc."):
        assert word not in update  # shown for both kinds of search, whatever the fact-check
    assert "cpPanel" not in script_function("updateFactcheck")
    load = script_function("loadCopy")
    deleted = re.search(
        r"\[([^\]]*)\]\.forEach\(function \(name\) \{ params\.delete\(name\); \}\)", load
    )
    assert deleted is not None
    assert set(re.findall(r"'(\w+)'", deleted.group(1))) == {"sentences", "repeat"}
    assert "getJSON('/api/prompt?' + params.toString())" in load
    assert "if (seq === cpSeq && lastBody !== null) renderCopy(body);" in load
    assert "cpSeq++" in script_function("resetCopy")
    assert page.count("/api/prompt?") == 1  # only when the box is opened
    render = script_function("renderCopy")
    assert "cpText.value = b.text;" in render
    for note in ("b.change_note", "b.figures_note", "b.answer_key_note"):
        assert note in render
    assert "navigator.clipboard.writeText(cpText.value)" in page
    assert "cpText.select();" in script_function("selectPrompt")
    for site in ("chatgpt", "openai", "claude.ai", "gemini"):
        assert site not in page.lower()
