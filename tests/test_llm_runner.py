"""The LLM runner end to end on the fake ``claude`` binary and a fake retriever: rows,
manifest, resume, refusal to resume a different run, stops and exit codes. No network, no
database."""

from __future__ import annotations

import json
import os
import signal
import threading

import pytest

from wikilense.llm.cli_backend import ClaudeCliBackend
from wikilense.llm.prompts import LABELS, load_prompt_set
from wikilense.llm.runner import (
    EXIT_BREAKER,
    EXIT_CONFIG,
    EXIT_FATAL,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_OVERAGE,
    EXIT_SIGNAL,
    EXIT_WINDOW,
    RunConfig,
    load_rows,
    run_with_lock,
)
from wikilense.llm.testing.fixtures import FakeRetriever, make_fake_binary, write_claims

PROMPT = load_prompt_set("generic_probe_v0")
CONDITIONS = ["no_retrieval", "retrieved", "gold"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """The fake binary, its log and a polluted parent environment."""
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "fake.log"))
    monkeypatch.setenv("CLAUDE_EFFORT", "xhigh")
    monkeypatch.setenv("CLAUDECODE", "1")
    return tmp_path


def backend_for(tmp_path, **kw):
    return ClaudeCliBackend(make_fake_binary(tmp_path / "bin"), tmp_path / "cwd", LABELS,
                            timeout_s=kw.pop("timeout_s", 5), max_attempts=kw.pop("max_attempts", 2),
                            backoff_base_s=0.01, **kw)


def config(tmp_path, texts, models=None, conditions=None, **kw):
    claims = write_claims(tmp_path / "claims.jsonl", texts)
    return RunConfig(run_dir=tmp_path / "runs" / "r1", claims_path=claims, prompt=PROMPT,
                     models=models or {"haiku": "claude-haiku-5-5"}, effort="low",
                     conditions=conditions or list(CONDITIONS), workers=3, min_free_mb=0, **kw)


def run(cfg, tmp_path, retriever=None, **backend_kw):
    return run_with_lock(cfg, backend_for(tmp_path, **backend_kw), retriever or FakeRetriever(),
                         cfg.run_dir.parent)


def calls(tmp_path):
    log = tmp_path / "fake.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_a_run_writes_one_final_row_per_item_and_a_manifest(env):
    cfg = config(env, ["LABEL_REFUTES a", "LABEL_NEI b"],
                 models={"sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"})
    assert run(cfg, env) == EXIT_OK
    rows = load_rows(cfg.run_dir / "results.jsonl")
    assert len(rows) == 2 * 3 * 2 and all(r["final"] and r["status"] == "ok" for r in rows)
    assert {r["label"] for r in rows if r["claim_id"] == 1} == {"REFUTES"}
    row = next(r for r in rows if r["condition"] == "retrieved" and r["claim_id"] == 2)
    assert row["model_requested"] in ("claude-sonnet-5-5", "claude-haiku-5-5")
    assert row["model_served"] == row["model_requested"]
    assert row["effort"] == "low" and row["prompt_id"] == "generic_probe_v0"
    assert row["prompt_hash"] == PROMPT.prompt_hash
    assert [h["chunk_id"] for h in row["retrieved"]] == [201, 202, 203, 204, 205]
    assert row["cli"]["list_price_usd_not_billed"] == 0.01
    assert row["computed_usage"]["prompt_tokens"] == 1503
    assert row["window_utilization"] == {"five_hour": 0.23, "seven_day": 0.5}
    assert (cfg.run_dir / row["raw_stream_file"]).is_file()
    manifest = json.loads((cfg.run_dir / "manifest.json").read_text())
    assert manifest["models"] == cfg.models and manifest["effort"] == "low"
    assert manifest["prompt"]["prompt_hash"] == PROMPT.prompt_hash
    assert manifest["retrieval"]["k"] == 5 and manifest["retrieval"]["database"] == "wikilense"
    assert manifest["retrieval"]["device"] == "cpu" and manifest["retrieval"]["ef_search"] == 100
    assert len(manifest["claims_file"]["sha256"]) == 64
    assert manifest["cli"]["version"] == "9.9.9 (Fake Claude Code)"
    assert manifest["invocations"][0]["exit_code"] == EXIT_OK and manifest["ended_at"]
    # the evidence the model saw: chunks with title and section path, then the claim
    stdin = next(c["stdin"] for c in calls(env) if "Chunk 1 of claim 2" in c["stdin"])
    assert "[1] Page 2 > History\nChunk 1 of claim 2." in stdin and "[2] Page 2\n" in stdin
    assert stdin.rstrip().endswith("CLAIM: LABEL_NEI b")
    gold = next(c["stdin"] for c in calls(env) if "Gold sentence of claim 1" in c["stdin"])
    assert "[1] Page 1\nGold sentence of claim 1." in gold


def test_resume_skips_final_keys_and_asks_errors_again(env):
    cfg = config(env, ["ok claim", "TRIGGER_ERROR_ALWAYS"], conditions=["no_retrieval"])
    assert run(cfg, env) == EXIT_INCOMPLETE
    first = len(calls(env))
    assert first == 1 + 2                              # 1 ok + 2 attempts of the error
    assert run(cfg, env) == EXIT_INCOMPLETE
    assert len(calls(env)) == first + 2                # only the error item is asked again
    rows = load_rows(cfg.run_dir / "results.jsonl")
    assert [r["status"] for r in rows].count("ok") == 1
    assert len(json.loads((cfg.run_dir / "manifest.json").read_text())["invocations"]) == 2


def test_a_parse_failure_and_a_refusal_are_final(env):
    cfg = config(env, ["TRIGGER_BADSCHEMA", "TRIGGER_REFUSE"], conditions=["gold"])
    assert run(cfg, env) == EXIT_OK
    n = len(calls(env))
    assert run(cfg, env) == EXIT_OK and len(calls(env)) == n
    statuses = {r["claim_id"]: (r["status"], r["parse_status"], r["label"])
                for r in load_rows(cfg.run_dir / "results.jsonl")}
    assert statuses == {1: ("parse_failed", "failed", None), 2: ("refusal", None, None)}


@pytest.mark.parametrize(("change", "value"), [
    ("models", {"sonnet": "claude-sonnet-5-5"}), ("effort", "medium"),
])
def test_resume_into_a_different_run_is_refused(env, change, value):
    cfg = config(env, ["a"], conditions=["no_retrieval"])
    assert run(cfg, env) == EXIT_OK
    setattr(cfg, change, value)
    assert run(cfg, env) == EXIT_CONFIG


def test_resume_with_another_database_k_or_claims_file_is_refused(env):
    cfg = config(env, ["a"], conditions=["no_retrieval"])
    assert run(cfg, env) == EXIT_OK
    assert run(cfg, env, retriever=FakeRetriever(database="wikilense_s1")) == EXIT_CONFIG
    assert run(cfg, env, retriever=FakeRetriever(k=10)) == EXIT_CONFIG
    write_claims(cfg.claims_path, ["a", "b"])
    assert run(cfg, env) == EXIT_CONFIG


def test_a_spent_window_stops_the_run_and_the_item_is_asked_again(env, monkeypatch):
    flag = env / "exhausted"
    flag.write_text("x")
    monkeypatch.setenv("FAKE_EXHAUST_FLAG", str(flag))
    cfg = config(env, ["TRIGGER_EXHAUST", "b", "c", "d"], conditions=["no_retrieval"])
    cfg.workers = 1
    assert run(cfg, env) == EXIT_WINDOW
    rows = load_rows(cfg.run_dir / "results.jsonl")
    assert [r["status"] for r in rows] == ["window"] and not rows[0]["final"]
    flag.unlink()
    assert run(cfg, env) == EXIT_OK
    assert sum(r["final"] for r in load_rows(cfg.run_dir / "results.jsonl")) == 4


def test_the_five_hour_and_weekly_thresholds_stop_the_run(env, monkeypatch):
    cfg = config(env, ["a", "b", "c"], conditions=["no_retrieval"])
    cfg.workers = 1
    monkeypatch.setenv("FAKE_UTIL5", "0.80")
    assert run(cfg, env) == EXIT_WINDOW
    assert len(load_rows(cfg.run_dir / "results.jsonl")) == 1   # the call is kept, then stop
    monkeypatch.setenv("FAKE_UTIL5", "0.10")
    monkeypatch.setenv("FAKE_UTIL7", "1.04")                    # past the weekly cap
    assert run(cfg, env) == EXIT_WINDOW
    manifest = json.loads((cfg.run_dir / "manifest.json").read_text())
    assert "seven_day" in manifest["invocations"][-1]["stop_reason"]


def test_missing_utilization_never_stops_the_run(env, monkeypatch):
    monkeypatch.setenv("FAKE_NO_UTIL", "1")
    cfg = config(env, ["a", "b"], conditions=["no_retrieval"])
    assert run(cfg, env) == EXIT_OK
    rows = load_rows(cfg.run_dir / "results.jsonl")
    assert all(r["window_utilization"] == {"five_hour": None, "seven_day": None} for r in rows)


@pytest.mark.parametrize(("trigger", "code"), [
    ("TRIGGER_OVERAGE", EXIT_OVERAGE), ("TRIGGER_5M", EXIT_OVERAGE),
    ("TRIGGER_WRONGMODEL", EXIT_FATAL),
])
def test_overage_and_fatal_results_stop_the_run(env, trigger, code):
    cfg = config(env, [trigger, "b", "c", "d"], conditions=["no_retrieval"])
    cfg.workers = 1
    assert run(cfg, env) == code
    assert len(calls(env)) == 1


def test_five_errors_in_a_row_trip_the_breaker(env):
    cfg = config(env, ["TRIGGER_ERROR_ALWAYS"] * 7, conditions=["no_retrieval"])
    cfg.workers = 1
    assert run(cfg, env, max_attempts=1) == EXIT_BREAKER
    assert len(load_rows(cfg.run_dir / "results.jsonl")) == 5


def test_a_signal_stops_the_run_and_kills_the_calls_in_flight(env, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "20")
    cfg = config(env, ["a", "b"], conditions=["no_retrieval"])
    timer = threading.Timer(1.5, os.kill, (os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        assert run(cfg, env, timeout_s=60) == EXIT_SIGNAL
    finally:
        timer.cancel()
    rows = load_rows(cfg.run_dir / "results.jsonl")
    assert all(not r["final"] for r in rows)


def test_one_runner_at_a_time(env):
    import fcntl

    cfg = config(env, ["a"], conditions=["no_retrieval"])
    cfg.run_dir.parent.mkdir(parents=True)
    with (cfg.run_dir.parent / ".llm_runner.lock").open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(cfg, env) == EXIT_CONFIG
    assert calls(env) == []


def test_claims_3000_needs_its_hash(env):
    cfg = config(env, ["a"], conditions=["no_retrieval"])
    target = cfg.claims_path.with_name("claims_3000.jsonl")
    cfg.claims_path.rename(target)
    cfg.claims_path = target
    assert run(cfg, env) == EXIT_CONFIG
    cfg.claims_sha256 = "0" * 64
    assert run(cfg, env) == EXIT_CONFIG
    assert calls(env) == []


def test_a_dry_run_makes_no_call_and_writes_nothing(env, capsys):
    cfg = config(env, ["a", "b"], dry_run=True)
    assert run(cfg, env) == EXIT_OK
    assert calls(env) == [] and not cfg.run_dir.exists()
    out = capsys.readouterr().out
    assert "<system prompt:" in out and "CLAIM: a" in out and "[1] Page 1 > History" in out
