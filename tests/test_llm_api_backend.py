"""The API backend against a fake Messages API (the real ``anthropic`` SDK over
``httpx.MockTransport``): request shape, key handling, money guards, errors, the backend switch.
No network, no real key, nothing billed. Skipped when the optional SDK is not installed."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from wikilense.llm import api_backend as ab
from wikilense.llm.api_backend import AnthropicApiBackend, ApiBackendError
from wikilense.llm.backend import CallRequest
from wikilense.llm.cli_backend import ClaudeCliBackend
from wikilense.llm.prompts import LABELS, load_prompt_set
from wikilense.llm.runner import (
    EXIT_CONFIG,
    EXIT_FATAL,
    EXIT_MONEY,
    EXIT_OK,
    RunConfig,
    group_items,
    load_rows,
    main,
    plan_items,
    request_parity,
    run_with_lock,
)
from wikilense.llm.testing.fixtures import (
    FakeRetriever,
    copy_prompt,
    make_fake_binary,
    write_claims,
)

try:
    import anthropic
    import httpx  # noqa: F401  (the fake API's transport)

    from wikilense.llm.testing.fake_api import SONNET_COST, FakeAPI, client_factory
except ImportError:
    anthropic = None

# Skipped test by test rather than as a module, so the suite collects the same tests with or
# without the optional SDK (requirements-llm.txt) and the stated number of tests stays true.
pytestmark = pytest.mark.skipif(anthropic is None,
                                reason="the optional anthropic package is not installed")

PROMPT = load_prompt_set("generic_probe_v0")
KEY = "sk-ant-test-0123456789abcdefSECRETVALUE"
SONNET = {"sonnet": "claude-sonnet-5-5"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A fake API, a .env holding the test key, and no ANTHROPIC* variable in the shell.

    The database password is set in the environment, so the runner's settings load in a checkout
    that has no ``.env`` of its own.
    """
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WIKILENSE_DB_PASSWORD", "x")
    env_file = tmp_path / ".env"
    env_file.write_text(f"WIKILENSE_DB_PASSWORD=x\nANTHROPIC_API_KEY={KEY}\n")
    fake = FakeAPI()
    fake.env_file = env_file
    return fake


def backend(fake, max_usd=5.0, **kw):
    return AnthropicApiBackend(fake.env_file, LABELS, max_usd=max_usd, backoff_base_s=0.01,
                               client_factory=client_factory(fake), **kw)


def config(tmp_path, texts, models=None, conditions=None, run="r1", prompt=PROMPT, **kw):
    claims = write_claims(tmp_path / "claims.jsonl", texts)
    return RunConfig(run_dir=tmp_path / "runs" / run, claims_path=claims, prompt=prompt,
                     models=models or dict(SONNET), effort="low",
                     conditions=conditions or ["no_retrieval", "retrieved", "gold"],
                     workers=3, min_free_mb=0, **kw)


def run(cfg, be):
    return run_with_lock(cfg, be, FakeRetriever(), cfg.run_dir.parent)


def files_text(folder: Path) -> str:
    return "\n".join(p.read_text(errors="replace") for p in folder.rglob("*") if p.is_file())


# ------------------------------------------------------------------------------------------
def test_rows_request_shape_cost_and_resume(api, tmp_path):
    cfg = config(tmp_path, ["LABEL_REFUTES a", "b"])
    assert run(cfg, backend(api)) == EXIT_OK
    rows = load_rows(cfg.run_dir / "results.jsonl")
    assert len(rows) == 6 and all(r["status"] == "ok" and r["final"] for r in rows)
    assert {r["label"] for r in rows if r["claim_id"] == 1} == {"REFUTES"}
    row = rows[0]
    assert row["backend"] == "anthropic_api" and row["model_served"] == "claude-sonnet-5-5"
    assert row["cost_usd"] == pytest.approx(SONNET_COST)
    assert row["api"]["usage"]["cache_read_input_tokens"] == 1000
    assert row["api"]["request_id"] == "req_test123" and row["cli"] is None
    assert row["computed_usage"]["prompt_tokens"] == 12 + 1000 + 300
    assert row["window_utilization"] is None
    body = api.calls[0]
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 4096
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["text"] == PROMPT.system
    assert body["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": PROMPT.schema}
    for forbidden in ("temperature", "top_p", "top_k", "tools", "tool_choice", "fallbacks"):
        assert forbidden not in body
    manifest = json.loads((cfg.run_dir / "manifest.json").read_text())
    assert manifest["backend"] == "anthropic_api"
    assert manifest["backend_info"]["version"] == f"anthropic {anthropic.__version__}"
    assert "price_source" in manifest["backend_info"]
    assert manifest["auth"]["models_api"] == {"claude-sonnet-5-5": "claude-sonnet-5-5"}
    n = len(api.calls)
    assert run(cfg, backend(api)) == EXIT_OK and len(api.calls) == n       # resumed, nothing new


def test_the_key_comes_from_env_file_only_and_is_never_written(api, tmp_path, monkeypatch):
    cfg = config(tmp_path, ["a"], conditions=["no_retrieval"])
    assert run(cfg, backend(api)) == EXIT_OK
    assert set(api.keys) == {KEY}                       # passed to the SDK explicitly
    assert KEY not in files_text(cfg.run_dir)           # rows, raw files, manifest
    assert "ANTHROPIC_API_KEY" not in __import__("os").environ
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-shell")    # ignored by the backend
    api.env_file.write_text("WIKILENSE_DB_PASSWORD=x\n")
    with pytest.raises(ApiBackendError, match="missing"):
        ab.load_api_key(api.env_file)


@pytest.mark.parametrize("value", ["", "sk-ant-REPLACE_ME", "<the key>", "not-a-key"])
def test_bad_key_values_are_refused_without_showing_them(tmp_path, value):
    env_file = tmp_path / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={value}\n")
    with pytest.raises(ApiBackendError) as info:
        ab.load_api_key(env_file)
    assert value == "" or value not in str(info.value)


def test_refused_environment_variables_stop_before_any_call(api, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    cfg = config(tmp_path, ["a"], conditions=["no_retrieval"])
    assert run(cfg, backend(api)) == EXIT_FATAL and api.calls == []


def test_max_usd_is_required_for_the_api_backend(api, tmp_path):
    with pytest.raises(ApiBackendError, match="max-usd"):
        AnthropicApiBackend(api.env_file, LABELS, max_usd=None)
    claims = write_claims(tmp_path / "claims.jsonl", ["a"])
    argv = ["--backend", "api", "--runs-root", str(tmp_path / "runs"), "--claims", str(claims),
            "--prompt", "generic_probe_v0", "--models", "sonnet", "--effort", "low",
            "--env-file", str(api.env_file)]
    assert main(argv) == EXIT_CONFIG
    assert main([*argv, "--dry-run", "--max-usd", "1"]) == EXIT_OK and api.calls == []


def test_the_cap_counts_earlier_rows_of_the_same_results_file(api, tmp_path):
    cfg = config(tmp_path, ["a", "b", "c"], conditions=["no_retrieval"])
    cfg.workers = 1
    assert run(cfg, backend(api, max_usd=SONNET_COST * 1.5)) == EXIT_MONEY
    assert len(api.calls) == 2                       # under the cap before each, then over
    assert run(cfg, backend(api, max_usd=SONNET_COST * 1.5)) == EXIT_MONEY
    assert len(api.calls) == 2                       # earlier spend counted: nothing sent


@pytest.mark.parametrize(("trigger", "code"), [("TRIGGER_CREDIT", EXIT_MONEY),
                                               ("TRIGGER_SPENDCAP", EXIT_MONEY),
                                               ("TRIGGER_AUTH", EXIT_FATAL)])
def test_account_limits_are_money_stops_and_a_bad_key_is_fatal(api, tmp_path, trigger, code):
    cfg = config(tmp_path, [trigger, "b", "c"], conditions=["no_retrieval"])
    cfg.workers = 1
    assert run(cfg, backend(api)) == code
    assert len(api.calls) == 1


def test_preflight_catches_a_bad_key_or_an_unknown_model_for_free(api, tmp_path):
    api.auth_fails = True
    cfg = config(tmp_path, ["a"], conditions=["no_retrieval"])
    assert run(cfg, backend(api)) == EXIT_FATAL and api.calls == []
    api.auth_fails = False
    api.unknown_models.add("claude-sonnet-5-5")
    assert run(cfg, backend(api)) == EXIT_FATAL and api.calls == []


def call(fake, text, **kw):
    be = backend(fake, **kw)
    be.preflight(["claude-sonnet-5-5"])
    req = CallRequest(model_id="claude-sonnet-5-5", effort="low", system=PROMPT.system,
                      user_text=f"CLAIM: {text}", schema=PROMPT.schema,
                      schema_json=PROMPT.schema_json, label_path=PROMPT.label_path)
    return be.call(req, threading.Event())


def test_a_refusal_is_final_with_its_category(api):
    res = call(api, "TRIGGER_REFUSE")
    assert res.status == "refusal" and res.final and "cyber" in res.error and res.label is None
    assert res.api["stop_details"]["category"] == "cyber"


def test_another_model_answering_is_a_refusal_never_a_label(api):
    res = call(api, "TRIGGER_OTHERMODEL")
    assert res.status == "refusal" and res.error_kind == "other_model" and res.label is None


def test_truncation_is_an_error_asked_again_later(api):
    res = call(api, "TRIGGER_TRUNC")
    assert res.status == "error" and res.error_kind == "max_tokens" and not res.final
    assert len(api.calls) == 1


def test_invalid_answers_end_as_parse_failed_after_the_retries(api):
    res = call(api, "TRIGGER_BADSCHEMA", max_attempts=3)
    assert res.status == "parse_failed" and res.parse_status == "failed" and res.attempts == 3
    assert res.cost_usd == pytest.approx(3 * SONNET_COST)        # every billed attempt counted


def test_an_invalid_answer_is_asked_again(api):
    res = call(api, "TRIGGER_BADONCE")
    assert res.status == "ok" and res.attempts == 2


def test_a_429_pauses_and_retries_without_using_an_attempt(api):
    res = call(api, "TRIGGER_RATE429", max_attempts=1)
    assert res.status == "ok" and res.attempts == 1
    assert [e["outcome"] for e in res.attempt_log] == ["429", "ok"]


def test_overloaded_backs_off_and_retries(api):
    res = call(api, "TRIGGER_OVERLOAD")
    assert res.status == "ok" and res.attempts == 2


def test_a_bad_request_is_an_error_not_retried(api):
    res = call(api, "TRIGGER_BADREQ")
    assert res.status == "error" and res.error_kind == "bad_request" and len(api.calls) == 1


def test_the_time_limit_cuts_the_stream_and_bills_it(api):
    res = call(api, "TRIGGER_SLOW", timeout_s=1.0)
    assert res.status == "error" and res.error_kind == "time_limit"
    assert res.cost_usd and res.cost_usd > 0


def test_the_cache_break_prefix_is_marked_and_the_text_is_unchanged(api, tmp_path):
    folder = copy_prompt(tmp_path / "prompts", "cached_v1")
    (folder / "user_gold.txt").write_text("EVIDENCE:\n{{evidence}}\n<<<CACHE_BREAK>>>\n"
                                          "CLAIM: {{claim}}\n")
    prompt = load_prompt_set("cached_v1", tmp_path / "prompts")
    cfg = config(tmp_path, ["a"], conditions=["gold"], prompt=prompt)
    assert run(cfg, backend(api)) == EXIT_OK
    content = api.calls[0]["messages"][0]["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[0]["text"].endswith("Gold sentence of claim 1.\n")
    assert content[1]["text"] == "CLAIM: a\n"
    assert "<<<CACHE_BREAK>>>" not in json.dumps(api.calls[0])


def test_group_by_keeps_items_sharing_a_field_together():
    claims = [{"id": i, "claim": "c", "label": "SUPPORTS", "doc": i % 2} for i in range(1, 5)]
    items = plan_items(claims, ["gold"], dict(SONNET), "low", "h")
    groups = group_items(items, "doc")
    assert [[i.claim_id for i in g] for g in groups] == [[1, 3], [2, 4]]
    assert [[i.claim_id for i in g] for g in group_items(items, None)] == [[1], [2], [3], [4]]


def test_cache_off_and_1h(api, tmp_path):
    assert backend(api, cache_ttl="off").request_kwargs(_req())["system"][0].get(
        "cache_control") is None
    assert backend(api, cache_ttl="1h").request_kwargs(_req())["system"][0]["cache_control"] == {
        "type": "ephemeral", "ttl": "1h"}


def _req():
    return CallRequest(model_id="claude-haiku-5-5", effort="low", system="s", user_text="u",
                       schema=PROMPT.schema, schema_json=PROMPT.schema_json,
                       label_path=PROMPT.label_path)


def test_the_api_request_matches_the_subscription_request(api, tmp_path, capsys):
    """Dry run, no key needed: for every condition the API sends the same model, effort,
    system text, user text and schema as the subscription request."""
    api.env_file.unlink()
    cfg = config(tmp_path, ["a", "b"], models={"haiku": "claude-haiku-5-5"}, dry_run=True)
    assert run(cfg, backend(api)) == EXIT_OK and api.calls == []
    out = capsys.readouterr().out
    assert out.count("parity with the subscription request") == 3
    assert "DIFFERS" not in out and out.count("MATCH") == 3
    assert request_parity(_req(), backend(api).request_kwargs(_req()) | {"model": "x"})
    assert not request_parity(_req(), backend(api).request_kwargs(_req()))


def test_switching_the_backend_keeps_keys_and_texts_and_refuses_mixed_folders(api, tmp_path,
                                                                             monkeypatch):
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "fake.log"))
    cfg_api = config(tmp_path, ["a", "b"], run="api_run")
    assert run(cfg_api, backend(api)) == EXIT_OK
    cli = ClaudeCliBackend(make_fake_binary(tmp_path / "bin"), tmp_path / "cwd", LABELS,
                           timeout_s=5, max_attempts=2, backoff_base_s=0.01)
    cfg_sub = config(tmp_path, ["a", "b"], run="sub_run")
    assert run(cfg_sub, cli) == EXIT_OK

    def fingerprint(folder):
        return sorted((r["key"], r["user_text_sha256"], r["prompt_hash"])
                      for r in load_rows(folder / "results.jsonl"))
    assert fingerprint(cfg_api.run_dir) == fingerprint(cfg_sub.run_dir)
    assert {r["backend"] for r in load_rows(cfg_sub.run_dir / "results.jsonl")} == {
        "claude_cli_subscription"}
    cfg_sub.run_dir = cfg_api.run_dir                 # the CLI into the API's folder
    assert run(cfg_sub, cli) == EXIT_CONFIG


def test_prices_and_cost(api):
    usage = {"input_tokens": 1000, "cache_read_input_tokens": 2000,
             "cache_creation_input_tokens": 300,
             "cache_creation": {"ephemeral_5m_input_tokens": 100,
                                "ephemeral_1h_input_tokens": 200},
             "output_tokens": 50}
    usd, tokens = ab.cost_of(usage, "claude-opus-5-5")
    assert usd == pytest.approx((1000 * 4 + 2000 * 0.20 + 100 * 5 + 200 * 8 + 50 * 20) / 1e6)
    assert tokens["prompt_tokens"] == 3300 and tokens["cache_write_1h"] == 200
    assert ab.cost_of(usage, "claude-haiku-5-5")[0] == pytest.approx(
        (1000 * 0.10 + 2000 * 0.01 + 100 * 0.125 + 200 * 0.20 + 50 * 0.50) / 1e6)
    long = dict(usage, input_tokens=100_001)
    assert ab.cost_of(long, "claude-haiku-5-5")[0] == pytest.approx(
        (100_001 * 0.50 + 2000 * 0.05 + 100 * 0.625 + 200 * 1.00 + 50 * 2.50) / 1e6)
    assert ab.cost_of(dict(usage, inference_geo="us"), "claude-sonnet-5-5")[0] == pytest.approx(
        1.1 * ab.cost_of(usage, "claude-sonnet-5-5")[0])
    with pytest.raises(ApiBackendError, match="no list price"):
        ab.cost_of(usage, "claude-unknown-9")


def test_served_model_is_strict_and_rate_limit_wait():
    assert ab.same_model("claude-sonnet-5-5", "claude-sonnet-5-5")
    assert ab.same_model("claude-sonnet-5-5-20261101", "claude-sonnet-5-5")
    assert not ab.same_model("claude-sonnet-5", "claude-sonnet-5-5")
    assert not ab.same_model("claude-sonnet-5-5x", "claude-sonnet-5-5")
    assert ab.rate_limit_wait({"retry-after": "3"}) == 3.0
    assert ab.rate_limit_wait({"retry-after": "999"}) == 120.0
    assert ab.rate_limit_wait({}) is None
