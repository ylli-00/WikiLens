"""The subscription backend against the fake ``claude`` binary: argv, child environment,
classification, served-model checks, parsing, retries and stop signals. No network."""

from __future__ import annotations

import json
import threading

import pytest

from wikilense.llm.backend import CallRequest
from wikilense.llm.cli_backend import (
    ClaudeCliBackend,
    CliBackendError,
    child_env,
    classify,
    overage_signal,
    served_names,
    utilization_of,
)
from wikilense.llm.models import MODEL_IDS, ModelError, resolve_models
from wikilense.llm.prompts import LABELS, load_prompt_set
from wikilense.llm.testing.fixtures import make_fake_binary

PROMPT = load_prompt_set("generic_probe_v0")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A backend on the fake binary, with a polluted parent environment."""
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    for name, value in {"CLAUDE_EFFORT": "xhigh", "CLAUDECODE": "1",
                        "ANTHROPIC_API_KEY": "sk-ant-should-not-be-used", "AI_AGENT_X": "1",
                        "FORCE_PROMPT_CACHING_5M": "1", "DISABLE_PROMPT_CACHING_SONNET": "1",
                        "MAX_THINKING_TOKENS": "9", "CLAUDE_BIN": "/x"}.items():
        monkeypatch.setenv(name, value)
    backend = ClaudeCliBackend(make_fake_binary(tmp_path / "bin"), tmp_path / "cwd", LABELS,
                               timeout_s=5, max_attempts=3, backoff_base_s=0.01)

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    backend.calls = calls
    return backend


def request(text, model="claude-sonnet-5-5"):
    return CallRequest(model_id=model, effort="low", system=PROMPT.system,
                       user_text=f"EVIDENCE: (none given)\n\nCLAIM: {text}\n",
                       schema=PROMPT.schema, schema_json=PROMPT.schema_json,
                       label_path=PROMPT.label_path)


def test_model_table_maps_option_names_to_full_ids_and_refuses_aliases():
    assert resolve_models("sonnet,opus,haiku") == MODEL_IDS
    assert MODEL_IDS == {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5",
                         "haiku": "claude-haiku-5-5"}
    with pytest.raises(ModelError):
        resolve_models("claude-sonnet-5-5")       # option names only
    with pytest.raises(ModelError):
        resolve_models("gpt")


def test_child_env_strips_then_adds_the_forced_variables():
    env = child_env({"CLAUDE_EFFORT": "xhigh", "CLAUDECODE": "1", "ANTHROPIC_API_KEY": "k",
                     "AI_AGENT_FOO": "1", "FORCE_PROMPT_CACHING_5M": "1",
                     "DISABLE_PROMPT_CACHING_OPUS": "1", "HOME": "/h", "PATH": "/p"})
    assert env == {"HOME": "/h", "PATH": "/p", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                   "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1"}


@pytest.mark.parametrize(("text", "rate", "kind"), [
    ("You've hit your 5-hour limit · resets 6am (Europe/Berlin)", None, "exhausted"),
    ("You’ve hit your Sonnet limit", None, "exhausted"),
    ("You've hit your usage credit limit", None, "exhausted"),
    ("You've hit your session limit · resets 3:20am", None, "exhausted"),
    ("API Error: 429", {"status": "rejected"}, "exhausted"),
    ("Server is temporarily limiting requests (not your usage limit)", None, "transient"),
    ("Claude's safeguards flagged this message (https://www.anthropic.com/legal/aup)", None,
     "refusal"),
    ("API Error: 529 overloaded", None, "transient"),
    ("Not logged in · Please run /login", None, "auth"),
    ("something else", None, ""),
])
def test_classify_follows_the_kit_order_with_the_2_1_322_wording(text, rate, kind):
    assert classify(text, rate) == kind


def test_utilization_is_a_fraction_and_missing_is_none():
    info = {"unifiedWindows": {"five_hour": {"utilization": 1.04}, "seven_day": {"status": "ok"}}}
    assert utilization_of(info) == {"five_hour": 1.04, "seven_day": None}


def test_overage_signals():
    assert overage_signal({"isUsingOverage": True})
    assert overage_signal({"overageInUse": True})
    assert overage_signal({"rateLimitType": "overage"})
    assert overage_signal({"rateLimitGraceActive": True, "overageStatus": "allowed_warning"})
    assert not overage_signal({"isUsingOverage": True,
                               "rateLimitType": "seven_day_overage_included"})
    assert not overage_signal({"status": "allowed", "isUsingOverage": False})


def test_served_model_is_exact_and_haiku_allows_no_suffix():
    usage = {"claude-opus-5-5[1m]": {}, "claude-opus-5": {}}
    assert served_names(usage, "claude-opus-5-5") == ["claude-opus-5-5[1m]"]
    assert served_names({"claude-opus-5": {}}, "claude-opus-5-5") == []
    assert served_names({"claude-haiku-5-5[1m]": {}}, "claude-haiku-5-5") == []
    assert served_names({"claude-haiku-5-5": {}}, "claude-haiku-5-5") == ["claude-haiku-5-5"]


def test_success_argv_stdin_env_and_fields(fake):
    res = fake.call(request("LABEL_REFUTES Paris is in Spain."), threading.Event())
    assert res.status == "ok" and res.final and res.label == "REFUTES"
    assert res.parse_status == "ok" and res.served_models == ["claude-sonnet-5-5"]
    assert res.cli["list_price_usd_not_billed"] == 0.01 and res.cli["num_turns"] == 2
    assert res.cli["model_usage"]["claude-sonnet-5-5"]["cacheReadInputTokens"] == 1200
    assert res.computed_usage["prompt_tokens"] == 1503
    assert res.window_utilization == {"five_hour": 0.23, "seven_day": 0.5}
    assert res.cli["last_rate_limit_event"]["type"] == "rate_limit_event"
    (call,) = fake.calls()
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5"
    assert argv[argv.index("--effort") + 1] == "low"
    assert argv[argv.index("--tools") + 1] == "none"
    assert "--json-schema" in argv and "--safe-mode" in argv and "--no-session-persistence" in argv
    assert not any("Paris" in a for a in argv), "the user turn must go on stdin, not argv"
    assert "Paris is in Spain." in call["stdin"]
    assert call["env"] == {k: None for k in call["env"]} | {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1"}
    assert call["cwd"] == str(fake.cwd)


def test_window_with_the_2_1_322_wording_is_not_retried(fake, tmp_path, monkeypatch):
    flag = tmp_path / "exhausted"
    flag.write_text("x")
    monkeypatch.setenv("FAKE_EXHAUST_FLAG", str(flag))
    res = fake.call(request("TRIGGER_EXHAUST"), threading.Event())
    assert res.status == "window" and not res.final and res.stop_signal.startswith("window")
    assert len(fake.calls()) == 1


def test_rejected_status_alone_counts_as_a_spent_window(fake):
    res = fake.call(request("TRIGGER_REJECTED_ONLY"), threading.Event())
    assert res.status == "window" and len(fake.calls()) == 1


def test_refusal_is_final_and_not_retried(fake):
    res = fake.call(request("TRIGGER_REFUSE"), threading.Event())
    assert res.status == "refusal" and res.final and res.label is None
    assert len(fake.calls()) == 1


def test_transient_error_is_retried(fake):
    res = fake.call(request("TRIGGER_TRANSIENT"), threading.Event())
    assert res.status == "ok" and res.attempts == 2


def test_errors_after_all_attempts_are_not_final(fake):
    res = fake.call(request("TRIGGER_ERROR_ALWAYS"), threading.Event())
    assert res.status == "error" and not res.final and res.attempts == 3


def test_wrong_model_is_fatal(fake):
    res = fake.call(request("TRIGGER_WRONGMODEL"), threading.Event())
    assert res.status == "fatal" and res.error_kind == "wrong_model"


def test_fallback_model_answer_is_a_refusal_never_a_label(fake):
    res = fake.call(request("TRIGGER_FALLBACK"), threading.Event())
    assert res.status == "refusal" and res.label is None and res.error_kind == "fallback"


def test_haiku_side_model_is_recorded_for_other_models(fake):
    res = fake.call(request("TRIGGER_SIDE_HAIKU"), threading.Event())
    assert res.status == "ok"
    assert res.side_models == {"claude-haiku-4-5": {"inputTokens": 500, "outputTokens": 10}}
    assert res.computed_usage["side_model_tokens"] == 510


def test_for_a_haiku_request_another_model_with_output_is_a_fallback(fake):
    res = fake.call(request("TRIGGER_SIDE_HAIKU", model="claude-haiku-5-5"), threading.Event())
    assert res.status == "refusal" and res.error_kind == "fallback"


def test_invalid_answers_end_as_parse_failed_after_the_retries(fake):
    res = fake.call(request("TRIGGER_BADSCHEMA"), threading.Event())
    assert res.status == "parse_failed" and res.final and res.parse_status == "failed"
    assert res.label is None and res.attempts == 3


def test_an_invalid_answer_is_asked_again(fake):
    res = fake.call(request("TRIGGER_BADONCE"), threading.Event())
    assert res.status == "ok" and res.attempts == 2


def test_a_loop_takes_the_last_valid_block(fake):
    res = fake.call(request("LABEL_NEI TRIGGER_LOOP"), threading.Event())
    assert res.status == "ok" and res.label == "NOT ENOUGH INFO"
    assert res.raw_response["answer_source"] == "tool_use[0]"


def test_a_timeout_after_a_complete_result_keeps_the_answer(fake):
    fake.timeout_s = 2
    res = fake.call(request("TRIGGER_HANG"), threading.Event())
    assert res.status == "ok" and res.attempts == 1
    assert res.attempt_log[0]["timed_out"] is True


def test_two_timeouts_end_the_item_as_an_error(fake):
    fake.timeout_s = 1
    res = fake.call(request("TRIGGER_SLEEP"), threading.Event())
    assert res.status == "error" and res.error_kind == "timeout" and res.attempts == 2


@pytest.mark.parametrize("trigger", ["TRIGGER_TOOLS", "TRIGGER_APIKEY"])
def test_unexpected_tools_or_an_api_key_source_are_fatal(fake, trigger):
    assert fake.call(request(trigger), threading.Event()).status == "fatal"


@pytest.mark.parametrize("trigger", ["TRIGGER_OVERAGE", "TRIGGER_5M"])
def test_signs_of_paid_extra_usage_set_a_stop_signal(fake, trigger):
    res = fake.call(request(trigger), threading.Event())
    assert res.status == "ok" and res.stop_signal.startswith("overage")


def test_no_model_usage_is_an_error_not_an_answer(fake):
    res = fake.call(request("TRIGGER_NOUSAGE"), threading.Event())
    assert res.status == "error" and res.error_kind == "no_model_usage" and res.label is None


def test_preflight_refuses_api_key_login_and_keeps_no_email(fake, monkeypatch):
    info = fake.preflight()
    assert info["authMethod"] == "claude.ai" and "email" not in info
    monkeypatch.setenv("FAKE_AUTH", "api_key")
    with pytest.raises(CliBackendError, match="bills"):
        fake.preflight()
    monkeypatch.setenv("FAKE_AUTH", "claude.ai")
    monkeypatch.setenv("FAKE_LOGGED_IN", "0")
    with pytest.raises(CliBackendError, match="not logged in"):
        fake.preflight()


def test_aliases_never_reach_the_cli(fake):
    with pytest.raises(ModelError):
        fake.argv(request("x", model="sonnet"))


def test_the_api_key_never_reaches_the_cli_child(fake, tmp_path, monkeypatch):
    """Even with ANTHROPIC_API_KEY in the shell and in a .env next to the run, the claude child
    sees no ANTHROPIC* variable and no key-shaped value: the CLI must never bill the API."""
    secret = "sk-ant-api03-THIS-MUST-NEVER-REACH-THE-CLI"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", secret)
    (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={secret}\n")
    monkeypatch.chdir(tmp_path)
    assert fake.call(request("a claim"), threading.Event()).status == "ok"
    (call,) = fake.calls()
    env = call["env_all"]
    assert not [k for k in env if k.upper().startswith("ANTHROPIC")]
    assert secret not in json.dumps(env) and "sk-ant-" not in json.dumps(env)
    assert call["cwd"] == str(fake.cwd)        # not the folder holding the .env
