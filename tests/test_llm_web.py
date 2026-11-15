"""The ``web`` condition: prompt sets and their conditions, argv of every condition, init tools,
the assistant-model rule, the turn cap, the trace of a real stream, the deny list, the audit,
and web runs end to end on the fake binary. No network, no database."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from wikilense.llm import web
from wikilense.llm.backend import CallRequest
from wikilense.llm.cli_backend import BASE_FLAGS, ClaudeCliBackend
from wikilense.llm.prompts import LABELS, PromptError, load_prompt_set
from wikilense.llm.runner import (
    EXIT_CONFIG,
    EXIT_FATAL,
    EXIT_OK,
    NoRetrieval,
    RunConfig,
    load_rows,
    run_with_lock,
)
from wikilense.llm.testing.fixtures import FakeRetriever, make_fake_binary, write_claims

FIXTURE = Path(web.__file__).parent / "testing" / "web_stream_deny_2026-11-15.jsonl"
WEB_PROMPT = load_prompt_set("fact_check_web_v1")
OLD_PROMPT = load_prompt_set("fact_check_v1")
#: The base flags as every call of the Haiku run of 2026-11-12 had them (its manifest).
BASE_FLAGS_2026_11_12 = ["--print", "--output-format", "stream-json", "--verbose", "--safe-mode",
                         "--strict-mcp-config", "--disable-slash-commands", "--setting-sources",
                         "", "--no-session-persistence", "--tools", "none"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "fake.log"))
    return tmp_path


def web_config(**kw) -> web.WebConfig:
    return web.WebConfig(deny=web.load_deny_hosts(), **({"timeout_s": 5.0} | kw))


def backend_for(tmp_path, with_web=True, **kw):
    return ClaudeCliBackend(make_fake_binary(tmp_path / "bin"), tmp_path / "cwd", LABELS,
                            timeout_s=5, max_attempts=kw.pop("max_attempts", 2),
                            backoff_base_s=0.01, web=web_config(**kw) if with_web else None)


def request(prompt=WEB_PROMPT, condition="web", text="CLAIM: x\n") -> CallRequest:
    return CallRequest(model_id="claude-haiku-5-5", effort="high", system=prompt.system,
                       user_text=text, schema=prompt.schema, schema_json=prompt.schema_json,
                       label_path=prompt.label_path, condition=condition)


def calls(tmp_path):
    log = tmp_path / "fake.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


# ---------------------------------------------------------------------------- prompt sets
def test_old_prompt_sets_load_unchanged():
    assert OLD_PROMPT.prompt_hash == \
        "901ae52a9a15c12196a66988b1176a68da54078140d5f4ad78ec44e7c5a1a984"
    assert load_prompt_set("generic_probe_v0").prompt_hash == \
        "599d682c18b8eb1955c54df6c7cd7d2d7bc0916a2b20b2410a56ad8d07b94ea7"
    assert OLD_PROMPT.conditions == ("no_retrieval", "retrieved", "gold")


def test_the_web_prompt_set():
    assert WEB_PROMPT.conditions == ("web",)
    assert WEB_PROMPT.user_templates == {"web": "CLAIM: {{claim}}\n"}
    old_dir, new_dir = OLD_PROMPT.directory, WEB_PROMPT.directory
    assert (new_dir / "schema.json").read_bytes() == (old_dir / "schema.json").read_bytes()
    old = (old_dir / "system.txt").read_text().split("\n")
    new = (new_dir / "system.txt").read_text().split("\n")
    assert len(old) == len(new) and new[-1] == ""                    # same lines, final newline
    assert [i for i, (a, b) in enumerate(zip(old, new)) if a != b] == [2, 8]
    assert new[2].startswith("You can search the web and open pages.")
    assert new[8].endswith("such as a page title). No other text.")


@pytest.mark.parametrize(("users", "error"), [({}, "no user template"),
                                              ({"memory": "u.txt"}, "unknown conditions")])
def test_a_prompt_set_needs_known_conditions(tmp_path, users, error):
    folder = tmp_path / "bad_v1"
    folder.mkdir()
    for name in ("system.txt", "u.txt"):
        (folder / name).write_text("x {{claim}}\n")
    (folder / "schema.json").write_text((OLD_PROMPT.directory / "schema.json").read_text())
    (folder / "prompt.json").write_text(json.dumps({"prompt_id": "bad_v1", "system": "system.txt",
                                                    "user": users, "schema": "schema.json"}))
    with pytest.raises(PromptError, match=error):
        load_prompt_set("bad_v1", tmp_path)


# ---------------------------------------------------------------------------- argv
@pytest.mark.parametrize("condition", ["no_retrieval", "retrieved", "gold"])
@pytest.mark.parametrize("with_web", [False, True])
def test_the_argv_of_the_three_old_conditions_is_unchanged(tmp_path, condition, with_web):
    backend = backend_for(tmp_path, with_web=with_web)
    req = request(OLD_PROMPT, condition)
    assert backend.argv(req) == [backend.binary, *BASE_FLAGS_2026_11_12,
                                 "--model", "claude-haiku-5-5", "--effort", "high",
                                 "--system-prompt", OLD_PROMPT.system,
                                 "--json-schema", OLD_PROMPT.schema_json]


def test_the_argv_of_web(tmp_path):
    backend = backend_for(tmp_path)
    rules = [f"WebFetch(domain:{h})" for h in web.load_deny_hosts().hosts]
    assert backend.argv(request()) == [
        backend.binary, *BASE_FLAGS_2026_11_12[:-2],
        "--tools", "WebSearch,WebFetch",
        "--allowedTools", "WebSearch", "WebFetch(domain:*)",
        "--disallowedTools", "mcp__*", *rules,
        "--permission-prompts", "none", "--max-turns", "20",
        "--model", "claude-haiku-5-5", "--effort", "high",
        "--system-prompt", WEB_PROMPT.system, "--json-schema", WEB_PROMPT.schema_json]
    assert tuple(BASE_FLAGS_2026_11_12) == BASE_FLAGS


def test_a_web_request_needs_the_web_settings(tmp_path):
    with pytest.raises(Exception, match="web settings"):
        backend_for(tmp_path, with_web=False).argv(request())


# ---------------------------------------------------------------------------- deny list
def test_the_deny_list_file():
    deny = web.load_deny_hosts()
    assert len(deny.hosts) == 17 and deny.hosts[:2] == ("fever.ai", "*.fever.ai")
    assert {"githubusercontent.com", "*.githubusercontent.com", "hf.co"} <= set(deny.hosts)
    assert deny.rules()[0] == "WebFetch(domain:fever.ai)"
    assert len(deny.sha256) == 64
    assert web_config().describe()["deny_hosts"] == list(deny.hosts)


@pytest.mark.parametrize("line", ["https://fever.ai/x", "fever.ai:443", "*", "Fever.AI",
                                  "fever.ai\nfever.ai"])
def test_a_bad_deny_list_is_refused(tmp_path, line):
    path = tmp_path / "deny.txt"
    path.write_text(f"# comment\n{line}\n")
    with pytest.raises(web.WebConfigError):
        web.load_deny_hosts(path)


def test_an_empty_deny_list_is_refused(tmp_path):
    (tmp_path / "deny.txt").write_text("# nothing\n\n")
    with pytest.raises(web.WebConfigError, match="no host"):
        web.load_deny_hosts(tmp_path / "deny.txt")


@pytest.mark.parametrize(("pattern", "host", "match"), [
    ("fever.ai", "fever.ai", True), ("fever.ai", "www.fever.ai", False),
    ("*.fever.ai", "fever.ai", False), ("*.fever.ai", "a.b.fever.ai", True),
    ("*.githubusercontent.com", "raw.githubusercontent.com", True),
    ("hf.co", "shf.co", False),
])
def test_host_patterns_match_like_the_cli(pattern, host, match):
    assert web.host_matches(pattern, host) is match


# ---------------------------------------------------------------------------- the real stream
def test_the_trace_of_a_recorded_stream():
    stdout = FIXTURE.read_text()
    result = next(json.loads(line) for line in stdout.splitlines() if '"type": "result"' in line)
    trace = web.web_trace(stdout, result)
    assert [t["tool"] for t in trace] == ["WebFetch", "WebFetch", "WebSearch", "WebSearch"]
    assert trace[0]["url"] == "https://fever.ai/dataset/feverous.html"
    assert trace[0]["denied"] and trace[0]["is_error"]
    assert trace[0]["error"] == "WebFetch denied access to domain:fever.ai."
    assert trace[1]["denied"] and "raw.githubusercontent.com" in trace[1]["error"]
    assert trace[2]["query"].startswith("FEVEROUS dataset number of claims")
    assert not trace[2]["denied"] and trace[2]["result_chars"] > 1000
    assert "https://zenodo.org/records/4911508" in trace[2]["result_urls"]
    assert all(t["model"] == "claude-haiku-5-5" for t in trace)
    counts = web.trace_counts(trace, result["modelUsage"])
    assert counts == {"n_web_search": 2, "n_web_fetch": 2, "n_denied": 2, "web_search_requests": 2}
    messages = web.assistant_messages(stdout)
    assert {m["model"] for m in messages} == {"claude-haiku-5-5"}
    assert sum(m["structured"] for m in messages) == 1
    assert web.foreign_messages(messages, ["claude-haiku-5-5"]) == []


# ---------------------------------------------------------------------------- calls
def test_a_web_call_records_its_trace(env):
    res = backend_for(env).call(request(text="CLAIM: TRIGGER_WEB x\n"), threading.Event())
    assert res.status == "ok" and res.label == "SUPPORTS"
    assert [t["tool"] for t in res.web["trace"]] == ["WebSearch", "WebFetch"]
    assert res.web["trace"][0]["result_urls"] == ["https://en.wikipedia.org/wiki/A",
                                                  "https://www.britannica.com/B"]
    assert res.web["trace"][1]["http_code"] == 200 and res.web["trace"][1]["bytes"] == 182090
    assert (res.web["n_web_search"], res.web["n_web_fetch"], res.web["n_denied"],
            res.web["web_search_requests"]) == (1, 1, 0, 1)


def test_a_denied_fetch_is_counted(env):
    res = backend_for(env).call(request(text="CLAIM: TRIGGER_WEB_DENY x\n"), threading.Event())
    assert res.status == "ok" and res.web["n_denied"] == 1
    assert res.web["trace"][0]["denied"]


@pytest.mark.parametrize("trigger", ["TRIGGER_TOOLS", "TRIGGER_NO_WEBFETCH"])
def test_web_init_tools_must_be_exactly_the_three(env, trigger):
    res = backend_for(env).call(request(text=f"CLAIM: {trigger}\n"), threading.Event())
    assert res.status == "fatal" and res.error_kind == "unexpected_tools"


def test_the_old_init_tools_rule_is_kept(env):
    res = backend_for(env).call(request(OLD_PROMPT, "no_retrieval", "TRIGGER_TOOLS"),
                                threading.Event())
    assert res.status == "fatal" and res.error_kind == "unexpected_tools"
    ok = backend_for(env).call(request(OLD_PROMPT, "gold", "fine"), threading.Event())
    assert ok.status == "ok" and ok.web == {}


def test_an_answer_from_another_model_is_a_refusal(env):
    res = backend_for(env).call(request(text="CLAIM: TRIGGER_WEB TRIGGER_FOREIGN\n"),
                                threading.Event())
    assert res.status == "refusal" and res.label is None and res.error_kind == "fallback"
    assert "claude-sonnet-5-5" in res.error


def test_side_model_output_is_allowed_only_after_a_web_tool(env):
    with_tool = backend_for(env).call(request(text="CLAIM: TRIGGER_WEB TRIGGER_SIDE_HAIKU\n"),
                                      threading.Event())
    assert with_tool.status == "ok" and "claude-haiku-4-5" in with_tool.side_models
    without = backend_for(env).call(request(text="CLAIM: TRIGGER_SIDE_HAIKU\n"), threading.Event())
    assert without.status == "refusal" and without.error_kind == "fallback"


def test_the_turn_cap_is_final_after_one_retry(env):
    res = backend_for(env, max_attempts=4).call(request(text="CLAIM: TRIGGER_MAXTURNS\n"),
                                                threading.Event())
    assert (res.status, res.final, res.parse_status, res.error_kind) == \
        ("parse_failed", True, "failed", "max_turns")
    assert res.attempts == 2 and len(calls(env)) == 2


def test_a_turn_cap_once_then_an_answer(env):
    res = backend_for(env).call(request(text="CLAIM: TRIGGER_MAXTURNS TRIGGER_MAXTURNS_ONCE\n"),
                                threading.Event())
    assert res.status == "ok" and res.attempts == 2


def test_a_web_call_uses_its_own_time_limit(env, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "2")
    res = backend_for(env, timeout_s=0.5, max_attempts=1).call(request(), threading.Event())
    assert res.status == "error" and res.error_kind == "timeout" and res.latency_s < 2


# ---------------------------------------------------------------------------- runs
def web_run(tmp_path, texts, backend=None, **kw):
    claims = write_claims(tmp_path / "claims.jsonl", texts)
    cfg = RunConfig(run_dir=tmp_path / "runs" / "web1", claims_path=claims, prompt=WEB_PROMPT,
                    models={"haiku": "claude-haiku-5-5"}, effort="high",
                    conditions=kw.pop("conditions", ["web"]), workers=2, min_free_mb=0, **kw)
    return cfg, run_with_lock(cfg, backend or backend_for(tmp_path), NoRetrieval(),
                              cfg.run_dir.parent)


def test_a_web_run_writes_web_rows_and_a_web_manifest(env):
    cfg, code = web_run(env, ["TRIGGER_WEB LABEL_REFUTES a", "TRIGGER_WEB_DENY b"])
    assert code == EXIT_OK
    rows = {r["claim_id"]: r for r in load_rows(cfg.run_dir / "results.jsonl")}
    assert rows[1]["label"] == "REFUTES" and rows[1]["n_web_fetch"] == 1
    assert rows[1]["web_trace"][1]["url"] == "https://en.wikipedia.org/wiki/A"
    assert rows[2]["n_denied"] == 1 and rows[2]["assistant_models"] == ["claude-haiku-5-5"]
    assert rows[1]["retrieved"] is None and rows[1]["condition"] == "web"
    manifest = json.loads((cfg.run_dir / "manifest.json").read_text())
    assert manifest["retrieval"]["database"] is None
    assert manifest["web"]["deny_hosts_sha256"] == web.load_deny_hosts().sha256
    assert manifest["web"]["max_turns"] == 20 and manifest["web"]["timeout_s"] == 5.0
    assert "--max-turns" in calls(env)[0]["argv"]


def test_a_condition_the_prompt_set_lacks_is_refused(env):
    _, code = web_run(env, ["a"], conditions=["no_retrieval"])
    assert code == EXIT_CONFIG and calls(env) == []


def test_web_needs_a_backend_with_web_settings(env):
    _, code = web_run(env, ["a"], backend=backend_for(env, with_web=False))
    assert code == EXIT_CONFIG and calls(env) == []


@pytest.mark.parametrize("change", ["deny", "max_turns"])
def test_resume_with_another_deny_list_or_turn_cap_is_refused(env, change):
    cfg, code = web_run(env, ["a"])
    assert code == EXIT_OK
    if change == "deny":
        other = env / "deny.txt"
        other.write_text("fever.ai\n")
        backend = ClaudeCliBackend(make_fake_binary(env / "bin"), env / "cwd", LABELS, timeout_s=5,
                                   web=web.WebConfig(deny=web.load_deny_hosts(other)))
    else:
        backend = backend_for(env, max_turns=10)
    assert run_with_lock(cfg, backend, NoRetrieval(), cfg.run_dir.parent) == EXIT_CONFIG


def test_an_old_condition_run_has_no_web_section(env):
    claims = write_claims(env / "claims.jsonl", ["a"])
    cfg = RunConfig(run_dir=env / "runs" / "old1", claims_path=claims, prompt=OLD_PROMPT,
                    models={"haiku": "claude-haiku-5-5"}, effort="high",
                    conditions=["no_retrieval"], workers=1, min_free_mb=0)
    assert run_with_lock(cfg, backend_for(env), FakeRetriever(), cfg.run_dir.parent) == EXIT_OK
    manifest = json.loads((cfg.run_dir / "manifest.json").read_text())
    assert "web" not in manifest and manifest["retrieval"]["database"] == "wikilense"
    row = load_rows(cfg.run_dir / "results.jsonl")[0]
    assert "web_trace" not in row and "--tools" in calls(env)[0]["argv"]
    assert calls(env)[0]["argv"][calls(env)[0]["argv"].index("--tools") + 1] == "none"


def test_wrong_tools_on_a_web_run_stop_it(env):
    _, code = web_run(env, ["TRIGGER_TOOLS"])
    assert code == EXIT_FATAL


# ---------------------------------------------------------------------------- audit
def audit_one(row, raw=None):
    return web.audit_row({"condition": "web", **row}, web.load_deny_hosts(), raw)


def test_a_clean_row_is_not_flagged():
    row = {"web_trace": [{"tool": "WebSearch", "query": "Hongze county history",
                          "result_urls": ["https://en.wikipedia.org/wiki/Hongze"]},
                         {"tool": "WebFetch", "url": "https://en.wikipedia.org/wiki/Hongze",
                          "prompt": "When was it dissolved?"}]}
    assert audit_one(row, "Hongze County was dissolved in 1950. See https://shf.com/x") == []


@pytest.mark.parametrize(("row", "raw", "reason"), [
    ({"web_trace": [{"tool": "WebSearch", "query": "feverous claim 85380 label"}]}, None,
     "mentions 'feverous'"),
    ({"web_trace": [{"tool": "WebFetch", "url": "https://hf.co/datasets/x"}]}, None,
     "denied host hf.co"),
    ({"web_trace": [{"tool": "WebSearch", "query": "q",
                     "result_urls": ["https://github.com/u/r/blob/main/dev.jsonl"]}]}, None,
     "GitHub data file"),
    ({"web_trace": [{"tool": "WebFetch", "url": "https://example.org/a"}]},
     "Claim 12. Verdict: REFUTES", "'Verdict' next to a label"),
    ({"web_trace": [{"tool": "WebFetch", "url": "https://mirror.example/x",
                     "final_url": "https://zenodo.org/records/4911508"}]}, None, "'zenodo'"),
])
def test_the_audit_flags(row, raw, reason):
    reasons = audit_one(row, raw)
    assert any(reason in r for r in reasons), reasons


def test_the_audit_script_writes_its_report(env):
    import importlib.util

    cfg, _ = web_run(env, ["TRIGGER_WEB a", "TRIGGER_WEB_DENY b"])
    script = Path(web.__file__).parents[2] / "scripts" / "llm_web_audit.py"
    spec = importlib.util.spec_from_file_location("llm_web_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.audit_run(cfg.run_dir)
    assert summary["web_rows"] == 2 and summary["denied_calls"] == 1
    assert summary["fetched_hosts"] == {"en.wikipedia.org": 1, "fever.ai": 1}
    assert [f["claim_id"] for f in summary["flagged"]] == [2]
    assert "flagged rows: 1" in (cfg.run_dir / "WEB_AUDIT.md").read_text()


def test_web_defaults_are_the_owners_choices():
    # owner, 2026-11-15: 200 s per web call (first proposal 900 s); turn cap 20, one retry
    assert web.DEFAULT_TIMEOUT_S == 200.0
    assert web.DEFAULT_MAX_TURNS == 20
    assert web.MAX_TURN_CAP_ATTEMPTS == 2
    assert web.WebConfig(deny=web.load_deny_hosts()).describe()["timeout_s"] == 200.0
