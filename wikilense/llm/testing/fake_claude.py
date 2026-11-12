"""A FAKE ``claude`` binary for the LLM runner's tests. It makes no network call.

Adapted from the kit's ``code/standalone/tests/fake_claude.py``, with the CLI 2.1.322 shapes
(the "5-hour limit" wording, ``rate_limit_info`` with ``unifiedWindows`` and overage fields).
It answers ``--version``, ``auth status --json`` and ``-p`` calls. What a ``-p`` call does is
chosen by TRIGGER_* words in the user turn read from stdin:

  (none)                  success; label SUPPORTS, or the label named LABEL_REFUTES / LABEL_NEI
  TRIGGER_EXHAUST         "You've hit your 5-hour limit" (status rejected) while
                          $FAKE_EXHAUST_FLAG exists, success afterwards
  TRIGGER_REJECTED_ONLY   an unknown error text, but rate_limit_info.status "rejected"
  TRIGGER_REFUSE          a usage-policy refusal (2.1.322 safeguards wording)
  TRIGGER_TRANSIENT       "529 overloaded" on the first call for this stdin, success after
  TRIGGER_ERROR_ALWAYS    "529 overloaded" on every call
  TRIGGER_WRONGMODEL      modelUsage names another model only
  TRIGGER_FALLBACK        the requested model writes 0 output, claude-opus-5 writes the answer
  TRIGGER_SIDE_HAIKU      a claude-haiku-4-5 side model with 10 output tokens
  TRIGGER_SLEEP           sleeps 30 s before answering (timeout test)
  TRIGGER_HANG            prints a complete success, then sleeps 30 s (timeout after a result)
  TRIGGER_BADSCHEMA       label MAYBE (outside the enum) on every call
  TRIGGER_BADONCE         label MAYBE on the first call for this stdin, valid after
  TRIGGER_LOOP            no structured_output; tool blocks [valid, $PARAMETER_NAME wrapper]
  TRIGGER_TOOLS           the init event lists Bash and Read
  TRIGGER_APIKEY          the init event says apiKeySource ANTHROPIC_API_KEY
  TRIGGER_OVERAGE         rate_limit_info.isUsingOverage true
  TRIGGER_5M              a 5-minute cache write
  TRIGGER_NOUSAGE         an empty modelUsage

Environment: FAKE_LOG (required for -p: one JSON line per call), FAKE_UTIL5 / FAKE_UTIL7
(utilization fractions, default 0.23 / 0.50), FAKE_NO_UTIL=1 (no utilization numbers),
FAKE_DELAY (seconds), FAKE_LOGGED_IN, FAKE_AUTH.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ENV_SEEN = ("CLAUDE_EFFORT", "CLAUDECODE", "ANTHROPIC_API_KEY", "AI_AGENT_X",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK",
            "FORCE_PROMPT_CACHING_5M", "DISABLE_PROMPT_CACHING_SONNET", "MAX_THINKING_TOKENS",
            "CLAUDE_BIN")


def first_time(stdin: str, tag: str) -> bool:
    """True on the first call with this stdin and tag (a marker file next to FAKE_LOG)."""
    marker = Path(os.environ["FAKE_LOG"] + f".{tag}." + hashlib.sha256(stdin.encode()).hexdigest()[:12])
    if marker.exists():
        return False
    marker.write_text("x")
    return True


def main() -> int:
    argv = sys.argv[1:]
    if argv == ["--version"]:
        print("9.9.9 (Fake Claude Code)")
        return 0
    if argv[:2] == ["auth", "status"]:
        print(json.dumps({"loggedIn": os.environ.get("FAKE_LOGGED_IN", "1") == "1",
                          "authMethod": os.environ.get("FAKE_AUTH", "claude.ai"),
                          "apiProvider": "firstParty", "subscriptionType": "max",
                          "planTier": "max_5x", "email": "someone@example.org"}))
        return 0
    stdin = sys.stdin.read()
    model = argv[argv.index("--model") + 1] if "--model" in argv else "?"
    record = {
        "argv": ["<sys>" if i > 0 and argv[i - 1] == "--system-prompt" else a
                 for i, a in enumerate(argv)],
        "env": {k: os.environ.get(k) for k in ENV_SEEN},
        "cwd": os.getcwd(), "stdin": stdin, "start": time.time(), "pid": os.getpid(),
    }

    def emit(obj):
        print(json.dumps(obj), flush=True)

    def finish(code):
        record["end"] = time.time()
        with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return code

    time.sleep(float(os.environ.get("FAKE_DELAY", "0")))
    if "TRIGGER_SLEEP" in stdin:
        time.sleep(30)
    tools = ["StructuredOutput", "Bash", "Read"] if "TRIGGER_TOOLS" in stdin else ["StructuredOutput"]
    emit({"type": "system", "subtype": "init", "model": model, "tools": tools,
          "apiKeySource": "ANTHROPIC_API_KEY" if "TRIGGER_APIKEY" in stdin else "none",
          "claude_code_version": "9.9.9"})
    reset5 = int(time.time()) + 3600
    no_util = os.environ.get("FAKE_NO_UTIL") == "1"

    def window(util, reset, status="allowed"):
        w = {"resetsAt": reset, "status": status}
        if not no_util:
            w["utilization"] = util
        return w

    def rate_event(status="allowed", **extra):
        info = {"status": status, "resetsAt": reset5, "rateLimitType": "five_hour",
                "unifiedWindows": {
                    "five_hour": window(float(os.environ.get("FAKE_UTIL5", "0.23")), reset5, status),
                    "seven_day": window(float(os.environ.get("FAKE_UTIL7", "0.50")), reset5 + 86400)}}
        info.update(extra)
        emit({"type": "rate_limit_event", "rate_limit_info": info})

    flag = os.environ.get("FAKE_EXHAUST_FLAG")
    if "TRIGGER_EXHAUST" in stdin and flag and Path(flag).exists():
        rate_event("rejected")
        emit({"type": "result", "subtype": "error", "is_error": True,
              "result": "You've hit your 5-hour limit · resets 6am (Europe/Berlin)"})
        return finish(1)
    if "TRIGGER_REJECTED_ONLY" in stdin:
        rate_event("rejected")
        emit({"type": "result", "subtype": "error", "is_error": True, "result": "API Error: 429"})
        return finish(1)
    if "TRIGGER_REFUSE" in stdin:
        emit({"type": "result", "subtype": "error", "is_error": True, "api_error_status": 400,
              "result": "API Error: Claude's safeguards flagged this message "
                        "(https://www.anthropic.com/legal/aup). This sometimes happens with safe, "
                        "normal conversations."})
        return finish(1)
    if ("TRIGGER_TRANSIENT" in stdin and first_time(stdin, "transient")) or "TRIGGER_ERROR_ALWAYS" in stdin:
        emit({"type": "result", "subtype": "error", "is_error": True,
              "result": "API Error: 529 overloaded"})
        return finish(1)

    usage = {model: {"inputTokens": 3, "outputTokens": 40, "cacheReadInputTokens": 1200,
                     "cacheCreationInputTokens": 300, "thinkingTokens": 0}}
    if "TRIGGER_WRONGMODEL" in stdin:
        usage = {"claude-opus-4-8[1m]": usage[model]}
    if "TRIGGER_FALLBACK" in stdin:
        usage[model]["outputTokens"] = 0
        usage["claude-opus-5"] = {"inputTokens": 5, "outputTokens": 33}
    if "TRIGGER_SIDE_HAIKU" in stdin:
        usage["claude-haiku-4-5"] = {"inputTokens": 500, "outputTokens": 10}
    if "TRIGGER_NOUSAGE" in stdin:
        usage = {}
    label = "SUPPORTS"
    if "LABEL_REFUTES" in stdin:
        label = "REFUTES"
    elif "LABEL_NEI" in stdin:
        label = "NOT ENOUGH INFO"
    if "TRIGGER_BADSCHEMA" in stdin or ("TRIGGER_BADONCE" in stdin and first_time(stdin, "bad")):
        label = "MAYBE"
    answer = {"result": {"label": label, "justification": "fake"}}
    blocks = [{"type": "thinking", "thinking": ""},
              {"type": "tool_use", "name": "StructuredOutput", "input": answer}]
    if "TRIGGER_LOOP" in stdin:
        blocks.append({"type": "tool_use", "name": "StructuredOutput",
                       "input": {"$PARAMETER_NAME": answer}})
    emit({"type": "assistant", "message": {"content": blocks}})
    extra = {"isUsingOverage": True} if "TRIGGER_OVERAGE" in stdin else {}
    rate_event(**extra)
    result = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2,
              "duration_ms": 1234, "duration_api_ms": 1100, "stop_reason": "tool_use",
              "modelUsage": usage, "total_cost_usd": 0.01,
              "usage": {"input_tokens": 3, "cache_creation": {
                  "ephemeral_1h_input_tokens": 300,
                  "ephemeral_5m_input_tokens": 100 if "TRIGGER_5M" in stdin else 0}}}
    if "TRIGGER_LOOP" in stdin:
        result["num_turns"] = 4
    else:
        result["structured_output"] = answer
    emit(result)
    if "TRIGGER_HANG" in stdin:
        time.sleep(30)
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
