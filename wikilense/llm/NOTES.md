# LLM runner: design notes

Which finding each design choice follows. Sources:
- **kit:** the owner's runner kit (`reviews/LLM_RUNNER_KIT_2026-11-11.zip`, unpacked in
  `llm_kit/LLM_RUNNER_KIT/`): its `REPORT.md` sections, the deviation records `D-NN` and
  `code/standalone/cc_runner.py`.
- **review:** the checked review of that kit (`reviews/LLM_RUNNER_KIT_REVIEW_2026-11-12.md`;
  ids such as C1, A1).
- **binary:** strings read from the CLI 2.1.322 binary itself (the copy in `llm_kit/bin/claude`).

## The call (`cli_backend.py`)

| choice | follows |
|---|---|
| The Claude Code CLI in print mode, one subprocess per attempt, no SDK and no API key. | kit REPORT §2.1 |
| Fixed flags (`BASE_FLAGS`), each explained in the code. | kit REPORT §2.1–2.2, D-42 |
| `--system-prompt` *replaces* Claude Code's own prompt (4,803 → 767 tokens); `--tools none` (25,126 → 680); `--safe-mode`. | kit D-42, REPORT §2.2 |
| The user turn goes on stdin, never on argv. The system prompt and the schema go on argv; `MAX_SYSTEM_BYTES` stays under Linux's 128 KiB per-argument limit. | kit REPORT §2.1, §2.10 |
| `--tools none`, not `""`: the kit tested only `none`. The 2.1.322 help documents `""`. Each call's `init.tools` must be `[StructuredOutput]`; anything else is fatal. | kit D-42; review A11, C7 |
| Full model ids only, and aliases are refused. | kit D-42 / `models.py`: `sonnet` once served `claude-sonnet-4-6` |
| The served model is checked on every call against `modelUsage`: the exact id, or the id plus a `[...]` suffix. For Haiku, the exact id only. The check is never a prefix test. | kit `_same_model`; review D17 |
| Child environment: strip `CLAUDE*`, `ANTHROPIC*`, `AI_AGENT*`, plus the caching and thinking variables without that prefix (`FORCE_PROMPT_CACHING_5M`, `DISABLE_PROMPT_CACHING*`, `MAX_THINKING_TOKENS`, ...). | kit D-42 (this session exports `CLAUDE_EFFORT`, `CLAUDECODE`); review A19, C22 |
| Then add `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, which turns off the hidden Haiku call (3,872 → 0 input tokens). | kit D-48 |
| Then add `CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1`, so a flagged claim is not re-sent to another model. | binary (changelog and env list); review C8 |
| A fixed, empty working folder (`<runs_root>/.empty_cwd`). | kit client docstring (the default prompt includes cwd and git status) |
| Minimal system prompt, one for all three conditions; only the user turn differs, and the claim is last. | kit REPORT §2.4 "order rule", §4 "Prompt layout" |
| Prompt caching: no explicit cache marks. The CLI caches the system prompt and the schema tool itself, with a 1-hour TTL (live: 1,113 tokens read back from call 2 on). The stream-json document mark is not used: retrieved chunks do not repeat in order across claims. | kit D-59, D-117; review D10 |
| Schema with one top-level property (`result`) and no `maxLength`/`maxItems`; the prompt says "one sentence of at most 30 words". The load is refused otherwise. | kit D-50/D-53/D-63, D-85/D-86 |
| The prompt asks for "exactly one StructuredOutput call whose only parameter is result" and "no text outside the tool call". | review B8 (`$PARAMETER_NAME` 5/9 → 0/9), B23 |
| Answer: `structured_output`, then the `StructuredOutput` blocks from last to first; the first one valid against the schema wins. Otherwise `parse_failed`, and the label is never guessed. | kit D-50; review A21, C15 |
| Failure classes in the kit's order: auth → not-your-limit → spent window → refusal → transient. | kit `classify_limit_failure` |
| Spent-window markers include the 2.1.322 wording ("You've hit your 5-hour limit", Sonnet / Fable / usage credit limit), plus a generic `hit your ... limit` pattern, plus `rate_limit_info.status == "rejected"`. | binary; review C1 |
| Refusals are final and never retried. The fallback (another model wrote the output) is a refusal, never a label. | kit D-95 |
| Retries: up to `max_attempts` (4) with 10/20/40 s back-off (±20%), and no back-off after the last attempt. At most 2 timeouts. A timeout whose output already holds a valid result keeps it. Usage is kept per attempt in `attempt_log`. | kit `cc_runner`; review C10, C13, C14 |
| After a kill, `communicate(timeout=10)`; each child gets its own process group. | review D13; kit `kill_group` |
| The auth gate fails closed: logged in, `authMethod == "claude.ai"`, `apiProvider == firstParty`. Only the plan fields are kept, never the email. `init.apiKeySource` other than `none` is fatal. | kit `assert_usable_subscription`; review C21 |
| Usage: the CLI's own fields are stored unchanged (`modelUsage`, `usage`, `num_turns`, `duration_ms`, the last `rate_limit_event`). Sums go in `computed_usage`: prompt = input + cacheRead + cacheCreation. | kit D-42/D-45 (input alone under-reports about 1,800 times) |
| `total_cost_usd` is stored as `list_price_usd_not_billed`. | kit naming (API list price; the subscription does not bill it) |
| Utilization is always read as a fraction, so 1.04 means past the cap; a missing reading is `None`, never 0. | binary (`values above 1 occur ...`); review C2, A3 |
| Signs of paid extra usage (`isUsingOverage`, `overageInUse`, `rateLimitType: overage`, a paid grace period, any 5-minute cache write) stop the run with exit 6. | kit D-59; review A1, B1, C2 |

## The run (`runner.py`)

| choice | follows |
|---|---|
| Resume key = (claim, condition, full model id, effort, prompt hash). A key is done once any row with it is final (ok, parse_failed, refusal); errors are asked again. Rows are indexed by key, not by id. | the prompt; review C4, D2 |
| No code version in the key. | review C3, D1 (the kit's date in the key re-asks everything) |
| Resuming is refused when the manifest differs in models, effort, prompt hash, database, k, strategy, ef_search, embedding model, claims-file hash or CLI version. | the prompt; review A7, C6, E12 (no mixing of CLI versions) |
| The pinned binary copy (`llm_kit/bin/claude`, through `CLAUDE_BIN`); the VS Code bundle is never picked by itself. | review C6, E12 |
| Separate stop thresholds for the five-hour and the weekly windows (0.80 both by default; the owner's probe rule). | review C9, E9, E29 |
| The runner never waits out a window itself: it exits 3 and you re-run the same command. | the prompt (no `--wait-on-window` in the probes) |
| Circuit breaker: 5 non-final results in a row stop the run. | kit D-49 |
| Order: claim by claim, with every condition and model of a claim before the next claim, so a stop leaves all conditions on the same claims. | review B17, E27 |
| 5 workers by default, 1–8 allowed, with a warning above 5. | the prompt; kit D-87 (5 the most ever measured) |
| One runner at a time on the machine (`<runs_root>/.llm_runner.lock`, flock), so at most `workers` claude processes run. | review C12, D3; the laptop rule (one heavy job, at most 5 claude processes) |
| The memory guard waits before a call while MemAvailable is under 2,500 MB. | kit `memory_gate` |
| SIGINT, SIGTERM and SIGHUP all stop the run and kill the calls in flight; a call started during the signal is killed too. | review C19 |
| Rows are appended, flushed and fsynced under a lock; a torn last line gets a newline before the next append. Raw streams are written atomically and named by claim, condition, model, key and time. | kit; review C17, C25 |
| The guard on the number of items (≤ 50 per model for `generic_probe*` even with `--full`; `--full` above 50 for any prompt), counting planned plus already-done items. | the prompt |
| `claims_3000.jsonl` needs `--claims-sha256`, and a file without a final newline is refused. | the contract with chat B |

## Retrieval (`retrieval.py`)

The query embedding runs on the CPU: `Embedder(settings.embedding_model, device="cpu")` with 4
torch threads. Then `search.search(conn, qvec, k, strategy="inline", ef_search=100,
query_text=claim)`, with `ef_search` always passed. The cache key is (database, k, strategy,
ef_search, embedding model, claim id); the strategy is added to the prompt's list so that an
`overfetch` run cannot reuse `inline` results. Each chunk is shown as
`[rank] Title > Section path` followed by `chunk.text`.

## Where the prompt and the kit differ (followed: the kit)

- The prompt said the kit caps workers at 8. That is `cc_runner`'s cap; the kit's reference
  project capped the route at 5 (`MAX_WORKERS_BY_ROUTE`) and never measured more. The runner
  allows 1–8 and warns above 5.
- The prompt quoted "3,500 to 4,000 calls per Max window at about 5,500 prompt tokens". The
  RefChecker window was about 3.4k prompt tokens per call (review B24), and the plan tier of
  those windows is unknown (review B2).
