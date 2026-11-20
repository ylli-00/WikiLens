"""The web page's live fact-check: a typed statement and its top passages to Haiku, as in the
LLM experiment (docs/LLM_EXTENSION.md, "On the web page").

**Turning it on.** ``WIKILENSE_FACTCHECK`` (``config.Settings.factcheck``) is ``off`` by default.
``subscription`` calls the Claude Code CLI with the machine's own login
(:class:`wikilense.llm.cli_backend.ClaudeCliBackend`; the binary is ``WIKILENSE_CLAUDE_BIN``, else
``CLAUDE_BIN``, else ``claude`` on the PATH). ``api`` calls the Anthropic API
(:class:`wikilense.llm.api_backend.AnthropicApiBackend`) with ``ANTHROPIC_API_KEY`` from the
environment, else from the repository's ``.env``. The key is read only when the setting is
``api``, so a key on its own turns nothing on. It is kept on the backend alone and removed from
every message this module returns (:meth:`FactChecker.scrub`).

**What is sent is the experiment's request, read from its record, never retyped.**
:func:`load_experiment` takes the original run of the ``retrieved`` condition from
``results/llm/runs.json``: the prompt id and hash, the model, the effort and the retrieval
settings. The prompt set is loaded with :func:`wikilense.llm.prompts.load_prompt_set`, and a set
whose hash differs from the recorded one is refused, so the measured figures belong to what is
sent. The user turn is built by the runner itself (:meth:`wikilense.llm.runner.Runner.request`)
from the passages in the form the runner's retriever stores them, and the answer is read by the
backend's own parsing: a schema-valid structured answer, never free text. Each click makes one
attempt, with no retry, so one click is one model call.

**The figures** (:func:`load_reliability`) are read from ``results/llm/llm_results.json`` when the
server starts. A figure the file lacks is reported as missing, never filled in. The verdict's
figure is the share of right answers among the experiment's answers with that verdict (its
``precision``), shown as a percentage with the file's rounding convention (one decimal, half up).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from wikilense.config import DEFAULT_ENV_FILE, REPO_ROOT, Settings
from wikilense.llm.backend import Backend, CallRequest, CallResult
from wikilense.llm.cli_backend import dig
from wikilense.llm.models import ModelError, check_full_id, option_of
from wikilense.llm.prompts import LABELS, PromptError, PromptSet, load_prompt_set
from wikilense.llm.retrieval import _hit_dict

OFF, SUBSCRIPTION, API = "off", "subscription", "api"

RUNS_FILE = REPO_ROOT / "results" / "llm" / "runs.json"
RESULTS_FILE = REPO_ROOT / "results" / "llm" / "llm_results.json"
#: The experiment's condition whose request the page sends: the claim plus the top-k passages.
CONDITION = "retrieved"
#: The ``run`` field of the first full run in runs.json (the repeats are ``r1`` and ``r2``).
ORIGINAL_RUN = "original"
#: The field next to the label in the answer schema that holds the one-sentence reason.
REASON_FIELD = "justification"

#: Seconds one call may take. The experiment's slowest call in this condition took 16.5 s
#: (runs.json, ``per_condition.retrieved.seconds_final_attempt.max``).
TIMEOUT_S = 60.0
#: The API route stops sending after this many US$ at list price per server start.
API_MAX_USD = 1.0
#: Longest statement accepted, in characters; about twice the longest benchmark claim (520).
MAX_STATEMENT_CHARS = 1000

#: The date the experiment's system prompt judges at; checked against the prompt at load.
JUDGED_AS_OF = "December 2020"
JUDGED_AS_OF_TEXT = f"judged as of {JUDGED_AS_OF}, the date of this Wikipedia copy"
VERDICT_TEXT: dict[str, str] = {
    "SUPPORTS": "Supported",
    "REFUTES": "Refuted",
    "NOT ENOUGH INFO": "Not enough info",
}
NO_CITATION_TEXT = "no passage cited: from the model's own knowledge"
REASON_NOTE = (
    "The reason is the model's own wording, and it can be wrong even when the verdict is right."
)
LIMIT_TEXT = (
    "These figures were measured on benchmark claims whose evidence was in the database; for "
    "your statement it may not be, so expect lower."
)
#: First words that mark a question (with a final "?"): the page then gives a hint, never blocks.
QUESTION_WORDS: tuple[str, ...] = (
    "who", "whom", "whose", "what", "when", "where", "which", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would",
    "should", "has", "have", "had",
)
QUESTION_HINT = (
    "This looks like a question. The fact-check labels a statement, such as "
    '"The Aare flows through Bern."'
)

#: The request header a fact-check needs: a browser sends a custom header to another site only
#: after a CORS preflight, which this server never answers. With :data:`LOOPBACK_HOSTS` it keeps
#: another site open in the same browser from making calls on this machine's login or key.
HEADER = "X-WikiLense-Factcheck"
HEADER_VALUE = "1"
DOCS_LINK = (
    "https://github.com/ylli-00/WikiLens/blob/main/docs/SETUP.md#turn-on-the-live-fact-check"
)
OFF_NOTE = (
    'Live fact-check with Haiku: off on this server. docs/SETUP.md, "Turn on the live '
    'fact-check", says how.'
)

#: Host names a fact-check request may be addressed to: this machine only. A server started with
#: ``--host 0.0.0.0`` then offers the search to the network but not the model calls, and a page
#: whose name is rebound to 127.0.0.1 (DNS rebinding) sends its own name and is refused.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
LOOPBACK_NOTE = (
    "The live fact-check answers only on this machine: open the page at http://127.0.0.1 or "
    "http://localhost."
)

_NUMBERS = r"\d+(?:(?:\s*(?:,|;|-|–|&|\band\b)\s*)+\d+)*"
#: A citation: numbers in brackets (``[2]``, ``[1, 3]``, ``[1, 2, and 3]``, ``[passage 2]``), or
#: ``passage 2`` / ``passages 1 and 2`` written out.
_CITATION_RE = re.compile(
    rf"\[\s*(?:passages?\s*)?({_NUMBERS})\s*\]|\bpassages?\s+({_NUMBERS})", re.IGNORECASE
)


class FactCheckError(RuntimeError):
    """The fact-check cannot run as configured; the message says why, in one sentence."""


class FactCheckRefused(Exception):
    """A request the fact-check answers without a model call (status code and detail)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FactCheckFailed(Exception):
    """A model call that gave no verdict; ``body`` is the JSON response."""

    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        super().__init__(body.get("detail"))
        self.status_code = status_code
        self.body = body


def _rel(path: Path) -> str:
    """Return ``path`` relative to the repository when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------------------------
# the experiment's record
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    """The request settings of the experiment's ``retrieved`` run, from its record."""

    run_folder: str
    prompt: PromptSet
    model_id: str
    effort: str
    backend: str
    cli_version: str | None
    database: str
    k: int
    strategy: str
    ef_search: int
    embedding_model: str
    source: str

    @property
    def model_name(self) -> str:
        """Return the model's short name for the page, e.g. ``Haiku``."""
        return option_of(self.model_id).capitalize()

    def describe(self) -> dict[str, Any]:
        """Return the settings as JSON (prompt id and hash, model, effort, retrieval)."""
        return {
            "run_folder": self.run_folder,
            "prompt_id": self.prompt.prompt_id,
            "prompt_hash": self.prompt.prompt_hash,
            "model": self.model_id,
            "model_name": self.model_name,
            "effort": self.effort,
            "backend": self.backend,
            "cli_version": self.cli_version,
            "database": self.database,
            "k": self.k,
            "strategy": self.strategy,
            "ef_search": self.ef_search,
            "embedding_model": self.embedding_model,
            "source": self.source,
        }


def load_experiment(path: Path = RUNS_FILE, prompts_dir: Path | None = None) -> Experiment:
    """Return the experiment's ``retrieved`` run from ``path`` with its prompt set loaded.

    Raises :class:`FactCheckError` when the record is missing or unreadable, when it does not hold
    exactly one original run of the condition with one full model id, when the prompt set's hash
    differs from the recorded one, when the system prompt does not judge as of
    :data:`JUDGED_AS_OF`, or when the answer schema has no :data:`REASON_FIELD` next to the label.
    """
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FactCheckError(f"cannot read the experiment's record {_rel(path)}: {exc}") from exc
    runs = [
        r for r in record.get("runs") or []
        if r.get("run") == ORIGINAL_RUN and r.get("conditions") == [CONDITION]
    ]
    if len(runs) != 1:
        raise FactCheckError(
            f"{_rel(path)} holds {len(runs)} original runs of the {CONDITION!r} condition, not one"
        )
    run = runs[0]
    try:
        models = list(run["model"])
        if len(models) != 1:
            raise FactCheckError(f"the {CONDITION!r} run names {len(models)} models, not one")
        model_id = check_full_id(str(models[0]))
        prompt = load_prompt_set(str(run["prompt_id"]), prompts_dir)
        retrieval = run["retrieval"]
        experiment = Experiment(
            run_folder=str(run["run_folder"]),
            prompt=prompt,
            model_id=model_id,
            effort=str(run["effort"]),
            backend=str(run["backend"]),
            cli_version=run.get("cli_version"),
            database=str(retrieval["database"]),
            k=int(retrieval["k"]),
            strategy=str(retrieval["strategy"]),
            ef_search=int(retrieval["ef_search"]),
            embedding_model=str(retrieval["embedding_model"]),
            source=f"{_rel(path)}: run {run['run_folder']}",
        )
    except (KeyError, TypeError, ValueError) as exc:
        kind = "record"
        if isinstance(exc, PromptError):
            kind = "prompt set"
        elif isinstance(exc, ModelError):
            kind = "model id"
        raise FactCheckError(f"the experiment's {kind} cannot be used: {exc}") from exc
    if prompt.prompt_hash != run.get("prompt_hash"):
        raise FactCheckError(
            f"the prompt set {prompt.prompt_id} differs from the one the experiment ran (hash "
            f"{str(run.get('prompt_hash'))[:12]} recorded, {prompt.prompt_hash[:12]} now), so the "
            "measured figures would not apply"
        )
    if CONDITION not in prompt.conditions:
        raise FactCheckError(f"the prompt set {prompt.prompt_id} has no {CONDITION!r} template")
    if f"as of {JUDGED_AS_OF}" not in prompt.system:
        raise FactCheckError(
            f"the system prompt of {prompt.prompt_id} does not judge as of {JUDGED_AS_OF}"
        )
    parent: Any = prompt.schema
    for part in prompt.label_path[:-1]:
        parent = (parent.get("properties") or {}).get(part) or {}
    if REASON_FIELD not in (parent.get("properties") or {}):
        raise FactCheckError(
            f"the answer schema of {prompt.prompt_id} has no {REASON_FIELD!r} next to the label"
        )
    return experiment


def experiment_request(
    experiment: Experiment, statement: str, passages: Sequence[Mapping[str, Any]]
) -> CallRequest:
    """Return the experiment's request for ``statement`` with ``passages``, built by the runner.

    ``passages`` are hit dicts in rank order as the runner's retriever stores them
    (``retrieval._hit_dict``). A :class:`~wikilense.llm.runner.Runner` configured like the
    experiment builds the request (:meth:`~wikilense.llm.runner.Runner.request`): the same system
    prompt, model, effort, schema and user turn (``claims.render_retrieved`` then the
    ``retrieved`` template). The runner only builds the request here; it writes nothing. It is
    imported here, not with this module, because it needs ``fcntl``, which Windows lacks.
    """
    from wikilense.llm.runner import Item, RunConfig, Runner, item_key

    model_id = experiment.model_id
    config = RunConfig(
        run_dir=Path("not-written"),
        claims_path=Path("not-read"),
        prompt=experiment.prompt,
        models={option_of(model_id): model_id},
        effort=experiment.effort,
        conditions=[CONDITION],
    )
    runner = Runner(config, backend=None, retriever=None)  # type: ignore[arg-type]
    claim = {"id": 0, "claim": statement}
    runner.hits[0] = [dict(p) for p in passages]
    item = Item(claim, CONDITION, option_of(model_id), model_id,
                item_key(0, CONDITION, model_id, experiment.effort, experiment.prompt.prompt_hash))
    return runner.request(item)


def passages_of(hits: Sequence[Any], k: int) -> list[dict[str, Any]]:
    """Return the first ``k`` ``search.Hit`` objects as the runner's hit dicts, ranked from 1."""
    return [_hit_dict(rank, hit) for rank, hit in enumerate(hits[:k], 1)]


# ---------------------------------------------------------------------------------------------
# the reliability figures
# ---------------------------------------------------------------------------------------------


def pct_text(fraction: float) -> str:
    """Return ``fraction`` as a percentage with one decimal, rounded half up (``0.7191`` -> ``71.9``).

    The value is read through its shortest decimal form (``repr``), as ``scripts/llm_results.py``
    rounds, so ``0.1235`` gives ``12.4``; the float's exact binary value would give ``12.3``.
    """
    value = Decimal(repr(float(fraction))) * 100
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class Figure:
    """One reliability figure from the results file, or a missing one (``pct`` None)."""

    pct: str | None
    right: int | None = None
    n: int | None = None
    interval_pct: tuple[str, str] | None = None
    source: str = ""

    @property
    def missing(self) -> bool:
        """Return True when the results file lacks the figure."""
        return self.pct is None

    def as_json(self) -> dict[str, Any]:
        """Return the figure as JSON, with ``missing``."""
        return {
            "pct": self.pct,
            "right": self.right,
            "n": self.n,
            "interval_pct": list(self.interval_pct) if self.interval_pct else None,
            "source": self.source,
            "missing": self.missing,
        }


@dataclass(frozen=True)
class Reliability:
    """The figures of the experiment's ``retrieved`` condition, read from the results file.

    ``per_verdict`` maps each label to the share of right answers among the answers with that
    label; ``overall`` is the three-class accuracy on all claims; ``supported_refuted`` the
    accuracy on the supported and refuted claims; ``missed_gold_page`` the accuracy on the claims
    whose top passages held no gold page (the lower end for a statement whose evidence may be
    missing).
    """

    per_verdict: dict[str, Figure]
    overall: Figure
    supported_refuted: Figure
    missed_gold_page: Figure
    file_error: str | None = None

    def figures(self) -> dict[str, Figure]:
        """Return every figure by a flat name (``verdict:SUPPORTS``, ``overall``, ...)."""
        out = {f"verdict:{label}": fig for label, fig in self.per_verdict.items()}
        out.update(overall=self.overall, supported_refuted=self.supported_refuted,
                   missed_gold_page=self.missed_gold_page)
        return out

    def missing(self) -> list[str]:
        """Return the sources of the figures the results file lacks."""
        return [fig.source for fig in self.figures().values() if fig.missing]

    def as_json(self) -> dict[str, Any]:
        """Return every figure as JSON, with the missing ones listed."""
        return {
            "per_verdict": {label: fig.as_json() for label, fig in self.per_verdict.items()},
            "overall": self.overall.as_json(),
            "supported_refuted": self.supported_refuted.as_json(),
            "missed_gold_page": self.missed_gold_page.as_json(),
            "missing": self.missing(),
            "file_error": self.file_error,
        }


def _path_get(obj: Any, dotted: str) -> Any:
    """Follow a dotted path through nested dicts; None when a step is missing."""
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def _accuracy_figure(data: Any, dotted: str, source: str) -> Figure:
    """Return an accuracy entry (``pct``, ``right``, ``n``, ``wilson_pct``) or a missing figure."""
    entry = _path_get(data, dotted)
    src = f"{source}: {dotted}"
    if not isinstance(entry, dict) or not isinstance(entry.get("pct"), str):
        return Figure(None, source=src)
    interval = entry.get("wilson_pct")
    interval_t = (str(interval[0]), str(interval[1])) if isinstance(interval, list) \
        and len(interval) == 2 else None
    right, n = entry.get("right"), entry.get("n")
    return Figure(
        entry["pct"],
        right=right if isinstance(right, int) else None,
        n=n if isinstance(n, int) else None,
        interval_pct=interval_t,
        source=src,
    )


def load_reliability(path: Path = RESULTS_FILE, condition: str = CONDITION) -> Reliability:
    """Return the figures of ``condition`` from ``path``; a figure it lacks is missing.

    Read: ``three_class.conditions.<condition>.per_label.<label>.precision`` per verdict and
    ``.accuracy`` overall, ``supported_refuted.conditions.<condition>.accuracy``, and
    ``retrieval_link.all_claims.gold_page_in_top5.no``. Nothing is computed beyond writing the
    precision as a percentage.
    """
    source = _rel(path)
    data: Any = None
    file_error = None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        file_error = f"cannot read {source}: {exc}"
    per_verdict: dict[str, Figure] = {}
    for label in LABELS:
        dotted = f"three_class.conditions.{condition}.per_label.{label}.precision"
        value = _path_get(data, dotted)
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        per_verdict[label] = Figure(pct_text(value) if ok else None, source=f"{source}: {dotted}")
    return Reliability(
        per_verdict=per_verdict,
        overall=_accuracy_figure(data, f"three_class.conditions.{condition}.accuracy", source),
        supported_refuted=_accuracy_figure(
            data, f"supported_refuted.conditions.{condition}.accuracy", source),
        missed_gold_page=_accuracy_figure(
            data, "retrieval_link.all_claims.gold_page_in_top5.no", source),
        file_error=file_error,
    )


def _count(fig: Figure) -> str:
    """Return ``" (2,127 of 3,000)"`` when the figure has its counts, else ``""``."""
    return f" ({fig.right:,} of {fig.n:,})" if fig.right is not None and fig.n is not None else ""


def reliability_lines(rel: Reliability, label: str, model: str, k: int) -> dict[str, str]:
    """Return the page's sentences for a verdict: ``verdict``, ``overall``, ``limit``,
    ``lower_end``; ``model`` is the model's short name and ``k`` the experiment's passages.

    Every number comes from ``rel``; a missing one is named as missing with its source.
    """
    verdict = VERDICT_TEXT[label]
    fig = rel.per_verdict[label]
    overall, sides, missed = rel.overall, rel.supported_refuted, rel.missed_gold_page
    scope = f"top {k} passages from MariaDB"
    if overall.n is not None:
        scope += f", all {overall.n:,} benchmark claims"
    if fig.missing:
        line = (f'How often "{verdict}" was right in the experiment: not in the results file '
                f"({fig.source}).")
    else:
        line = (f'When {model} said "{verdict}" in the experiment ({scope}), that was right '
                f"{fig.pct}% of the time.")
    if overall.missing:
        total = f"Overall accuracy: not in the results file ({overall.source})."
    else:
        total = f"Overall it was right on {overall.pct}% of the claims{_count(overall)}"
        total += (f" and on {sides.pct}% of the supported or refuted ones{_count(sides)}."
                  if not sides.missing else
                  f"; the supported/refuted accuracy is not in the results file ({sides.source}).")
    if missed.missing:
        lower = (f"The accuracy when retrieval missed the gold page is not in the results file "
                 f"({missed.source}).")
    else:
        interval = (f", interval {missed.interval_pct[0]}–{missed.interval_pct[1]}%"
                    if missed.interval_pct else "")
        if missed.right is not None and missed.n is not None:
            lower = (f"On the {missed.n:,} claims whose top {k} passages held no gold page, "
                     f"{model} was right on {missed.right:,} ({missed.pct}%{interval}): take "
                     "that as the lower end.")
        else:
            lower = (f"On the claims whose top {k} passages held no gold page, {model} was right "
                     f"{missed.pct}% of the time{interval}: take that as the lower end.")
    return {"verdict": line, "overall": total, "limit": LIMIT_TEXT, "lower_end": lower}


# ---------------------------------------------------------------------------------------------
# the statement and the reason
# ---------------------------------------------------------------------------------------------


def looks_like_question(text: str) -> bool:
    """Return True when ``text`` ends with ``?`` or starts with a word of :data:`QUESTION_WORDS`."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    words = re.findall(r"[A-Za-z]+", stripped.lower())
    return bool(words) and words[0] in QUESTION_WORDS


def _citation_numbers(group: str) -> list[int]:
    """Return the numbers of one bracket's text: ``2``, ``1, 3``, ``2-4``, ``1 and 2``."""
    numbers: list[int] = []
    for token in re.split(r"\s*(?:,|;|&|\band\b)\s*", group):
        bounds = re.split(r"\s*[-–]\s*", token.strip())
        if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
            low, high = int(bounds[0]), int(bounds[1])
            if low <= high <= low + 100:
                numbers.extend(range(low, high + 1))
        elif len(bounds) == 1 and bounds[0].isdigit():
            numbers.append(int(bounds[0]))
    return numbers


def reason_parts(reason: str, n_passages: int) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Split ``reason`` into text parts and citation parts; return them, the cited passage numbers
    and the numbers outside 1..``n_passages``.

    A citation is a bracket of numbers such as ``[2]``, ``[1, 3]``, ``[2-3]``, ``[1, 2, and 3]``
    or ``[passage 2]``, or the words ``passage 2`` / ``passages 1 and 2``; its part carries
    ``passages``, the numbers that name a passage that was sent.
    """
    parts: list[dict[str, Any]] = []
    cited: list[int] = []
    unknown: list[int] = []
    pos = 0
    for match in _CITATION_RE.finditer(reason):
        if match.start() > pos:
            parts.append({"text": reason[pos:match.start()]})
        numbers = _citation_numbers(match.group(1) or match.group(2))
        known = [n for n in numbers if 1 <= n <= n_passages]
        unknown.extend(n for n in numbers if not 1 <= n <= n_passages)
        parts.append({"text": match.group(0), "passages": known})
        cited.extend(known)
        pos = match.end()
    if pos < len(reason):
        parts.append({"text": reason[pos:]})
    return parts, sorted(set(cited)), sorted(set(unknown))


# ---------------------------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------------------------


def _failure(res: CallResult, mode: str, timeout_s: float) -> tuple[int, str]:
    """Return the status code and the plain message of a call that gave no verdict."""
    status, kind = res.status, res.error_kind
    if status == "error" and kind in ("timeout", "time_limit"):   # CLI and API time limits
        return 504, f"No answer within {timeout_s:.0f} s, so there is no verdict. Try again later."
    if status == "error" and kind == "rate_limit":
        return 503, ("The API's per-minute rate limit was hit, so there is no verdict. Try again "
                     "in a minute.")
    if status == "window":
        return 503, ("The usage limit of this machine's Claude login is reached, so there is no "
                     "verdict. Try again after it resets.")
    if status == "money":
        return 503, (f"The API route's spending cap (US${API_MAX_USD:.2f} at list price per "
                     "server start) or the account's limit is reached, so there is no verdict.")
    if status == "fatal" and kind == "auth":
        if mode == SUBSCRIPTION:
            return 503, ("The Claude Code CLI on this machine is not logged in, so there is no "
                         "verdict. Log in with the CLI, then restart the server.")
        return 503, "The API key was refused, so there is no verdict."
    if status == "parse_failed":
        return 502, "The model answered, but not in the expected form, so there is no verdict."
    if status == "refusal":
        return 502, ("The model declined, or another model answered (which would not count as "
                     "Haiku's verdict), so there is no verdict.")
    if status == "fatal":
        return 503, "The model cannot be reached as this server is set up, so there is no verdict."
    return 502, "The model call failed, so there is no verdict."


@dataclass
class FactChecker:
    """The fact-check of one server: its mode, backend, the experiment and the figures.

    ``enabled`` is True when the mode is not ``off`` and nothing stopped it (``unavailable``
    names what did). ``secrets`` are removed from every message (:meth:`scrub`). One call runs
    at a time. ``checked`` is set by :meth:`check`, which :meth:`run` calls before the first call
    when the server has not. After a call, the runner's stop rules turn the fact-check off for
    the rest of the server's life: a stop signal (a sign of paid extra usage, a five-minute cache
    write, the API spending cap) or a fatal result (login, key, wrong model, unexpected tools).
    """

    mode: str = OFF
    backend: Backend | None = None
    experiment: Experiment | None = None
    reliability: Reliability | None = None
    unavailable: str | None = None
    timeout_s: float = TIMEOUT_S
    backend_version: str | None = None
    checked: bool = False
    secrets: tuple[str, ...] = field(default=(), repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def enabled(self) -> bool:
        """Return True when a fact-check can be made."""
        return (self.mode != OFF and self.unavailable is None and self.backend is not None
                and self.experiment is not None)

    def scrub(self, text: str | None) -> str | None:
        """Return ``text`` with every secret replaced by ``[hidden]``."""
        if text is None:
            return None
        for secret in self.secrets:
            if secret:
                text = text.replace(secret, "[hidden]")
        return text

    def off_detail(self) -> str:
        """Return why no fact-check is made: off, or turned on but unavailable."""
        if self.mode == OFF:
            return OFF_NOTE
        return self.scrub(f"The live fact-check is set to {self.mode} but cannot run: "
                          f"{self.unavailable}") or ""

    def check(self) -> str:
        """Check the backend without a model call and return one line for the server's log.

        ``subscription``: ``claude auth status`` must show the claude.ai login
        (``ClaudeCliBackend.preflight``); a failure makes the fact-check unavailable. ``api``: no
        call is made (the key was read when the backend was built).
        """
        self.checked = True
        if self.mode == OFF:
            return "off (WIKILENSE_FACTCHECK=subscription or api turns it on)"
        if not self.enabled:
            return self.off_detail()
        assert self.backend is not None and self.experiment is not None
        if self.mode == SUBSCRIPTION:
            from wikilense.llm.cli_backend import CliBackendError

            try:
                info = self.backend.preflight([self.experiment.model_id])
            except CliBackendError as exc:
                self.unavailable = str(exc)
                return self.off_detail()
            version = getattr(self.backend, "version", None)
            self.backend_version = version() if callable(version) else None
            login = f"login {info.get('authMethod')}, plan {info.get('subscriptionType')}"
        else:
            login = "key read (not shown)"
        exp = self.experiment
        return (f"{self.mode}, {exp.model_id} at effort {exp.effort}, prompt "
                f"{exp.prompt.prompt_id} ({exp.prompt.prompt_hash[:12]}); {login}"
                + (f"; CLI {self.backend_version}" if self.backend_version else ""))

    def status(self) -> dict[str, Any]:
        """Return what the page needs before a call: on or off, the notes, the experiment."""
        body: dict[str, Any] = {
            "mode": self.mode,
            "enabled": self.enabled,
            "note": None if self.enabled else self.off_detail(),
            "docs_link": DOCS_LINK,
        }
        if not self.enabled:
            return body
        assert self.experiment is not None
        body.update(
            header=HEADER,
            header_value=HEADER_VALUE,
            model=self.experiment.model_id,
            passages=self.experiment.k,
            judged_as_of=JUDGED_AS_OF_TEXT,
            question_words=list(QUESTION_WORDS),
            question_hint=QUESTION_HINT,
            max_statement_chars=MAX_STATEMENT_CHARS,
            experiment=self.experiment.describe(),
        )
        return body

    def caveats(self, differences: Sequence[str]) -> list[str]:
        """Return the caveats of a verdict: settings that differ, then the route or CLI version."""
        assert self.experiment is not None
        exp = self.experiment
        out = []
        if differences:
            out.append(
                "Your search differs from the experiment's (" + "; ".join(differences) + "): "
                f"the figures were measured with the experiment's settings on {exp.database} "
                f"(k {exp.k}, strategy {exp.strategy}, mhnsw_ef_search {exp.ef_search}, no "
                "filters)."
            )
        if self.mode == API:
            out.append("The figures were measured through the Claude Code CLI on a subscription; "
                       "this server sends the same request through the Anthropic API, a route "
                       "the experiment did not measure.")
        elif self.backend_version and exp.cli_version and self.backend_version != exp.cli_version:
            out.append(f"This server's Claude Code CLI is {self.backend_version}; the experiment "
                       f"ran {exp.cli_version}.")
        return out

    def _apply_stop_rules(self, res: CallResult) -> str | None:
        """Turn the fact-check off after a call the runner would stop a run on; return the
        sentence that says so, or None."""
        if res.stop_signal:
            why = f"a call carried the stop signal {res.stop_signal!r}"
        elif res.status == "fatal":
            why = f"a call ended with a fatal {res.error_kind or 'error'}"
        else:
            return None
        self.unavailable = self.scrub(f"{why}; restart the server to turn it on again")
        return f"The live fact-check is now off on this server: {self.unavailable}."

    def run(self, statement: str, hits: Sequence[Any], differences: Sequence[str]) -> dict[str, Any]:
        """Make one call for ``statement`` with the first passages of ``hits``; return the verdict.

        ``hits`` are the reader's search's ``search.Hit`` objects in rank order; the first
        ``experiment.k`` are sent. Raises :class:`FactCheckRefused` (no call made) when the
        fact-check is off, the statement is too long, there is no passage, or a call is already
        running; :class:`FactCheckFailed` when the call gave no verdict.
        """
        if not self.enabled:
            raise FactCheckRefused(403, self.off_detail())
        if not self.checked:
            self.check()
            if not self.enabled:
                raise FactCheckRefused(503, self.off_detail())
        assert self.backend is not None and self.experiment is not None
        exp = self.experiment
        statement = statement.strip()
        if len(statement) > MAX_STATEMENT_CHARS:
            raise FactCheckRefused(422, f"the statement has {len(statement):,} characters; the "
                                        f"fact-check takes at most {MAX_STATEMENT_CHARS:,}")
        passages = passages_of(hits, exp.k)
        if not passages:
            raise FactCheckRefused(422, "the search returned no passages, so there is nothing "
                                        "to check the statement against")
        request = experiment_request(exp, statement, passages)
        if not self.lock.acquire(blocking=False):
            raise FactCheckRefused(429, "a fact-check is already running on this server; wait "
                                        "for it to finish")
        try:
            start = time.perf_counter()
            try:
                res = self.backend.call(request, threading.Event())
            except Exception as exc:  # noqa: BLE001 - a backend raising is a failed call too
                res = CallResult("error", error=f"{type(exc).__name__}: {exc}",
                                 error_kind="backend")
            call_ms = round((time.perf_counter() - start) * 1000.0, 3)
            stopped = self._apply_stop_rules(res)
        finally:
            self.lock.release()
        timing = {"model_call": call_ms}
        reason = dig(res.answer, (*exp.prompt.label_path[:-1], REASON_FIELD)) \
            if res.answer is not None else None
        if res.status != "ok" or res.label not in VERDICT_TEXT or not isinstance(reason, str):
            code, message = _failure(res, self.mode, self.timeout_s)
            raise FactCheckFailed(code, {
                "detail": message + (f" {stopped}" if stopped else ""),
                "status": res.status,
                "error_kind": res.error_kind,
                "error": (self.scrub(res.error) or "")[:300] or None,
                "turned_off": stopped,
                "timing_ms": timing,
            })
        parts, cited, unknown = reason_parts(reason, len(passages))
        assert self.reliability is not None
        caveats = self.caveats(differences)
        if stopped:
            caveats.append(stopped)
        return {
            "statement": statement,
            "verdict": res.label,
            "verdict_text": VERDICT_TEXT[res.label],
            "judged_as_of": JUDGED_AS_OF_TEXT,
            "reason_label": f"{exp.model_name}'s reason",
            "reason": self.scrub(reason),
            "reason_parts": [{**p, "text": self.scrub(p["text"])} for p in parts],
            "cited": cited,
            "cited_unknown": unknown,
            "no_citation": NO_CITATION_TEXT if not cited else None,
            "reason_note": REASON_NOTE,
            "passages": [{k: p[k] for k in ("rank", "chunk_id", "title", "section_path")}
                         for p in passages],
            "reliability": self.reliability.as_json(),
            "reliability_lines": reliability_lines(self.reliability, res.label, exp.model_name,
                                                   exp.k),
            "caveats": caveats,
            "turned_off": stopped,
            "settings_differ": list(differences),
            "question_hint": QUESTION_HINT if looks_like_question(statement) else None,
            "mode": self.mode,
            "model": exp.model_id,
            "experiment": exp.describe(),
            "timing_ms": timing,
        }


def resolve_api_key(environ: Mapping[str, str], env_file: Path) -> str:
    """Return ``ANTHROPIC_API_KEY`` from ``environ``, else from ``env_file``; the value is never
    in a message.

    Raises ``ApiBackendError`` for a missing, empty or placeholder key, or one that does not start
    with ``sk-ant-``.
    """
    from wikilense.llm.api_backend import PLACEHOLDER_KEYS, ApiBackendError, load_api_key

    value = (environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not value:
        return load_api_key(env_file)
    if value in PLACEHOLDER_KEYS or not value.startswith("sk-ant-"):
        raise ApiBackendError("ANTHROPIC_API_KEY in the environment is a placeholder or does "
                              "not start with sk-ant-")
    return value


def build_factchecker(
    settings: Settings,
    environ: Mapping[str, str] | None = None,
    env_file: Path = DEFAULT_ENV_FILE,
    runs_file: Path = RUNS_FILE,
    results_file: Path = RESULTS_FILE,
) -> FactChecker:
    """Return the fact-check ``settings.factcheck`` asks for; never raises for a setup problem.

    ``off`` builds nothing and reads no key. Otherwise the experiment and the figures are read,
    then the backend is built with one attempt per call (on the API route also one try on a
    per-minute rate limit) and :data:`TIMEOUT_S`; a problem (no record, a changed prompt, a runner
    that cannot be imported, no CLI binary, no key, no SDK, a variable that redirects the SDK)
    leaves the fact-check unavailable with the reason.
    """
    mode = settings.factcheck
    if mode == OFF:
        return FactChecker(OFF)
    env = os.environ if environ is None else environ
    checker = FactChecker(mode)
    try:
        checker.experiment = load_experiment(runs_file)
    except FactCheckError as exc:
        checker.unavailable = str(exc)
        return checker
    checker.reliability = load_reliability(results_file)
    try:
        importlib.import_module("wikilense.llm.runner")   # builds the request; needs fcntl
    except ImportError as exc:
        checker.unavailable = (f"the request is built by the LLM runner, which cannot be "
                               f"imported on this system ({exc})")
        return checker
    from wikilense.llm.api_backend import AnthropicApiBackend, ApiBackendError, check_environment
    from wikilense.llm.cli_backend import ClaudeCliBackend, CliBackendError

    try:
        if mode == SUBSCRIPTION:
            binary = settings.claude_bin or env.get("CLAUDE_BIN") or shutil.which("claude")
            if not binary:
                raise FactCheckError("no Claude Code CLI found: set WIKILENSE_CLAUDE_BIN to its "
                                     "path")
            cwd = Path(tempfile.mkdtemp(prefix="wikilense-factcheck-"))
            checker.backend = ClaudeCliBackend(binary, cwd, LABELS, timeout_s=TIMEOUT_S,
                                               max_attempts=1)
        else:
            check_environment(dict(env))
            key = resolve_api_key(env, env_file)
            checker.secrets = (key,)
            backend = AnthropicApiBackend(env_file, LABELS, max_usd=API_MAX_USD,
                                          timeout_s=TIMEOUT_S, max_attempts=1, api_key=key,
                                          rate_limit_attempts=1)
            if backend.sdk_version() is None:
                raise FactCheckError("the anthropic package is not installed: pip install -r "
                                     "requirements-llm.txt")
            checker.backend = backend
    except (FactCheckError, CliBackendError, ApiBackendError) as exc:
        checker.backend = None
        checker.unavailable = checker.scrub(str(exc))
    return checker
