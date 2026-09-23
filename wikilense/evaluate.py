"""Evaluation harness: recall and latency of the semantic search against the FEVEROUS ground truth.

The corpus claims are the queries (``claim.text``); their gold evidence (``claim_evidence``) is
the answer. Per claim the ground truth is the set of gold pages (any evidence set, resolved
``page_id`` only) and the sentence-only evidence sets (every element a sentence or list item
with a resolved ``sentence_id``); :func:`load_ground_truth` builds it from the tables.

:func:`evaluate` embeds every claim once with ``embed_queries``, runs ``search.search`` for each
claim and each ``k`` in ``ks`` (one untimed warm-up pass, then ``repeats`` timed passes) and
reports per ``k``:

- article recall@k: the share of claims with a gold page among the pages of the first ``k``
  hits (denominator: every claim);
- evidence recall@k: the share of claims for which every unit of at least one sentence-only
  evidence set lies inside the first ``k`` chunks (denominator: the claims that have such a set);
- unit coverage@k: the mean, over the same claims, of the best share of a set's units covered;
- SQL latency: p50 / p95 / mean / min / max of one ``search`` call in milliseconds, over
  ``repeats`` x claims samples.

The recall values at ``k`` are computed on the hits of the ``LIMIT k`` search of the first timed
pass, the same statement whose latency is reported at ``k``. For ``none`` and ``inline`` these
are the first ``k`` hits of the ``LIMIT max(ks)`` search; for ``overfetch`` and ``rrf`` they are
not, because the inner candidate list is ``k * overfetch`` long (``prefix_mismatches`` counts
the claims where they differ). The per-claim ranks in ``claims`` come from ``LIMIT max(ks)``.

The query-embedding latency (one ``embed_queries([text])`` call per claim and repeat, after a
warm-up call) is reported separately; the vectors used for the searches come from one batched
``embed_queries`` call, so the SQL timing never includes the model.

Every ``search.search`` call also receives the claim text as ``query_text`` (the ``rrf``
strategy fuses the vector ranking with a full-text ranking of these words; the other strategies
ignore it). ``strategy`` accepts whatever ``search.STRATEGIES`` lists. ``mhnsw_ef_search``
defaults to ``settings.ef_search`` (``WIKILENSE_EF_SEARCH``); ``ef_search=SERVER_DEFAULT_EF_SEARCH``
leaves the server's value in place. The parameters of a run record the effective session
``mhnsw_ef_search``, the global ``mhnsw_max_cache_size``, the ``M`` and ``DISTANCE`` of the
vector index as ``SHOW CREATE TABLE chunk`` reports them, and the chunk count, so that a result
file says which index it was measured on.

The metrics (:func:`article_recall_at_k`, :func:`evidence_recall_at_k`,
:func:`unit_coverage_at_k`, :func:`gold_page_rank`, :func:`evidence_rank`,
:func:`latency_stats`) are pure functions over lists of hit page ids or hit sentence-id sets in
the order ``search`` returned the chunks: "top-k chunks" means the first ``k`` hits.
:func:`write_results` writes the JSON and the Markdown tables to ``results/``.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import platform
import re
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pymysql

from wikilense import __version__, db
from wikilense import search as searchmod
from wikilense.config import REPO_ROOT, Settings, load_settings
from wikilense.search import EF_SEARCH_VARIABLE, Filters, Hit, ef_search_session

logger = logging.getLogger(__name__)

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10, 20)
DEFAULT_REPEATS = 5
DEFAULT_STRATEGY = "none"
DEFAULT_OVERFETCH = 10
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_NAME = "baseline"

SERVER_DEFAULT_EF_SEARCH = "server"
"""Pass as ``ef_search`` to leave ``mhnsw_ef_search`` at the server's value for the run."""

#: Strategies whose statement uses the ``overfetch`` factor (recorded in the parameters).
OVERFETCH_STRATEGIES: frozenset[str] = frozenset({"overfetch", "rrf"})

#: Element types that are text units (rows of ``sentence``); the others are table content.
TEXT_UNIT_TYPES = frozenset({"sentence", "item"})

#: Percentiles reported by :func:`latency_stats` (numpy's linear interpolation).
PERCENTILES: tuple[int, int] = (50, 95)

#: Largest ``IN (...)`` list sent when the hit chunks are joined to their sentence ids.
CHUNK_ID_BATCH = 500

#: Distributions whose versions are recorded (importlib.metadata names).
VERSIONED_DISTRIBUTIONS: tuple[str, ...] = ("PyMySQL", "numpy", "torch", "sentence-transformers")

# ---------------------------------------------------------------------------------------------
# SQL (every value is a parameter; the only variable text is the length of an IN list)
# ---------------------------------------------------------------------------------------------

CLAIMS_SQL = "SELECT claim_id, split, text, label FROM claim ORDER BY claim_id"
EVIDENCE_SQL = (
    "SELECT ce.claim_id, ce.evidence_set, ce.element_type, ce.page_id, page.title, "
    "ce.sentence_id "
    "FROM claim_evidence AS ce LEFT JOIN page ON page.page_id = ce.page_id "
    "ORDER BY ce.claim_id, ce.evidence_set, ce.position"
)
#: ``WHERE chunk_id IN (%s, ...)``; the placeholder list is built by :func:`_placeholders`.
CHUNK_SENTENCES_PREFIX = "SELECT chunk_id, sentence_id FROM chunk_sentence WHERE chunk_id IN ("
CHUNK_SENTENCES_SUFFIX = ")"
INGEST_META_SQL = "SELECT `key`, `value` FROM ingest_meta ORDER BY `key`"
CORPUS_COUNTS_SQL = (
    "SELECT (SELECT COUNT(*) FROM page), (SELECT COUNT(*) FROM chunk), "
    "(SELECT COUNT(*) FROM sentence)"
)
VERSION_SQL = "SELECT VERSION()"
CACHE_SIZE_SQL = "SELECT @@GLOBAL.mhnsw_max_cache_size"
SHOW_CREATE_CHUNK_SQL = "SHOW CREATE TABLE chunk"
#: ``M`` and ``DISTANCE`` as SHOW CREATE TABLE prints a vector index on 11.8:
#: ``VECTOR KEY `embedding` (`embedding`) `M`='16' `DISTANCE`='cosine'``.
_INDEX_M_RE = re.compile(r"`M`\s*=\s*'?(\d+)'?")
_INDEX_DISTANCE_RE = re.compile(r"`DISTANCE`\s*=\s*'?([A-Za-z]+)'?")

# ---------------------------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimTruth:
    """The ground truth of one claim.

    ``gold_pages`` are the resolved ``page_id`` values over every evidence set and
    ``gold_titles`` their titles; ``sentence_only_sets`` holds, per evidence set whose elements
    are all sentences or list items with a resolved ``sentence_id``, the set of those ids, in
    evidence-set order. A claim without such a set is not eligible for evidence recall.
    """

    claim_id: int
    split: str
    label: str
    text: str
    gold_pages: set[int]
    sentence_only_sets: list[set[int]]
    gold_titles: set[str] = field(default_factory=set)

    @property
    def eligible(self) -> bool:
        """Return True when the claim has at least one sentence-only evidence set."""
        return bool(self.sentence_only_sets)


@dataclass(frozen=True)
class LatencyStats:
    """Summary of timing samples in milliseconds; ``n`` is the number of samples."""

    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class KMetrics:
    """The metrics at one ``k``: hit counts, recall values and the SQL latency of ``LIMIT k``.

    ``article_recall = article_hits / n_claims`` and
    ``evidence_recall = evidence_hits / n_evidence_claims`` of the :class:`EvalResult`.
    """

    k: int
    article_hits: int
    article_recall: float
    evidence_hits: int
    evidence_recall: float
    unit_coverage: float
    sql_latency: LatencyStats


@dataclass(frozen=True)
class ClaimOutcome:
    """What the search did for one claim, for the failure-mode discussion.

    ``gold_page_rank`` is the 1-based rank of the first hit on a gold page within the
    ``max(ks)`` hits, or None; ``evidence_rank`` the smallest number of top hits that covers a
    sentence-only set, or None (also None when the claim is not eligible).
    """

    claim_id: int
    split: str
    label: str
    text: str
    gold_titles: tuple[str, ...]
    eligible: bool
    gold_page_rank: int | None
    evidence_rank: int | None
    unit_coverage: float
    hit_chunk_ids: tuple[int, ...]


@dataclass
class EvalResult:
    """The output of :func:`evaluate`; ``to_dict`` gives the JSON form written by ``write_results``.

    ``n_claims`` is the denominator of article recall, ``n_evidence_claims`` that of evidence
    recall and unit coverage. ``unstable_claims`` lists the claims whose ``max(ks)`` hits
    differed between repeats; ``prefix_mismatches`` counts, per ``k``, the claims whose
    ``LIMIT k`` hits were not the first ``k`` hits of the ``LIMIT max(ks)`` query.
    """

    parameters: dict[str, Any]
    ingest_meta: dict[str, str]
    machine: dict[str, Any]
    versions: dict[str, str]
    n_claims: int
    n_evidence_claims: int
    per_k: list[KMetrics]
    embedding_latency: LatencyStats
    claims: list[ClaimOutcome]
    unstable_claims: list[int]
    prefix_mismatches: dict[int, int]
    notes: list[str]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the result as plain dicts, lists and scalars (JSON serialisable)."""
        data = asdict(self)
        data["prefix_mismatches"] = {str(k): v for k, v in self.prefix_mismatches.items()}
        return data

    def metrics_at(self, k: int) -> KMetrics:
        """Return the :class:`KMetrics` for ``k``; raises KeyError when ``k`` was not evaluated."""
        for item in self.per_k:
            if item.k == k:
                return item
        raise KeyError(f"k={k} was not evaluated; ks = {[m.k for m in self.per_k]}")


class QueryEmbedderLike(Protocol):
    """What :func:`evaluate` needs from an embedder (``embedding.Embedder`` or a test double)."""

    def embed_queries(
        self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dim)`` with unit rows."""


#: A function that returns the :class:`Filters` to use for one claim (oracle runs).
ClaimFilters = Callable[[ClaimTruth], "Filters | None"]

# ---------------------------------------------------------------------------------------------
# pure metric functions (no database)
# ---------------------------------------------------------------------------------------------


def _check_k(k: int) -> None:
    """Raise ValueError unless ``k`` is a positive int."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}")


def article_recall_at_k(hit_pages: Sequence[int], gold_pages: Collection[int], k: int) -> bool:
    """Return True when one of the first ``k`` hit page ids is a gold page.

    ``hit_pages`` is the page id of every hit in search order (a page may repeat). A ``k``
    larger than the number of hits uses every hit; an empty ``gold_pages`` is never recalled.
    """
    _check_k(k)
    gold = set(gold_pages)
    return any(page in gold for page in hit_pages[:k])


def gold_page_rank(hit_pages: Sequence[int], gold_pages: Collection[int]) -> int | None:
    """Return the 1-based rank of the first hit on a gold page, or None when there is none."""
    gold = set(gold_pages)
    for rank, page in enumerate(hit_pages, start=1):
        if page in gold:
            return rank
    return None


def covered_units(hit_units: Sequence[Collection[int]], k: int) -> set[int]:
    """Return the union of the sentence ids of the first ``k`` hits.

    ``hit_units`` holds, per hit in search order, the sentence ids the chunk contains.
    """
    _check_k(k)
    covered: set[int] = set()
    for units in hit_units[:k]:
        covered.update(units)
    return covered


def _set_covered(gold_set: Collection[int], covered: set[int]) -> bool:
    """Return True when ``gold_set`` is non-empty and every unit of it is in ``covered``."""
    units = set(gold_set)
    return bool(units) and units <= covered


def evidence_recall_at_k(
    hit_units: Sequence[Collection[int]], sentence_only_sets: Sequence[Collection[int]], k: int
) -> bool:
    """Return True when every unit of at least one sentence-only set is in the first ``k`` hits.

    ``sentence_only_sets`` are the alternative gold sets of one claim (sentence ids); an empty
    list, or only empty sets, gives False. The caller decides which claims count (those with at
    least one set).
    """
    covered = covered_units(hit_units, k)
    return any(_set_covered(gold_set, covered) for gold_set in sentence_only_sets)


def unit_coverage_at_k(
    hit_units: Sequence[Collection[int]], sentence_only_sets: Sequence[Collection[int]], k: int
) -> float:
    """Return the best share, over the sentence-only sets, of a set's units in the first ``k`` hits.

    The value is between 0 and 1 and equals 1.0 exactly when :func:`evidence_recall_at_k` is
    True. Without a (non-empty) set the coverage is 0.0.
    """
    covered = covered_units(hit_units, k)
    shares = [
        len(set(gold_set) & covered) / len(set(gold_set))
        for gold_set in sentence_only_sets
        if len(gold_set) > 0
    ]
    return max(shares) if shares else 0.0


def evidence_rank(
    hit_units: Sequence[Collection[int]], sentence_only_sets: Sequence[Collection[int]]
) -> int | None:
    """Return the smallest ``k`` for which :func:`evidence_recall_at_k` holds, or None."""
    covered: set[int] = set()
    for rank, units in enumerate(hit_units, start=1):
        covered.update(units)
        if any(_set_covered(gold_set, covered) for gold_set in sentence_only_sets):
            return rank
    return None


def latency_stats(samples_ms: Sequence[float]) -> LatencyStats:
    """Return p50 / p95 (linear interpolation), mean, min and max of the samples, in ms.

    Raises ValueError for an empty sample list.
    """
    if len(samples_ms) == 0:
        raise ValueError("latency_stats needs at least one sample")
    arr = np.asarray(samples_ms, dtype=np.float64)
    p50, p95 = np.percentile(arr, PERCENTILES)
    return LatencyStats(
        n=int(arr.size),
        p50_ms=float(p50),
        p95_ms=float(p95),
        mean_ms=float(arr.mean()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )


# ---------------------------------------------------------------------------------------------
# ground truth and lookups
# ---------------------------------------------------------------------------------------------


def load_ground_truth(conn: pymysql.Connection) -> list[ClaimTruth]:
    """Return one :class:`ClaimTruth` per row of ``claim``, ordered by ``claim_id``.

    Gold pages are the resolved ``claim_evidence.page_id`` values of every evidence set. An
    evidence set is sentence-only when every one of its elements is a ``sentence`` or ``item``
    with a resolved ``sentence_id``; a set with a cell or caption, or with a text unit that did
    not resolve, is left out. A claim with no evidence rows has empty gold pages and no sets.
    """
    with conn.cursor() as cur:
        cur.execute(CLAIMS_SQL)
        claim_rows = cur.fetchall()
        cur.execute(EVIDENCE_SQL)
        evidence_rows = cur.fetchall()
    gold_pages: dict[int, set[int]] = defaultdict(set)
    gold_titles: dict[int, set[str]] = defaultdict(set)
    elements: dict[tuple[int, int], list[tuple[str, int | None]]] = defaultdict(list)
    for claim_id, evidence_set, element_type, page_id, title, sentence_id in evidence_rows:
        cid = int(claim_id)
        if page_id is not None:
            gold_pages[cid].add(int(page_id))
            gold_titles[cid].add(str(title))
        elements[(cid, int(evidence_set))].append(
            (str(element_type), None if sentence_id is None else int(sentence_id))
        )
    sentence_only: dict[int, list[set[int]]] = defaultdict(list)
    for (cid, _), members in elements.items():  # insertion order = evidence_set order
        if all(kind in TEXT_UNIT_TYPES and sid is not None for kind, sid in members):
            sentence_only[cid].append({sid for _, sid in members if sid is not None})
    return [
        ClaimTruth(
            claim_id=int(claim_id),
            split=str(split),
            label=str(label),
            text=str(text),
            gold_pages=set(gold_pages.get(int(claim_id), set())),
            sentence_only_sets=[set(s) for s in sentence_only.get(int(claim_id), [])],
            gold_titles=set(gold_titles.get(int(claim_id), set())),
        )
        for claim_id, split, text, label in claim_rows
    ]


def oracle_title_filters(truth: ClaimTruth) -> Filters:
    """Return :class:`Filters` that restrict the hits to the claim's gold pages (an oracle bound).

    With this as ``claim_filters`` the evaluation asks whether the right chunk surfaces once
    the page is known. A claim without a resolved gold page gets ``titles=()`` and no hits.
    """
    return Filters(titles=sorted(truth.gold_titles))


def _placeholders(n: int) -> str:
    """Return ``n`` comma-separated ``%s`` placeholders (``n`` >= 1)."""
    if n < 1:
        raise ValueError("an IN list needs at least one placeholder")
    return ", ".join(["%s"] * n)


def chunk_sentence_ids(conn: pymysql.Connection, chunk_ids: Collection[int]) -> dict[int, set[int]]:
    """Return ``{chunk_id: {sentence_id, ...}}`` through ``chunk_sentence`` for the given chunks.

    Every requested id is a key (a chunk without sentences maps to an empty set). The lookup is
    sent in batches of :data:`CHUNK_ID_BATCH` ids; an empty request runs no statement.
    """
    ids = sorted({int(cid) for cid in chunk_ids})
    result: dict[int, set[int]] = {cid: set() for cid in ids}
    with conn.cursor() as cur:
        for start in range(0, len(ids), CHUNK_ID_BATCH):
            batch = ids[start : start + CHUNK_ID_BATCH]
            sql = CHUNK_SENTENCES_PREFIX + _placeholders(len(batch)) + CHUNK_SENTENCES_SUFFIX
            cur.execute(sql, tuple(batch))
            for chunk_id, sentence_id in cur.fetchall():
                result[int(chunk_id)].add(int(sentence_id))
    return result


def read_ingest_meta(conn: pymysql.Connection) -> dict[str, str]:
    """Return the ``ingest_meta`` table as ``{key: value}`` (empty before an ingest)."""
    with conn.cursor() as cur:
        cur.execute(INGEST_META_SQL)
        return {str(key): str(value) for key, value in cur.fetchall()}


def _corpus_counts(conn: pymysql.Connection) -> dict[str, int]:
    """Return the row counts of ``page``, ``chunk`` and ``sentence``."""
    with conn.cursor() as cur:
        cur.execute(CORPUS_COUNTS_SQL)
        n_pages, n_chunks, n_sentences = cur.fetchone()
    return {"n_pages": int(n_pages), "n_chunks": int(n_chunks), "n_sentences": int(n_sentences)}


def _server_version(conn: pymysql.Connection) -> str:
    """Return the server's ``VERSION()`` string."""
    with conn.cursor() as cur:
        cur.execute(VERSION_SQL)
        return str(cur.fetchone()[0])


def global_cache_size(conn: pymysql.Connection) -> int:
    """Return the global ``mhnsw_max_cache_size`` in bytes (the HNSW graph cache limit)."""
    with conn.cursor() as cur:
        cur.execute(CACHE_SIZE_SQL)
        return int(cur.fetchone()[0])


def parse_vector_index(create_table: str) -> dict[str, Any]:
    """Return ``{"index_m": int | None, "index_distance": str | None}`` from a CREATE TABLE text.

    The values are the ``M`` and ``DISTANCE`` options of the vector index as ``SHOW CREATE
    TABLE`` prints them; an option that is not printed (server default) gives None.
    """
    m_match = _INDEX_M_RE.search(create_table)
    distance_match = _INDEX_DISTANCE_RE.search(create_table)
    return {
        "index_m": int(m_match.group(1)) if m_match else None,
        "index_distance": distance_match.group(1).lower() if distance_match else None,
    }


def vector_index_info(conn: pymysql.Connection) -> dict[str, Any]:
    """Return :func:`parse_vector_index` of ``SHOW CREATE TABLE chunk`` on this connection."""
    with conn.cursor() as cur:
        cur.execute(SHOW_CREATE_CHUNK_SQL)
        row = cur.fetchone()
    return parse_vector_index(str(row[1]) if row else "")


# ---------------------------------------------------------------------------------------------
# machine and versions
# ---------------------------------------------------------------------------------------------


def _cpu_model() -> str:
    """Return the CPU model from /proc/cpuinfo, else ``platform.processor()`` or ``"unknown"``."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _memory_gb() -> float | None:
    """Return the total memory in GB (from /proc/meminfo, one decimal), or None."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kib = float(line.split()[1])
                    return round(kib / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gpu_name(device: str | None) -> str | None:
    """Return the CUDA device name when ``device`` is a cuda device and torch sees it, else None."""
    if not device or not device.startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return str(torch.cuda.get_device_name(index))
    except (ImportError, RuntimeError, ValueError, AssertionError):
        return None


def machine_info(embedder: object | None = None) -> dict[str, Any]:
    """Return the platform, CPU, core count, memory, embedding device and GPU of this machine.

    The device is read from ``embedder.device`` when the embedder has it; the GPU name is
    looked up only for a cuda device.
    """
    device = getattr(embedder, "device", None)
    device = str(device) if device is not None else None
    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "embedding_device": device,
        "gpu": _gpu_name(device),
    }


def versions_info(conn: pymysql.Connection) -> dict[str, str]:
    """Return the Python, WikiLense, MariaDB and library versions (``"not installed"`` if absent)."""
    versions = {
        "python": platform.python_version(),
        "wikilense": __version__,
        "mariadb": _server_version(conn),
    }
    for dist in VERSIONED_DISTRIBUTIONS:
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = "not installed"
    return versions


# ---------------------------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------------------------


def _check_eval_args(
    ks: Sequence[int],
    repeats: int,
    strategy: str,
    filters: Filters | None,
    claim_filters: ClaimFilters | None,
    overfetch: int,
    ef_search: int | str | None = None,
) -> tuple[int, ...]:
    """Validate the arguments of :func:`evaluate` and return ``ks`` sorted, without duplicates.

    ``strategy`` must be one of ``search.STRATEGIES`` (read at call time); ``ef_search`` must be
    None, a positive int or :data:`SERVER_DEFAULT_EF_SEARCH`. Raises ValueError / TypeError
    with the offending argument in the message.
    """
    if isinstance(ks, (str, bytes)) or len(ks) == 0:
        raise ValueError("ks must be a non-empty sequence of positive ints")
    for k in ks:
        _check_k(k)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"repeats must be a positive int, got {repeats!r}")
    if isinstance(overfetch, bool) or not isinstance(overfetch, int) or overfetch < 1:
        raise ValueError(f"overfetch must be a positive int, got {overfetch!r}")
    strategies = tuple(searchmod.STRATEGIES)
    if strategy not in strategies:
        raise ValueError(f"unknown strategy {strategy!r}; choose one of {strategies}")
    explicit = ef_search is not None and ef_search != SERVER_DEFAULT_EF_SEARCH
    if explicit and (
        isinstance(ef_search, bool) or not isinstance(ef_search, int) or ef_search < 1
    ):
        raise ValueError(
            f"ef_search must be a positive int, None (settings.ef_search) or "
            f"{SERVER_DEFAULT_EF_SEARCH!r}, got {ef_search!r}"
        )
    if filters is not None and not isinstance(filters, Filters):
        raise TypeError(f"filters must be a Filters or None, got {type(filters).__name__}")
    if claim_filters is not None and not callable(claim_filters):
        raise TypeError("claim_filters must be a function of one ClaimTruth returning Filters")
    if filters is not None and not filters.is_empty() and claim_filters is not None:
        raise ValueError("give either filters or claim_filters, not both")
    if strategy == "none" and (
        (filters is not None and not filters.is_empty()) or claim_filters is not None
    ):
        others = ", ".join(repr(name) for name in strategies if name != "none")
        raise ValueError(f"strategy 'none' ignores filters; use one of {others} with them")
    return tuple(sorted(set(ks)))


def _resolve_ef_search(
    ef_search: int | str | None, settings: Settings | None
) -> tuple[int | None, str]:
    """Return ``(value to set for the run or None, source)`` for the ``ef_search`` argument.

    An int is used as given (source ``"argument"``); None takes ``settings.ef_search``, loading
    the settings when none were passed (source ``"settings"``); :data:`SERVER_DEFAULT_EF_SEARCH`
    leaves the session untouched (source ``"server"``). Raises ``config.SettingsError`` when the
    settings have to be loaded and cannot be.
    """
    if ef_search == SERVER_DEFAULT_EF_SEARCH:
        return None, "server"
    if ef_search is None:
        if settings is None:
            settings = load_settings()
        return int(settings.ef_search), "settings"
    return int(ef_search), "argument"


def _search_kwargs(query_text: str) -> dict[str, str]:
    """Return ``{"query_text": query_text}`` when ``search.search`` accepts it, else ``{}``.

    The signature is inspected so that the harness runs with a search function that has no
    ``query_text`` parameter yet (or a test double without one).
    """
    parameters = inspect.signature(searchmod.search).parameters
    if "query_text" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return {"query_text": query_text}
    return {}


def _embed_claims(
    embedder: QueryEmbedderLike, texts: Sequence[str], repeats: int
) -> tuple[np.ndarray, list[float]]:
    """Return the batched query vectors and the per-claim embedding times in ms.

    One warm-up call loads the model; then all texts are embedded in one batch (these vectors
    are used for every search), and finally every text is embedded on its own ``repeats``
    times with the wall-clock time of each call recorded.
    """
    embedder.embed_queries([texts[0]])
    vectors = np.asarray(embedder.embed_queries(texts), dtype=np.float32)
    if vectors.shape != (len(texts), db.VECTOR_DIM):
        raise ValueError(
            f"embedder returned shape {vectors.shape}, expected {(len(texts), db.VECTOR_DIM)} "
            f"(chunk.embedding is VECTOR({db.VECTOR_DIM}))"
        )
    samples: list[float] = []
    for _ in range(repeats):
        for text in texts:
            start = time.perf_counter()
            embedder.embed_queries([text])
            samples.append((time.perf_counter() - start) * 1000.0)
    return vectors, samples


def _run_searches(
    conn: pymysql.Connection,
    truths: Sequence[ClaimTruth],
    vectors: np.ndarray,
    ks: tuple[int, ...],
    repeats: int,
    strategy: str,
    filters: Filters | None,
    claim_filters: ClaimFilters | None,
    overfetch: int,
) -> tuple[list[dict[int, list[Hit]]], dict[int, list[float]], list[int], dict[int, int]]:
    """Run every (claim, k) search once untimed, then ``repeats`` times timed.

    Every call gets the claim text as ``query_text`` when ``search.search`` takes it (see
    :func:`_search_kwargs`). Returns, per claim, the hits of every ``LIMIT k`` search of the
    first timed pass (``{k: hits}``), the SQL times in ms per ``k``, the ids of the claims whose
    ``max(ks)`` hits changed between passes, and the number of claims per ``k`` whose ``LIMIT
    k`` hits are not the first ``k`` of ``max(ks)``.
    """
    max_k = ks[-1]
    per_claim_filters = [
        claim_filters(truth) if claim_filters is not None else filters for truth in truths
    ]
    per_claim_kwargs = [_search_kwargs(truth.text) for truth in truths]
    hits_first: list[dict[int, list[Hit]]] = [{} for _ in truths]
    sql_ms: dict[int, list[float]] = {k: [] for k in ks}
    unstable: set[int] = set()
    prefix_mismatches: dict[int, int] = {k: 0 for k in ks if k != max_k}
    for pass_index in range(-1, repeats):  # -1 is the untimed warm-up pass
        for index, truth in enumerate(truths):
            hits_by_k: dict[int, list[Hit]] = {}
            for k in ks:
                start = time.perf_counter()
                hits = searchmod.search(
                    conn,
                    vectors[index],
                    k=k,
                    filters=per_claim_filters[index],
                    strategy=strategy,
                    overfetch=overfetch,
                    **per_claim_kwargs[index],
                )
                elapsed = (time.perf_counter() - start) * 1000.0
                if pass_index < 0:
                    continue
                sql_ms[k].append(elapsed)
                hits_by_k[k] = hits
            if pass_index < 0:
                continue
            ids_max = [h.chunk_id for h in hits_by_k[max_k]]
            if pass_index == 0:
                hits_first[index] = hits_by_k
                for k in prefix_mismatches:
                    if [h.chunk_id for h in hits_by_k[k]] != ids_max[:k]:
                        prefix_mismatches[k] += 1
            elif ids_max != [h.chunk_id for h in hits_first[index][max_k]]:
                unstable.add(truth.claim_id)
    return hits_first, sql_ms, sorted(unstable), prefix_mismatches


def evaluate(
    conn: pymysql.Connection,
    embedder: QueryEmbedderLike,
    ks: Sequence[int] = DEFAULT_KS,
    repeats: int = DEFAULT_REPEATS,
    strategy: str = DEFAULT_STRATEGY,
    filters: Filters | None = None,
    ef_search: int | str | None = None,
    overfetch: int = DEFAULT_OVERFETCH,
    claim_filters: ClaimFilters | None = None,
    settings: Settings | None = None,
) -> EvalResult:
    """Run the recall and latency harness over every claim in ``conn``'s database.

    ``ks`` are the cut-offs (sorted, duplicates dropped); ``repeats`` the number of timed
    passes after one warm-up pass; ``strategy`` (one of ``search.STRATEGIES``), ``filters`` and
    ``overfetch`` go to ``search.search`` unchanged, together with the claim text as
    ``query_text`` (``filters`` apply to every claim; ``claim_filters`` is a function returning
    the Filters for one claim, e.g. :func:`oracle_title_filters`, and cannot be combined with
    non-empty ``filters``). ``ef_search`` sets ``mhnsw_ef_search`` for the whole run (restored
    afterwards): an int is used as given, None (the default) takes ``settings.ef_search``
    (``settings`` defaults to ``load_settings()``), and :data:`SERVER_DEFAULT_EF_SEARCH` keeps
    the server's value; the effective session value, its source, the global
    ``mhnsw_max_cache_size`` and the vector index's ``M`` and ``DISTANCE`` are recorded in the
    parameters. The metrics and the SQL latency at ``k`` both come from the ``LIMIT k`` query
    of each claim; the per-claim ranks in ``claims`` from the ``LIMIT max(ks)`` query. Returns
    the :class:`EvalResult`.

    Raises ValueError for invalid ``ks`` / ``repeats`` / ``overfetch`` / ``strategy`` /
    ``ef_search``, for filters combined with strategy ``none``, when the database holds no
    claims, or when the embedder's vectors do not have ``db.VECTOR_DIM`` dimensions; TypeError
    for a ``filters`` value that is not a Filters; ``config.SettingsError`` when the settings
    are needed for ``ef_search`` and cannot be loaded.
    """
    ks_sorted = _check_eval_args(
        ks, repeats, strategy, filters, claim_filters, overfetch, ef_search
    )
    ef_requested, ef_source = _resolve_ef_search(ef_search, settings)
    truths = load_ground_truth(conn)
    if not truths:
        raise ValueError("the claim table is empty: run the ingest before evaluating")
    counts = _corpus_counts(conn)
    if counts["n_chunks"] == 0:
        raise ValueError("the chunk table is empty: run the ingest before evaluating")
    meta = read_ingest_meta(conn)
    index_info = vector_index_info(conn)
    cache_size = global_cache_size(conn)
    logger.info(
        "evaluate: %d claims (%d with a sentence-only set), %d chunks, ks=%s, repeats=%d, "
        "strategy=%s, ef_search=%s (%s), index M=%s DISTANCE=%s, cache %d bytes",
        len(truths),
        sum(1 for t in truths if t.eligible),
        counts["n_chunks"],
        list(ks_sorted),
        repeats,
        strategy,
        ef_requested,
        ef_source,
        index_info["index_m"],
        index_info["index_distance"],
        cache_size,
    )

    texts = [truth.text for truth in truths]
    embed_start = time.perf_counter()
    vectors, embed_samples = _embed_claims(embedder, texts, repeats)
    logger.info("embedded %d claims in %.2f s", len(texts), time.perf_counter() - embed_start)

    with ef_search_session(conn, ef_requested):
        ef_effective = db.get_session_var(conn, EF_SEARCH_VARIABLE)
        sql_start = time.perf_counter()
        hits_first, sql_ms, unstable, prefix_mismatches = _run_searches(
            conn, truths, vectors, ks_sorted, repeats, strategy, filters, claim_filters, overfetch
        )
    logger.info(
        "searched %d claims x %d ks x %d passes in %.2f s",
        len(truths),
        len(ks_sorted),
        repeats + 1,
        time.perf_counter() - sql_start,
    )

    all_chunk_ids = {
        hit.chunk_id for by_k in hits_first for hits in by_k.values() for hit in hits
    }
    units_of = chunk_sentence_ids(conn, all_chunk_ids)

    n_eligible = 0
    article_hits = dict.fromkeys(ks_sorted, 0)
    evidence_hits = dict.fromkeys(ks_sorted, 0)
    coverage_sum = dict.fromkeys(ks_sorted, 0.0)
    outcomes: list[ClaimOutcome] = []
    for truth, by_k in zip(truths, hits_first):
        # The metrics at k use the LIMIT k query, the statement whose latency is reported at k.
        for k in ks_sorted:
            hit_pages_k = [hit.page_id for hit in by_k[k]]
            article_hits[k] += int(article_recall_at_k(hit_pages_k, truth.gold_pages, k))
        if truth.eligible:
            n_eligible += 1
            for k in ks_sorted:
                units_k = [units_of.get(hit.chunk_id, set()) for hit in by_k[k]]
                sets = truth.sentence_only_sets
                evidence_hits[k] += int(evidence_recall_at_k(units_k, sets, k))
                coverage_sum[k] += unit_coverage_at_k(units_k, sets, k)
        # The per-claim diagnostics (ranks, hit ids) come from the LIMIT max(ks) query.
        hits = by_k[ks_sorted[-1]]
        hit_pages = [hit.page_id for hit in hits]
        hit_units = [units_of.get(hit.chunk_id, set()) for hit in hits]
        outcomes.append(
            ClaimOutcome(
                claim_id=truth.claim_id,
                split=truth.split,
                label=truth.label,
                text=truth.text,
                gold_titles=tuple(sorted(truth.gold_titles)),
                eligible=truth.eligible,
                gold_page_rank=gold_page_rank(hit_pages, truth.gold_pages),
                evidence_rank=(
                    evidence_rank(hit_units, truth.sentence_only_sets) if truth.eligible else None
                ),
                unit_coverage=(
                    unit_coverage_at_k(hit_units, truth.sentence_only_sets, ks_sorted[-1])
                    if truth.eligible
                    else 0.0
                ),
                hit_chunk_ids=tuple(hit.chunk_id for hit in hits),
            )
        )
    n_claims = len(truths)
    per_k = [
        KMetrics(
            k=k,
            article_hits=article_hits[k],
            article_recall=article_hits[k] / n_claims,
            evidence_hits=evidence_hits[k],
            evidence_recall=(evidence_hits[k] / n_eligible) if n_eligible else 0.0,
            unit_coverage=(coverage_sum[k] / n_eligible) if n_eligible else 0.0,
            sql_latency=latency_stats(sql_ms[k]),
        )
        for k in ks_sorted
    ]
    parameters: dict[str, Any] = {
        "ks": list(ks_sorted),
        "max_k": ks_sorted[-1],
        "repeats": repeats,
        "strategy": strategy,
        "overfetch": overfetch if strategy in OVERFETCH_STRATEGIES else None,
        "filters": asdict(filters) if filters is not None and not filters.is_empty() else None,
        "claim_filters": getattr(claim_filters, "__name__", None) if claim_filters else None,
        "ef_search": ef_requested,
        "ef_search_source": ef_source,
        "ef_search_effective": ef_effective,
        "mhnsw_max_cache_size": cache_size,
        **index_info,
        "embedding_model": getattr(embedder, "model_name", None),
        "embedding_device": getattr(embedder, "device", None),
        "n_claims": n_claims,
        "n_evidence_claims": n_eligible,
        **counts,
    }
    notes = [
        (
            "article_recall = article_hits / n_claims; evidence_recall = evidence_hits / "
            "n_evidence_claims (claims with a sentence-only evidence set whose units all "
            "resolved); unit_coverage = mean over the same claims of the best share of a set's "
            "units covered."
        ),
        (
            "Ranking: the metrics at k use the hits of the LIMIT k search of each claim (first "
            "timed pass), in the order search() returned them; the per-claim ranks use the "
            "LIMIT max_k search."
        ),
        (
            "sql_latency: wall-clock milliseconds of one search() call (LIMIT k) as seen by the "
            "Python client, repeats x n_claims samples after one untimed warm-up pass; p50 and "
            "p95 use linear interpolation."
        ),
        (
            "embedding_latency: wall-clock milliseconds of one embed_queries([text]) call per "
            "claim and repeat after a warm-up call; the searches use vectors from one batched "
            "call."
        ),
    ]
    return EvalResult(
        parameters=parameters,
        ingest_meta=meta,
        machine=machine_info(embedder),
        versions=versions_info(conn),
        n_claims=n_claims,
        n_evidence_claims=n_eligible,
        per_k=per_k,
        embedding_latency=latency_stats(embed_samples),
        claims=outcomes,
        unstable_claims=unstable,
        prefix_mismatches=prefix_mismatches,
        notes=notes,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------


def worst_claims(result: EvalResult, n: int = 10) -> list[ClaimOutcome]:
    """Return up to ``n`` claims with the worst rank of their gold page (missing first, then largest).

    Ties keep the claim id order.
    """
    ordered = sorted(
        result.claims,
        key=lambda c: (0, 0) if c.gold_page_rank is None else (1, -c.gold_page_rank),
    )
    return ordered[:n]


def _md_cell(value: Any) -> str:
    """Return ``value`` as one Markdown table cell (None as ``-``, pipes escaped, no newlines)."""
    text = "-" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Return a Markdown table with the given header cells and rows."""
    lines = [
        "| " + " | ".join(_md_cell(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(v) for v in row) + " |")
    return "\n".join(lines)


def _md_latency_rows(stats: LatencyStats) -> list[Any]:
    """Return the cells of one latency row: n, p50, p95, mean, min, max (ms, 2 decimals)."""
    return [
        stats.n,
        f"{stats.p50_ms:.2f}",
        f"{stats.p95_ms:.2f}",
        f"{stats.mean_ms:.2f}",
        f"{stats.min_ms:.2f}",
        f"{stats.max_ms:.2f}",
    ]


def results_markdown(result: EvalResult, name: str = DEFAULT_NAME) -> str:
    """Return the Markdown report: parameters first, then one table per metric."""
    parts = [
        f"# WikiLense evaluation: {name}",
        "",
        (
            f"Generated {result.created_at}. {result.n_claims} claims; "
            f"{result.n_evidence_claims} with a sentence-only evidence set (the denominator of "
            "evidence recall and unit coverage)."
        ),
        "",
        "## Parameters",
        "",
        _md_table(
            ("Parameter", "Value"),
            [(key, json.dumps(value) if isinstance(value, (dict, list)) else value)
             for key, value in result.parameters.items()],
        ),
        "",
        "## Ingest parameters (ingest_meta)",
        "",
        _md_table(("Key", "Value"), list(result.ingest_meta.items())),
        "",
        "## Machine and versions",
        "",
        _md_table(
            ("Item", "Value"),
            [*result.machine.items(), *result.versions.items()],
        ),
        "",
        "## Article recall@k",
        "",
        "A claim counts when a gold page is among the pages of its first k chunks.",
        "",
        _md_table(
            ("k", "claims recalled", "of", "article recall"),
            [(m.k, m.article_hits, result.n_claims, f"{m.article_recall:.3f}") for m in result.per_k],
        ),
        "",
        "## Evidence recall@k",
        "",
        (
            "A claim counts when every unit of one of its sentence-only evidence sets is inside "
            "its first k chunks."
        ),
        "",
        _md_table(
            ("k", "claims recalled", "of", "evidence recall"),
            [
                (m.k, m.evidence_hits, result.n_evidence_claims, f"{m.evidence_recall:.3f}")
                for m in result.per_k
            ],
        ),
        "",
        "## Unit coverage@k",
        "",
        (
            "Mean, over the same claims, of the best share of a set's gold units inside the "
            "first k chunks."
        ),
        "",
        _md_table(
            ("k", "of", "unit coverage"),
            [(m.k, result.n_evidence_claims, f"{m.unit_coverage:.3f}") for m in result.per_k],
        ),
        "",
        "## SQL latency per k (ms)",
        "",
        "One search() call with LIMIT k, measured by the client; warm.",
        "",
        _md_table(
            ("k", "n", "p50", "p95", "mean", "min", "max"),
            [[m.k, *_md_latency_rows(m.sql_latency)] for m in result.per_k],
        ),
        "",
        "## Query embedding latency (ms)",
        "",
        "One embed_queries([text]) call per claim and repeat; warm.",
        "",
        _md_table(
            ("n", "p50", "p95", "mean", "min", "max"),
            [_md_latency_rows(result.embedding_latency)],
        ),
        "",
        "## Claims with the worst rank of their gold page",
        "",
        (
            f"Rank within the first {result.parameters.get('max_k')} chunks; "
            "\"not in top k\" means no chunk of a gold page was retrieved."
        ),
        "",
        _md_table(
            ("claim", "label", "gold page", "gold page rank", "evidence rank", "claim text"),
            [
                (
                    c.claim_id,
                    c.label,
                    ", ".join(c.gold_titles) or "(unresolved)",
                    c.gold_page_rank if c.gold_page_rank is not None else "not in top k",
                    c.evidence_rank if c.evidence_rank is not None
                    else ("not in top k" if c.eligible else "n/a"),
                    c.text[:80],
                )
                for c in worst_claims(result)
            ],
        ),
        "",
    ]
    if result.unstable_claims:
        parts.append(
            f"Claims whose top-{result.parameters.get('max_k')} hits changed between repeats: "
            + ", ".join(str(c) for c in result.unstable_claims)
            + "."
        )
        parts.append("")
    mismatches = {k: v for k, v in result.prefix_mismatches.items() if v}
    if mismatches:
        parts.append(
            "Claims whose LIMIT k hits were not the first k of LIMIT max_k, per k: "
            + json.dumps({str(k): v for k, v in mismatches.items()})
            + "."
        )
        parts.append("")
    parts.append("## Notes")
    parts.append("")
    parts.extend(f"- {note}" for note in result.notes)
    parts.append("")
    return "\n".join(parts)


def write_results(
    result: EvalResult, out_dir: str | Path = DEFAULT_RESULTS_DIR, name: str = DEFAULT_NAME
) -> tuple[Path, Path]:
    """Write ``<name>.json`` and ``<name>.md`` into ``out_dir`` and return their paths.

    ``out_dir`` is created when missing. ``name`` must be a plain file stem (no directory
    separators, not empty); otherwise ValueError.
    """
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"name must be a plain file stem without separators, got {name!r}")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{name}.json"
    md_path = directory / f"{name}.md"
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(results_markdown(result, name), encoding="utf-8")
    return json_path, md_path


def run_eval(
    settings: Settings,
    ks: Sequence[int] = DEFAULT_KS,
    repeats: int = DEFAULT_REPEATS,
    strategy: str = DEFAULT_STRATEGY,
    filters: Filters | None = None,
    ef_search: int | str | None = None,
    overfetch: int = DEFAULT_OVERFETCH,
    claim_filters: ClaimFilters | None = None,
    out_dir: str | Path = DEFAULT_RESULTS_DIR,
    name: str = DEFAULT_NAME,
    device: str | None = None,
) -> tuple[EvalResult, Path, Path]:
    """Evaluate ``settings.db_name`` with ``embedding.Embedder(settings.embedding_model)``.

    Opens the connection, runs :func:`evaluate` (with ``settings`` for the ``ef_search``
    default), writes the results with :func:`write_results` and returns
    ``(result, json_path, md_path)``. ``device`` is passed to the Embedder. The connection is
    closed afterwards; nothing is committed.
    """
    from wikilense.embedding import Embedder

    embedder = Embedder(settings.embedding_model, device=device)
    conn = db.connect(settings)
    try:
        result = evaluate(
            conn,
            embedder,
            ks=ks,
            repeats=repeats,
            strategy=strategy,
            filters=filters,
            ef_search=ef_search,
            overfetch=overfetch,
            claim_filters=claim_filters,
            settings=settings,
        )
    finally:
        conn.close()
    json_path, md_path = write_results(result, out_dir=out_dir, name=name)
    return result, json_path, md_path
