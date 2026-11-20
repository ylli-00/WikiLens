"""The web page's benchmark-claim picker: which claims a database offers, their gold evidence, and
the gold marks and recall summary of one search, by the evaluation's own definitions.

Claim files (:data:`CLAIM_FILES`), in two formats:

- ``data/corpus/claims.jsonl``: the demo corpus's 75 claims in FEVEROUS's format, the evidence
  sets as ``evidence[i]["content"]``;
- ``data/scaled/claims_3000.jsonl``: the scaled run's 3,000 claims, the evidence sets as
  ``evidence_sets[i]["elements"]`` (``wikilense.llm.claims`` has the format).

:func:`read_claim_files` reads both into :class:`FileClaim` records. A claim that is in both files
(the same FEVEROUS id) must have the same text, label and evidence sets in both; it is kept once,
with both files as its sources.

:func:`build_catalog` decides what one database offers:

1. A claim file *belongs* to the database when the database's ``claim`` table holds exactly the
   file's claim ids: the ingest writes that table from the claims file of the corpus it loads.
   When no file belongs, the catalog is empty and the page hides the picker.
2. A claim of either file is *offered* when every page its evidence cites (every evidence set,
   table sets included) is a title of the ``page`` table and every sentence or list item of its
   evidence resolves to a row of ``sentence``. Elements resolve with ``ingest.resolve_element``,
   the rule ``claim_evidence`` is written with. A claim whose pages are all there but one of whose
   text units does not resolve is left out and listed in ``dropped`` with the reason.
3. The ground truth of an offered claim is ``evaluate.ground_truth_from_rows`` over the resolved
   rows, the function ``evaluate.load_ground_truth`` uses on ``claim_evidence``. For the claims
   of a file that belongs to the database it is also compared with ``load_ground_truth`` on the
   database itself; a claim where the two differ is dropped with the reason.

:func:`search_marks` gives the marks of one search's hits and the claim's summary at ``k``:
``evaluate.article_recall_at_k``, ``gold_page_rank``, ``covered_units``,
``evidence_recall_at_k`` and ``unit_coverage_at_k`` on the hits in search order, with each
chunk's sentence ids from ``evaluate.chunk_sentence_ids``, as the evaluation computes them.

The ready-made examples of the page are fixed claims and inputs (:data:`EXAMPLES`), chosen by
the rule of :func:`pick_examples` from the claims offered on both the demo database and the
scaled database S1; ``EXAMPLE_*`` hold the result, so the examples are the same on every server,
also one without S1.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymysql

from wikilense.config import REPO_ROOT
from wikilense.corpus import iter_claims, parse_element_id
from wikilense.evaluate import (
    TEXT_UNIT_TYPES,
    ClaimTruth,
    article_recall_at_k,
    chunk_sentence_ids,
    covered_units,
    evidence_recall_at_k,
    gold_page_rank,
    ground_truth_from_rows,
    load_ground_truth,
    unit_coverage_at_k,
)
from wikilense.ingest import resolve_element
from wikilense.search import Hit

logger = logging.getLogger(__name__)

#: The claim files the picker reads, in the order the picker lists their claims.
CLAIM_FILES: tuple[Path, ...] = (
    REPO_ROOT / "data" / "corpus" / "claims.jsonl",
    REPO_ROOT / "data" / "scaled" / "claims_3000.jsonl",
)

LABELS: tuple[str, ...] = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")

#: What :func:`claim_kind` returns, in words for the page.
KIND_TEXT: dict[str, str] = {
    "single": "one page",
    "needs2": "two or more pages (every evidence set)",
    "either": "one page or several, depending on the evidence set",
    "none": "no evidence",
}

# Every value is a parameter; the only variable text is the length of the IN list.
PAGE_TITLES_SQL = "SELECT page_id, title FROM page"
CLAIM_IDS_SQL = "SELECT claim_id FROM claim"
SIZE_SQL = "SELECT (SELECT COUNT(*) FROM page), (SELECT COUNT(*) FROM chunk)"
PAGE_LIST_SQL = "SELECT title FROM page ORDER BY title"
UNITS_PREFIX = (
    "SELECT page_id, element_key, sentence_id, text FROM sentence WHERE (page_id, element_key) IN ("
)
UNITS_SUFFIX = ")"
#: Largest number of ``(page_id, element_key)`` pairs sent in one lookup.
UNIT_BATCH = 500

# ---------------------------------------------------------------------------------------------
# claim files
# ---------------------------------------------------------------------------------------------


class ClaimFileError(ValueError):
    """A claim file has a record that cannot be read, or two files disagree on one claim."""


@dataclass(frozen=True)
class FileClaim:
    """One claim as a claim file holds it: the evidence sets are FEVEROUS element ids."""

    claim_id: int
    split: str
    label: str
    text: str
    evidence_sets: tuple[tuple[str, ...], ...]
    sources: tuple[str, ...]


def source_name(path: Path) -> str:
    """Return ``path`` relative to the repository (POSIX form), or its file name outside it."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield the claim records of one file of either format, header lines skipped."""
    with open(path, encoding="utf-8") as handle:
        first = handle.readline()
    if first.strip() and "evidence_sets" in json.loads(first):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from iter_claims(path)


def _evidence_sets(record: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Return the element ids of every evidence set of a record of either format."""
    if "evidence_sets" in record:
        return tuple(tuple(str(e) for e in s["elements"]) for s in record["evidence_sets"])
    return tuple(tuple(str(e) for e in s["content"]) for s in record["evidence"])


def read_claim_files(paths: Sequence[Path]) -> dict[int, FileClaim]:
    """Return ``{claim_id: FileClaim}`` over the files that exist, in file order, then line order.

    A missing file is skipped with a warning. Raises :class:`ClaimFileError` for a line that is not
    JSON, a record without the fields, an evidence element that is not a FEVEROUS element id, or a
    claim id in two files whose text, label or evidence sets differ.
    """
    claims: dict[int, FileClaim] = {}
    for path in paths:
        if not Path(path).is_file():
            logger.warning("claim file %s not found; the picker leaves it out", path)
            continue
        name = source_name(Path(path))
        try:
            records = list(_records(Path(path)))
        except ValueError as exc:  # a line that is not JSON, or a FEVEROUS record without split
            raise ClaimFileError(f"{name}: unreadable claim record: {exc}") from exc
        for record in records:
            try:
                claim = FileClaim(
                    claim_id=int(record["id"]),
                    split=str(record["split"]),
                    label=str(record["label"]),
                    text=str(record["claim"]),
                    evidence_sets=_evidence_sets(record),
                    sources=(name,),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ClaimFileError(f"{name}: unreadable claim record: {exc}") from exc
            for element_id in (e for s in claim.evidence_sets for e in s):
                try:
                    parse_element_id(element_id)
                except ValueError as exc:
                    raise ClaimFileError(f"{name}: claim {claim.claim_id}: {exc}") from exc
            seen = claims.get(claim.claim_id)
            if seen is None:
                claims[claim.claim_id] = claim
                continue
            same = (seen.text, seen.label, seen.evidence_sets) == (
                claim.text, claim.label, claim.evidence_sets
            )
            if not same:
                raise ClaimFileError(
                    f"claim {claim.claim_id} differs between {', '.join(seen.sources)} and {name}"
                )
            if name not in seen.sources:
                claims[claim.claim_id] = FileClaim(
                    seen.claim_id, seen.split, seen.label, seen.text, seen.evidence_sets,
                    seen.sources + (name,),
                )
    return claims


def file_claim_ids(claims: Mapping[int, FileClaim]) -> dict[str, frozenset[int]]:
    """Return ``{claim file: the ids of its claims}`` from the merged claims."""
    ids: dict[str, set[int]] = {}
    for claim in claims.values():
        for source in claim.sources:
            ids.setdefault(source, set()).add(claim.claim_id)
    return {source: frozenset(members) for source, members in ids.items()}


def claim_kind(evidence_sets: Sequence[Sequence[str]]) -> str:
    """Return ``"single"``, ``"needs2"``, ``"either"`` or ``"none"`` for a claim's evidence sets.

    The rule of ``scripts/scaled_build.py`` (``describe``), which gave ``claims_3000.jsonl`` its
    ``kind``: ``"needs2"`` when every non-empty set cites two or more pages, ``"single"`` when all
    sets together cite one page, ``"either"`` otherwise, ``"none"`` without a non-empty set.
    """
    pages = [{parse_element_id(e)[0] for e in s} for s in evidence_sets if s]
    if not pages:
        return "none"
    if all(len(p) >= 2 for p in pages):
        return "needs2"
    if len(set().union(*pages)) == 1:
        return "single"
    return "either"


# ---------------------------------------------------------------------------------------------
# the catalog of one database
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldElement:
    """One element of a gold evidence set, resolved in one database.

    ``sentence_id`` and ``text`` (the ``sentence`` row's text) are set for a sentence or list
    item, and None for a table element (tables are not indexed).
    """

    element_id: str
    page_title: str
    element_type: str
    element_key: str
    page_id: int
    sentence_id: int | None
    text: str | None

    @property
    def in_table(self) -> bool:
        """Return True for a table cell, header cell or caption."""
        return self.element_type not in TEXT_UNIT_TYPES


@dataclass(frozen=True)
class OfferedClaim:
    """A claim the picker offers on one database, with its resolved evidence and ground truth."""

    claim: FileClaim
    truth: ClaimTruth
    sets: tuple[tuple[GoldElement, ...], ...]
    kind: str

    @property
    def claim_id(self) -> int:
        """Return the FEVEROUS claim id."""
        return self.claim.claim_id

    def gold_units(self) -> dict[int, str]:
        """Return ``{sentence_id: element_key}`` of every resolved text unit of every set."""
        return {
            e.sentence_id: e.element_key for s in self.sets for e in s if e.sentence_id is not None
        }

    def listing(self) -> dict[str, Any]:
        """Return the picker's entry: id, label, text, kind and the claim files."""
        return {
            "id": self.claim_id,
            "label": self.claim.label,
            "text": self.claim.text,
            "kind": self.kind,
            "sources": list(self.claim.sources),
        }

    def details(self) -> dict[str, Any]:
        """Return the claim's details for the API: label, kind, sources and the evidence sets."""
        return {
            **self.listing(),
            "kind_text": KIND_TEXT[self.kind],
            "split": self.claim.split,
            "evidence_sets": [
                {
                    "sentence_only": not any(e.in_table for e in s),
                    "elements": [
                        {
                            "element_id": e.element_id,
                            "page": e.page_title,
                            "type": e.element_type,
                            "element_key": e.element_key,
                            "in_table": e.in_table,
                            "text": e.text,
                        }
                        for e in s
                    ],
                }
                for s in self.sets
            ],
        }


@dataclass(frozen=True)
class Catalog:
    """What one database offers: its size, the claim files that belong to it and the claims.

    ``claims`` is in the order of :data:`CLAIM_FILES` (the files that belong to the database
    first), then line order; ``dropped`` lists ``(claim_id, reason)`` for the claims whose pages
    are all in the database but which are left out.
    """

    database: str
    n_pages: int
    n_chunks: int
    own_files: tuple[str, ...]
    claims: dict[int, OfferedClaim]
    dropped: tuple[tuple[int, str], ...]


def database_size(conn: pymysql.Connection) -> tuple[int, int]:
    """Return ``(pages, chunks)`` of the connection's database."""
    with conn.cursor() as cur:
        cur.execute(SIZE_SQL)
        n_pages, n_chunks = cur.fetchone()
    return int(n_pages), int(n_chunks)


def page_titles(conn: pymysql.Connection) -> list[str]:
    """Return every page title of the connection's database, sorted."""
    with conn.cursor() as cur:
        cur.execute(PAGE_LIST_SQL)
        return [str(title) for (title,) in cur.fetchall()]


def _resolve_units(
    conn: pymysql.Connection, pairs: Iterable[tuple[int, str]]
) -> tuple[dict[tuple[int, str], int], dict[int, str]]:
    """Return ``({(page_id, element_key): sentence_id}, {sentence_id: text})`` for the pairs found."""
    wanted = sorted(set(pairs))
    ids: dict[tuple[int, str], int] = {}
    texts: dict[int, str] = {}
    with conn.cursor() as cur:
        for start in range(0, len(wanted), UNIT_BATCH):
            batch = wanted[start : start + UNIT_BATCH]
            sql = UNITS_PREFIX + ", ".join(["(%s, %s)"] * len(batch)) + UNITS_SUFFIX
            cur.execute(sql, tuple(v for pair in batch for v in pair))
            for page_id, key, sentence_id, text in cur.fetchall():
                ids[(int(page_id), str(key))] = int(sentence_id)
                texts[int(sentence_id)] = str(text)
    return ids, texts


def build_catalog(
    conn: pymysql.Connection, database: str, claims: Mapping[int, FileClaim]
) -> Catalog:
    """Return the :class:`Catalog` of the connection's database (the rule of the module docstring).

    ``database`` is the name the catalog is reported under; ``claims`` comes from
    :func:`read_claim_files`.
    """
    n_pages, n_chunks = database_size(conn)
    with conn.cursor() as cur:
        cur.execute(PAGE_TITLES_SQL)
        page_ids = {str(title): int(page_id) for page_id, title in cur.fetchall()}
        cur.execute(CLAIM_IDS_SQL)
        table_ids = frozenset(int(cid) for (cid,) in cur.fetchall())
    files = file_claim_ids(claims)
    own = tuple(name for name, ids in files.items() if table_ids and ids == table_ids)
    if not own:
        return Catalog(database, n_pages, n_chunks, (), {}, ())

    order = list(own) + [name for name in files if name not in own]
    ordered = sorted(
        claims.values(), key=lambda c: min(order.index(s) for s in c.sources)
    )  # stable: line order within a file
    candidates = [
        c
        for c in ordered
        if any(c.evidence_sets)
        and all(parse_element_id(e)[0] in page_ids for s in c.evidence_sets for e in s)
    ]
    pairs = []
    for c in candidates:
        for s in c.evidence_sets:
            for e in s:
                title, element_type, key = parse_element_id(e)
                if element_type in TEXT_UNIT_TYPES:
                    pairs.append((page_ids[title], key))
    sentence_ids, texts = _resolve_units(conn, pairs)

    dropped: list[tuple[int, str]] = []
    resolved: dict[int, tuple[FileClaim, tuple[tuple[GoldElement, ...], ...]]] = {}
    for c in candidates:
        sets = []
        missing = None
        for s in c.evidence_sets:
            elements = []
            for e in s:
                title, element_type, key, page_id, sentence_id = resolve_element(
                    e, page_ids, sentence_ids
                )
                if page_id is None or (element_type in TEXT_UNIT_TYPES and sentence_id is None):
                    missing = e  # page_id is set: every page was checked above
                    continue
                text = texts.get(sentence_id) if sentence_id is not None else None
                elements.append(
                    GoldElement(e, title, element_type, key, page_id, sentence_id, text)
                )
            sets.append(tuple(elements))
        if missing is not None:
            dropped.append((c.claim_id, f"{missing} is not a row of sentence"))
            continue
        resolved[c.claim_id] = (c, tuple(sets))

    claim_rows = [(c.claim_id, c.split, c.text, c.label) for c, _ in resolved.values()]
    evidence_rows = [
        (cid, index, e.element_type, e.page_id, e.page_title, e.sentence_id)
        for cid, (_, sets) in resolved.items()
        for index, s in enumerate(sets)
        for e in s
    ]
    truths = {t.claim_id: t for t in ground_truth_from_rows(claim_rows, evidence_rows)}
    stored = {t.claim_id: t for t in load_ground_truth(conn)}
    offered: dict[int, OfferedClaim] = {}
    for cid, (c, sets) in resolved.items():
        truth = truths[cid]
        if cid in table_ids and stored.get(cid) != truth:
            dropped.append((cid, "its resolved evidence differs from the claim_evidence rows"))
            continue
        offered[cid] = OfferedClaim(c, truth, sets, claim_kind(c.evidence_sets))
    if dropped:
        logger.warning("%s: %d claims left out of the picker: %s", database, len(dropped), dropped)
    return Catalog(database, n_pages, n_chunks, own, offered, tuple(dropped))


# ---------------------------------------------------------------------------------------------
# marks and summary of one search
# ---------------------------------------------------------------------------------------------


def search_marks(
    conn: pymysql.Connection, offered: OfferedClaim, hits: Sequence[Hit], k: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(marks, summary)`` of a search's hits for the claim, at ``k``.

    One mark per hit, in hit order: ``gold_page`` (the hit's page is a gold page), ``gold_evidence``
    (the chunk holds a gold text unit of any evidence set) and ``gold_sentence_keys`` (the element
    keys of those units). The summary applies the evaluation's functions to the hits in search
    order: ``gold_page_found`` (article recall@k), ``gold_page_rank`` (rank of the first hit on a
    gold page, or None), and, when the claim has a sentence-only evidence set,
    ``evidence_recall`` (evidence recall@k), ``unit_coverage`` and the units of the best-covered
    set (``units_covered`` of ``units_total``); without such a set ``evidence_recall_applies`` is
    False and those are None, as the evaluation leaves the claim out of evidence recall.
    ``table_sets`` counts the evidence sets with a table element (not indexed).
    """
    truth = offered.truth
    hit_pages = [hit.page_id for hit in hits]
    units_by_chunk = chunk_sentence_ids(conn, [hit.chunk_id for hit in hits])
    hit_units = [units_by_chunk[hit.chunk_id] for hit in hits]
    gold_units = offered.gold_units()
    marks = [
        {
            "chunk_id": hit.chunk_id,
            "gold_page": hit.page_id in truth.gold_pages,
            "gold_evidence": bool(units & gold_units.keys()),
            "gold_sentence_keys": sorted(gold_units[u] for u in units & gold_units.keys()),
        }
        for hit, units in zip(hits, hit_units, strict=True)
    ]
    summary: dict[str, Any] = {
        "k": k,
        "n_hits": len(hits),
        "gold_page_found": article_recall_at_k(hit_pages, truth.gold_pages, k),
        "gold_page_rank": gold_page_rank(hit_pages[:k], truth.gold_pages),
        "evidence_recall_applies": truth.eligible,
        "evidence_recall": None,
        "unit_coverage": None,
        "units_covered": None,
        "units_total": None,
        "n_evidence_sets": len(offered.sets),
        "table_sets": sum(1 for s in offered.sets if any(e.in_table for e in s)),
    }
    if truth.eligible:
        covered = covered_units(hit_units, k)
        coverage = unit_coverage_at_k(hit_units, truth.sentence_only_sets, k)
        # the set unit_coverage_at_k takes its share from: the first one with the best share
        best = max(truth.sentence_only_sets, key=lambda s: len(s & covered) / len(s))
        summary.update(
            evidence_recall=evidence_recall_at_k(hit_units, truth.sentence_only_sets, k),
            unit_coverage=coverage,
            units_covered=len(best & covered),
            units_total=len(best),
        )
    return marks, summary


# ---------------------------------------------------------------------------------------------
# the ready-made examples
# ---------------------------------------------------------------------------------------------

#: The salt of the example rule (:func:`example_rank`).
EXAMPLE_SALT = "example-2026-11-20"


def example_rank(claim_id: int) -> str:
    """Return the hex SHA-256 of ``"<EXAMPLE_SALT>:<claim_id>"``; the lowest one is picked."""
    return hashlib.sha256(f"{EXAMPLE_SALT}:{claim_id}".encode()).hexdigest()


def _pages(offered: OfferedClaim) -> set[str]:
    return {e.page_title for s in offered.sets for e in s}


def pick_examples(demo: Catalog, s1: Catalog) -> dict[str, int]:
    """Return the example claim ids by the example rule.

    The candidates are the claims offered on both catalogs that have a sentence-only evidence
    set (so the summary has an evidence recall to show). For SUPPORTS, then REFUTES, then NOT
    ENOUGH INFO, the example is the candidate of that label with the lowest
    :func:`example_rank` among those that cite no page an earlier example cites. The two-page
    example is the claim offered on ``s1`` whose every evidence set needs two or more pages
    (kind ``needs2``), with a sentence-only set, with the lowest rank. The keys are the labels
    and ``"two_page"``. Raises ValueError when a group has no candidate.
    """
    both = [c for cid, c in demo.claims.items() if cid in s1.claims and c.truth.eligible]
    picked: dict[str, int] = {}
    used: set[str] = set()
    for label in LABELS:
        pool = [c for c in both if c.claim.label == label and not (_pages(c) & used)]
        if not pool:
            raise ValueError(f"no example candidate for {label}")
        best = min(pool, key=lambda c: example_rank(c.claim_id))
        picked[label] = best.claim_id
        used |= _pages(best)
    two = [c for c in s1.claims.values() if c.kind == "needs2" and c.truth.eligible]
    if not two:
        raise ValueError("no two-page example candidate")
    picked["two_page"] = min(two, key=lambda c: example_rank(c.claim_id)).claim_id
    return picked


#: :func:`pick_examples` on the demo database and S1 (the check is in the docs).
EXAMPLE_CLAIM_IDS: dict[str, int] = {
    "SUPPORTS": 36003,
    "REFUTES": 22563,
    "NOT ENOUGH INFO": 87976,
    "two_page": 58230,
}
#: The filter example's page: the supported example's one gold page.
EXAMPLE_FILTER_PAGE = "Andorra"
#: One word below ``page.n_words`` of :data:`EXAMPLE_FILTER_PAGE` (8,376 on the demo and on S1),
#: so ``max_words`` keeps every page shorter than it and excludes the gold page.
EXAMPLE_FILTER_MAX_WORDS = 8375
#: A question about a page that both databases hold ("Black Sea").
EXAMPLE_FREE_TEXT = "Which countries border the Black Sea?"
EXAMPLE_FREE_TEXT_PAGE = "Black Sea"


@dataclass(frozen=True)
class Example:
    """One ready-made example: a button that opens the page with these form parameters."""

    key: str
    label: str
    params: dict[str, str]
    s1_only: bool = False


EXAMPLES: tuple[Example, ...] = (
    Example(
        "supported",
        "a supported claim",
        {"claim_id": str(EXAMPLE_CLAIM_IDS["SUPPORTS"]), "sentences": "1"},
    ),
    Example(
        "refuted",
        "a refuted claim",
        {"claim_id": str(EXAMPLE_CLAIM_IDS["REFUTES"]), "sentences": "1"},
    ),
    Example(
        "nei",
        "a not-enough-info claim",
        {"claim_id": str(EXAMPLE_CLAIM_IDS["NOT ENOUGH INFO"]), "sentences": "1"},
    ),
    Example(
        "filter",
        "a filter that hides the answer",
        {
            "claim_id": str(EXAMPLE_CLAIM_IDS["SUPPORTS"]),
            "max_words": str(EXAMPLE_FILTER_MAX_WORDS),
            "sentences": "1",
        },
    ),
    Example("free_text", "free text", {"q": EXAMPLE_FREE_TEXT}),
    Example(
        "two_page",
        "a two-page claim (S1 only)",
        {"claim_id": str(EXAMPLE_CLAIM_IDS["two_page"]), "sentences": "1"},
        s1_only=True,
    ),
)
