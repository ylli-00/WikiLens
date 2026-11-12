#!/usr/bin/env python3
"""Independent self-check of the scaled claim set and the S0/S1 corpora in OUT_DIR.

Every rule of the scaled build is re-derived here from its specification, with separate code,
and compared with what the build wrote. Nothing is changed: the only file written is
``OUT_DIR/SELFCHECK.md`` (a summary table, then every measured number and up to
``MAX_EXAMPLES`` offenders per failed item). The checks:

A. The raw FEVEROUS claim files, read with ``json`` and this script's own element-id regex:
   record counts, header lines, train/dev id overlap, ids that do not parse, and agreement of
   the regex with ``wikilense.corpus.parse_element_id``.
B. The claims file (``--claims``): line count, key order (top level; nested objects: the key
   sets) and types, unique ids, each claim equal to its raw record (label, claim, challenge),
   kind, per-set elements / text_only / findable (findable recomputed from ``S0/pages.jsonl``
   with ``parse_page`` and ``chunk_page(240, 1)``), gold_pages, gold_set, gold_evidence (texts
   = ``TextUnit.text``), 700 single + 300 needs2 claims per label.
C. ``pool.jsonl``: the eligible claims recomputed from the raw files (same claims, same order,
   same kinds), the findable flags of the picked claims, and the seeded sample re-run on the
   file's findable flags, which must give the claims file's (split, id) sequence exactly.
D. With the FEVEROUS database (``WIKILENSE_FEVEROUS_DB``, opened read-only; skipped with
   ``--no-db``): every unfindable pool claim is unfindable in the database, a seeded sample of
   300 findable unpicked pool claims is findable, rule D is re-derived completely and compared
   with ``distractors_D.jsonl`` / ``distractors_D.json``, and the S0/S1 page lines equal the
   database pages (NFC).
E. ``S0/`` and ``S1/``: every line parses, all strings NFC, titles unique, S0 titles = gold
   pages present in the database (sorted), S1 = S0's bytes + the rule-D pages in order, no
   rule-D page is gold, ``claims.jsonl`` identical in both and equal to the raw records plus
   ``split`` in claims-file order, every evidence id accepted by ``parse_element_id``.
F. The ingest's own page and claim loaders (``wikilense.ingest._Ingest``) run on each folder
   without a database: pages, chunks, chunk words, unit words and the SHA-256 of the embedding
   texts against corpora.json, MANIFEST.md and this script's own chunking; every claim's gold
   set resolved to sentence ids that are in a chunk; evidence id counts per folder.
G. MANIFEST.md and corpora.json: bytes, SHA-256 and line counts of pages.jsonl and claims.jsonl
   recomputed, and the exact MANIFEST rows with the recomputed numbers.

Run from the repository root (the database path comes from the environment, never from here):

    WIKILENSE_FEVEROUS_DB=<path to feverous_wikiv1.db> .venv/bin/python scripts/scaled_check.py \\
        --out-dir <OUT_DIR> --claims <OUT_DIR>/claims_3000.jsonl.tmp

``--no-db`` skips check D. ``--train`` / ``--dev`` default to ``data/feverous/`` in the
repository. Exit status 0 when no check fails, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
import traceback
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wikilense import ingest as ing
from wikilense.chunking import chunk_page, embedding_text
from wikilense.config import REPO_ROOT, Settings, SettingsError, load_settings
from wikilense.corpus import PAGE_META_FILE, parse_element_id
from wikilense.wikitext import extract_links, parse_page

LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
KINDS = ("single", "needs2")
SPLITS = ("train", "dev")
TEXT_TYPES = frozenset({"sentence", "item"})
DEFAULT_TRAIN = REPO_ROOT / "data" / "feverous" / "feverous_train_challenges.jsonl"
DEFAULT_DEV = REPO_ROOT / "data" / "feverous" / "feverous_dev_challenges.jsonl"
DB_ENV = "WIKILENSE_FEVEROUS_DB"
MAX_EXAMPLES = 10
PROGRESS_EVERY = 1000

CLAIM_KEYS = (
    "id", "split", "label", "claim", "kind", "challenge",
    "gold_pages", "evidence_sets", "gold_set", "gold_evidence",
)  # fmt: skip
SET_KEYS = ("elements", "text_only", "findable")
GOLD_EVIDENCE_KEYS = ("element", "page", "text")
POOL_KEYS = ("id", "split", "label", "kind", "findable")
DISTRACTOR_KEYS = ("title", "gold", "walk_index", "rank", "times_linked")
SKIP_REASONS = ("gold", "taken", "missing")
CORPUS_NUMBERS = ("pages", "chunks", "chunk_words", "unit_words", "embedding_text_sha256")
FOLDERS = ("S0", "S1")

# ``<page>_<type>_<numbers>`` matched over the whole id: the page group is lazy, so the leftmost
# type that leaves only numbers wins and ``X_header_cell_0_1_1`` is (X, header_cell), never
# (X_header, cell); a page part may contain underscores.
_ELEMENT_RE = re.compile(
    r"(?P<page>.+?)_(?P<type>header_cell|table_caption|sentence|cell|item)_(?P<nums>\d+(?:_\d+)*)"
)
_EMBEDDING_HEX_RE = re.compile(r"Embedding texts.*?`([0-9a-fA-F]{64})`")


@dataclass(frozen=True)
class Params:
    """The numbers the build is specified with; tests shrink ``per_kind`` for a tiny world."""

    per_kind: dict[str, int] = field(default_factory=lambda: {"single": 700, "needs2": 300})
    sample_seed: int = 20261111
    rule_d_seed: int = 7
    max_words: int = 240
    overlap_units: int = 1
    db_sample: int = 300
    db_sample_seed: int = 20261112

    @property
    def n_claims(self) -> int:
        """Return the number of claims the claims file must hold (3,000 by default)."""
        return sum(self.per_kind[kind] for kind in KINDS) * len(LABELS)


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------


def log(message: str) -> None:
    """Print one progress line, flushed at once."""
    print(f"[scaled_check {time.strftime('%H:%M:%S')}] {message}", flush=True)


def nfc(text: str) -> str:
    """Return ``text`` in Unicode NFC."""
    return unicodedata.normalize("NFC", text)


def nfc_deep(value: Any) -> Any:
    """Return ``value`` with every string (dict keys and values, recursively) in NFC."""
    if isinstance(value, str):
        return nfc(value)
    if isinstance(value, list):
        return [nfc_deep(item) for item in value]
    if isinstance(value, dict):
        return {nfc_deep(key): nfc_deep(item) for key, item in value.items()}
    return value


def first_non_nfc(value: Any) -> str | None:
    """Return the first string inside ``value`` (keys included) that is not NFC, else None."""
    if isinstance(value, str):
        return None if unicodedata.is_normalized("NFC", value) else value
    items: list[Any] = []
    if isinstance(value, dict):
        items = [part for pair in value.items() for part in pair]
    elif isinstance(value, list):
        items = value
    for item in items:
        found = first_non_nfc(item)
        if found is not None:
            return found
    return None


def is_int(value: Any) -> bool:
    """Return True for an int that is not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def short(value: Any, limit: int = 160) -> str:
    """Return ``repr(value)`` cut to ``limit`` characters for an example line."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def tag(split: Any, claim_id: Any) -> str:
    """Return the ``split:id`` name of a claim used in examples."""
    return f"{split}:{claim_id}"


class Offenders:
    """Counts offenders and keeps the first ``MAX_EXAMPLES`` descriptions."""

    def __init__(self) -> None:
        self.count = 0
        self.examples: list[str] = []

    def add(self, example: str) -> None:
        """Count one offender; keep its description while there is room."""
        self.count += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(example)


# ---------------------------------------------------------------------------------------------
# check records
# ---------------------------------------------------------------------------------------------


@dataclass
class Item:
    """One measured statement inside a check."""

    name: str
    ok: bool
    detail: str
    examples: list[str] = field(default_factory=list)


@dataclass
class Check:
    """One lettered check: its items, key numbers for the summary table, free notes."""

    letter: str
    title: str
    items: list[Item] = field(default_factory=list)
    key_numbers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: str | None = None

    @property
    def failed(self) -> bool:
        """Return True when an item failed."""
        return any(not item.ok for item in self.items)

    @property
    def result(self) -> str:
        """Return PASS, FAIL or SKIPPED."""
        if self.failed:
            return "FAIL"
        return "SKIPPED" if self.skipped else "PASS"

    def expect(self, name: str, ok: bool, detail: str, examples: list[str] | None = None) -> None:
        """Record one item."""
        self.items.append(Item(name, bool(ok), detail, list(examples or [])[:MAX_EXAMPLES]))

    def equal(self, name: str, got: Any, want: Any) -> None:
        """Record an item that passes when ``got == want``."""
        detail = f"{got!r}" if got == want else f"got {short(got)}, expected {short(want)}"
        self.expect(name, got == want, detail)

    def none(self, name: str, offenders: Offenders, checked: int, what: str = "") -> None:
        """Record an item that passes when nothing was counted in ``offenders``."""
        detail = f"{offenders.count:,} of {checked:,} {what}".rstrip()
        self.expect(name, offenders.count == 0, detail, offenders.examples)

    def key(self, text: str) -> None:
        """Add one key number for the summary table."""
        self.key_numbers.append(text)


# ---------------------------------------------------------------------------------------------
# element ids, evidence sets, kinds
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Element:
    """One parsed element id: NFC page title, element type and page-object key."""

    page: str
    type: str
    key: str

    @property
    def full_id(self) -> str:
        """Return the NFC element id ``<page>_<key>``."""
        return f"{self.page}_{self.key}"


def parse_element(element_id: Any) -> Element | None:
    """Return the ``Element`` of a FEVEROUS element id, or None when it does not parse."""
    if not isinstance(element_id, str):
        return None
    match = _ELEMENT_RE.fullmatch(element_id)
    if match is None:
        return None
    return Element(nfc(match["page"]), match["type"], f"{match['type']}_{match['nums']}")


@dataclass(frozen=True)
class SetView:
    """One raw evidence set: its ``content`` ids and their parses (None = did not parse)."""

    raw: tuple[str, ...]
    elements: tuple[Element | None, ...]

    @property
    def pages(self) -> frozenset[str]:
        """Return the distinct NFC pages the set cites."""
        return frozenset(e.page for e in self.elements if e is not None)

    @property
    def text_only(self) -> bool:
        """Return True when the set is non-empty and every element is a sentence or item."""
        return bool(self.elements) and all(
            e is not None and e.type in TEXT_TYPES for e in self.elements
        )

    @property
    def full_ids(self) -> list[str | None]:
        """Return the NFC full ids in content order."""
        return [None if e is None else e.full_id for e in self.elements]

    def pairs(self) -> list[tuple[str, str]]:
        """Return the ``(page, key)`` pairs of the parsed elements."""
        return [(e.page, e.key) for e in self.elements if e is not None]


def set_views(record: dict) -> list[SetView]:
    """Return one ``SetView`` per entry of the record's raw ``evidence`` list, in order."""
    views = []
    for evidence_set in record.get("evidence") or []:
        content = evidence_set.get("content") if isinstance(evidence_set, dict) else None
        content = tuple(content) if isinstance(content, list) else ()
        views.append(SetView(content, tuple(parse_element(el) for el in content)))
    return views


def claim_kind(views: list[SetView]) -> str:
    """Return none / needs2 / single / either for a claim's evidence sets (empty sets ignored)."""
    nonempty = [view for view in views if view.raw]
    if not nonempty:
        return "none"
    if all(len(view.pages) >= 2 for view in nonempty):
        return "needs2"
    union = frozenset().union(*(view.pages for view in nonempty))
    return "single" if len(union) == 1 else "either"


def claim_gold_pages(views: list[SetView]) -> list[str]:
    """Return the sorted NFC pages of every element of every set."""
    return sorted(frozenset().union(*(view.pages for view in views)))


# ---------------------------------------------------------------------------------------------
# A: the raw claim files
# ---------------------------------------------------------------------------------------------


@dataclass
class PoolEntry:
    """An eligible claim recomputed from the raw files: kind single/needs2, a text-only set."""

    split: str
    id: int
    label: str
    kind: str
    text_sets: list[list[tuple[str, str]]]


@dataclass
class RawData:
    """What the raw scan keeps: counts, the eligible pool, and the raw records of picked claims."""

    records: Counter = field(default_factory=Counter)
    headers: Counter = field(default_factory=Counter)
    kinds: Counter = field(default_factory=Counter)
    pool: list[PoolEntry] = field(default_factory=list)
    picked: dict[tuple[str, int], dict] = field(default_factory=dict)
    ids: dict[str, set] = field(default_factory=lambda: {split: set() for split in SPLITS})
    duplicate_ids: Offenders = field(default_factory=Offenders)
    bad_lines: Offenders = field(default_factory=Offenders)
    unparseable: Offenders = field(default_factory=Offenders)
    disagree: Offenders = field(default_factory=Offenders)
    n_element_ids: int = 0
    errors: list[str] = field(default_factory=list)


def _compare_with_library(element_id: str, element: Element, raw: RawData) -> None:
    """Count an element id whose parse differs from ``wikilense.corpus.parse_element_id``."""
    try:
        theirs = parse_element_id(element_id)
    except ValueError:
        theirs = None
    if theirs != (element.page, element.type, element.key):
        raw.disagree.add(f"{element_id!r}: mine {element}, library {theirs}")


def _scan_record(record: dict, split: str, raw: RawData, picked: set) -> None:
    """Add one raw record to the counts, the pool and the kept picked records."""
    claim_id = record.get("id")
    if claim_id in raw.ids[split]:
        raw.duplicate_ids.add(tag(split, claim_id))
    raw.ids[split].add(claim_id)
    raw.records[split] += 1
    views = set_views(record)
    for view in views:
        for element_id, element in zip(view.raw, view.elements):
            raw.n_element_ids += 1
            if element is None:
                raw.unparseable.add(f"{tag(split, claim_id)} {element_id!r}")
            else:
                _compare_with_library(element_id, element, raw)
    kind = claim_kind(views)
    raw.kinds[(split, kind)] += 1
    text_sets = [view.pairs() for view in views if view.text_only]
    if kind in KINDS and text_sets:
        raw.pool.append(PoolEntry(split, claim_id, str(record.get("label")), kind, text_sets))
    if (split, claim_id) in picked:
        raw.picked[(split, claim_id)] = {**record, "split": split}


def _scan_line_raw(line: str, number: int, split: str, raw: RawData, picked: set) -> None:
    """Parse one raw line: count a header (id == ""), else scan the record."""
    if not line.strip():
        return
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raw.bad_lines.add(f"{split} line {number}: {exc}")
        return
    if not isinstance(record, dict):
        raw.bad_lines.add(f"{split} line {number}: not an object")
    elif record.get("id") == "":
        raw.headers[(split, number)] += 1
    else:
        _scan_record(record, split, raw, picked)


def scan_raw(train: Path, dev: Path, picked: set) -> RawData:
    """Read both raw claim files (train first), header lines skipped, in file order."""
    raw = RawData()
    for split, path in (("train", train), ("dev", dev)):
        log(f"reading raw claims ({split}): {path.name}")
        if not path.is_file():
            raw.errors.append(f"{split} file {path.name} not found")
            continue
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                _scan_line_raw(line, number, split, raw, picked)
    log(f"raw claims: {raw.records['train']:,} train, {raw.records['dev']:,} dev, "
        f"{len(raw.pool):,} eligible")
    return raw


def check_a(ctx: Context) -> Check:
    """A: the raw claim files themselves."""
    check = Check("A", "Raw claim files")
    raw = ctx.raw
    for error in raw.errors:
        check.expect("raw file readable", False, error)
    for split in SPLITS:
        headers = sorted(number for s, number in raw.headers if s == split)
        check.expect(f"{split}: one header line (id == ''), the first line",
                     headers == [1], f"header lines at {headers}")
    check.none("every raw line parses as a JSON object", raw.bad_lines,
               sum(raw.records.values()), "records")
    overlap = raw.ids["train"] & raw.ids["dev"]
    check.expect("train and dev ids do not overlap", not overlap,
                 f"{len(overlap):,} shared ids", [short(i) for i in sorted(overlap, key=str)])
    check.none("ids unique within a split", raw.duplicate_ids, sum(raw.records.values()), "ids")
    check.none("every element id parses with this script's regex", raw.unparseable,
               raw.n_element_ids, "element ids")
    check.none("this regex agrees with wikilense.corpus.parse_element_id", raw.disagree,
               raw.n_element_ids, "element ids")
    kinds = ", ".join(f"{split} {kind} {n:,}" for (split, kind), n in sorted(raw.kinds.items()))
    check.notes.append(f"Kinds over all records: {kinds}.")
    check.key(f"{raw.records['train']:,} train + {raw.records['dev']:,} dev records")
    check.key(f"{raw.n_element_ids:,} element ids")
    check.key(f"{len(raw.pool):,} eligible")
    return check


# ---------------------------------------------------------------------------------------------
# reading the build's files
# ---------------------------------------------------------------------------------------------


@dataclass
class JsonLines:
    """The parsed lines of a JSONL file (None for a line that did not parse)."""

    path: Path
    exists: bool
    records: list[Any] = field(default_factory=list)
    bad: Offenders = field(default_factory=Offenders)
    blank: int = 0


def read_jsonl(path: Path) -> JsonLines:
    """Return every line of a small JSONL file parsed, in order (None for a broken line)."""
    result = JsonLines(path, path.is_file())
    if not result.exists:
        return result
    with open(path, "rb") as handle:
        for number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                result.blank += 1
                continue
            try:
                result.records.append(json.loads(raw_line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                result.bad.add(f"line {number}: {exc}")
                result.records.append(None)
    return result


def read_json(path: Path) -> tuple[Any, str | None]:
    """Return ``(parsed, None)`` for a JSON file, or ``(None, reason)``."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name}: {exc}"


def read_text(path: Path) -> str | None:
    """Return a text file's content, or None when it cannot be read."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


@dataclass
class FileStats:
    """Bytes, SHA-256 and lines of one file (lines = newline-terminated lines, plus a tail)."""

    exists: bool
    n_bytes: int = 0
    sha256: str = ""
    lines: int = 0


@dataclass
class PagesScan:
    """One streamed pass over a pages.jsonl: file stats, titles, own chunking totals."""

    stats: FileStats
    titles: list[str] = field(default_factory=list)
    line_sha: dict[str, str] = field(default_factory=dict)
    bad: Offenders = field(default_factory=Offenders)
    blank: int = 0
    non_nfc: Offenders = field(default_factory=Offenders)
    not_dumps: Offenders = field(default_factory=Offenders)
    duplicates: Offenders = field(default_factory=Offenders)
    pages: int = 0
    chunks: int = 0
    chunk_words: int = 0
    unit_words: int = 0
    embedding_sha: str = ""
    landing: set[tuple[str, str]] = field(default_factory=set)
    texts: dict[tuple[str, str], str] = field(default_factory=dict)


def _scan_page(page: dict, scan: PagesScan, params: Params, wanted: dict, digest: Any) -> str:
    """Parse and chunk one page, add it to the totals, record wanted landings; return title."""
    parsed = parse_page(page)
    chunks = chunk_page(parsed, max_words=params.max_words, overlap_units=params.overlap_units)
    scan.pages += 1
    scan.chunks += len(chunks)
    scan.chunk_words += sum(chunk.n_words for chunk in chunks)
    scan.unit_words += parsed.stats["n_words"]
    for chunk in chunks:
        path = parsed.sections[chunk.section_ordinal].path
        digest.update((embedding_text(parsed.title, path, chunk.text) + "\n").encode("utf-8"))
    keys = wanted.get(parsed.title)
    if keys:
        in_chunks = {key for chunk in chunks for key in chunk.element_keys}
        texts = {unit.element_key: unit.text for unit in parsed.units}
        for key in keys:
            if key in in_chunks:
                scan.landing.add((parsed.title, key))
            if key in texts:
                scan.texts[(parsed.title, key)] = texts[key]
    return parsed.title


def _scan_line(raw_line: bytes, number: int, scan: PagesScan, params: Params, wanted: dict,
               digest: Any) -> None:
    """Check one pages.jsonl line (parse, NFC, json.dumps form, title) and scan its page."""
    text_line = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
    if not text_line.strip():
        scan.blank += 1
        return
    try:
        text = text_line.decode("utf-8")
        page = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        scan.bad.add(f"line {number}: {exc}")
        return
    if not isinstance(page, dict):
        scan.bad.add(f"line {number}: not an object")
        return
    found = first_non_nfc(page)
    if found is not None:
        scan.non_nfc.add(f"line {number}: {short(found, 80)}")
    if json.dumps(page, ensure_ascii=False) != text:
        scan.not_dumps.add(f"line {number}")
    title = _scan_page(page, scan, params, wanted, digest)
    if title in scan.line_sha:
        scan.duplicates.add(f"line {number}: {title!r}")
    scan.titles.append(title)
    scan.line_sha[title] = hashlib.sha256(text_line).hexdigest()


def scan_pages(path: Path, params: Params, wanted: dict | None = None) -> PagesScan:
    """Stream ``path`` once: file bytes/SHA/lines, per-line checks, own chunking totals."""
    scan = PagesScan(FileStats(path.is_file()))
    if not scan.stats.exists:
        return scan
    log(f"scanning {path.parent.name}/{path.name}")
    file_digest = hashlib.sha256()
    embed_digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for number, raw_line in enumerate(handle, start=1):
            file_digest.update(raw_line)
            scan.stats.n_bytes += len(raw_line)
            scan.stats.lines += 1
            _scan_line(raw_line, number, scan, params, wanted or {}, embed_digest)
            if number % PROGRESS_EVERY == 0:
                log(f"  {path.parent.name}: {number:,} lines, {scan.chunks:,} chunks")
    scan.stats.sha256 = file_digest.hexdigest()
    scan.embedding_sha = embed_digest.hexdigest()
    log(f"  {path.parent.name}: {scan.pages:,} pages, {scan.chunks:,} chunks")
    return scan


def file_stats(path: Path) -> FileStats:
    """Return bytes, SHA-256 and line count of a file."""
    stats = FileStats(path.is_file())
    if not stats.exists:
        return stats
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            stats.n_bytes += len(raw_line)
            stats.lines += 1
    stats.sha256 = digest.hexdigest()
    return stats


def starts_with(path: Path, prefix_path: Path) -> bool:
    """Return True when the bytes of ``path`` begin with every byte of ``prefix_path``."""
    with open(prefix_path, "rb") as prefix, open(path, "rb") as whole:
        while True:
            block = prefix.read(1 << 20)
            if not block:
                return True
            if whole.read(len(block)) != block:
                return False


# ---------------------------------------------------------------------------------------------
# the shared context
# ---------------------------------------------------------------------------------------------


@dataclass
class Context:
    """Everything the checks share, read once."""

    params: Params
    out_dir: Path
    claims_path: Path
    use_db: bool
    env_file: Path | None
    claims: JsonLines
    raw: RawData
    picked: list[tuple[Any, Any]]
    gold: set[str]
    pool: JsonLines
    distractors: JsonLines
    stats: tuple[Any, str | None]
    scans: dict[str, PagesScan]
    claims_files: dict[str, FileStats]
    manifests: dict[str, str | None]
    corpora: tuple[Any, str | None]
    db_name: str | None = None

    @property
    def claim_records(self) -> list[dict]:
        """Return the claims-file records that are JSON objects."""
        return [rec for rec in self.claims.records if isinstance(rec, dict)]

    @property
    def distractor_titles(self) -> list[str]:
        """Return the titles of distractors_D.jsonl in file order."""
        return [rec.get("title") for rec in self.distractors.records if isinstance(rec, dict)]

    @property
    def gold_missing(self) -> list[str]:
        """Return distractors_D.json's ``gold_missing_from_db`` (empty when unreadable)."""
        stats = self.stats[0]
        missing = stats.get("gold_missing_from_db") if isinstance(stats, dict) else None
        return list(missing) if isinstance(missing, list) else []


def _hashable(value: Any) -> bool:
    """Return True when ``value`` can be a dict key."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


def wanted_pairs(ctx_raw: RawData, picked: list[tuple[Any, Any]]) -> dict[str, set[str]]:
    """Return ``{page: {keys}}`` for every element of every set of the picked claims."""
    wanted: dict[str, set[str]] = defaultdict(set)
    for key in picked:
        record = ctx_raw.picked.get(key)
        if record is None:
            continue
        for view in set_views(record):
            for page, element_key in view.pairs():
                wanted[page].add(element_key)
    return wanted


def gold_of(raw: RawData, picked: list[tuple[Any, Any]]) -> set[str]:
    """Return the union of every page cited by any element of any set of the picked claims."""
    gold: set[str] = set()
    for key in picked:
        record = raw.picked.get(key)
        if record is not None:
            gold.update(claim_gold_pages(set_views(record)))
    return gold


def load_context(out_dir: Path, claims_path: Path, train: Path, dev: Path, use_db: bool,
                 params: Params, env_file: Path | None) -> Context:
    """Read the claims file, the raw files, and stream both pages files once."""
    claims = read_jsonl(claims_path)
    picked = [(rec.get("split"), rec.get("id")) for rec in claims.records if isinstance(rec, dict)]
    picked = [key if _hashable(key) else (repr(key[0]), repr(key[1])) for key in picked]
    raw = scan_raw(train, dev, set(picked))
    wanted = wanted_pairs(raw, picked)
    scans = {
        "S0": scan_pages(out_dir / "S0" / "pages.jsonl", params, wanted),
        "S1": scan_pages(out_dir / "S1" / "pages.jsonl", params),
    }
    return Context(
        params=params,
        out_dir=out_dir,
        claims_path=claims_path,
        use_db=use_db,
        env_file=env_file,
        claims=claims,
        raw=raw,
        picked=picked,
        gold=gold_of(raw, picked),
        pool=read_jsonl(out_dir / "pool.jsonl"),
        distractors=read_jsonl(out_dir / "distractors_D.jsonl"),
        stats=read_json(out_dir / "distractors_D.json"),
        scans=scans,
        claims_files={f: file_stats(out_dir / f / "claims.jsonl") for f in FOLDERS},
        manifests={f: read_text(out_dir / f / "MANIFEST.md") for f in FOLDERS},
        corpora=read_json(out_dir / "corpora.json"),
    )


# ---------------------------------------------------------------------------------------------
# B: the claims file
# ---------------------------------------------------------------------------------------------


def _sets_shape(sets: Any) -> list[str]:
    """Return the shape problems of an ``evidence_sets`` value."""
    if not isinstance(sets, list):
        return ["evidence_sets is not a list"]
    problems = []
    for index, entry in enumerate(sets):
        if not isinstance(entry, dict) or set(entry) != set(SET_KEYS):
            problems.append(f"evidence_sets[{index}] keys {short(list(entry or {}))}")
            continue
        elements = entry["elements"]
        if not isinstance(elements, list) or not all(isinstance(e, str) for e in elements):
            problems.append(f"evidence_sets[{index}].elements not a list of str")
        if not isinstance(entry["text_only"], bool) or not isinstance(entry["findable"], bool):
            problems.append(f"evidence_sets[{index}] text_only/findable not bool")
    return problems


def _gold_evidence_shape(entries: Any) -> list[str]:
    """Return the shape problems of a ``gold_evidence`` value."""
    if not isinstance(entries, list):
        return ["gold_evidence is not a list"]
    problems = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != set(GOLD_EVIDENCE_KEYS):
            problems.append(f"gold_evidence[{index}] keys {short(list(entry or {}))}")
        elif not all(isinstance(entry[k], str) for k in GOLD_EVIDENCE_KEYS):
            problems.append(f"gold_evidence[{index}] values not str")
    return problems


def shape_problems(rec: Any) -> list[str]:
    """Return what is wrong with the keys and types of one claims-file line."""
    if not isinstance(rec, dict):
        return ["not a JSON object"]
    if list(rec) != list(CLAIM_KEYS):
        return [f"keys {list(rec)}"]
    problems = []
    checks = {
        "id": is_int(rec["id"]),
        "split": rec["split"] in SPLITS,
        "label": rec["label"] in LABELS,
        "claim": isinstance(rec["claim"], str),
        "kind": rec["kind"] in KINDS,
        "gold_set": is_int(rec["gold_set"]),
    }
    problems += [f"{name}={short(rec[name], 60)}" for name, ok in checks.items() if not ok]
    pages = rec["gold_pages"]
    if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
        problems.append("gold_pages not a list of str")
    elif pages != sorted(set(pages)):
        problems.append("gold_pages not sorted and unique")
    return problems + _sets_shape(rec["evidence_sets"]) + _gold_evidence_shape(rec["gold_evidence"])


class ClaimVerifier:
    """Compares each claims-file line with its raw record and with S0's chunks."""

    CATEGORIES = (
        "top-level key order, keys and types",
        "(split, id) exists in the raw files",
        "label, claim and challenge equal the raw record",
        "kind recomputed",
        "one evidence_sets entry per raw evidence set",
        "elements recomputed (NFC page + key, content order)",
        "text_only recomputed",
        "findable recomputed from S0/pages.jsonl",
        "picked claim has a findable set in S0",
        "gold_pages recomputed",
        "gold_set = first set findable in S0",
        "gold_set = first set the file marks findable",
        "gold_evidence elements and pages = the gold set's elements",
        "gold_evidence texts = TextUnit.text in S0",
    )

    def __init__(self, raw: RawData, scan: PagesScan) -> None:
        self.raw = raw
        self.scan = scan
        self.offenders = {name: Offenders() for name in self.CATEGORIES}

    def flag(self, category: str, example: str) -> None:
        """Count one offender of a category."""
        self.offenders[category].add(example)

    def findable(self, view: SetView) -> bool:
        """Return True when the set is text-only and every element lands in an S0 chunk."""
        return view.text_only and all(pair in self.scan.landing for pair in view.pairs())

    def verify(self, rec: Any) -> None:
        """Check one claims-file line."""
        problems = shape_problems(rec)
        if problems:
            name = tag(rec.get("split"), rec.get("id")) if isinstance(rec, dict) else "?"
            self.flag("top-level key order, keys and types", f"{name}: {'; '.join(problems)[:200]}")
        if not isinstance(rec, dict):
            return
        name = tag(rec.get("split"), rec.get("id"))
        raw = self.raw.picked.get((rec.get("split"), rec.get("id")))
        if raw is None:
            self.flag("(split, id) exists in the raw files", name)
            return
        for key in ("label", "claim", "challenge"):
            if rec.get(key) != raw.get(key):
                self.flag("label, claim and challenge equal the raw record", f"{name}: {key}")
        views = set_views(raw)
        kind = claim_kind(views)
        if rec.get("kind") != kind:
            self.flag("kind recomputed", f"{name}: file {rec.get('kind')!r}, recomputed {kind!r}")
        if rec.get("gold_pages") != claim_gold_pages(views):
            self.flag("gold_pages recomputed", f"{name}: {short(rec.get('gold_pages'), 100)}")
        self._verify_sets(rec, name, views)

    def _verify_sets(self, rec: dict, name: str, views: list[SetView]) -> None:
        """Check evidence_sets, gold_set and gold_evidence of one line."""
        sets = rec.get("evidence_sets")
        sets = sets if isinstance(sets, list) else []
        if len(sets) != len(views):
            self.flag("one evidence_sets entry per raw evidence set",
                      f"{name}: {len(sets)} entries, {len(views)} raw sets")
        flags = [self.findable(view) for view in views]
        for index, (view, entry) in enumerate(zip(views, sets)):
            entry = entry if isinstance(entry, dict) else {}
            if entry.get("elements") != view.full_ids:
                self.flag("elements recomputed (NFC page + key, content order)",
                          f"{name} set {index}")
            if entry.get("text_only") is not view.text_only:
                self.flag("text_only recomputed", f"{name} set {index}: file "
                          f"{entry.get('text_only')!r}, recomputed {view.text_only!r}")
            if entry.get("findable") is not flags[index]:
                self.flag("findable recomputed from S0/pages.jsonl", f"{name} set {index}: file "
                          f"{entry.get('findable')!r}, recomputed {flags[index]!r}")
        first = next((i for i, ok in enumerate(flags) if ok), None)
        marked = next((i for i, e in enumerate(sets)
                       if isinstance(e, dict) and e.get("findable") is True), None)
        if first is None:
            self.flag("picked claim has a findable set in S0", name)
        if rec.get("gold_set") != first:
            self.flag("gold_set = first set findable in S0",
                      f"{name}: file {rec.get('gold_set')!r}, recomputed {first!r}")
        if rec.get("gold_set") != marked:
            self.flag("gold_set = first set the file marks findable",
                      f"{name}: gold_set {rec.get('gold_set')!r}, first marked {marked!r}")
        if first is not None:
            self._verify_gold_evidence(rec, name, views[first])

    def _verify_gold_evidence(self, rec: dict, name: str, view: SetView) -> None:
        """Check gold_evidence against the recomputed gold set and S0's unit texts."""
        entries = rec.get("gold_evidence")
        entries = [e if isinstance(e, dict) else {} for e in entries or []]
        got = [(e.get("element"), e.get("page")) for e in entries]
        want = [(None, None) if e is None else (e.full_id, e.page) for e in view.elements]
        if got != want:
            self.flag("gold_evidence elements and pages = the gold set's elements", name)
            return
        for entry, element in zip(entries, view.elements):
            text = self.scan.texts.get((element.page, element.key)) if element else None
            if entry.get("text") != text:
                self.flag("gold_evidence texts = TextUnit.text in S0",
                          f"{name} {element.full_id if element else '?'}: "
                          f"file {short(entry.get('text'), 60)}, S0 {short(text, 60)}")


def check_b(ctx: Context) -> Check:
    """B: the claims file against the raw records and S0's chunks."""
    check = Check("B", "Claims file")
    claims, params = ctx.claims, ctx.params
    if not claims.exists:
        check.expect("claims file exists", False, f"{ctx.claims_path.name} not found")
        return check
    n_lines = len(claims.records) + claims.blank
    check.equal("lines", n_lines, params.n_claims)
    check.none("every line parses", claims.bad, n_lines, "lines")
    records = ctx.claim_records
    ids = Counter(rec.get("id") for rec in records)
    dup = [short(i) for i, n in ids.items() if n > 1]
    check.expect("ids unique", not dup, f"{len(dup):,} ids occur more than once", dup)
    verifier = ClaimVerifier(ctx.raw, ctx.scans["S0"])
    for rec in claims.records:
        verifier.verify(rec)
    if not ctx.scans["S0"].stats.exists:
        check.expect("S0/pages.jsonl exists (needed for findable)", False, "not found")
    for name in ClaimVerifier.CATEGORIES:
        check.none(name, verifier.offenders[name], len(claims.records), "claims")
    groups = Counter((rec.get("label"), rec.get("kind")) for rec in records)
    want = {(label, kind): params.per_kind[kind] for label in LABELS for kind in KINDS}
    wrong = [f"{label}/{kind} {groups.get((label, kind), 0):,} (expected {n:,})"
             for (label, kind), n in want.items() if groups.get((label, kind), 0) != n]
    wrong += [f"{short(k, 60)} {n:,} (expected 0)" for k, n in groups.items() if k not in want]
    sizes = ", ".join(f"{kind} {params.per_kind[kind]:,}" for kind in KINDS)
    check.expect(f"claims per label: {sizes}", not wrong,
                 "; ".join(wrong) if wrong else "every group exact")
    check.key(f"{len(records):,} claims")
    findable_sets = sum(1 for rec in records for e in rec.get("evidence_sets") or []
                        if isinstance(e, dict) and e.get("findable") is True)
    check.key(f"{findable_sets:,} findable sets")
    return check


# ---------------------------------------------------------------------------------------------
# C: pool.jsonl and the seeded sample
# ---------------------------------------------------------------------------------------------


def pool_shape(rec: Any) -> str | None:
    """Return what is wrong with the keys and types of one pool.jsonl line, or None."""
    if not isinstance(rec, dict):
        return "not a JSON object"
    findable = rec.get("findable")
    want = set(POOL_KEYS) | (set() if findable is True else {"reasons"})
    if set(rec) != want:
        return f"keys {list(rec)}"
    if not is_int(rec["id"]) or rec["split"] not in SPLITS or not isinstance(findable, bool):
        return "id/split/findable types"
    if rec["label"] not in LABELS or rec["kind"] not in KINDS:
        return f"label/kind {rec['label']!r}/{rec['kind']!r}"
    reasons = rec.get("reasons", [])
    if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
        return "reasons not a list of str"
    if reasons != sorted(reasons):
        return "reasons not sorted"
    return None


def reproduce_sample(pool: list[Any], params: Params) -> list[tuple[Any, Any]]:
    """Re-run the seeded sample on the pool lines' findable flags; return (split, id) in order.

    Raises ``ValueError`` (from ``random.sample``) when a group has too few findable claims.
    """
    groups: dict[tuple[str, str], list[tuple[Any, Any]]] = defaultdict(list)
    for rec in pool:
        if isinstance(rec, dict) and rec.get("findable") is True:
            groups[(rec.get("label"), rec.get("kind"))].append((rec.get("split"), rec.get("id")))
    rng = random.Random(params.sample_seed)
    picked: list[tuple[Any, Any]] = []
    for label in LABELS:
        for kind in KINDS:
            picked.extend(rng.sample(groups[(label, kind)], params.per_kind[kind]))
    return picked


def _first_difference(got: list, want: list) -> str:
    """Return a one-line description of where two sequences first differ."""
    for index, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return f"first difference at {index}: {short(a)} vs {short(b)}"
    return f"lengths {len(got):,} and {len(want):,}"


def _pool_membership(check: Check, ctx: Context, lines: list[dict]) -> None:
    """Pool lines = the eligible claims recomputed from the raw files, same order and kinds."""
    got = [(rec.get("split"), rec.get("id")) for rec in lines]
    want = [(entry.split, entry.id) for entry in ctx.raw.pool]
    detail = f"{len(got):,} lines, {len(want):,} eligible claims recomputed"
    if got != want:
        missing, extra = len(set(want) - set(got)), len(set(got) - set(want))
        detail += f"; {missing:,} missing, {extra:,} extra; {_first_difference(got, want)}"
    check.expect("pool = every eligible claim, in pool order, nothing else", got == want, detail)
    by_key = {(entry.split, entry.id): entry for entry in ctx.raw.pool}
    wrong = Offenders()
    for rec in lines:
        entry = by_key.get((rec.get("split"), rec.get("id")))
        if entry and (rec.get("label"), rec.get("kind")) != (entry.label, entry.kind):
            wrong.add(f"{tag(entry.split, entry.id)}: file {rec.get('label')}/{rec.get('kind')},"
                      f" recomputed {entry.label}/{entry.kind}")
    check.none("label and kind equal the recomputed ones", wrong, len(lines), "pool lines")


def _pool_table(ctx: Context, lines: list[dict]) -> list[str]:
    """Return the label x kind table: eligible, findable, picked, spare."""
    picked = set(ctx.picked)
    eligible = Counter((entry.label, entry.kind) for entry in ctx.raw.pool)
    findable = Counter((r.get("label"), r.get("kind")) for r in lines if r.get("findable") is True)
    chosen = Counter((r.get("label"), r.get("kind")) for r in lines
                     if (r.get("split"), r.get("id")) in picked)
    rows = ["| Label | Kind | Eligible | Findable | Picked | Spare |", "|---|---|---:|---:|---:|---:|"]
    for label in LABELS:
        for kind in KINDS:
            key = (label, kind)
            rows.append(f"| {label} | {kind} | {eligible[key]:,} | {findable[key]:,} | "
                        f"{chosen[key]:,} | {findable[key] - chosen[key]:,} |")
    return rows


def check_c(ctx: Context) -> Check:
    """C: pool.jsonl against the raw files, and the seeded sample."""
    check = Check("C", "Pool and seeded sample")
    pool = ctx.pool
    if not pool.exists:
        check.expect("pool.jsonl exists", False, "not found")
        return check
    lines = [rec for rec in pool.records if isinstance(rec, dict)]
    check.none("every line parses", pool.bad, len(pool.records), "lines")
    shapes = Offenders()
    for rec in pool.records:
        problem = pool_shape(rec)
        if problem:
            shapes.add(f"{tag(*_key(rec))}: {problem}")
    check.none("keys, types, reasons only when not findable (sorted)", shapes,
               len(pool.records), "lines")
    _pool_membership(check, ctx, lines)
    flags = {(rec.get("split"), rec.get("id")): rec.get("findable") for rec in lines}
    bad = Offenders()
    for key in ctx.picked:
        if flags.get(key) is not True:
            bad.add(f"{tag(*key)}: pool findable = {flags.get(key)!r}")
    check.none("every picked claim is in the pool and findable", bad, len(ctx.picked), "claims")
    try:
        sample = reproduce_sample(pool.records, ctx.params)
        same = sample == ctx.picked
        check.expect("seeded sample on pool.jsonl reproduces the claims file's (split, id) order",
                     same, f"{len(sample):,} drawn" + ("" if same else
                                                     f"; {_first_difference(sample, ctx.picked)}"))
    except ValueError as exc:
        check.expect("seeded sample on pool.jsonl reproduces the claims file", False, str(exc))
    check.notes.extend(_pool_table(ctx, lines))
    n_findable = sum(1 for rec in lines if rec.get("findable") is True)
    check.key(f"{len(lines):,} pool lines, {n_findable:,} findable")
    return check


def _key(rec: Any) -> tuple[Any, Any]:
    """Return ``(split, id)`` of a record, or ``(None, None)`` for a non-object."""
    return (rec.get("split"), rec.get("id")) if isinstance(rec, dict) else (None, None)


# ---------------------------------------------------------------------------------------------
# D: the FEVEROUS database
# ---------------------------------------------------------------------------------------------


class FeverousDB:
    """Read-only access to ``wiki(id PRIMARY KEY, data json)``; ids stored NFD."""

    def __init__(self, path: Path) -> None:
        uri = f"file:{urllib.parse.quote(str(path))}?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.execute("SELECT 1 FROM wiki LIMIT 1").fetchall()  # fails early on a wrong file
        self._exists: dict[str, bool] = {}
        self.fetches = 0
        self.lookups = 0

    def page(self, title: str) -> dict | None:
        """Return the NFC page object stored under ``NFD(title)``, or None."""
        self.fetches += 1
        row = self.conn.execute(
            "SELECT data FROM wiki WHERE id = ?", (unicodedata.normalize("NFD", title),)
        ).fetchone()
        self._exists[title] = row is not None
        return None if row is None else nfc_deep(json.loads(row[0]))

    def exists(self, title: str) -> bool:
        """Return True when a row with id ``NFD(title)`` exists (cached)."""
        if title not in self._exists:
            self.lookups += 1
            row = self.conn.execute(
                "SELECT 1 FROM wiki WHERE id = ?", (unicodedata.normalize("NFD", title),)
            ).fetchone()
            self._exists[title] = row is not None
        return self._exists[title]

    def close(self) -> None:
        """Close the connection."""
        self.conn.close()


def db_path(env_file: Path | None) -> str | None:
    """Return the database path from the environment variable only (it is not a .env setting);
    ``env_file`` is accepted for the callers' symmetry and not read."""
    del env_file
    return os.environ.get(DB_ENV, "") or None


class DbChunkKeys:
    """Chunk element keys of database pages, computed once per title."""

    def __init__(self, db: FeverousDB, params: Params) -> None:
        self.db = db
        self.params = params
        self.cache: dict[str, frozenset[str] | None] = {}

    def keys(self, title: str) -> frozenset[str] | None:
        """Return the keys in chunks of page ``title``; None when missing or titled otherwise."""
        if title not in self.cache:
            page = self.db.page(title)
            result = None
            if page is not None:
                parsed = parse_page(page)
                if parsed.title == title:
                    chunks = chunk_page(parsed, self.params.max_words, self.params.overlap_units)
                    result = frozenset(k for chunk in chunks for k in chunk.element_keys)
            self.cache[title] = result
        return self.cache[title]

    def findable_sets(self, entry: PoolEntry) -> list[int]:
        """Return the indexes (among the text-only sets) of the sets findable in the database."""
        found = []
        for index, pairs in enumerate(entry.text_sets):
            if all(key in (self.keys(page) or ()) for page, key in pairs):
                found.append(index)
        return found


def _check_pool_flags(check: Check, ctx: Context, db: FeverousDB) -> None:
    """Unfindable pool claims stay unfindable; a seeded sample of spares is findable."""
    by_key = {(entry.split, entry.id): entry for entry in ctx.raw.pool}
    lines = [rec for rec in ctx.pool.records if isinstance(rec, dict)]
    keys = DbChunkKeys(db, ctx.params)
    unfindable = _select(lines, by_key, findable=False)
    log(f"DB: rechecking {len(unfindable):,} unfindable pool claims")
    wrong = Offenders()
    for number, entry in enumerate(unfindable, start=1):
        found = keys.findable_sets(entry)
        if found:
            wrong.add(f"{tag(entry.split, entry.id)}: text-only set(s) {found} findable in the DB")
        if number % PROGRESS_EVERY == 0:
            log(f"  {number:,} of {len(unfindable):,}, {db.fetches:,} pages fetched")
    check.none("every unfindable pool claim is unfindable in the DB", wrong, len(unfindable),
               "claims")
    picked = set(ctx.picked)
    spares = [e for e in _select(lines, by_key, findable=True) if (e.split, e.id) not in picked]
    sample = random.Random(ctx.params.db_sample_seed).sample(
        spares, min(ctx.params.db_sample, len(spares)))
    log(f"DB: rechecking a sample of {len(sample):,} findable unpicked pool claims")
    wrong = Offenders()
    for entry in sample:
        if not keys.findable_sets(entry):
            wrong.add(f"{tag(entry.split, entry.id)}: no text-only set findable in the DB")
    check.none(f"sample (seed {ctx.params.db_sample_seed}) of findable unpicked pool claims is "
               "findable in the DB", wrong, len(sample), "claims")
    check.key(f"{len(unfindable):,} unfindable + {len(sample):,} sampled rechecked")


def _select(lines: list[dict], by_key: dict, findable: bool) -> list[PoolEntry]:
    """Return the raw pool entries of the pool lines whose findable flag is ``findable``."""
    return [by_key[_key(rec)] for rec in lines
            if rec.get("findable") is findable and _key(rec) in by_key]


@dataclass
class RuleD:
    """The re-derived rule D."""

    records: list[dict] = field(default_factory=list)
    skips: Counter = field(default_factory=Counter)
    without_link: list[str] = field(default_factory=list)
    missing_gold: list[str] = field(default_factory=list)
    gold_line_sha: dict[str, str] = field(default_factory=dict)
    title_mismatch: Offenders = field(default_factory=Offenders)


def ranked_links(page: dict) -> tuple[list[str], Counter]:
    """Return the distinct link titles ranked by (-count, first mention) and the counts."""
    counts: Counter = Counter()
    first: dict[str, int] = {}
    for position, (title, _source) in enumerate(extract_links(page)):
        counts[title] += 1
        first.setdefault(title, position)
    return sorted(counts, key=lambda t: (-counts[t], first[t])), counts


def _walk_article(gold_title: str, page: dict, walk_index: int, result: RuleD, gold: set[str],
                  taken: set[str], db: FeverousDB) -> None:
    """Walk one gold article's ranked links; take the first usable title."""
    ranked, counts = ranked_links(page)
    for rank, title in enumerate(ranked):
        if title in gold:
            result.skips["gold"] += 1
        elif title in taken:
            result.skips["taken"] += 1
        elif not db.exists(title):
            result.skips["missing"] += 1
        else:
            taken.add(title)
            result.records.append({"title": title, "gold": gold_title, "walk_index": walk_index,
                                   "rank": rank, "times_linked": counts[title]})
            return
    result.without_link.append(gold_title)


def derive_rule_d(gold: set[str], db: FeverousDB, seed: int) -> RuleD:
    """Return rule D re-derived from the database (one page per gold article, no cap)."""
    order = sorted(gold)
    random.Random(seed).shuffle(order)
    result = RuleD()
    taken: set[str] = set()
    for walk_index, gold_title in enumerate(order):
        page = db.page(gold_title)
        if page is None:
            result.missing_gold.append(gold_title)
            continue
        line = json.dumps(page, ensure_ascii=False).encode("utf-8")
        result.gold_line_sha[gold_title] = hashlib.sha256(line).hexdigest()
        if str(page.get("title", "")) != gold_title:  # page is NFC already
            result.title_mismatch.add(f"{gold_title!r} has title {page.get('title')!r}")
        _walk_article(gold_title, page, walk_index, result, gold, taken, db)
        if (walk_index + 1) % PROGRESS_EVERY == 0:
            log(f"  rule D: {walk_index + 1:,} of {len(order):,} gold articles walked")
    return result


def _compare_rule_d(check: Check, ctx: Context, derived: RuleD) -> None:
    """Compare the re-derived rule D with distractors_D.jsonl and distractors_D.json."""
    got = ctx.distractors.records
    shapes = Offenders()
    for rec in got:
        if not isinstance(rec, dict) or set(rec) != set(DISTRACTOR_KEYS):
            shapes.add(short(rec))
    check.none("distractors_D.jsonl keys", shapes, len(got), "lines")
    detail = f"{len(got):,} lines, {len(derived.records):,} re-derived"
    if got != derived.records:
        detail += "; " + _first_difference(got, derived.records)
    check.expect("distractors_D.jsonl = re-derived rule D (title, gold, walk_index, rank, "
                 "times_linked)", got == derived.records, detail)
    stats, error = ctx.stats
    if error or not isinstance(stats, dict):
        check.expect("distractors_D.json readable", False, error or "not an object")
        return
    want_skips = {reason: derived.skips[reason] for reason in SKIP_REASONS}
    check.equal("distractors_D.json skips", stats.get("skips"), want_skips)
    pages = stats.get("pages")
    pages_ok = pages == len(derived.records) or pages == [r["title"] for r in derived.records]
    check.expect("distractors_D.json pages", pages_ok,
                 f"{short(pages, 60)}, re-derived {len(derived.records):,}")
    listed = stats.get("gold_without_usable_link")
    titles = [e.get("title") for e in listed if isinstance(e, dict)] if isinstance(
        listed, list) else None
    check.equal("distractors_D.json gold_without_usable_link (titles, as a set)",
                sorted(titles, key=str) if titles is not None else None,
                sorted(derived.without_link))
    check.equal("distractors_D.json gold_missing_from_db (as a set)",
                sorted(ctx.gold_missing, key=str), sorted(derived.missing_gold))


def _compare_lines_with_db(check: Check, ctx: Context, derived: RuleD, db: FeverousDB) -> None:
    """S0 gold lines and S1 rule-D lines equal json.dumps of the NFC database pages."""
    s0, s1 = ctx.scans["S0"], ctx.scans["S1"]
    wrong = Offenders()
    for title, sha in derived.gold_line_sha.items():
        if s0.line_sha.get(title) != sha:
            wrong.add(f"{title!r}")
    check.none("S0 page lines = NFC database pages", wrong, len(derived.gold_line_sha), "pages")
    check.none("gold pages' NFC title = their id", derived.title_mismatch,
               len(derived.gold_line_sha), "pages")
    wrong = Offenders()
    titles = [rec["title"] for rec in derived.records]
    for title in titles:
        page = db.page(title)
        line = json.dumps(page, ensure_ascii=False).encode("utf-8") if page else b""
        if s1.line_sha.get(title) != hashlib.sha256(line).hexdigest():
            wrong.add(f"{title!r}")
    check.none("S1 rule-D page lines = NFC database pages", wrong, len(titles), "pages")


def check_d(ctx: Context) -> Check:
    """D: everything that needs the FEVEROUS database."""
    check = Check("D", "FEVEROUS database rechecks and rule D")
    if not ctx.use_db:
        check.skipped = "--no-db given: the database checks did not run"
        check.key("not run (--no-db)")
        return check
    path = db_path(ctx.env_file)
    if not path:
        check.expect("database path known", False, f"{DB_ENV} is not set")
        return check
    try:
        db = FeverousDB(Path(path))
    except sqlite3.Error as exc:
        check.expect("database opens read-only", False, f"{Path(path).name}: {exc}")
        return check
    ctx.db_name = Path(path).name
    try:
        _check_pool_flags(check, ctx, db)
        log(f"DB: re-deriving rule D over {len(ctx.gold):,} gold pages")
        derived = derive_rule_d(ctx.gold, db, ctx.params.rule_d_seed)
        _compare_rule_d(check, ctx, derived)
        _compare_lines_with_db(check, ctx, derived, db)
        skips = ", ".join(f"{r} {derived.skips[r]:,}" for r in SKIP_REASONS)
        check.key(f"{len(derived.records):,} rule-D pages re-derived (skips: {skips})")
        check.notes.append(f"Database pages fetched: {db.fetches:,}; existence lookups: "
                           f"{db.lookups:,}; gold missing from the DB: "
                           f"{len(derived.missing_gold):,}; gold without a usable link: "
                           f"{len(derived.without_link):,}.")
    finally:
        db.close()
    return check


# ---------------------------------------------------------------------------------------------
# E: S0 and S1 files
# ---------------------------------------------------------------------------------------------


def _pages_file_items(check: Check, folder: str, scan: PagesScan) -> None:
    """Per-line items of one pages.jsonl."""
    check.expect(f"{folder}/pages.jsonl exists", scan.stats.exists, "")
    if not scan.stats.exists:
        return
    lines = scan.stats.lines
    check.none(f"{folder}: every line parses as a JSON object", scan.bad, lines, "lines")
    check.expect(f"{folder}: no blank lines", scan.blank == 0, f"{scan.blank:,} blank")
    check.none(f"{folder}: every string NFC (keys and values)", scan.non_nfc, lines, "lines")
    check.none(f"{folder}: lines are json.dumps(page, ensure_ascii=False)", scan.not_dumps,
               lines, "lines")
    check.none(f"{folder}: titles unique", scan.duplicates, lines, "lines")


def _s0_titles(check: Check, ctx: Context, s0: PagesScan) -> None:
    """S0 titles = gold pages minus those missing from the DB, sorted."""
    missing = set(ctx.gold_missing)
    want = ctx.gold - missing
    got = set(s0.titles)
    absent = sorted(ctx.gold - got)
    check.expect("S0 titles = gold pages minus gold_missing_from_db", got == want,
                 f"{len(got):,} titles; {len(ctx.gold):,} gold, {len(missing):,} listed missing; "
                 f"{len(got - want):,} extra, {len(want - got):,} absent",
                 [f"extra {t!r}" for t in sorted(got - want)][:5]
                 + [f"absent {t!r}" for t in sorted(want - got)][:5])
    check.equal("gold pages absent from S0 = gold_missing_from_db", absent,
                sorted(missing, key=str))
    check.expect("S0 sorted by title", s0.titles == sorted(s0.titles), "")
    check.key(f"{len(ctx.gold):,} gold pages, {len(absent):,} absent from S0")


def _s1_layout(check: Check, ctx: Context, s0: PagesScan, s1: PagesScan) -> None:
    """S1 = S0 bytes, then the rule-D pages in distractors_D.jsonl order."""
    s0_path, s1_path = (ctx.out_dir / f / "pages.jsonl" for f in FOLDERS)
    prefix = s0_path.is_file() and s1_path.is_file() and starts_with(s1_path, s0_path)
    check.expect("S1/pages.jsonl starts with S0/pages.jsonl byte for byte", prefix,
                 f"S0 {s0.stats.n_bytes:,} bytes, S1 {s1.stats.n_bytes:,} bytes")
    extra = s1.titles[len(s0.titles):]
    d_titles = ctx.distractor_titles
    detail = f"{len(extra):,} extra titles, {len(d_titles):,} rule-D titles"
    if extra != d_titles:
        detail += "; " + _first_difference(extra, d_titles)
    check.expect("S1 titles after S0 = distractors_D.jsonl titles in order", extra == d_titles,
                 detail)
    gold_d = sorted(set(d_titles) & ctx.gold, key=str)
    check.expect("no rule-D title is a gold page", not gold_d, f"{len(gold_d):,}",
                 [repr(t) for t in gold_d])
    check.key(f"S1 = {len(s0.titles):,} + {len(extra):,} pages")


def _claims_jsonl(check: Check, ctx: Context) -> None:
    """claims.jsonl: identical in S0 and S1, raw records + split, claims-file order, ids parse."""
    s0, s1 = ctx.claims_files["S0"], ctx.claims_files["S1"]
    check.expect("claims.jsonl identical in S0 and S1", s0.exists and s1.exists and
                 (s0.n_bytes, s0.sha256) == (s1.n_bytes, s1.sha256),
                 f"S0 {s0.n_bytes:,} bytes {s0.sha256[:12]}, S1 {s1.n_bytes:,} bytes "
                 f"{s1.sha256[:12]}")
    check.equal("claims.jsonl lines", s0.lines, ctx.params.n_claims)
    if not s0.exists:
        return
    wrong, ids_bad, n_ids = Offenders(), Offenders(), 0
    with open(ctx.out_dir / "S0" / "claims.jsonl", encoding="utf-8", newline="") as handle:
        lines = [line.rstrip("\n") for line in handle]
    for index, line in enumerate(lines):
        key = ctx.picked[index] if index < len(ctx.picked) else None
        raw = ctx.raw.picked.get(key) if key else None
        if raw is None or line != json.dumps(raw, ensure_ascii=False):
            same = raw is not None and _loads(line) == raw
            wrong.add(f"line {index + 1} ({tag(*key) if key else 'beyond the claims file'})"
                      + (": same object, different bytes" if same else ""))
        for element_id in _content_ids(_loads(line)):
            n_ids += 1
            try:
                parse_element_id(element_id)
            except (ValueError, TypeError):
                ids_bad.add(f"line {index + 1}: {element_id!r}")
    check.none("claims.jsonl line i = json.dumps(raw record + split) of claims-file line i",
               wrong, len(lines), "lines")
    check.none("every evidence id parses with wikilense.corpus.parse_element_id", ids_bad,
               n_ids, "ids")


def _loads(line: str) -> Any:
    """Return ``json.loads(line)``, or None when it does not parse."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _content_ids(record: Any) -> list[Any]:
    """Return every ``content`` id of every evidence set of a raw record (any shape)."""
    if not isinstance(record, dict) or not isinstance(record.get("evidence"), list):
        return []
    return [element_id for evidence_set in record["evidence"] if isinstance(evidence_set, dict)
            for element_id in evidence_set.get("content") or []]


def check_e(ctx: Context) -> Check:
    """E: the S0 and S1 files."""
    check = Check("E", "S0 / S1 files")
    s0, s1 = ctx.scans["S0"], ctx.scans["S1"]
    _pages_file_items(check, "S0", s0)
    _pages_file_items(check, "S1", s1)
    if ctx.stats[1]:
        check.expect("distractors_D.json readable", False, ctx.stats[1])
    if not ctx.distractors.exists:
        check.expect("distractors_D.jsonl exists", False, "not found")
    _s0_titles(check, ctx, s0)
    _s1_layout(check, ctx, s0, s1)
    _claims_jsonl(check, ctx)
    return check


# ---------------------------------------------------------------------------------------------
# F: the ingest's own loaders, without a database
# ---------------------------------------------------------------------------------------------


class _FakeConn:
    """Stands in for the pymysql connection: the loaders only commit."""

    def commit(self) -> None:
        """Do nothing."""

    def rollback(self) -> None:
        """Do nothing."""


class _CaptureLoader:
    """Replaces ``ingest._Loader``: counts rows, keeps claim_evidence rows and page words."""

    def __init__(self) -> None:
        self.inserted: dict[str, int] = dict.fromkeys(ing._INSERT_COLUMNS, 0)
        self.evidence: list[tuple[Any, ...]] = []
        self.page_words = 0
        self._words_at = ing._INSERT_COLUMNS["page"].index("n_words")

    def add(self, table: str, row: Any) -> None:
        """Count one row; keep it when it is a claim_evidence row."""
        self.inserted[table] = self.inserted.get(table, 0) + 1
        if table == "claim_evidence":
            self.evidence.append(tuple(row))
        elif table == "page":
            self.page_words += int(row[self._words_at])

    def flush(self) -> None:
        """Do nothing: no database."""


@dataclass
class DryRun:
    """What one dry run of the ingest loaders measured."""

    pages: int = 0
    chunks: int = 0
    chunk_words: int = 0
    unit_words: int = 0
    embedding_sha: str = ""
    claims: int = 0
    evidence: tuple[int, int, int] = (0, 0, 0)
    gold_unresolved: Offenders = field(default_factory=Offenders)
    error: str | None = None


def _gold_recount(loader: _CaptureLoader, chunked: set[int], gold_sets: dict[Any, Any],
                  run: DryRun) -> None:
    """Every element of every claim's gold set has a sentence_id that is in some chunk."""
    columns = ing._INSERT_COLUMNS["claim_evidence"]
    at = {name: columns.index(name) for name in ("claim_id", "evidence_set", "sentence_id")}
    by_set: dict[tuple[Any, Any], list[Any]] = defaultdict(list)
    for row in loader.evidence:
        by_set[(row[at["claim_id"]], row[at["evidence_set"]])].append(row[at["sentence_id"]])
    for claim_id, gold_set in gold_sets.items():
        ids = by_set.get((claim_id, gold_set), [])
        if not ids or not all(sid is not None and sid in chunked for sid in ids):
            run.gold_unresolved.add(f"claim {claim_id} set {gold_set!r}")
    page_at, sentence_at = columns.index("page_id"), columns.index("sentence_id")
    run.evidence = (len(loader.evidence),
                    sum(1 for row in loader.evidence if row[page_at] is not None),
                    sum(1 for row in loader.evidence if row[sentence_at] is not None))


def ingest_dry_run(folder: Path, settings: Settings, gold_sets: dict[Any, Any]) -> DryRun:
    """Run ``_Ingest._read_page_meta``, ``_load_pages`` and ``_load_claims`` on ``folder``."""
    run = DryRun()
    log(f"ingest dry run on {folder.name}")
    try:
        obj = ing._Ingest(settings, _FakeConn(), folder / "pages.jsonl", folder / "claims.jsonl",
                          folder / PAGE_META_FILE, reset=True, embedder=None, batch_size=64,
                          use_prefix=True, progress=False)
        loader = _CaptureLoader()
        obj.loader = loader
        obj.next_id = {table: 1 for table in ing._ID_COLUMNS}
        obj._read_page_meta()
        obj._load_pages()
        obj._load_claims()
    except Exception as exc:  # noqa: BLE001 - any ingest error is a FAIL to report, not a crash
        message = str(exc).replace(str(folder.parent) + os.sep, "")  # no home paths in the report
        run.error = f"{type(exc).__name__}: {message}"
        return run
    digest = hashlib.sha256()
    for chunk in obj.chunks:
        digest.update((chunk.embedding_text + "\n").encode("utf-8"))
    run.pages, run.chunks, run.claims = obj.report.n_pages, len(obj.chunks), obj.report.n_claims
    run.chunk_words = sum(chunk.n_words for chunk in obj.chunks)
    run.unit_words = loader.page_words
    run.embedding_sha = digest.hexdigest()
    chunked = {sid for chunk in obj.chunks for sid in chunk.sentence_ids}
    _gold_recount(loader, chunked, gold_sets, run)
    return run


def _settings(check: Check, ctx: Context) -> Settings:
    """Return ``load_settings()`` (checked: 240 words, overlap 1), or a fallback after a FAIL."""
    params = ctx.params
    try:
        settings = load_settings(ctx.env_file)
    except SettingsError as exc:
        check.expect("load_settings() works", False, str(exc).split(":")[0])
        return Settings(db_password="unused", chunk_max_words=params.max_words,
                        chunk_overlap_units=params.overlap_units)
    check.equal("load_settings(): chunk_max_words, chunk_overlap_units",
                (settings.chunk_max_words, settings.chunk_overlap_units),
                (params.max_words, params.overlap_units))
    return settings


def _manifest_numbers(text: str | None) -> dict[str, Any]:
    """Return the Chunks / Words rows and the Embedding texts hex found in a MANIFEST.md."""
    found: dict[str, Any] = {}
    if text is None:
        return found
    labels = {"Chunks": "chunks", "Words in chunks (overlap counted in each chunk)": "chunk_words",
              "Words in text units (page.n_words summed)": "unit_words",
              "Pages (lines in pages.jsonl)": "pages"}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] in labels:
            found[labels[cells[0]]] = cells[1]
        match = _EMBEDDING_HEX_RE.search(line)
        if match:
            found["embedding_text_sha256"] = match.group(1)
    return found


def _compare_run(check: Check, ctx: Context, folder: str, run: DryRun) -> None:
    """Dry-run numbers = own chunking = corpora.json = MANIFEST.md, for one folder."""
    scan = ctx.scans[folder]
    corpora = ctx.corpora[0] if isinstance(ctx.corpora[0], dict) else {}
    entry = corpora.get(folder) if isinstance(corpora.get(folder), dict) else {}
    manifest = _manifest_numbers(ctx.manifests[folder])
    mine = {"pages": scan.pages, "chunks": scan.chunks, "chunk_words": scan.chunk_words,
            "unit_words": scan.unit_words, "embedding_text_sha256": scan.embedding_sha}
    ingest = {"pages": run.pages, "chunks": run.chunks, "chunk_words": run.chunk_words,
              "unit_words": run.unit_words, "embedding_text_sha256": run.embedding_sha}
    for name in CORPUS_NUMBERS:
        shown = f"{ingest[name]:,}" if isinstance(ingest[name], int) else ingest[name]
        values = {"ingest": ingest[name], "own chunking": mine[name],
                  "corpora.json": entry.get(name), "MANIFEST.md": manifest.get(name)}
        ok = (ingest[name] == mine[name] == entry.get(name)
              and manifest.get(name) == shown)
        detail = shown if ok else ", ".join(f"{k} {short(v, 70)}" for k, v in values.items())
        check.expect(f"{folder} {name}: ingest = own chunking = corpora.json = MANIFEST.md", ok,
                     detail)


def check_f(ctx: Context) -> Check:
    """F: chunks exactly as the ingest would make them, and the ingest-level findable recount."""
    check = Check("F", "Ingest dry run (chunks as the ingest makes them)")
    settings = _settings(check, ctx)
    gold_sets = {rec.get("id"): rec.get("gold_set") for rec in ctx.claim_records}
    runs: dict[str, DryRun] = {}
    for folder in FOLDERS:
        run = ingest_dry_run(ctx.out_dir / folder, settings, gold_sets)
        runs[folder] = run
        check.expect(f"{folder}: page and claim loaders ran", run.error is None, run.error or "")
        if run.error:
            continue
        _compare_run(check, ctx, folder, run)
        check.equal(f"{folder}: claims loaded", run.claims, ctx.params.n_claims)
        check.none(f"{folder}: every gold-set element has a sentence_id inside a chunk",
                   run.gold_unresolved, len(gold_sets), "claims")
        total, with_page, with_sentence = run.evidence
        check.notes.append(f"{folder}: {total:,} evidence ids, {with_page:,} with page_id, "
                           f"{with_sentence:,} with sentence_id.")
        check.key(f"{folder} {run.chunks:,} chunks, {total:,} evidence ids "
                  f"({with_sentence:,} with sentence_id)")
    if all(run.error is None for run in runs.values()):
        check.equal("evidence id counts (total, page_id, sentence_id) equal in S0 and S1",
                    runs["S1"].evidence, runs["S0"].evidence)
    return check


# ---------------------------------------------------------------------------------------------
# G: MANIFEST.md and corpora.json
# ---------------------------------------------------------------------------------------------


def _expected_rows(ctx: Context, folder: str) -> tuple[list[str], list[str]]:
    """Return the exact MANIFEST rows to find, and the accepted Gold-pages rows."""
    scan, claims = ctx.scans[folder], ctx.claims_files[folder]
    d_titles = set(ctx.distractor_titles)
    gold_here = sum(1 for title in scan.titles if title in ctx.gold)
    rule_d = sum(1 for title in scan.titles if title in d_titles)
    rows = [
        f"| Pages (lines in pages.jsonl) | {scan.stats.lines:,} |",
        f"| Rule D pages | {rule_d:,} |",
        f"| Chunks | {scan.chunks:,} |",
        f"| Words in chunks (overlap counted in each chunk) | {scan.chunk_words:,} |",
        f"| Words in text units (page.n_words summed) | {scan.unit_words:,} |",
        f"| Claims (lines in claims.jsonl) | {claims.lines:,} |",
        f"| pages.jsonl | {scan.stats.n_bytes} | {scan.stats.sha256} |",
        f"| claims.jsonl | {claims.n_bytes} | {claims.sha256} |",
    ]
    gold_rows = [f"| Gold pages | {gold_here:,} |", f"| Gold pages | {len(ctx.gold):,} |"]
    return rows, gold_rows


def _check_manifest(check: Check, ctx: Context, folder: str) -> None:
    """The exact MANIFEST rows and the Embedding texts hash of one folder."""
    text = ctx.manifests[folder]
    if text is None:
        check.expect(f"{folder}/MANIFEST.md exists", False, "not found")
        return
    lines = {line.strip() for line in text.splitlines()}
    rows, gold_rows = _expected_rows(ctx, folder)
    missing = [row for row in rows if row not in lines]
    check.expect(f"{folder}/MANIFEST.md rows with the recomputed numbers", not missing,
                 f"{len(rows) - len(missing)} of {len(rows)} rows found", missing)
    gold_found = [row for row in gold_rows if row in lines]
    check.expect(f"{folder}/MANIFEST.md Gold pages row", bool(gold_found),
                 gold_found[0] if gold_found else f"expected {gold_rows[0]}")
    hexes = _EMBEDDING_HEX_RE.findall(text)
    want = ctx.scans[folder].embedding_sha
    check.expect(f"{folder}/MANIFEST.md Embedding texts hash", want in hexes,
                 f"found {short(hexes, 90)}, recomputed {want[:16]}...")


def _check_corpora(check: Check, ctx: Context, folder: str) -> None:
    """corpora.json entries of one folder against the recomputed numbers and files."""
    corpora, error = ctx.corpora
    entry = corpora.get(folder) if isinstance(corpora, dict) else None
    if not isinstance(entry, dict):
        check.expect(f"corpora.json has {folder}", False, error or "missing")
        return
    scan = ctx.scans[folder]
    want = {"pages": scan.pages, "chunks": scan.chunks, "chunk_words": scan.chunk_words,
            "unit_words": scan.unit_words, "embedding_text_sha256": scan.embedding_sha}
    check.equal(f"corpora.json {folder} numbers", {k: entry.get(k) for k in want}, want)
    files = entry.get("files") if isinstance(entry.get("files"), dict) else {}
    for name, stats in (("pages.jsonl", scan.stats), ("claims.jsonl", ctx.claims_files[folder])):
        got = files.get(name) if isinstance(files.get(name), dict) else {}
        check.equal(f"corpora.json {folder} files.{name} (bytes, sha256, lines)",
                    (got.get("bytes"), got.get("sha256"), got.get("lines")),
                    (stats.n_bytes, stats.sha256, stats.lines))


def check_g(ctx: Context) -> Check:
    """G: MANIFEST.md and corpora.json against the files."""
    check = Check("G", "MANIFEST.md and corpora.json")
    for folder in FOLDERS:
        _check_manifest(check, ctx, folder)
        _check_corpora(check, ctx, folder)
        scan = ctx.scans[folder]
        check.key(f"{folder} pages.jsonl {scan.stats.n_bytes:,} B, {scan.stats.lines:,} lines")
    return check


# ---------------------------------------------------------------------------------------------
# H: the report
# ---------------------------------------------------------------------------------------------


def guarded(letter: str, title: str, function: Callable[[Context], Check], ctx: Context) -> Check:
    """Run one check; an unexpected exception becomes a FAIL item instead of a crash."""
    log(f"check {letter}: {title}")
    try:
        return function(ctx)
    except Exception as exc:  # noqa: BLE001 - a self-check reports a crash as a FAIL item
        check = Check(letter, title)
        last = traceback.extract_tb(exc.__traceback__)[-1]
        check.expect("check ran to the end", False,
                     f"{type(exc).__name__}: {exc} (line {last.lineno}, {last.name})")
        return check


def render(checks: list[Check], ctx: Context) -> str:
    """Return the SELFCHECK.md text: summary table first, then every item."""
    failed = [check.letter for check in checks if check.failed]
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    db_line = (f"ran on `{ctx.db_name}`" if ctx.db_name else
               "did not run (--no-db)" if not ctx.use_db else "did not run (see check D)")
    out = [
        "# Self-check of the scaled data", "",
        (f"Written by `scripts/scaled_check.py` on {when}. Folder `{ctx.out_dir.name}`, claims "
         f"file `{ctx.claims_path.name}`, {ctx.params.n_claims:,} claims expected."), "",
        (f"**Result: {'FAIL (' + ', '.join(failed) + ')' if failed else 'PASS'}.** "
         f"Database checks (D): {db_line}."), "",
        "| Check | Result | Key numbers |", "|---|---|---|",
    ]
    for check in checks:
        out.append(f"| {check.letter}. {check.title} | {check.result} | "
                   f"{'; '.join(check.key_numbers)} |")
    for check in checks:
        out += ["", f"## {check.letter}. {check.title}: {check.result}", ""]
        if check.skipped:
            out.append(f"Skipped: {check.skipped}.")
        for item in check.items:
            out.append(f"- {'PASS' if item.ok else 'FAIL'}: {item.name}"
                       + (f": {item.detail}" if item.detail else ""))
            out += [f"  - `{example}`" for example in item.examples if not item.ok]
        if check.notes:
            out.append("")
            out += check.notes
    return "\n".join(out) + "\n"


CHECKS: tuple[tuple[str, str, Callable[[Context], Check]], ...] = (
    ("A", "Raw claim files", check_a),
    ("B", "Claims file", check_b),
    ("C", "Pool and seeded sample", check_c),
    ("D", "FEVEROUS database rechecks and rule D", check_d),
    ("E", "S0 / S1 files", check_e),
    ("F", "Ingest dry run (chunks as the ingest makes them)", check_f),
    ("G", "MANIFEST.md and corpora.json", check_g),
)


def run_checks(out_dir: Path, claims_path: Path, train: Path = DEFAULT_TRAIN,
               dev: Path = DEFAULT_DEV, use_db: bool = True, params: Params | None = None,
               env_file: Path | None = None) -> list[Check]:
    """Run checks A to G, write ``OUT_DIR/SELFCHECK.md`` and return the checks."""
    ctx = load_context(Path(out_dir), Path(claims_path), Path(train), Path(dev), use_db,
                       params or Params(), env_file)
    checks = [guarded(letter, title, function, ctx) for letter, title, function in CHECKS]
    (ctx.out_dir / "SELFCHECK.md").write_text(render(checks, ctx), encoding="utf-8")
    return checks


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Return the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True, help="the build's output folder")
    parser.add_argument("--claims", type=Path, required=True, help="the claims file to check")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="raw train claims")
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV, help="raw dev claims")
    parser.add_argument("--no-db", action="store_true", help="skip check D (no database)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, params: Params | None = None,
         env_file: Path | None = None) -> int:
    """Run the self-check from the command line; return 0 when no check failed, else 1."""
    args = parse_args(argv)
    if not args.out_dir.is_dir():
        print(f"no such folder: {args.out_dir}", file=sys.stderr)
        return 1
    checks = run_checks(args.out_dir, args.claims, args.train, args.dev, not args.no_db,
                        params, env_file)
    print(flush=True)
    for check in checks:
        print(f"{check.letter}. {check.title}: {check.result}  ({'; '.join(check.key_numbers)})")
        for item in check.items:
            if not item.ok:
                print(f"     FAIL {item.name}: {item.detail}")
    print(f"written: {args.out_dir.name}/SELFCHECK.md", flush=True)
    return 1 if any(check.failed for check in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
