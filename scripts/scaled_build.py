"""Build the data of the scaled experiment: 3,000 claims, the rule-D distractor pages and the two
corpus folders S0 (gold pages) and S1 (gold pages + rule D), as the owner decided on 2026-11-11
(decisions 1, 2 and 3 of the scaling plan).

Run from the repository root, one subcommand at a time and in this order; each one reads what the
one before wrote into OUT_DIR:

    claims       the pick      -> claims_3000.jsonl.tmp, pool.jsonl, selection.json
    distractors  rule D        -> distractors_D.jsonl, distractors_D.json
    corpora      the folders   -> S0/ and S1/ (pages.jsonl, claims.jsonl, MANIFEST.md), corpora.json
    report       no database   -> SELECTION.md, DISTRACTORS_D.md

    WIKILENSE_FEVEROUS_DB=<feverous_wikiv1.db> .venv/bin/python scripts/scaled_build.py claims \
        --out-dir <OUT_DIR>
    ... distractors --out-dir <OUT_DIR> --claims <OUT_DIR>/claims_3000.jsonl.tmp   (same for corpora, report)

The claims step writes only ``claims_3000.jsonl.tmp``; after the independent self-check
(``scripts/scaled_check.py``) the file is renamed to ``claims_3000.jsonl`` by hand, so this script
can never overwrite the final file. The FEVEROUS DB (``WIKILENSE_FEVEROUS_DB``) is opened
read-only; titles are looked up NFD and every string that comes back is normalised to NFC.

Rules:

1. Claims (decision 1). From train + dev (train first, then dev, each in file order):
   - ``kind`` is ``describe()`` of reviews/SCALING_PLAN_2026-11-11_claim_pool.py, unchanged:
     "single" when every evidence set cites the same one page, "needs2" when every set cites two
     or more pages (sets with an empty ``content`` are ignored);
   - a set is *text-only* when it is not empty and holds only sentences and list items;
   - an element *lands in a chunk* when its page is in the DB, the parsed page title equals the
     evidence page, and its key is in the ``element_keys`` of a chunk made by
     ``chunking.chunk_page(parse_page(page), 240, 1)``; a set is *findable* when it is text-only
     and all its elements land in a chunk; a claim is findable when one of its sets is;
   - eligible are the single and needs2 claims with a text-only set; unfindable ones are left
     out before sampling;
   - ``random.Random(20261111)``; per label (SUPPORTS, REFUTES, NOT ENOUGH INFO)
     ``rng.sample(single, 700)`` then ``rng.sample(needs2, 300)``, as the planning script did.
     A group with too few findable claims stops the step (exit 3): the rule is the owner's.
2. Distractors (decision 2, rule D). Gold articles in ``sorted(gold)`` order shuffled with
   ``random.Random(7)``; each takes its most-linked title (``wikitext.extract_links``, duplicates
   kept, ties broken by the first mention), skipping gold pages, titles taken by an earlier
   article and titles missing from the DB; one page per gold article, no cap.
3. Corpora (decision 3). S0 = every page cited by any evidence set of the 3,000 claims (table
   sets too), sorted by title; S1 = S0's lines, then the rule-D pages in the order taken.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sqlite3
import statistics
import sys
import time
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wikilense.chunking import Chunk, chunk_page, embedding_text
from wikilense.config import DEFAULT_CHUNK_MAX_WORDS, DEFAULT_CHUNK_OVERLAP_UNITS, REPO_ROOT
from wikilense.corpus import iter_claims, parse_element_id
from wikilense.wikitext import ParsedPage, TextUnit, extract_links, parse_page

FEVEROUS_DB_VARIABLE = "WIKILENSE_FEVEROUS_DB"
DEFAULT_TRAIN = REPO_ROOT / "data" / "feverous" / "feverous_train_challenges.jsonl"
DEFAULT_DEV = REPO_ROOT / "data" / "feverous" / "feverous_dev_challenges.jsonl"

SEED = 20261111
WALK_SEED = 7
LABELS: tuple[str, ...] = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
KINDS: tuple[str, ...] = ("single", "needs2")
PER_GROUP: dict[str, int] = {"single": 700, "needs2": 300}
TEXT_TYPES = frozenset({"sentence", "item"})
MAX_WORDS = 240
OVERLAP_UNITS = 1

CLAIMS_TMP = "claims_3000.jsonl.tmp"
POOL_FILE = "pool.jsonl"
SELECTION_JSON = "selection.json"
D_LIST = "distractors_D.jsonl"
D_JSON = "distractors_D.json"
CORPORA_JSON = "corpora.json"
FOLDERS = ("S0", "S1")

PAGE_SQL = "SELECT data FROM wiki WHERE id = ?"
EXISTS_SQL = "SELECT 1 FROM wiki WHERE id = ?"

EXIT_SHORT_GROUP = 3

REASON_PAGE_MISSING = "page not in the DB"
REASON_TITLE = "page title differs from the evidence page"
REASON_KEY = "key not in the parsed page"
REASON_EMPTY = "empty after cleaning"
REASON_HATNOTE = "hatnote"
REASON_OTHER = "chunkable unit in no chunk"


class GroupTooSmall(RuntimeError):
    """A label x kind group has fewer findable claims than the pick needs."""

    def __init__(self, short: dict[tuple[str, str], tuple[int, int]]) -> None:
        self.short = short
        text = ", ".join(f"{lab} {kind}: {have} of {need}" for (lab, kind), (have, need) in short.items())
        super().__init__(f"not enough findable claims: {text}")


# ---- small helpers ------------------------------------------------------------------------------


def log(message: str) -> None:
    """Print one progress line, flushed (the runs go to a log file)."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def nfc_deep(value: Any) -> Any:
    """Return ``value`` with every string in it (dict keys too) normalised to NFC.

    Raises ``ValueError`` when two keys of one dict become equal after normalisation.
    """
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [nfc_deep(item) for item in value]
    if isinstance(value, dict):
        result = {nfc_deep(key): nfc_deep(item) for key, item in value.items()}
        if len(result) != len(value):
            raise ValueError("two keys of one object are equal after NFC normalisation")
        return result
    return value


def write_atomic(path: Path, lines: Iterable[str]) -> None:
    """Write ``lines`` to ``path`` through a ``.part`` file and an atomic rename."""
    part = path.with_name(path.name + ".part")
    with open(part, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as indented JSON (non-ASCII kept) through ``write_atomic``."""
    write_atomic(path, [json.dumps(data, ensure_ascii=False, indent=1) + "\n"])


def read_jsonl(path: Path) -> list[dict]:
    """Return the JSON objects of a JSONL file, one per non-blank line."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def count_lines(path: Path) -> int:
    """Return the number of lines of a text file."""
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_stats(values: list[int]) -> dict[str, float | int]:
    """Return n, median, mean, p90 (``numpy.percentile``, linear interpolation), min and max."""
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median": float(statistics.median(values)),
        "mean": round(statistics.fmean(values), 2),
        "p90": float(np.percentile(values, 90)),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


class FeverousDB:
    """Read-only access to ``feverous_wikiv1.db``: NFD lookups, NFC results."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"{FEVEROUS_DB_VARIABLE} points at {path}, which is not a file")
        uri = f"file:{urllib.parse.quote(str(path))}?mode=ro"
        self.con = sqlite3.connect(uri, uri=True)
        self.n_page_reads = 0
        self.n_exists = 0

    @classmethod
    def from_env(cls) -> FeverousDB:
        """Open the database named by ``WIKILENSE_FEVEROUS_DB``; exit with a message if unset."""
        value = os.environ.get(FEVEROUS_DB_VARIABLE, "")
        if not value:
            raise SystemExit(f"set {FEVEROUS_DB_VARIABLE} to the path of feverous_wikiv1.db")
        return cls(value)

    def page(self, title: str) -> dict | None:
        """Return the NFC page object stored under ``title``, or None when there is none."""
        self.n_page_reads += 1
        row = self.con.execute(PAGE_SQL, (unicodedata.normalize("NFD", title),)).fetchone()
        return None if row is None else nfc_deep(json.loads(row[0]))

    def exists(self, title: str) -> bool:
        """Return True when the database has a row with id ``title`` (NFD)."""
        self.n_exists += 1
        row = self.con.execute(EXISTS_SQL, (unicodedata.normalize("NFD", title),)).fetchone()
        return row is not None


def chunk(parsed: ParsedPage) -> list[Chunk]:
    """Return the page's chunks as the ingest makes them (240 words, overlap 1)."""
    return chunk_page(parsed, max_words=MAX_WORDS, overlap_units=OVERLAP_UNITS)


def check_chunk_settings() -> None:
    """Stop unless the repo's defaults are the tuned 240 words and overlap 1."""
    if (DEFAULT_CHUNK_MAX_WORDS, DEFAULT_CHUNK_OVERLAP_UNITS) != (MAX_WORDS, OVERLAP_UNITS):
        raise SystemExit(
            f"wikilense.config has chunk settings {DEFAULT_CHUNK_MAX_WORDS}/"
            f"{DEFAULT_CHUNK_OVERLAP_UNITS}, expected {MAX_WORDS}/{OVERLAP_UNITS}"
        )


# ---- step 1: claims -----------------------------------------------------------------------------


def describe(c: dict) -> tuple[str, bool, bool, set[str]]:
    """Return (kind, any_text, all_text, all_pages); verbatim from the planning script
    reviews/SCALING_PLAN_2026-11-11_claim_pool.py (only ``TEXT`` renamed)."""
    sets = []
    for s in c["evidence"]:
        parsed = [parse_element_id(e) for e in s["content"]]
        if not parsed:
            continue
        pages = {p for p, _, _ in parsed}
        text_only = all(t in TEXT_TYPES for _, t, _ in parsed)
        sets.append((pages, text_only))
    if not sets:
        return "none", False, False, set()
    all_pages = set().union(*(p for p, _ in sets))
    if all(len(p) >= 2 for p, _ in sets):
        kind = "needs2"
    elif len(all_pages) == 1:
        kind = "single"
    else:
        kind = "either"
    any_text = any(t for _, t in sets)
    all_text = all(t for _, t in sets)
    return kind, any_text, all_text, all_pages


@dataclass
class EvidenceSet:
    """One raw evidence set: full NFC element ids in content order and their (page, key) pairs."""

    elements: list[str]
    units: list[tuple[str, str]]
    text_only: bool
    findable: bool = False


@dataclass
class Candidate:
    """An eligible claim (single or needs2 with a text-only set) and what the pick needs of it."""

    id: int
    split: str
    label: str
    claim: str
    kind: str
    challenge: Any
    gold_pages: list[str]
    sets: list[EvidenceSet]
    findable: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def gold_set(self) -> int:
        """Return the index of the first findable set (the claim must be findable)."""
        return next(i for i, s in enumerate(self.sets) if s.findable)


def parse_sets(record: dict) -> list[EvidenceSet]:
    """Return one ``EvidenceSet`` per entry of the record's raw ``evidence`` list, in order."""
    result = []
    for evidence_set in record["evidence"]:
        parsed = [parse_element_id(e) for e in evidence_set["content"]]
        result.append(
            EvidenceSet(
                elements=[f"{page}_{key}" for page, _, key in parsed],
                units=[(page, key) for page, _, key in parsed],
                text_only=bool(parsed) and all(t in TEXT_TYPES for _, t, _ in parsed),
            )
        )
    return result


def read_candidates(
    train: Path, dev: Path
) -> tuple[list[Candidate], Counter, Counter, dict[str, int]]:
    """Return the eligible claims in pool order, the counts of all claims and of claims with a
    text-only set per (label, kind), and the number of claims per split."""
    candidates: list[Candidate] = []
    all_claims: Counter = Counter()
    text_claims: Counter = Counter()
    per_split: dict[str, int] = {}
    for split, path in (("train", train), ("dev", dev)):
        per_split[split] = 0
        for record in iter_claims(path, split=split):
            per_split[split] += 1
            kind, any_text, _, pages = describe(record)
            all_claims[(record["label"], kind)] += 1
            if any_text:
                text_claims[(record["label"], kind)] += 1
            if kind not in KINDS or not any_text:
                continue
            candidates.append(
                Candidate(
                    id=int(record["id"]),
                    split=split,
                    label=record["label"],
                    claim=record["claim"],
                    kind=kind,
                    challenge=record.get("challenge"),
                    gold_pages=sorted(pages),
                    sets=parse_sets(record),
                )
            )
    return candidates, all_claims, text_claims, per_split


@dataclass
class ChunkLookup:
    """For the needed (page, key) pairs: the text of those inside a chunk, the reason for others."""

    text: dict[tuple[str, str], str] = field(default_factory=dict)
    reason: dict[tuple[str, str], str] = field(default_factory=dict)
    pages_missing: list[str] = field(default_factory=list)
    pages_title_differs: list[str] = field(default_factory=list)
    n_pages: int = 0


def classify(units: dict[str, TextUnit], in_chunks: set[str], key: str) -> str | None:
    """Return None when ``key`` is in a chunk, else why it is not (``units``: key -> unit)."""
    if key in in_chunks:
        return None
    unit = units.get(key)
    if unit is None:
        return REASON_KEY
    if not unit.text:
        return REASON_EMPTY
    if unit.is_hatnote:
        return REASON_HATNOTE
    return REASON_OTHER


def lookup_chunks(db: FeverousDB, needed: dict[str, set[str]], every: int = 2000) -> ChunkLookup:
    """Fetch every needed page once (sorted titles), chunk it, and classify its needed keys."""
    out = ChunkLookup()
    start = time.perf_counter()
    for number, title in enumerate(sorted(needed), start=1):
        page = db.page(title)
        if page is None:
            out.pages_missing.append(title)
            for key in needed[title]:
                out.reason[(title, key)] = REASON_PAGE_MISSING
            continue
        out.n_pages += 1
        parsed = parse_page(page)
        if parsed.title != title:
            out.pages_title_differs.append(title)
            for key in needed[title]:
                out.reason[(title, key)] = REASON_TITLE
            continue
        in_chunks = {key for ch in chunk(parsed) for key in ch.element_keys}
        units = {u.element_key: u for u in parsed.units}
        for key in needed[title]:
            why = classify(units, in_chunks, key)
            if why is None:
                out.text[(title, key)] = units[key].text
            else:
                out.reason[(title, key)] = why
        if number % every == 0:
            log(f"chunked {number:,} of {len(needed):,} pages [{time.perf_counter() - start:.0f} s]")
    return out


def mark_findable(candidates: list[Candidate], found: ChunkLookup) -> None:
    """Set ``findable`` on every set and claim, and the sorted reasons of unfindable claims."""
    for cand in candidates:
        reasons: set[str] = set()
        for s in cand.sets:
            s.findable = s.text_only and all(u in found.text for u in s.units)
            if s.text_only and not s.findable:
                reasons.update(found.reason[u] for u in s.units if u not in found.text)
        cand.findable = any(s.findable for s in cand.sets)
        cand.reasons = [] if cand.findable else sorted(reasons)


def pick(
    candidates: list[Candidate],
    seed: int = SEED,
    per_group: dict[str, int] | None = None,
    labels: tuple[str, ...] = LABELS,
) -> list[Candidate]:
    """Return the seeded pick: per label, ``rng.sample(single, n)`` then ``rng.sample(needs2, n)``
    over the findable candidates in pool order. Raises ``GroupTooSmall`` before sampling when a
    group cannot be filled."""
    per_group = PER_GROUP if per_group is None else per_group
    pools = {
        (lab, kind): [c for c in candidates if c.label == lab and c.kind == kind and c.findable]
        for lab in labels
        for kind in KINDS
    }
    short = {
        key: (len(pool), per_group[key[1]])
        for key, pool in pools.items()
        if len(pool) < per_group[key[1]]
    }
    if short:
        raise GroupTooSmall(short)
    rng = random.Random(seed)
    picked: list[Candidate] = []
    for lab in labels:
        for kind in KINDS:
            picked += rng.sample(pools[(lab, kind)], per_group[kind])
    return picked


def claim_line(cand: Candidate, found: ChunkLookup) -> str:
    """Return the claims_3000.jsonl line of one picked claim (keys in the contract's order)."""
    gold_set = cand.gold_set
    record = {
        "id": cand.id,
        "split": cand.split,
        "label": cand.label,
        "claim": cand.claim,
        "kind": cand.kind,
        "challenge": cand.challenge,
        "gold_pages": cand.gold_pages,
        "evidence_sets": [
            {"elements": s.elements, "text_only": s.text_only, "findable": s.findable}
            for s in cand.sets
        ],
        "gold_set": gold_set,
        "gold_evidence": [
            {"element": element, "page": page, "text": found.text[(page, key)]}
            for element, (page, key) in zip(
                cand.sets[gold_set].elements, cand.sets[gold_set].units, strict=True
            )
        ],
    }
    return json.dumps(record, ensure_ascii=False) + "\n"


def nested(counter: Counter, outer: Iterable[str], inner: Iterable[str]) -> dict:
    """Return ``{a: {b: counter[(a, b)]}}`` for the given key orders."""
    inner = list(inner)
    return {a: {b: counter[(a, b)] for b in inner} for a in outer}


def selection_stats(
    candidates: list[Candidate],
    picked: list[Candidate],
    all_claims: Counter,
    text_claims: Counter,
    per_split: dict[str, int],
    found: ChunkLookup,
    needed: dict[str, set[str]],
    gold_missing: list[str],
) -> dict:
    """Return the measured numbers of the claims step (selection.json)."""
    eligible = Counter((c.label, c.kind) for c in candidates)
    findable = Counter((c.label, c.kind) for c in candidates if c.findable)
    n_picked = Counter((c.label, c.kind) for c in picked)
    pool = {
        lab: {
            kind: {
                "claims": all_claims[(lab, kind)],
                "with_text_only_set": text_claims[(lab, kind)],
                "eligible": eligible[(lab, kind)],
                "findable": findable[(lab, kind)],
                "unfindable": eligible[(lab, kind)] - findable[(lab, kind)],
                "picked": n_picked[(lab, kind)],
                "spare": findable[(lab, kind)] - n_picked[(lab, kind)],
            }
            for kind in KINDS
        }
        for lab in LABELS
    }
    others = {
        lab: {
            kind: {"claims": all_claims[(lab, kind)], "with_text_only_set": text_claims[(lab, kind)]}
            for kind in ("either", "none")
        }
        for lab in LABELS
    }
    unfindable: dict = {lab: {kind: Counter() for kind in KINDS} for lab in LABELS}
    for c in candidates:
        if not c.findable:
            unfindable[c.label][c.kind][" + ".join(c.reasons)] += 1
    element_reasons = Counter(found.reason.values())
    gold = sorted(set().union(*(c.gold_pages for c in picked)))
    gold_from_gold_sets = set()
    gold_text_sets = set()
    for c in picked:
        gold_from_gold_sets.update(p for p, _ in c.sets[c.gold_set].units)
        for s in c.sets:
            if s.text_only:
                gold_text_sets.update(p for p, _ in s.units)
    group = lambda c: f"{c.label} / {c.kind}"
    split = defaultdict(Counter)
    challenge = defaultdict(Counter)
    for c in picked:
        split[group(c)][c.split] += 1
        challenge[group(c)][str(c.challenge)] += 1
    n_gold_pages = Counter(min(len(c.gold_pages), 5) for c in picked if c.kind == "needs2")
    n_findable_sets = Counter(sum(s.findable for s in c.sets) for c in picked)
    n_sets = Counter(len(c.sets) for c in picked)
    return {
        "inputs": {"claims_per_split": per_split},
        "rule": {
            "seed": SEED,
            "per_group": PER_GROUP,
            "labels": list(LABELS),
            "chunking": {"max_words": MAX_WORDS, "overlap_units": OVERLAP_UNITS},
        },
        "pool": pool,
        "other_kinds": others,
        "unfindable_claims_by_reason": {
            lab: {kind: dict(unfindable[lab][kind].most_common()) for kind in KINDS} for lab in LABELS
        },
        "unfindable_elements_by_reason": dict(element_reasons.most_common()),
        "candidate_pages": {
            "pages_needed": len(needed),
            "pages_read": found.n_pages,
            "pages_missing_from_db": len(found.pages_missing),
            "pages_title_differs": len(found.pages_title_differs),
            "examples_missing": found.pages_missing[:20],
            "examples_title_differs": found.pages_title_differs[:20],
            "elements_needed": sum(len(v) for v in needed.values()),
            "elements_in_a_chunk": len(found.text),
        },
        "picked": {
            "n": len(picked),
            "groups": nested(n_picked, LABELS, KINDS),
            "split": {k: dict(sorted(v.items())) for k, v in split.items()},
            "split_total": dict(sorted(Counter(c.split for c in picked).items())),
            "challenge": {k: dict(v.most_common()) for k, v in challenge.items()},
            "challenge_total": dict(Counter(str(c.challenge) for c in picked).most_common()),
            "needs2_gold_pages_per_claim": {
                ("5+" if k == 5 else str(k)): v for k, v in sorted(n_gold_pages.items())
            },
            "evidence_sets_per_claim": {str(k): v for k, v in sorted(n_sets.items())},
            "findable_sets_per_claim": {str(k): v for k, v in sorted(n_findable_sets.items())},
            "claims_with_a_table_set": sum(any(not s.text_only for s in c.sets) for c in picked),
            "claim_text_not_nfc": [
                c.id for c in picked if unicodedata.normalize("NFC", c.claim) != c.claim
            ],
            "gold_evidence_elements": sum(len(c.sets[c.gold_set].units) for c in picked),
        },
        "gold_pages": {
            "n": len(gold),
            "in_a_gold_set": len(gold_from_gold_sets),
            "in_a_text_only_set": len(gold_text_sets),
            "only_in_sets_with_tables": len(set(gold) - gold_text_sets),
            "missing_from_db": gold_missing,
            "with_underscore": sum("_" in t for t in gold),
        },
    }


def run_claims(args: argparse.Namespace) -> int:
    """The claims step: pool, findable check, seeded pick, claims_3000.jsonl.tmp."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    check_chunk_settings()
    db = FeverousDB.from_env()
    candidates, all_claims, text_claims, per_split = read_candidates(args.train, args.dev)
    log(f"read {sum(per_split.values()):,} claims {per_split}; {len(candidates):,} eligible")
    needed: dict[str, set[str]] = defaultdict(set)
    for cand in candidates:
        for s in cand.sets:
            if s.text_only:
                for page, key in s.units:
                    needed[page].add(key)
    log(f"findable check: {len(needed):,} pages, {sum(len(v) for v in needed.values()):,} elements")
    found = lookup_chunks(db, needed)
    mark_findable(candidates, found)
    log(
        f"findable: {sum(c.findable for c in candidates):,} of {len(candidates):,}; "
        f"pages missing {len(found.pages_missing)}, title differs {len(found.pages_title_differs)}"
    )
    write_atomic(
        out / POOL_FILE,
        (
            json.dumps(
                {"id": c.id, "split": c.split, "label": c.label, "kind": c.kind, "findable": c.findable}
                | ({} if c.findable else {"reasons": c.reasons}),
                ensure_ascii=False,
            )
            + "\n"
            for c in candidates
        ),
    )
    try:
        picked = pick(candidates)
    except GroupTooSmall as exc:
        log(f"STOP: {exc}")
        stats = selection_stats(candidates, [], all_claims, text_claims, per_split, found, needed, [])
        stats["stopped"] = str(exc)
        write_json(out / SELECTION_JSON, stats)
        return EXIT_SHORT_GROUP
    gold = sorted(set().union(*(c.gold_pages for c in picked)))
    read_ok = set(needed) - set(found.pages_missing)
    gold_missing = [t for t in gold if t not in read_ok and not db.exists(t)]
    write_atomic(out / CLAIMS_TMP, (claim_line(c, found) for c in picked))
    stats = selection_stats(
        candidates, picked, all_claims, text_claims, per_split, found, needed, gold_missing
    )
    stats["seconds"] = round(time.perf_counter() - start, 1)
    stats["db_reads"] = {"pages": db.n_page_reads, "exists": db.n_exists}
    write_json(out / SELECTION_JSON, stats)
    log(
        f"wrote {len(picked):,} claims to {CLAIMS_TMP}; {len(gold):,} gold pages "
        f"({len(gold_missing)} missing from the DB) [{stats['seconds']} s]"
    )
    return 0


# ---- step 2: rule D ----------------------------------------------------------------------------


def rank_links(links: list[tuple[str, str]]) -> list[tuple[str, int]]:
    """Return the distinct link titles as (title, times linked), most linked first, ties by the
    first mention."""
    count: Counter = Counter()
    first: dict[str, int] = {}
    for position, (title, _) in enumerate(links):
        count[title] += 1
        first.setdefault(title, position)
    return [(t, count[t]) for t in sorted(count, key=lambda t: (-count[t], first[t]))]


def gold_titles(claims: list[dict]) -> set[str]:
    """Return the union of the claims' ``gold_pages``."""
    return set().union(*(c["gold_pages"] for c in claims))


def walk_order(gold: set[str], seed: int = WALK_SEED) -> list[str]:
    """Return ``sorted(gold)`` shuffled with ``random.Random(seed)``."""
    order = sorted(gold)
    random.Random(seed).shuffle(order)
    return order


def rule_d(db: FeverousDB, gold: set[str], every: int = 1000) -> tuple[list[dict], dict]:
    """Walk the gold articles and return (the rule-D pages in the order taken, the statistics)."""
    order = walk_order(gold)
    taken: set[str] = set()
    exists: dict[str, bool] = {}
    chosen: list[dict] = []
    skips: Counter = Counter()
    without: list[dict] = []
    missing_gold: list[str] = []
    n_links: list[int] = []
    start = time.perf_counter()
    for walk_index, g in enumerate(order):
        page = db.page(g)
        if page is None:
            missing_gold.append(g)
            continue
        links = extract_links(page)
        ranked = rank_links(links)
        n_links.append(len(links))
        mine: Counter = Counter()
        for rank, (title, times) in enumerate(ranked):
            if title in gold:
                mine["gold"] += 1
                continue
            if title in taken:
                mine["taken"] += 1
                continue
            if title not in exists:
                exists[title] = db.exists(title)
            if not exists[title]:
                mine["missing"] += 1
                continue
            taken.add(title)
            chosen.append(
                {
                    "title": title,
                    "gold": g,
                    "walk_index": walk_index,
                    "rank": rank,
                    "times_linked": times,
                    "skipped": dict(mine),
                }
            )
            break
        else:
            without.append(
                {"title": g, "n_links": len(links), "n_distinct": len(ranked), "skipped": dict(mine)}
            )
        skips.update(mine)
        if (walk_index + 1) % every == 0:
            log(f"walked {walk_index + 1:,} of {len(order):,} [{time.perf_counter() - start:.0f} s]")
    ranks = Counter(c["rank"] for c in chosen)
    stats = {
        "rule": {
            "walk": f"sorted(gold) shuffled with random.Random({WALK_SEED})",
            "links": "wikitext.extract_links, duplicates kept; ties by first mention",
            "skip_order": ["gold", "taken", "missing"],
        },
        "gold_articles": len(order),
        "gold_missing_from_db": missing_gold,
        "pages": len(chosen),
        "gold_without_usable_link": without,
        "rank_taken": {str(k): v for k, v in sorted(ranks.items())},
        "skips": {k: skips[k] for k in ("gold", "taken", "missing")},
        "times_linked_of_taken": {
            str(k): v for k, v in sorted(Counter(min(c["times_linked"], 10) for c in chosen).items())
        },
        "gold_links_per_article": summary_stats(n_links),
        "distinct_titles_checked_in_db": len(exists),
        "distinct_titles_missing_from_db": sum(not v for v in exists.values()),
        "seconds": round(time.perf_counter() - start, 1),
    }
    return chosen, stats


def run_distractors(args: argparse.Namespace) -> int:
    """The distractors step: rule D over the gold pages of the claims file."""
    out = Path(args.out_dir)
    db = FeverousDB.from_env()
    claims = read_jsonl(Path(args.claims))
    gold = gold_titles(claims)
    log(f"rule D over {len(gold):,} gold articles")
    chosen, stats = rule_d(db, gold)
    stats["db_reads"] = {"pages": db.n_page_reads, "exists": db.n_exists}
    stats["claims_sha256"] = sha256_of_file(Path(args.claims))
    write_atomic(
        out / D_LIST,
        (
            json.dumps({k: c[k] for k in ("title", "gold", "walk_index", "rank", "times_linked")},
                       ensure_ascii=False) + "\n"
            for c in chosen
        ),
    )
    write_json(out / D_JSON, stats)
    log(
        f"rule D: {stats['pages']:,} pages; {len(stats['gold_without_usable_link'])} gold articles "
        f"without a usable link; skips {stats['skips']} [{stats['seconds']} s]"
    )
    return 0


# ---- step 3: corpora ---------------------------------------------------------------------------


def raw_claim_records(
    wanted: list[tuple[str, int]], train: Path, dev: Path
) -> Iterator[dict]:
    """Yield the raw FEVEROUS records (plus ``split``) of ``wanted`` (split, id), in that order."""
    keys = set(wanted)
    found: dict[tuple[str, int], dict] = {}
    for split, path in (("train", train), ("dev", dev)):
        for record in iter_claims(path, split=split):
            key = (split, int(record["id"]))
            if key in keys:
                found[key] = record
    missing = keys - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} claims not found in the FEVEROUS files, e.g. {min(missing)}")
    for key in wanted:
        yield found[key]


@dataclass
class FolderStats:
    """What MANIFEST.md and corpora.json report for one corpus folder."""

    pages: int = 0
    gold_pages: int = 0
    d_pages: int = 0
    chunks: int = 0
    chunk_words: int = 0
    unit_words: int = 0
    units: int = 0
    units_empty: int = 0
    units_hatnote: int = 0
    sections: int = 0
    pages_without_chunks: int = 0
    digest: Any = field(default_factory=hashlib.sha256)

    def add(self, parsed: ParsedPage, chunks: list[Chunk], gold: bool) -> None:
        """Count one page and feed its chunks' embedding texts to the digest, in ingest order."""
        self.pages += 1
        self.gold_pages += int(gold)
        self.d_pages += int(not gold)
        self.chunks += len(chunks)
        self.chunk_words += sum(ch.n_words for ch in chunks)
        self.unit_words += parsed.stats["n_words"]
        self.units += len(parsed.units)
        self.units_empty += sum(not u.text for u in parsed.units)
        self.units_hatnote += sum(u.is_hatnote for u in parsed.units)
        self.sections += len(parsed.sections)
        self.pages_without_chunks += int(not chunks)
        for ch in chunks:
            text = embedding_text(parsed.title, parsed.sections[ch.section_ordinal].path, ch.text)
            self.digest.update((text + "\n").encode("utf-8"))

    def as_dict(self) -> dict:
        """Return the counts and the hex digest."""
        data = {k: v for k, v in self.__dict__.items() if k != "digest"}
        data["embedding_text_sha256"] = self.digest.hexdigest()
        return data


def evidence_resolution(claims: list[dict], units_of: dict[str, set[str]]) -> dict[str, int]:
    """Return how many evidence ids the ingest would resolve to a page and to a text unit."""
    total = with_page = with_unit = 0
    for c in claims:
        for s in c["evidence_sets"]:
            for element in s["elements"]:
                page, element_type, key = parse_element_id(element)
                total += 1
                if page in units_of:
                    with_page += 1
                    with_unit += int(element_type in TEXT_TYPES and key in units_of[page])
    return {"evidence_ids": total, "with_a_page": with_page, "with_a_text_unit": with_unit}


def manifest(name: str, stats: dict, groups: dict, d_note: str) -> str:
    """Return the MANIFEST.md of one corpus folder."""
    files = stats["files"]
    rows = [
        ("Pages (lines in pages.jsonl)", stats["pages"]),
        ("Gold pages", stats["gold_pages"]),
        ("Rule D pages", stats["d_pages"]),
        ("Chunks", stats["chunks"]),
        ("Words in chunks (overlap counted in each chunk)", stats["chunk_words"]),
        ("Words in text units (page.n_words summed)", stats["unit_words"]),
        ("Text units (sentences and list items)", stats["units"]),
        ("Text units empty after cleaning (in no chunk)", stats["units_empty"]),
        ("Text units that are hatnotes (in no chunk)", stats["units_hatnote"]),
        ("Sections (lead included)", stats["sections"]),
        ("Pages without any chunk", stats["pages_without_chunks"]),
        ("Claims (lines in claims.jsonl)", stats["claims"]),
        ("Evidence element ids", stats["evidence"]["evidence_ids"]),
        ("Evidence ids whose page is in the folder", stats["evidence"]["with_a_page"]),
        ("Evidence ids that are a text unit of the folder", stats["evidence"]["with_a_text_unit"]),
    ]
    group_rows = "\n".join(
        f"| {lab} | {groups[lab]['single']:,} | {groups[lab]['needs2']:,} |" for lab in LABELS
    )
    lines = [
        f"# Corpus {name}",
        "",
        "Built by `scripts/scaled_build.py corpora` for the scaled experiment (owner's decisions of",
        "2026-11-11: 3,000 claims, rule D, the sizes S0 and S1). Load it with",
        f"`WIKILENSE_DB_NAME=wikilense_{name.lower()} wikilense ingest --corpus-dir <this folder>`.",
        "",
        "## Contents",
        "",
        "- `pages.jsonl`: one FEVEROUS page object per line, read from `feverous_wikiv1.db` (the",
        "  December 2020 Wikipedia of FEVEROUS) and normalised to Unicode NFC (keys and values).",
        "  First the gold pages, sorted by title: every page cited by any evidence set of the",
        "  3,000 claims, table-only sets included.",
        f"  {d_note}",
        "- `claims.jsonl`: the 3,000 raw FEVEROUS claim records plus `\"split\"`, in the order of",
        "  `claims_3000.jsonl` (the same file in S0 and S1).",
        "",
        "## Counts",
        "",
        "| Count | Value |",
        "|---|---|",
        *(f"| {label} | {value:,} |" for label, value in rows),
        "",
        "Claims per group:",
        "",
        "| Label | single | needs2 |",
        "|---|---|---|",
        group_rows,
        "",
        "## Chunking",
        "",
        f"`chunking.chunk_page(parse_page(page), {MAX_WORDS}, {OVERLAP_UNITS})`, the ingest's own path;",
        "embedding text `chunking.embedding_text(title, section path, chunk text)` (the title prefix).",
        f"Embedding texts (SHA-256 over each text plus a newline, in ingest order): `{stats['embedding_text_sha256']}`",
        "",
        "## Files",
        "",
        "| File | Bytes | SHA-256 |",
        "|---|---|---|",
        f"| pages.jsonl | {files['pages.jsonl']['bytes']} | {files['pages.jsonl']['sha256']} |",
        f"| claims.jsonl | {files['claims.jsonl']['bytes']} | {files['claims.jsonl']['sha256']} |",
        "",
    ]
    return "\n".join(lines)


def run_corpora(args: argparse.Namespace) -> int:
    """The corpora step: write S0/ and S1/ and corpora.json."""
    out = Path(args.out_dir)
    start = time.perf_counter()
    check_chunk_settings()
    db = FeverousDB.from_env()
    claims = read_jsonl(Path(args.claims))
    gold = sorted(gold_titles(claims))
    d_list = read_jsonl(out / D_LIST)
    made_from = json.loads((out / D_JSON).read_text(encoding="utf-8")).get("claims_sha256")
    if made_from != sha256_of_file(Path(args.claims)):
        raise SystemExit(f"{D_LIST} was made from another claims file; run the distractors step again")
    for name in FOLDERS:
        (out / name).mkdir(exist_ok=True)
    parts = {name: out / name / "pages.jsonl.part" for name in FOLDERS}
    stats = {name: FolderStats() for name in FOLDERS}
    units_of: dict[str, set[str]] = {}
    chunks_per_page: dict[str, dict[str, int]] = {"gold": {}, "D": {}}
    gold_missing: list[str] = []
    seen: set[str] = set()
    with contextlib.ExitStack() as stack:
        handles = {
            name: stack.enter_context(open(path, "w", encoding="utf-8")) for name, path in parts.items()
        }
        for number, title in enumerate(gold, start=1):
            page = db.page(title)
            if page is None:
                gold_missing.append(title)
                continue
            parsed = parse_page(page)
            if parsed.title != title or title in seen:
                raise SystemExit(f"gold page {title!r}: parsed title {parsed.title!r} or duplicate")
            seen.add(title)
            chunks = chunk(parsed)
            line = json.dumps(page, ensure_ascii=False) + "\n"
            for name in FOLDERS:
                handles[name].write(line)
                stats[name].add(parsed, chunks, gold=True)
            units_of[title] = {u.element_key for u in parsed.units}
            chunks_per_page["gold"][title] = len(chunks)
            if number % 1000 == 0:
                log(f"gold pages {number:,} of {len(gold):,} [{time.perf_counter() - start:.0f} s]")
        for number, entry in enumerate(d_list, start=1):
            title = entry["title"]
            page = db.page(title)
            if page is None:
                raise SystemExit(f"rule-D page {title!r} is not in the DB")
            parsed = parse_page(page)
            if parsed.title != title or title in seen:
                raise SystemExit(f"rule-D page {title!r}: parsed title {parsed.title!r} or duplicate")
            seen.add(title)
            chunks = chunk(parsed)
            handles["S1"].write(json.dumps(page, ensure_ascii=False) + "\n")
            stats["S1"].add(parsed, chunks, gold=False)
            chunks_per_page["D"][title] = len(chunks)
            if number % 1000 == 0:
                log(f"rule-D pages {number:,} of {len(d_list):,} [{time.perf_counter() - start:.0f} s]")
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
    records = list(raw_claim_records([(c["split"], c["id"]) for c in claims], args.train, args.dev))
    claim_lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in records]
    groups = {lab: {kind: 0 for kind in KINDS} for lab in LABELS}
    for c in claims:
        groups[c["label"]][c["kind"]] += 1
    resolution = evidence_resolution(claims, units_of)
    result: dict[str, Any] = {}
    for name in FOLDERS:
        folder = out / name
        os.replace(parts[name], folder / "pages.jsonl")
        write_atomic(folder / "claims.jsonl", claim_lines)
        data = stats[name].as_dict()
        data["claims"] = len(claim_lines)
        data["evidence"] = resolution
        data["files"] = {
            fn: {
                "bytes": (folder / fn).stat().st_size,
                "sha256": sha256_of_file(folder / fn),
                "lines": count_lines(folder / fn),
            }
            for fn in ("pages.jsonl", "claims.jsonl")
        }
        note = (
            "No other pages." if name == "S0" else
            "Then the rule-D pages, in the order the walk took them (`distractors_D.jsonl`)."
        )
        write_atomic(folder / "MANIFEST.md", [manifest(name, data, groups, note)])
        result[name] = data
    result["gold_missing_from_db"] = gold_missing
    result["chunks_per_gold_page"] = summary_stats(list(chunks_per_page["gold"].values()))
    result["chunks_per_d_page"] = summary_stats(list(chunks_per_page["D"].values()))
    result["d_pages_without_chunks"] = sum(not n for n in chunks_per_page["D"].values())
    result["chunks_per_page"] = chunks_per_page
    result["seconds"] = round(time.perf_counter() - start, 1)
    write_json(out / CORPORA_JSON, result)
    log(
        f"S0: {stats['S0'].pages:,} pages, {stats['S0'].chunks:,} chunks; "
        f"S1: {stats['S1'].pages:,} pages, {stats['S1'].chunks:,} chunks [{result['seconds']} s]"
    )
    return 0


# ---- step 4: the reports -----------------------------------------------------------------------


def table(header: list[str], rows: list[list[Any]]) -> list[str]:
    """Return a Markdown table; ints get thousands separators."""
    fmt = lambda v: f"{v:,}" if isinstance(v, int) else str(v)
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(fmt(v) for v in row) + " |" for row in rows),
    ]


def selection_md(sel: dict, corpora: dict, claims_name: str) -> str:
    """Return SELECTION.md."""
    pool = sel["pool"]
    picked = sel["picked"]
    gp = sel["gold_pages"]
    cp = corpora["chunks_per_gold_page"]
    out = [
        "# Selection of the 3,000 claims (scaled experiment)",
        "",
        "Written by `scripts/scaled_build.py report` from `selection.json` and `corpora.json`;",
        "every number below was measured by `scripts/scaled_build.py claims` and `corpora` on the",
        "FEVEROUS train + dev files and `feverous_wikiv1.db`.",
        "",
        "## Rule (owner's decision 1, 2026-11-11)",
        "",
        "- Kinds as `describe()` in the planning script: **single** = every evidence set cites the",
        "  same one page; **needs2** = every set cites two or more pages.",
        "- Eligible: a single or needs2 claim with a text-only evidence set (sentences and list",
        "  items only).",
        f"- Findable: a text-only set whose every element is in a chunk of `chunk_page(parse_page(page), {MAX_WORDS}, {OVERLAP_UNITS})`",
        "  (the ingest's chunker; hatnotes and empty units are in no chunk by design). Unfindable",
        "  claims are left out before sampling.",
        f"- `random.Random({SEED})`; per label `sample(single, 700)` then `sample(needs2, 300)`.",
        "- The challenge tag is recorded, not balanced on.",
        "",
        "## Pool per group, before and after the findable check",
        "",
        *table(
            ["Label", "Kind", "All claims of the kind", "Eligible (text-only set)", "Findable",
             "Left out (unfindable)", "Picked", "Spare"],
            [
                [lab, kind, g["claims"], g["eligible"], g["findable"], g["unfindable"], g["picked"],
                 g["spare"]]
                for lab in LABELS
                for kind in KINDS
                for g in [pool[lab][kind]]
            ],
        ),
        "",
        "Why the unfindable claims fail (all reasons over their text-only sets):",
        "",
        *table(
            ["Label", "Kind", "Reason", "Claims"],
            [
                [lab, kind, reason, n]
                for lab in LABELS
                for kind in KINDS
                for reason, n in sel["unfindable_claims_by_reason"][lab][kind].items()
            ],
        ),
        "",
        "Elements of text-only sets in no chunk, by reason:",
        "",
        *table(["Reason", "Elements"], [[k, v] for k, v in sel["unfindable_elements_by_reason"].items()]),
        "",
        ("Pages read for the findable check: "
        f"{sel['candidate_pages']['pages_needed']:,} needed, "
        f"{sel['candidate_pages']['pages_missing_from_db']:,} not in the DB, "
        f"{sel['candidate_pages']['pages_title_differs']:,} whose parsed title differs; "
        f"{sel['candidate_pages']['elements_in_a_chunk']:,} of "
        f"{sel['candidate_pages']['elements_needed']:,} needed elements are in a chunk."),
        "",
        "## The 3,000 claims",
        "",
        *table(["Label", "single", "needs2", "Total"],
               [[lab, picked["groups"][lab]["single"], picked["groups"][lab]["needs2"],
                 picked["groups"][lab]["single"] + picked["groups"][lab]["needs2"]] for lab in LABELS]),
        "",
        "Split per group:",
        "",
        *table(["Group", "train", "dev"],
               [[g, v.get("train", 0), v.get("dev", 0)] for g, v in picked["split"].items()]
               + [["all", picked["split_total"].get("train", 0), picked["split_total"].get("dev", 0)]]),
        "",
        "Challenge tags per group (recorded, not balanced on):",
        "",
    ]
    tags = list(picked["challenge_total"])
    out += table(
        ["Group", *tags],
        [[g, *(v.get(t, 0) for t in tags)] for g, v in picked["challenge"].items()]
        + [["all", *(picked["challenge_total"][t] for t in tags)]],
    )
    out += [
        "",
        f"- needs2 claims by number of gold pages: {picked['needs2_gold_pages_per_claim']}.",
        (f"- Evidence sets per claim: {picked['evidence_sets_per_claim']}; findable sets per claim: "
        f"{picked['findable_sets_per_claim']}."),
        (f"- Claims with a set that holds a table cell, header cell or caption: "
        f"{picked['claims_with_a_table_set']:,} (their pages are gold pages too)."),
        f"- Elements in the gold sets (`gold_evidence`): {picked['gold_evidence_elements']:,}.",
        f"- Claim texts that are not NFC (kept as FEVEROUS has them): {picked['claim_text_not_nfc'] or 'none'}.",
        "",
        "## Gold pages",
        "",
        f"- Gold pages (every page cited by any evidence set of the 3,000 claims): **{gp['n']:,}**;",
        f"  {gp['in_a_gold_set']:,} of them are in a claim's gold set, {gp['in_a_text_only_set']:,} in some",
        f"  text-only set, {gp['only_in_sets_with_tables']:,} only in sets with table elements.",
        f"- Missing from the FEVEROUS DB: {len(gp['missing_from_db'])}"
        + (f" ({', '.join(gp['missing_from_db'])})" if gp["missing_from_db"] else "") + ".",
        (f"- Chunks per gold page: median {cp['median']:g}, mean {cp['mean']:g}, p90 {cp['p90']:g}"
        f" (`numpy.percentile`, linear), max {cp['max']:,}; min {cp['min']:,}; total {cp['sum']:,}."),
        f"- Gold pages without any chunk: {corpora['S0']['pages_without_chunks']:,}.",
        "",
        f"The claims file is `{claims_name}`; its format is the contract with the LLM runner (chat A).",
        "",
    ]
    return "\n".join(out)


def distractors_md(dj: dict, corpora: dict) -> str:
    """Return DISTRACTORS_D.md."""
    cd = corpora["chunks_per_d_page"]
    without = dj["gold_without_usable_link"]
    out = [
        "# Rule D distractor pages (scaled experiment)",
        "",
        "Written by `scripts/scaled_build.py report` from `distractors_D.json` and `corpora.json`",
        "(measured by the `distractors` and `corpora` steps). The list is `distractors_D.jsonl`.",
        "",
        "## Rule (owner's decision 2, 2026-11-11)",
        "",
        "Per gold article, count its links (`wikitext.extract_links`, duplicates kept, page order);",
        "its favourite is the most-linked title, ties broken by the first mention. Skip gold pages,",
        "titles another article already took, and titles missing from the FEVEROUS DB; then take",
        f"the next favourite. Gold articles in `sorted(gold)` order shuffled with `random.Random({WALK_SEED})`;",
        "one page per gold article with a usable link, no cap.",
        "",
        "## Results",
        "",
        f"- Gold articles walked: {dj['gold_articles']:,} (missing from the DB: {len(dj['gold_missing_from_db'])}).",
        f"- Rule-D pages: **{dj['pages']:,}**.",
        f"- Gold articles without a usable link: {len(without)}.",
        (f"- Skips during the walk: gold page {dj['skips']['gold']:,}, already taken {dj['skips']['taken']:,}, "
        f"missing from the DB {dj['skips']['missing']:,}."),
        (f"- Distinct titles looked up in the DB: {dj['distinct_titles_checked_in_db']:,}, of which "
        f"{dj['distinct_titles_missing_from_db']:,} missing."),
        (f"- Links per gold article: median {dj['gold_links_per_article']['median']:g}, mean "
        f"{dj['gold_links_per_article']['mean']:g}, max {dj['gold_links_per_article']['max']:,}."),
        (f"- Chunks per rule-D page: median {cd['median']:g}, mean {cd['mean']:g}, p90 {cd['p90']:g}, "
        f"max {cd['max']:,}; total {cd['sum']:,}; pages without any chunk "
        f"{corpora['d_pages_without_chunks']:,}."),
        "",
        "Which favourite was taken (0 = the most-linked title; a higher rank means the titles",
        "before it were skipped):",
        "",
        *table(["Rank", "Pages"], [[k, v] for k, v in dj["rank_taken"].items()]),
        "",
        "How often the gold article links to the page it took (10 = 10 or more):",
        "",
        *table(["Times linked", "Pages"], [[k, v] for k, v in dj["times_linked_of_taken"].items()]),
        "",
    ]
    if without:
        out += ["Gold articles without a usable link:", ""]
        out += table(
            ["Gold article", "Links", "Distinct titles", "Skipped"],
            [[w["title"], w["n_links"], w["n_distinct"], json.dumps(w["skipped"])] for w in without],
        )
        out.append("")
    out += [
        "## Corpus sizes",
        "",
        *table(
            ["Folder", "Pages", "Gold pages", "Rule-D pages", "Chunks"],
            [[n, corpora[n]["pages"], corpora[n]["gold_pages"], corpora[n]["d_pages"], corpora[n]["chunks"]]
             for n in FOLDERS],
        ),
        "",
    ]
    return "\n".join(out)


def run_report(args: argparse.Namespace) -> int:
    """The report step: SELECTION.md and DISTRACTORS_D.md (no database)."""
    out = Path(args.out_dir)
    sel = json.loads((out / SELECTION_JSON).read_text(encoding="utf-8"))
    dj = json.loads((out / D_JSON).read_text(encoding="utf-8"))
    corpora = json.loads((out / CORPORA_JSON).read_text(encoding="utf-8"))
    write_atomic(out / "SELECTION.md", [selection_md(sel, corpora, Path(args.claims).name)])
    write_atomic(out / "DISTRACTORS_D.md", [distractors_md(dj, corpora)])
    log("wrote SELECTION.md and DISTRACTORS_D.md")
    return 0


# ---- command line ------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="step", required=True)
    for step in ("claims", "distractors", "corpora", "report"):
        p = sub.add_parser(step)
        p.add_argument("--out-dir", required=True, type=Path, help="the scaled_data folder")
        p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
        p.add_argument("--dev", type=Path, default=DEFAULT_DEV)
        if step != "claims":
            p.add_argument("--claims", required=True, type=Path, help="the claims file to read")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one step and return its exit status."""
    args = parse_args(argv)
    steps = {
        "claims": run_claims,
        "distractors": run_distractors,
        "corpora": run_corpora,
        "report": run_report,
    }
    return steps[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
