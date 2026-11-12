"""Tests for scripts/scaled_check.py on a tiny synthetic world, without MariaDB, the embedding
model or the real FEVEROUS database.

The world: six pages in a tiny SQLite ``wiki`` table stored NFD (two link targets and one gold
page are missing), thirteen raw claims (train and dev files with their header line) covering
every kind, an empty evidence set, a table set, a hatnote and decomposed ids, and the output
folder built by hand here for one claim per (label, kind) group (``PER_KIND``). The checker
must pass on that folder and fail on each of the tampered copies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sqlite3
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from wikilense.chunking import chunk_page, embedding_text
from wikilense.config import REPO_ROOT
from wikilense.corpus import parse_element_id
from wikilense.wikitext import extract_links, parse_page

SCRIPT = REPO_ROOT / "scripts" / "scaled_check.py"
PER_KIND = {"single": 1, "needs2": 1}
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
CAFE = "Café Noir"  # NFC; the raw ids below also use the decomposed form


def _cell(cell_id: str, value: str, header: bool = False) -> dict:
    return {"id": cell_id, "value": value, "is_header": header, "row_span": 1, "column_span": 1}


PAGES: dict[str, dict] = {
    "Alpha": {
        "title": "Alpha",
        "order": ["sentence_0", "sentence_1", "list_0", "section_0", "sentence_2",
                  "sentence_3", "table_0"],
        "sentence_0": "Alpha is a [[Beta|beta]] river in [[Gamma]].",
        "sentence_1": "It flows past [[Beta]] and [[Epsilon]].",
        "list_0": {"type": "unordered", "list": [
            {"id": "item_0_0", "value": "A [[Gamma]] bridge."},
            {"id": "item_0_1", "value": "A mill."},
        ]},
        "section_0": {"value": "History", "level": 2},
        "sentence_2": "Main article: [[Beta]]",  # a hatnote: in no chunk
        "sentence_3": "Alpha was named after [[Zeta]] in 1900.",
        "table_0": {"type": "infobox", "table": [[
            _cell("header_cell_0_0_0", "Length", header=True),
            _cell("cell_0_0_1", "[[Gamma]] 10 km"),
        ]]},
    },
    "Beta": {
        "title": "Beta",
        "order": ["sentence_0", "sentence_1", "sentence_2"],
        "sentence_0": "Beta is a town on the [[Alpha]].",
        "sentence_1": "Beta has a [[Gamma|market]] and a [[Delta]] square near [[Gamma]].",
        "sentence_2": "Beta hosts the [[Epsilon]] festival.",
    },
    CAFE: {
        "title": CAFE,
        "order": ["sentence_0", "sentence_1"],
        "sentence_0": f"{CAFE} is a song by [[Delta]].",  # Delta and Zeta are not in the DB
        "sentence_1": "It appeared on [[Zeta]].",
    },
    "Gamma": {"title": "Gamma", "order": ["sentence_0"],
              "sentence_0": "Gamma is a région of [[Alpha]]."},
    "Epsilon": {"title": "Epsilon", "order": ["sentence_0"], "sentence_0": "A festival."},
    "Theta": {"title": "Theta", "order": ["sentence_0"], "sentence_0": "Nobody links here."},
}  # fmt: skip

CAFE_NFD = unicodedata.normalize("NFD", CAFE)


def _set(*ids: str) -> dict:
    return {"content": list(ids), "context": {i: [] for i in ids}}


def _claim(claim_id: int, label: str, *sets: dict) -> dict:
    return {"evidence": list(sets), "id": claim_id, "claim": f"Claim number {claim_id}.",
            "label": label, "annotator_operations": [], "challenge": "Other"}


TRAIN = [
    _claim(101, "SUPPORTS", _set("Alpha_cell_0_0_1"), _set("Alpha_sentence_0", "Alpha_item_0_1")),
    _claim(102, "SUPPORTS", _set("Alpha_sentence_3")),
    _claim(103, "SUPPORTS", _set("Alpha_sentence_1", "Beta_sentence_0")),
    _claim(104, "REFUTES", _set(), _set("Beta_sentence_1")),  # the empty set is ignored for kind
    _claim(105, "REFUTES", _set("Beta_sentence_2", f"{CAFE_NFD}_sentence_0")),
    _claim(106, "NOT ENOUGH INFO", _set("Alpha_sentence_2")),  # hatnote: unfindable
    _claim(107, "SUPPORTS", _set("Alpha_sentence_0"), _set("Beta_sentence_0")),  # either
    _claim(108, "REFUTES", _set("Alpha_header_cell_0_0_0", "Beta_sentence_0")),  # no text set
    _claim(109, "SUPPORTS", _set()),  # none
    _claim(110, "REFUTES", _set("Beta_sentence_0")),
]
DEV = [
    _claim(201, "NOT ENOUGH INFO", _set(f"{CAFE}_sentence_1")),
    _claim(202, "NOT ENOUGH INFO", _set("Alpha_sentence_0", "Missing Page_sentence_1"),
           _set("Beta_sentence_0", f"{CAFE_NFD}_sentence_0")),
    _claim(203, "NOT ENOUGH INFO", _set("Alpha_sentence_9", "Beta_sentence_0")),  # no sentence_9
]
HEADER = {"id": "", "claim": "", "label": "", "evidence": "", "annotator_operations": "",
          "challenge": ""}


@pytest.fixture(scope="module")
def sc() -> Any:
    spec = importlib.util.spec_from_file_location("scaled_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scaled_check"] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("scaled_check", None)


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dummy password for load_settings(); no chunk or database overrides from outside."""
    monkeypatch.setenv("WIKILENSE_DB_PASSWORD", "dummy-test-password")
    for name in ("WIKILENSE_CHUNK_MAX_WORDS", "WIKILENSE_CHUNK_OVERLAP_UNITS",
                 "WIKILENSE_EF_SEARCH", "WIKILENSE_DB_NAME", "WIKILENSE_FEVEROUS_DB"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------------------------
# a small reference build of the world (written for these tests, not shared with the checker)
# ---------------------------------------------------------------------------------------------


def _deep(form: str, value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize(form, value)
    if isinstance(value, list):
        return [_deep(form, v) for v in value]
    if isinstance(value, dict):
        return {_deep(form, k): _deep(form, v) for k, v in value.items()}
    return value


def _in_chunks(page: dict) -> set[str]:
    return {key for chunk in chunk_page(parse_page(page), 240, 1) for key in chunk.element_keys}


def _dump_lines(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")


@dataclass
class World:
    root: Path
    pages: dict[str, dict] = field(default_factory=lambda: PAGES)
    written: dict[str, dict] | None = None  # pages written to S0/S1 instead of the DB's

    @property
    def out(self) -> Path:
        return self.root / "out"

    @property
    def claims(self) -> Path:
        return self.out / "claims_3000.jsonl.tmp"

    def paths(self) -> tuple[Path, Path, Path]:
        return self.root / "train.jsonl", self.root / "dev.jsonl", self.root / "wiki.db"

    # -- inputs ---------------------------------------------------------------------------------

    def write_inputs(self) -> None:
        train, dev, db = self.paths()
        for path, records in ((train, TRAIN), (dev, DEV)):
            path.write_text("".join(json.dumps(r) + "\n" for r in [HEADER, *records]),
                            encoding="utf-8")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE wiki (id PRIMARY KEY, data json)")
        for title, page in self.pages.items():
            conn.execute("INSERT INTO wiki VALUES (?, ?)", (
                unicodedata.normalize("NFD", title),
                json.dumps(_deep("NFD", page), ensure_ascii=False)))
        conn.commit()
        conn.close()

    # -- selection ------------------------------------------------------------------------------

    def describe(self, split: str, record: dict) -> dict:
        """Kind, sets and findability of one raw record, computed with parse_element_id."""
        sets = []
        for evidence_set in record["evidence"]:
            parsed = [parse_element_id(el) for el in evidence_set["content"]]
            text_only = bool(parsed) and {t for _, t, _ in parsed} <= {"sentence", "item"}
            lands = all(page in self.pages and key in _in_chunks(self.pages[page])
                        for page, _, key in parsed)
            sets.append({"parsed": parsed, "text_only": text_only,
                         "findable": text_only and lands})
        nonempty = [s for s in sets if s["parsed"]]
        pages = [{p for p, _, _ in s["parsed"]} for s in nonempty]
        if not nonempty:
            kind = "none"
        elif min(len(p) for p in pages) >= 2:
            kind = "needs2"
        else:
            kind = "single" if len(set().union(*pages)) == 1 else "either"
        eligible = kind in PER_KIND and any(s["text_only"] for s in sets)
        return {"split": split, "record": record, "sets": sets, "kind": kind,
                "eligible": eligible, "findable": any(s["findable"] for s in sets)}

    def select(self) -> tuple[list[dict], list[dict]]:
        described = [self.describe("train", r) for r in TRAIN]
        described += [self.describe("dev", r) for r in DEV]
        pool = [d for d in described if d["eligible"]]
        rng = random.Random(20261111)
        picked = []
        for label in LABELS:
            for kind in ("single", "needs2"):
                group = [d for d in pool if d["findable"] and d["kind"] == kind
                         and d["record"]["label"] == label]
                picked += rng.sample(group, PER_KIND[kind])
        return pool, picked

    # -- outputs --------------------------------------------------------------------------------

    def claim_line(self, d: dict) -> dict:
        record = d["record"]
        gold_set = next(i for i, s in enumerate(d["sets"]) if s["findable"])
        units = {}
        for page, _, key in d["sets"][gold_set]["parsed"]:
            texts = {u.element_key: u.text for u in parse_page(self.pages[page]).units}
            units[(page, key)] = texts[key]
        return {
            "id": record["id"], "split": d["split"], "label": record["label"],
            "claim": record["claim"], "kind": d["kind"], "challenge": record["challenge"],
            "gold_pages": sorted({p for s in d["sets"] for p, _, _ in s["parsed"]}),
            "evidence_sets": [{"elements": [f"{p}_{k}" for p, _, k in s["parsed"]],
                               "text_only": s["text_only"], "findable": s["findable"]}
                              for s in d["sets"]],
            "gold_set": gold_set,
            "gold_evidence": [{"element": f"{p}_{k}", "page": p, "text": units[(p, k)]}
                              for p, _, k in d["sets"][gold_set]["parsed"]],
        }  # fmt: skip

    def rule_d(self, gold: set[str]) -> tuple[list[dict], dict]:
        order = sorted(gold)
        random.Random(7).shuffle(order)
        taken: list[dict] = []
        skips = Counter({"gold": 0, "taken": 0, "missing": 0})
        stats: dict[str, Any] = {"gold_without_usable_link": [], "gold_missing_from_db": []}
        for walk_index, g in enumerate(order):
            if g not in self.pages:
                stats["gold_missing_from_db"].append(g)
                continue
            titles = [t for t, _ in extract_links(self.pages[g])]
            counts = Counter(titles)
            ranked = sorted(counts, key=lambda t: (-counts[t], titles.index(t)))
            for rank, title in enumerate(ranked):
                reason = ("gold" if title in gold else
                          "taken" if title in {r["title"] for r in taken} else
                          "missing" if title not in self.pages else None)
                if reason:
                    skips[reason] += 1
                    continue
                taken.append({"title": title, "gold": g, "walk_index": walk_index,
                              "rank": rank, "times_linked": counts[title]})
                break
            else:
                stats["gold_without_usable_link"].append({"title": g})
        stats.update(skips=dict(skips), pages=len(taken))
        return taken, stats

    def write_corpus(self, folder: str, titles: list[str], n_gold: int, n_d: int,
                     claims_lines: list[str]) -> dict:
        pages = self.written or self.pages
        lines = [json.dumps(pages[t], ensure_ascii=False) + "\n" for t in titles]
        directory = self.out / folder
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pages.jsonl").write_text("".join(lines), encoding="utf-8")
        (directory / "claims.jsonl").write_text("".join(claims_lines), encoding="utf-8")
        digest, chunks, chunk_words, unit_words = hashlib.sha256(), 0, 0, 0
        for title in titles:
            parsed = parse_page(pages[title])
            unit_words += parsed.stats["n_words"]
            for chunk in chunk_page(parsed, 240, 1):
                chunks += 1
                chunk_words += chunk.n_words
                path = parsed.sections[chunk.section_ordinal].path
                digest.update((embedding_text(parsed.title, path, chunk.text) + "\n").encode())
        files = {}
        for name in ("pages.jsonl", "claims.jsonl"):
            data = (directory / name).read_bytes()
            files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                           "lines": data.count(b"\n")}
        entry = {"pages": len(titles), "chunks": chunks, "chunk_words": chunk_words,
                 "unit_words": unit_words, "embedding_text_sha256": digest.hexdigest(),
                 "files": files}
        (directory / "MANIFEST.md").write_text("\n".join([
            f"# Corpus {folder}", "", "| Measure | Value |", "|---|---:|",
            f"| Pages (lines in pages.jsonl) | {len(titles):,} |",
            f"| Gold pages | {n_gold:,} |", f"| Rule D pages | {n_d:,} |",
            f"| Chunks | {chunks:,} |",
            f"| Words in chunks (overlap counted in each chunk) | {chunk_words:,} |",
            f"| Words in text units (page.n_words summed) | {unit_words:,} |",
            f"| Claims (lines in claims.jsonl) | {len(claims_lines):,} |", "",
            "| File | Bytes | SHA-256 |", "|---|---:|---|",
            *(f"| {n} | {f['bytes']} | {f['sha256']} |" for n, f in files.items()), "",
            f"Embedding texts (one per chunk, ingest order): `{digest.hexdigest()}`", "",
        ]), encoding="utf-8")  # fmt: skip
        return entry

    def build(self) -> World:
        self.write_inputs()
        pool, picked = self.select()
        _dump_lines(self.claims, [self.claim_line(d) for d in picked])
        _dump_lines(self.out / "pool.jsonl", [
            {"id": d["record"]["id"], "split": d["split"], "label": d["record"]["label"],
             "kind": d["kind"], "findable": d["findable"]}
            | ({} if d["findable"] else {"reasons": ["element not in a chunk"]})
            for d in pool])  # fmt: skip
        gold = {p for d in picked for s in d["sets"] for p, _, _ in s["parsed"]}
        taken, stats = self.rule_d(gold)
        _dump_lines(self.out / "distractors_D.jsonl", taken)
        (self.out / "distractors_D.json").write_text(json.dumps(stats), encoding="utf-8")
        s0 = sorted(g for g in gold if g in self.pages)
        claims_lines = [json.dumps({**d["record"], "split": d["split"]}, ensure_ascii=False)
                        + "\n" for d in picked]
        corpora = {
            "S0": self.write_corpus("S0", s0, len(s0), 0, claims_lines),
            "S1": self.write_corpus("S1", s0 + [r["title"] for r in taken], len(s0), len(taken),
                                    claims_lines),
        }
        (self.out / "corpora.json").write_text(json.dumps(corpora, indent=1), encoding="utf-8")
        return self


def _run(sc: Any, world: World, monkeypatch: pytest.MonkeyPatch, use_db: bool = True) -> dict:
    """Run every check on the world; return ``{letter: Check}``."""
    train, dev, db = world.paths()
    monkeypatch.setenv("WIKILENSE_FEVEROUS_DB", str(db))
    checks = sc.run_checks(world.out, world.claims, train, dev, use_db,
                           sc.Params(per_kind=dict(PER_KIND)), env_file=world.root / "none.env")
    return {check.letter: check for check in checks}


def _failed_items(check: Any) -> list[str]:
    return [item.name for item in check.items if not item.ok]


def _rewrite(path: Path, change: Any) -> None:
    """Replace the JSONL records of ``path`` by ``change(records)``."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _dump_lines(path, change(records))


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path).build()


# ---------------------------------------------------------------------------------------------
# unit rules
# ---------------------------------------------------------------------------------------------


def test_element_regex_prefers_header_cell_and_keeps_underscores(sc: Any) -> None:
    assert sc.parse_element("X_header_cell_0_1_1") == sc.Element("X", "header_cell",
                                                                 "header_cell_0_1_1")
    assert sc.parse_element("A_b_sentence_1_x_item_2_3") == sc.Element(
        "A_b_sentence_1_x", "item", "item_2_3")
    assert sc.parse_element(f"{CAFE_NFD}_table_caption_0").page == CAFE
    assert sc.parse_element("Alpha_title") is None
    assert sc.parse_element("_sentence_0") is None


def test_kind_rules(sc: Any) -> None:
    def kind(*sets: tuple[str, ...]) -> str:
        return sc.claim_kind(sc.set_views({"evidence": [{"content": list(s)} for s in sets]}))

    assert kind() == "none"
    assert kind(()) == "none"
    assert kind(("A_sentence_0", "B_sentence_1")) == "needs2"
    assert kind((), ("A_sentence_0", "B_cell_0_0_1"), ("C_sentence_0", "A_item_0_0")) == "needs2"
    assert kind(("A_sentence_0",), (), ("A_cell_0_1_1",)) == "single"
    assert kind(("A_sentence_0",), ("A_sentence_1", "B_sentence_0")) == "either"


# ---------------------------------------------------------------------------------------------
# the whole check
# ---------------------------------------------------------------------------------------------


def test_consistent_world_passes(sc: Any, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    checks = _run(sc, world, monkeypatch)
    assert {letter: check.result for letter, check in checks.items()} == dict.fromkeys(
        "ABCDEFG", "PASS"), {k: _failed_items(c) for k, c in checks.items()}
    # the world exercises the edge cases the rules name
    stats = json.loads((world.out / "distractors_D.json").read_text())
    assert stats["gold_missing_from_db"] == ["Missing Page"]
    assert stats["gold_without_usable_link"] == [{"title": CAFE}]
    assert stats["skips"]["gold"] >= 1 and stats["skips"]["taken"] == 1
    taken = (world.out / "distractors_D.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["title"] for line in taken) == ["Epsilon", "Gamma"]
    pool = (world.out / "pool.jsonl").read_text(encoding="utf-8").splitlines()
    pool = [json.loads(line) for line in pool]
    assert [p["id"] for p in pool if not p["findable"]] == [106, 203]
    report = (world.out / "SELFCHECK.md").read_text(encoding="utf-8")
    assert report.startswith("# Self-check of the scaled data")
    assert "**Result: PASS.** Database checks (D): ran on `wiki.db`." in report


def test_no_db_skips_check_d_and_exits_zero(sc: Any, world: World,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    checks = _run(sc, world, monkeypatch, use_db=False)
    assert checks["D"].result == "SKIPPED"
    assert all(c.result == "PASS" for k, c in checks.items() if k != "D")
    train, dev, _ = world.paths()
    argv = ["--out-dir", str(world.out), "--claims", str(world.claims), "--train", str(train),
            "--dev", str(dev), "--no-db"]
    assert sc.main(argv, sc.Params(per_kind=dict(PER_KIND)), world.root / "none.env") == 0
    assert "did not run (--no-db)" in (world.out / "SELFCHECK.md").read_text(encoding="utf-8")


def test_wrong_kind_fails(sc: Any, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def change(records: list[dict]) -> list[dict]:
        records[0]["kind"] = "needs2" if records[0]["kind"] == "single" else "single"
        return records

    _rewrite(world.claims, change)
    checks = _run(sc, world, monkeypatch)
    assert "kind recomputed" in _failed_items(checks["B"])
    train, dev, _ = world.paths()
    argv = ["--out-dir", str(world.out), "--claims", str(world.claims), "--train", str(train),
            "--dev", str(dev)]
    assert sc.main(argv, sc.Params(per_kind=dict(PER_KIND)), world.root / "none.env") == 1


def test_gold_element_in_a_hatnote_fails(sc: Any, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    # The DB (used for the selection) has a normal Beta_sentence_0, but the pages written to S0
    # and S1 turn it into a hatnote: the gold element of claims 103 and 202 is in no chunk.
    beta = dict(PAGES["Beta"], sentence_0="See also: [[Alpha]]")
    world = World(tmp_path, written=dict(PAGES, Beta=beta)).build()
    checks = _run(sc, world, monkeypatch)
    assert "findable recomputed from S0/pages.jsonl" in _failed_items(checks["B"])
    assert "S0: every gold-set element has a sentence_id inside a chunk" in _failed_items(
        checks["F"])


@pytest.mark.parametrize("target", ["manifest", "corpora"])
def test_tampered_sha256_fails(sc: Any, world: World, monkeypatch: pytest.MonkeyPatch,
                               target: str) -> None:
    real = hashlib.sha256((world.out / "S0" / "pages.jsonl").read_bytes()).hexdigest()
    fake = ("0" if real[0] != "0" else "1") + real[1:]
    path = world.out / ("S0/MANIFEST.md" if target == "manifest" else "corpora.json")
    text = path.read_text(encoding="utf-8")
    assert real in text
    path.write_text(text.replace(real, fake), encoding="utf-8")
    failed = _failed_items(_run(sc, world, monkeypatch)["G"])
    want = ("S0/MANIFEST.md rows with the recomputed numbers" if target == "manifest"
            else "corpora.json S0 files.pages.jsonl (bytes, sha256, lines)")
    assert want in failed


def test_s1_not_starting_with_s0_fails(sc: Any, world: World,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    path = world.out / "S1" / "pages.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    path.write_text("".join(lines), encoding="utf-8")
    failed = _failed_items(_run(sc, world, monkeypatch)["E"])
    assert "S1/pages.jsonl starts with S0/pages.jsonl byte for byte" in failed


def test_duplicate_title_fails(sc: Any, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    path = world.out / "S1" / "pages.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines + lines[-1:]), encoding="utf-8")
    checks = _run(sc, world, monkeypatch)
    assert "S1: titles unique" in _failed_items(checks["E"])
    run_item = next(i for i in checks["F"].items if i.name == "S1: page and claim loaders ran")
    assert not run_item.ok and "duplicate page title" in run_item.detail


def test_wrong_rule_d_rank_fails(sc: Any, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def change(records: list[dict]) -> list[dict]:
        records[0]["rank"] += 1
        return records

    _rewrite(world.out / "distractors_D.jsonl", change)
    failed = _failed_items(_run(sc, world, monkeypatch)["D"])
    assert ("distractors_D.jsonl = re-derived rule D (title, gold, walk_index, rank, "
            "times_linked)") in failed


def test_pool_without_an_eligible_claim_fails(sc: Any, world: World,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    _rewrite(world.out / "pool.jsonl", lambda records: [r for r in records if r["id"] != 203])
    failed = _failed_items(_run(sc, world, monkeypatch)["C"])
    assert "pool = every eligible claim, in pool order, nothing else" in failed
