"""Tests for scripts/scaled_build.py without the real FEVEROUS data: the claim kinds, the findable
check, the seeded pick, rule D and the corpus folders, on a tiny FEVEROUS database (NFD ids, as
the real one) built in a temporary directory."""

from __future__ import annotations

import importlib.util
import json
import random
import sqlite3
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from wikilense.config import REPO_ROOT
from wikilense.corpus import iter_claims, iter_pages

SCRIPT = REPO_ROOT / "scripts" / "scaled_build.py"
CAFE = "Café"  # NFC; the database stores it decomposed


@pytest.fixture(scope="module")
def sb() -> Iterator[Any]:
    spec = importlib.util.spec_from_file_location("scaled_build", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def nfd_deep(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFD", value)
    if isinstance(value, list):
        return [nfd_deep(v) for v in value]
    if isinstance(value, dict):
        return {nfd_deep(k): nfd_deep(v) for k, v in value.items()}
    return value


def cell(cell_id: str, value: str) -> dict:
    return {"id": cell_id, "value": value, "is_header": False, "row_span": 1, "column_span": 1}


PAGES: dict[str, dict] = {
    CAFE: {
        "title": CAFE,
        "order": ["sentence_0", "section_0", "sentence_1", "sentence_2", "list_0"],
        "sentence_0": "A café serves [[Gold two|coffee]], [[Gold two]], [[Gold two]] and [[Gold two]].",
        "section_0": {"value": "History", "level": 2},
        "sentence_1": "Main article: [[History of cafés]]",
        "sentence_2": "The [[Missing page]] [[Missing page]] [[Missing page]] stood near [[Bar]], "
        "[[Baz]], [[Bar]] and [[Baz]].",
        "list_0": {"type": "unordered", "list": [{"id": "item_0_0", "value": "Espresso is common."}]},
    },
    "Gold two": {
        "title": "Gold two",
        "order": ["sentence_0", "sentence_1"],
        "sentence_0": "Gold two links to [[Bar]] once.",
        "sentence_1": f"It also links to [[Qux]] and [[{CAFE}]].",
    },
    "Tab": {
        "title": "Tab",
        "order": ["table_0", "sentence_0"],
        "table_0": {"type": "infobox", "table": [[cell("cell_0_0_0", "x [[Baz]]")]]},
        "sentence_0": "Tab has one sentence.",
    },
    "Bar": {"title": "Bar", "order": ["sentence_0"], "sentence_0": "Bar is a place."},
    "Baz": {"title": "Baz", "order": ["sentence_0"], "sentence_0": "Baz is a place."},
    "Qux": {"title": "Qux", "order": ["sentence_0"], "sentence_0": "Qux is a place."},
}

CAFE_NFD = unicodedata.normalize("NFD", CAFE)


def claim(cid: int, label: str, *sets: list[str]) -> dict:
    return {
        "evidence": [{"content": list(s), "context": {}} for s in sets],
        "id": cid,
        "claim": f"Claim number {cid}.",
        "label": label,
        "annotator_operations": [],
        "challenge": "Other" if cid % 2 else "Multi-hop Reasoning",
    }


TRAIN = [
    claim(1, "SUPPORTS", [f"{CAFE_NFD}_sentence_0"]),
    claim(2, "SUPPORTS", [f"{CAFE}_sentence_2", "Gold two_sentence_0"]),
    claim(3, "SUPPORTS", [f"{CAFE}_sentence_1"]),  # a hatnote: unfindable
    claim(4, "REFUTES", ["Gold two_sentence_1"], ["Gold two_cell_0_0_0"]),
    claim(5, "REFUTES", ["Tab_cell_0_0_0", f"{CAFE}_sentence_0"],
          [f"{CAFE}_item_0_0", "Gold two_sentence_1"]),
]
DEV = [
    claim(6, "NOT ENOUGH INFO", [f"{CAFE}_item_0_0"]),
    claim(7, "NOT ENOUGH INFO", [f"{CAFE}_sentence_0", "Gold two_sentence_0"]),
    claim(8, "NOT ENOUGH INFO", ["Nowhere_sentence_0", f"{CAFE}_sentence_0"]),  # page missing
    claim(9, "SUPPORTS", [f"{CAFE}_sentence_0"], ["Gold two_sentence_0"]),  # "either"
]
HEADER = {"id": "", "claim": "", "label": "", "evidence": []}


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sb: Any) -> dict[str, Path]:
    db_path = tmp_path / "wiki.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE wiki (id PRIMARY KEY, data json)")
    for title, page in PAGES.items():
        con.execute("INSERT INTO wiki VALUES (?, ?)",
                    (unicodedata.normalize("NFD", title), json.dumps(nfd_deep(page), ensure_ascii=False)))
    con.commit()
    con.close()
    paths = {"train": tmp_path / "train.jsonl", "dev": tmp_path / "dev.jsonl", "out": tmp_path / "out"}
    for key, records in (("train", TRAIN), ("dev", DEV)):
        paths[key].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [HEADER, *records]),
                              encoding="utf-8")
    monkeypatch.setenv(sb.FEVEROUS_DB_VARIABLE, str(db_path))
    monkeypatch.setattr(sb, "PER_GROUP", {"single": 1, "needs2": 1})
    return paths


def run(sb: Any, world: dict[str, Path], step: str) -> int:
    args = [step, "--out-dir", str(world["out"]), "--train", str(world["train"]), "--dev", str(world["dev"])]
    if step != "claims":
        args += ["--claims", str(world["out"] / sb.CLAIMS_TMP)]
    return sb.main(args)


def build_all(sb: Any, world: dict[str, Path]) -> None:
    for step in ("claims", "distractors", "corpora", "report"):
        assert run(sb, world, step) == 0, step


# ---- the pure rules ----------------------------------------------------------------------------


def test_describe_kinds(sb: Any) -> None:
    assert sb.describe(claim(1, "SUPPORTS", ["A_sentence_0"], ["A_cell_0_0_1"]))[0] == "single"
    assert sb.describe(claim(1, "SUPPORTS", ["A_sentence_0", "B_sentence_1"]))[0] == "needs2"
    assert sb.describe(claim(1, "SUPPORTS", ["A_sentence_0"], ["B_sentence_0"]))[0] == "either"
    assert sb.describe(claim(1, "SUPPORTS", []))[0] == "none"
    # text-only means sentences and list items only
    assert sb.describe(claim(1, "SUPPORTS", ["A_item_0_1"]))[1] is True
    assert sb.describe(claim(1, "SUPPORTS", ["A_header_cell_0_0_1"]))[1] is False


def test_nfc_deep_normalises_keys_and_values(sb: Any) -> None:
    data = sb.nfc_deep({CAFE_NFD: [CAFE_NFD, {"k": CAFE_NFD}], "n": 1})
    assert data == {CAFE: [CAFE, {"k": CAFE}], "n": 1}
    with pytest.raises(ValueError, match="equal after NFC"):
        sb.nfc_deep({CAFE: 1, CAFE_NFD: 2})


def test_rank_links_most_linked_first_ties_by_first_mention(sb: Any) -> None:
    links = [("B", "s0"), ("A", "s0"), ("C", "s1"), ("A", "s1"), ("B", "s2"), ("C", "s2"), ("C", "s3")]
    assert sb.rank_links(links) == [("C", 3), ("B", 2), ("A", 2)]


def _cands(sb: Any, n: int) -> list[Any]:
    return [
        sb.Candidate(id=i, split="train", label=lab, claim="", kind=kind, challenge=None,
                     gold_pages=["P"], sets=[], findable=True)
        for i, (lab, kind) in enumerate((lab, kind) for lab in sb.LABELS for kind in sb.KINDS for _ in range(n))
    ]


def test_pick_matches_the_planning_scripts_call_order(sb: Any) -> None:
    cands = _cands(sb, 5)
    picked = sb.pick(cands, seed=11, per_group={"single": 2, "needs2": 1})
    rng = random.Random(11)
    expected = []
    for lab in sb.LABELS:
        single = [c for c in cands if c.label == lab and c.kind == "single"]
        needs2 = [c for c in cands if c.label == lab and c.kind == "needs2"]
        expected += rng.sample(single, 2) + rng.sample(needs2, 1)
    assert [c.id for c in picked] == [c.id for c in expected]


def test_pick_stops_before_sampling_when_a_group_is_short(sb: Any) -> None:
    cands = _cands(sb, 3)
    for c in cands:
        if c.label == "NOT ENOUGH INFO" and c.kind == "needs2":
            c.findable = c.id % 3 == 0  # one of three stays findable
    with pytest.raises(sb.GroupTooSmall) as info:
        sb.pick(cands, per_group={"single": 2, "needs2": 2})
    assert info.value.short == {("NOT ENOUGH INFO", "needs2"): (1, 2)}


# ---- the whole build on the tiny database ------------------------------------------------------


def test_claims_step_writes_the_contract(sb: Any, world: dict[str, Path]) -> None:
    assert run(sb, world, "claims") == 0
    out = world["out"]
    assert not (out / "claims_3000.jsonl").exists()  # only the .tmp name is ever written
    claims = [json.loads(line) for line in (out / sb.CLAIMS_TMP).read_text(encoding="utf-8").splitlines()]
    assert [c["id"] for c in claims] == [1, 2, 4, 5, 6, 7]
    assert list(claims[0]) == ["id", "split", "label", "claim", "kind", "challenge", "gold_pages",
                               "evidence_sets", "gold_set", "gold_evidence"]
    first = claims[0]
    assert first["evidence_sets"] == [{"elements": [f"{CAFE}_sentence_0"], "text_only": True, "findable": True}]
    assert first["gold_pages"] == [CAFE]
    assert first["gold_evidence"] == [
        {"element": f"{CAFE}_sentence_0", "page": CAFE, "text": "A café serves coffee, Gold two, Gold two and Gold two."}
    ]
    five = claims[3]
    assert five["kind"] == "needs2" and five["gold_set"] == 1
    assert five["gold_pages"] == [CAFE, "Gold two", "Tab"]
    assert [s["text_only"] for s in five["evidence_sets"]] == [False, True]
    assert [e["text"] for e in five["gold_evidence"]] == ["Espresso is common.", f"It also links to Qux and {CAFE}."]
    four = claims[2]
    assert four["kind"] == "single" and [s["findable"] for s in four["evidence_sets"]] == [True, False]
    pool = {(p["split"], p["id"]): p for p in map(json.loads, (out / sb.POOL_FILE).read_text().splitlines())}
    assert ("dev", 9) not in pool  # "either" is not eligible
    assert pool[("train", 3)]["reasons"] == ["hatnote"]
    assert pool[("dev", 8)]["reasons"] == ["page not in the DB"]
    sel = json.loads((out / sb.SELECTION_JSON).read_text())
    assert sel["pool"]["SUPPORTS"]["single"] == {
        "claims": 2, "with_text_only_set": 2, "eligible": 2, "findable": 1, "unfindable": 1, "picked": 1, "spare": 0
    }
    assert sel["unfindable_elements_by_reason"] == {"hatnote": 1, "page not in the DB": 1}
    assert sel["gold_pages"]["n"] == 3 and sel["gold_pages"]["missing_from_db"] == []


def test_short_group_stops_without_a_claims_file(sb: Any, world: dict[str, Path],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sb, "PER_GROUP", {"single": 1, "needs2": 2})
    assert run(sb, world, "claims") == sb.EXIT_SHORT_GROUP
    assert not (world["out"] / sb.CLAIMS_TMP).exists()
    assert "not enough findable claims" in json.loads((world["out"] / sb.SELECTION_JSON).read_text())["stopped"]


def test_rule_d_follows_the_walk(sb: Any, world: dict[str, Path]) -> None:
    run(sb, world, "claims")
    assert run(sb, world, "distractors") == 0
    taken = [json.loads(line) for line in (world["out"] / sb.D_LIST).read_text().splitlines()]
    # usable favourites per gold article, with their rank in its own ranked link list
    prefs = {CAFE: [("Bar", 2), ("Baz", 3)], "Gold two": [("Bar", 0), ("Qux", 1)], "Tab": [("Baz", 0)]}
    expected, used = [], set()
    for walk_index, g in enumerate(sb.walk_order({CAFE, "Gold two", "Tab"})):
        for title, rank in prefs[g]:
            if title not in used:
                used.add(title)
                expected.append({"title": title, "gold": g, "walk_index": walk_index, "rank": rank})
                break
    assert [{k: d[k] for k in ("title", "gold", "walk_index", "rank")} for d in taken] == expected
    stats = json.loads((world["out"] / sb.D_JSON).read_text())
    assert stats["pages"] == len(expected)
    assert len(stats["gold_without_usable_link"]) == 3 - len(expected)


def test_corpora_folders_load_like_the_ingest_reads_them(sb: Any, world: dict[str, Path]) -> None:
    build_all(sb, world)
    out = world["out"]
    s0 = (out / "S0" / "pages.jsonl").read_bytes()
    s1 = (out / "S1" / "pages.jsonl").read_bytes()
    assert s1.startswith(s0)
    titles0 = [p["title"] for p in iter_pages(out / "S0" / "pages.jsonl")]
    titles1 = [p["title"] for p in iter_pages(out / "S1" / "pages.jsonl")]
    d_titles = [json.loads(line)["title"] for line in (out / sb.D_LIST).read_text().splitlines()]
    assert titles0 == [CAFE, "Gold two", "Tab"]  # NFC, sorted
    assert titles1 == titles0 + d_titles and len(set(titles1)) == len(titles1)
    assert unicodedata.normalize("NFC", s1.decode("utf-8")) == s1.decode("utf-8")
    raw = {(r["split"], r["id"]): r for r in [*({**r, "split": "train"} for r in TRAIN),
                                              *({**r, "split": "dev"} for r in DEV)]}
    for name in ("S0", "S1"):
        records = list(iter_claims(out / name / "claims.jsonl"))
        assert [(r["split"], r["id"]) for r in records] == [
            ("train", 1), ("train", 2), ("train", 4), ("train", 5), ("dev", 6), ("dev", 7)]
        assert all(r == raw[(r["split"], r["id"])] for r in records)
        assert list(records[0])[-1] == "split"
    assert (out / "S0" / "claims.jsonl").read_bytes() == (out / "S1" / "claims.jsonl").read_bytes()
    corpora = json.loads((out / sb.CORPORA_JSON).read_text())
    manifest = (out / "S1" / "MANIFEST.md").read_text()
    assert f"| Pages (lines in pages.jsonl) | {3 + len(d_titles)} |" in manifest
    assert f"| Chunks | {corpora['S1']['chunks']:,} |" in manifest
    assert corpora["S1"]["files"]["pages.jsonl"]["sha256"] in manifest
    assert f"`{corpora['S1']['embedding_text_sha256']}`" in manifest
    # Café: the lead sentence, then its History section (the hatnote is in no chunk, the sentence
    # and the list item fit one chunk); Gold two and Tab one chunk each
    assert corpora["chunks_per_page"]["gold"] == {CAFE: 2, "Gold two": 1, "Tab": 1}
    assert corpora["S0"]["units_hatnote"] == 1
    # claims 1, 2, 4, 5, 6, 7 carry 1 + 2 + 2 + 4 + 1 + 2 ids, all on S0 pages; two are cells
    assert corpora["S0"]["evidence"] == {"evidence_ids": 12, "with_a_page": 12, "with_a_text_unit": 10}
    assert (out / "SELECTION.md").is_file() and (out / "DISTRACTORS_D.md").is_file()
