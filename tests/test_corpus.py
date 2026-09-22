"""Tests for wikilense.corpus and scripts/build_corpus.py (synthetic data; no database)."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from wikilense.corpus import (
    _link_targets,
    claim_pages,
    iter_claims,
    iter_pages,
    link_targets,
    load_corpus,
    nfc,
    parse_element_id,
    select_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
HEADER = {
    "id": "",
    "claim": "",
    "label": "",
    "evidence": "",
    "annotator_operations": "",
    "challenge": "",
}


# --- element ids -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "element_id, expected",
    [
        ("Aare_sentence_7", ("Aare", "sentence", "sentence_7")),
        ("Aare_cell_0_1_1", ("Aare", "cell", "cell_0_1_1")),
        ("Aare_header_cell_0_2_0", ("Aare", "header_cell", "header_cell_0_2_0")),
        ("Aare_table_caption_3", ("Aare", "table_caption", "table_caption_3")),
        ("Aare_item_0_1", ("Aare", "item", "item_0_1")),
        ("Michael Folivi_cell_1_12_0", ("Michael Folivi", "cell", "cell_1_12_0")),
        # titles containing an underscore, including one that ends like an element id
        ("Foo_Bar_item_0_1", ("Foo_Bar", "item", "item_0_1")),
        ("A_sentence_1_sentence_2", ("A_sentence_1", "sentence", "sentence_2")),
        ("Cell_biology_cell_0_1_1", ("Cell_biology", "cell", "cell_0_1_1")),
    ],
)
def test_parse_element_id(element_id, expected):
    assert parse_element_id(element_id) == expected


def test_parse_element_id_normalises_decomposed_title_to_nfc():
    decomposed = "Café"  # e + combining acute
    page, element_type, key = parse_element_id(f"{decomposed}_sentence_2")
    assert page == "Café"
    assert page == nfc(decomposed)
    assert (element_type, key) == ("sentence", "sentence_2")


@pytest.mark.parametrize(
    "bad", ["Aare", "Aare_sentence_", "Aare_paragraph_1", "Aare_title", "_sentence_1", ""]
)
def test_parse_element_id_rejects_non_ids(bad):
    with pytest.raises(ValueError):
        parse_element_id(bad)


def test_claim_pages_unions_every_evidence_set():
    record = {
        "evidence": [
            {"content": ["Algebraic logic_sentence_0", "Lindenbaum–Tarski algebra_sentence_1"]},
            {"content": ["Algebraic logic_cell_0_1_0", "Café_item_0_0"]},
        ]
    }
    assert claim_pages(record) == {"Algebraic logic", "Lindenbaum–Tarski algebra", "Café"}


# --- synthetic shard and claim files ------------------------------------------------------------


def _page(title: str, sentences: list[str], items: list[str] = (), cells: list[str] = ()) -> dict:
    """Return a page object in the FEVEROUS shape with the given sentences, list and table."""
    page = {"title": title, "order": []}
    for i, text in enumerate(sentences):
        page["order"].append(f"sentence_{i}")
        page[f"sentence_{i}"] = text
    if items:
        page["order"].append("list_0")
        page["list_0"] = {
            "list": [{"id": f"item_0_{i}", "value": v, "level": 1} for i, v in enumerate(items)],
            "type": "unordered",
        }
    if cells:
        page["order"].append("table_0")
        row = [
            {"id": f"cell_0_0_{j}", "value": v, "is_header": False, "row_span": "1",
             "column_span": "1"}
            for j, v in enumerate(cells)
        ]
        page["table_0"] = {"table": [row], "type": "general"}
    return page


SHARD_PAGES = [
    _page("Zeta", ["Plain."] * 25),
    _page(
        "Alpha",
        ["See [[Gamma|the gamma]] and [[Delta]].", "Also [[Missing_page|missing]] and [[Beta#History|Beta]]."],
        items=["[[Epsilon|e]]", "no link"],
        cells=["[[Gamma]] again", "[[#Top|fragment only]]"],
    ),
    _page("Beta", ["[[Delta|d]] [[Epsilon]] [[Alpha]]"]),
    _page("Gamma", ["One.", "Two.", "Three."]),
    _page("Delta", ["One."]),
    _page("Epsilon", ["One."]),
    _page("Eta", ["Short."] * 3),
    _page("Theta", ["Exactly twenty."] * 20),
    _page("Iota", ["Long."] * 30),
]


def _claim(claim_id: int, label: str, sets: list[list[str]]) -> dict:
    """Return a FEVEROUS-shaped claim record with the given evidence sets."""
    return {
        "evidence": [{"content": content, "context": {}} for content in sets],
        "id": claim_id,
        "claim": f"claim {claim_id}",
        "label": label,
        "annotator_operations": [],
        "challenge": "Other",
    }


TRAIN_CLAIMS = [
    _claim(1, "SUPPORTS", [["Alpha_sentence_0"], ["Beta_cell_0_0_0", "Alpha_item_0_0"]]),
    _claim(2, "REFUTES", [["Missing page_sentence_0"]]),
    _claim(3, "REFUTES", [["Alpha_sentence_1", "Missing page_sentence_2"]]),
    _claim(5, "NOT ENOUGH INFO", [[]]),
]
DEV_CLAIMS = [_claim(4, "REFUTES", [["Beta_sentence_0"]])]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


@pytest.fixture
def synthetic(tmp_path: Path) -> dict[str, Path]:
    """Write the synthetic shard and claim files; return their paths."""
    paths = {
        "shard": tmp_path / "wiki_synthetic.jsonl",
        "train": tmp_path / "train.jsonl",
        "dev": tmp_path / "dev.jsonl",
        "out": tmp_path / "corpus",
    }
    _write_jsonl(paths["shard"], SHARD_PAGES)
    _write_jsonl(paths["train"], [HEADER, *TRAIN_CLAIMS])
    _write_jsonl(paths["dev"], [HEADER, *DEV_CLAIMS])
    return paths


def test_link_targets_reads_sentences_items_and_cells():
    alpha = SHARD_PAGES[1]
    assert link_targets(alpha) == {"Gamma", "Delta", "Missing page", "Beta", "Epsilon"}
    assert link_targets(SHARD_PAGES[0]) == set()
    assert link_targets({"title": "X", "order": ["sentence_0"], "sentence_0": "[[A_b#c|x]]"}) == {"A b"}
    assert _link_targets is link_targets  # the former private name still works


def test_iter_pages_keeps_file_order(synthetic):
    assert [p["title"] for p in iter_pages(synthetic["shard"])] == [p["title"] for p in SHARD_PAGES]


def test_iter_claims_skips_header_and_adds_split(synthetic):
    records = list(iter_claims(synthetic["train"], "train"))
    assert [r["id"] for r in records] == [1, 2, 3, 5]
    assert all(r["split"] == "train" for r in records)
    assert "" not in {r["id"] for r in records}


def test_iter_claims_without_split_requires_stored_split(synthetic):
    with pytest.raises(ValueError):
        list(iter_claims(synthetic["train"]))


def test_select_corpus_in_shard_claims_and_evidence(synthetic):
    sel = select_corpus([synthetic["shard"]], synthetic["train"], synthetic["dev"], n_fill=2)
    assert [(c["id"], c["split"]) for c in sel.claims] == [(1, "train"), (4, "dev")]
    assert sel.evidence_titles == ["Alpha", "Beta"]


def test_filler_ranking_by_distinct_linking_pages_then_title(synthetic):
    # Delta and Epsilon are linked from both evidence pages (tie -> title order); Gamma is linked
    # twice from Alpha but counts once; Missing page is not a shard title; the "#Top" link is empty.
    sel = select_corpus([synthetic["shard"]], synthetic["train"], synthetic["dev"], n_fill=2)
    assert sel.filler_titles == ["Delta", "Epsilon"]
    assert (sel.filler_from_links, sel.filler_from_order) == (2, 0)
    assert sel.filler_link_counts == {"Delta": 2, "Epsilon": 2}
    sel3 = select_corpus([synthetic["shard"]], synthetic["train"], synthetic["dev"], n_fill=3)
    assert sel3.filler_titles == ["Delta", "Epsilon", "Gamma"]
    assert sel3.filler_link_counts["Gamma"] == 1


def test_filler_fallback_follows_shard_order_with_min_sentences(synthetic):
    sel = select_corpus(
        [synthetic["shard"]], synthetic["train"], synthetic["dev"], n_fill=5, min_sentences=20
    )
    # Zeta (25) comes first in shard order; Eta (3) is too short; Theta has exactly 20.
    assert sel.filler_titles == ["Delta", "Epsilon", "Gamma", "Zeta", "Theta"]
    assert (sel.filler_from_links, sel.filler_from_order) == (3, 2)
    assert sel.titles == {"Alpha", "Beta", "Delta", "Epsilon", "Gamma", "Zeta", "Theta"}
    sel_all = select_corpus(
        [synthetic["shard"]], synthetic["train"], synthetic["dev"], n_fill=50, min_sentences=1
    )
    assert sel_all.filler_titles == ["Delta", "Epsilon", "Gamma", "Zeta", "Eta", "Theta", "Iota"]
    none = select_corpus([synthetic["shard"]], synthetic["train"], synthetic["dev"], n_fill=0)
    assert none.filler_titles == [] and none.evidence_titles == ["Alpha", "Beta"]


def _load_build_script():
    spec = importlib.util.spec_from_file_location(
        "build_corpus", REPO_ROOT / "scripts" / "build_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_counts(text: str) -> dict[str, int]:
    """Return the ``| name | number |`` rows of a manifest as a dict."""
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"^\| (.+?) \| (\d+) \|$", text, re.MULTILINE)}


def test_build_script_writes_raw_lines_claims_and_manifest(synthetic):
    build = _load_build_script()
    argv = [
        "--shard", str(synthetic["shard"]), "--train", str(synthetic["train"]),
        "--dev", str(synthetic["dev"]), "--out-dir", str(synthetic["out"]),
        "--n-fill", "5", "--min-sentences", "20",
    ]
    assert build.main(argv) == 0
    raw_lines = {json.loads(line)["title"]: line for line in synthetic["shard"].read_bytes().splitlines(True)}
    out_lines = (synthetic["out"] / "pages.jsonl").read_bytes().splitlines(True)
    titles = [json.loads(line)["title"] for line in out_lines]
    assert titles == ["Zeta", "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Theta"]  # shard order
    assert all(line == raw_lines[title] for title, line in zip(titles, out_lines))

    pages, claims = load_corpus(synthetic["out"])
    assert [p["title"] for p in pages] == titles
    assert [(c["id"], c["split"], c["label"]) for c in claims] == [
        (1, "train", "SUPPORTS"), (4, "dev", "REFUTES")
    ]
    assert set(claims[0]) == set(TRAIN_CLAIMS[0]) | {"split"}

    manifest = (synthetic["out"] / "MANIFEST.md").read_text(encoding="utf-8")
    counts = _manifest_counts(manifest)
    assert counts["Pages (lines in pages.jsonl)"] == 7
    assert counts["Evidence pages"] == 2
    assert counts["Filler pages"] == 5
    assert counts["Filler pages from links"] == 3
    assert counts["Filler pages from shard order"] == 2
    assert counts["Claims (lines in claims.jsonl)"] == 2
    assert counts["Claims, train"] == 1 and counts["Claims, dev"] == 1
    assert counts["Claims, SUPPORTS"] == 1 and counts["Claims, REFUTES"] == 1
    assert counts["Claims, NOT ENOUGH INFO"] == 0
    assert counts["Claims with a sentence-only evidence set"] == 2
    assert counts["Evidence element ids"] == 4
    assert "1. Delta (linked from 2)" in manifest and "5. Theta (shard order)" in manifest
    assert hashlib.sha256((synthetic["out"] / "pages.jsonl").read_bytes()).hexdigest() in manifest
    # deterministic: a second run reproduces the manifest byte for byte
    assert build.main(argv) == 0
    assert (synthetic["out"] / "MANIFEST.md").read_text(encoding="utf-8") == manifest


def test_load_corpus_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path)


# --- the committed corpus ------------------------------------------------------------------------


@pytest.mark.skipif(
    not all((CORPUS_DIR / n).is_file() for n in ("pages.jsonl", "claims.jsonl", "MANIFEST.md")),
    reason="data/corpus files absent (run scripts/build_corpus.py)",
)
def test_manifest_counts_match_corpus_files():
    manifest = (CORPUS_DIR / "MANIFEST.md").read_text(encoding="utf-8")
    counts = _manifest_counts(manifest)
    pages, claims = load_corpus(CORPUS_DIR)
    titles = [nfc(p["title"]) for p in pages]
    evidence = set()
    for record in claims:
        evidence |= claim_pages(record)
    assert evidence <= set(titles)

    assert counts["Pages (lines in pages.jsonl)"] == len(pages)
    assert counts["Evidence pages"] == len(evidence)
    assert counts["Filler pages"] == len(pages) - len(evidence)
    assert counts["Filler pages from links"] + counts["Filler pages from shard order"] == (
        counts["Filler pages"]
    )
    assert counts["Claims (lines in claims.jsonl)"] == len(claims)
    for split in ("train", "dev"):
        assert counts[f"Claims, {split}"] == sum(1 for c in claims if c["split"] == split)
    for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
        assert counts[f"Claims, {label}"] == sum(1 for c in claims if c["label"] == label)
    sentence_only = sum(
        1
        for c in claims
        if any(
            s["content"] and all(parse_element_id(el)[1] == "sentence" for el in s["content"])
            for s in c["evidence"]
        )
    )
    assert counts["Claims with a sentence-only evidence set"] == sentence_only
    assert counts["Evidence element ids"] == sum(
        len(s["content"]) for c in claims for s in c["evidence"]
    )
    for name in ("pages.jsonl", "claims.jsonl"):
        data = (CORPUS_DIR / name).read_bytes()
        assert f"| {name} | {len(data)} | {hashlib.sha256(data).hexdigest()} |" in manifest
    for title in sorted(evidence):
        assert f"- {title} (" in manifest
    for title in set(titles) - evidence:
        assert re.search(rf"^\d+\. {re.escape(title)} \(", manifest, re.MULTILINE)
