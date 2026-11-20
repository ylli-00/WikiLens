"""Tests for wikilense.picker without a database: reading the two claim formats, the claim kind,
the example rule and the fixed examples against the committed data, plus the two functions the
picker shares with the ingest and the evaluation (``ingest.resolve_element`` and
``evaluate.ground_truth_from_rows``), so that the picker's gold evidence is the evaluation's.

The tests that need the database (the catalog, the marks and the summary) are in
tests/test_web.py, next to the synthetic corpus they run on.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from wikilense import picker
from wikilense import web as webmod
from wikilense.config import REPO_ROOT
from wikilense.corpus import iter_claims, iter_pages, parse_element_id
from wikilense.evaluate import TEXT_UNIT_TYPES, ClaimTruth, ground_truth_from_rows
from wikilense.ingest import resolve_element
from wikilense.picker import (
    Catalog,
    ClaimFileError,
    FileClaim,
    GoldElement,
    OfferedClaim,
    claim_kind,
    example_rank,
    pick_examples,
    read_claim_files,
)
from wikilense.wikitext import page_stats

SCALED_BUILD = REPO_ROOT / "scripts" / "scaled_build.py"
DEMO_CLAIMS = REPO_ROOT / "data" / "corpus" / "claims.jsonl"
DEMO_PAGES = REPO_ROOT / "data" / "corpus" / "pages.jsonl"
SCALED_CLAIMS = REPO_ROOT / "data" / "scaled" / "claims_3000.jsonl"


def _feverous_record(cid: int, label: str, text: str, sets: list[list[str]]) -> dict[str, Any]:
    """Return a claim record in the format of data/corpus/claims.jsonl."""
    return {
        "evidence": [{"content": s, "context": {}} for s in sets],
        "id": cid,
        "claim": text,
        "label": label,
        "annotator_operations": [],
        "challenge": "Other",
        "split": "train",
    }


def _scaled_record(cid: int, label: str, text: str, sets: list[list[str]]) -> dict[str, Any]:
    """Return a claim record in the format of data/scaled/claims_3000.jsonl."""
    return {
        "id": cid,
        "split": "train",
        "label": label,
        "claim": text,
        "kind": claim_kind(sets),
        "challenge": "Other",
        "evidence_sets": [{"elements": s, "text_only": True, "findable": True} for s in sets],
    }


def _write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------
# read_claim_files
# ---------------------------------------------------------------------------------------------


def test_read_claim_files_reads_both_formats_and_merges_a_shared_claim(tmp_path: Path) -> None:
    shared = (7, "SUPPORTS", "Alpha has two sentences.", [["Alpha_sentence_0", "Alpha_sentence_1"]])
    demo = _write(
        tmp_path / "demo.jsonl",
        [_feverous_record(*shared), _feverous_record(3, "REFUTES", "Beta.", [["Beta_cell_0_0_0"]])],
    )
    scaled = _write(
        tmp_path / "scaled.jsonl",
        [_scaled_record(5, "NOT ENOUGH INFO", "Gamma.", [["Gamma_item_0_1"]]),
         _scaled_record(*shared)],
    )
    claims = read_claim_files([demo, scaled])
    # file order, then line order; the shared claim keeps its first place
    assert list(claims) == [7, 3, 5]
    assert claims[7] == FileClaim(
        claim_id=7,
        split="train",
        label="SUPPORTS",
        text="Alpha has two sentences.",
        evidence_sets=(("Alpha_sentence_0", "Alpha_sentence_1"),),
        sources=("demo.jsonl", "scaled.jsonl"),
    )
    assert claims[3].sources == ("demo.jsonl",) and claims[3].evidence_sets == (("Beta_cell_0_0_0",),)
    assert claims[5].sources == ("scaled.jsonl",) and claims[5].label == "NOT ENOUGH INFO"
    assert picker.file_claim_ids(claims) == {
        "demo.jsonl": frozenset({7, 3}),
        "scaled.jsonl": frozenset({5, 7}),
    }


@pytest.mark.parametrize(
    "changed",
    [
        (7, "SUPPORTS", "Alpha has three sentences.", [["Alpha_sentence_0"]]),  # text
        (7, "REFUTES", "Alpha has two sentences.", [["Alpha_sentence_0"]]),  # label
        (7, "SUPPORTS", "Alpha has two sentences.", [["Alpha_sentence_1"]]),  # evidence
        (7, "SUPPORTS", "Alpha has two sentences.", [["Alpha_sentence_0"], ["Beta_sentence_0"]]),
    ],
)
def test_read_claim_files_refuses_two_files_that_disagree_on_a_claim(
    tmp_path: Path, changed: tuple[Any, ...]
) -> None:
    demo = _write(
        tmp_path / "demo.jsonl",
        [_feverous_record(7, "SUPPORTS", "Alpha has two sentences.", [["Alpha_sentence_0"]])],
    )
    scaled = _write(tmp_path / "scaled.jsonl", [_scaled_record(*changed)])
    with pytest.raises(ClaimFileError, match="claim 7 differs between demo.jsonl and scaled.jsonl"):
        read_claim_files([demo, scaled])


def test_read_claim_files_skips_a_missing_file_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    demo = _write(tmp_path / "demo.jsonl", [_feverous_record(7, "SUPPORTS", "A.", [["A_sentence_0"]])])
    with caplog.at_level(logging.WARNING, logger="wikilense.picker"):
        claims = read_claim_files([tmp_path / "absent.jsonl", demo])
    assert list(claims) == [7] and claims[7].sources == ("demo.jsonl",)
    assert "absent.jsonl" in caplog.text and "not found" in caplog.text
    assert read_claim_files([tmp_path / "absent.jsonl"]) == {}


def test_read_claim_files_skips_the_feverous_header_line(tmp_path: Path) -> None:
    header = {"id": "", "claim": "", "label": "", "evidence": [], "annotator_operations": []}
    demo = _write(
        tmp_path / "demo.jsonl", [header, _feverous_record(7, "SUPPORTS", "A.", [["A_sentence_0"]])]
    )
    assert list(read_claim_files([demo])) == [7]


def test_read_claim_files_refuses_a_scaled_record_without_a_field(tmp_path: Path) -> None:
    record = _scaled_record(7, "SUPPORTS", "A.", [["A_sentence_0"]])
    del record["label"]
    scaled = _write(tmp_path / "scaled.jsonl", [record])
    with pytest.raises(ClaimFileError, match="scaled.jsonl: unreadable claim record"):
        read_claim_files([scaled])


def test_read_claim_files_refuses_a_feverous_record_without_split(tmp_path: Path) -> None:
    """A record without the fields is a ClaimFileError in either format."""
    record = _feverous_record(7, "SUPPORTS", "A.", [["A_sentence_0"]])
    del record["split"]
    demo = _write(tmp_path / "demo.jsonl", [record])
    with pytest.raises(ClaimFileError, match="demo.jsonl: unreadable claim record"):
        read_claim_files([demo])


def test_read_claim_files_refuses_a_line_that_is_not_json(tmp_path: Path) -> None:
    """A broken line stops the read with the file named, not with a bare JSON error."""
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"id": 7, "claim": \n', encoding="utf-8")
    with pytest.raises(ClaimFileError, match="broken.jsonl: unreadable claim record"):
        read_claim_files([broken])


def test_the_committed_claim_files_agree_on_every_shared_claim() -> None:
    """The page reads both files at start-up; a disagreement would stop every claim request."""
    claims = read_claim_files(picker.CLAIM_FILES)
    ids = picker.file_claim_ids(claims)
    assert set(ids) == {"data/corpus/claims.jsonl", "data/scaled/claims_3000.jsonl"}
    assert len(ids["data/corpus/claims.jsonl"]) == 75
    assert len(ids["data/scaled/claims_3000.jsonl"]) == 3000


# ---------------------------------------------------------------------------------------------
# claim_kind
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scaled_build() -> Iterator[Any]:
    """scripts/scaled_build.py imported as a module (it is a script, not part of the package)."""
    spec = importlib.util.spec_from_file_location("scaled_build", SCALED_BUILD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.mark.parametrize(
    ("sets", "kind"),
    [
        ([["A_sentence_0"]], "single"),
        ([["A_sentence_0", "A_cell_0_1_0"], ["A_item_0_2"]], "single"),
        ([[], ["A_sentence_0"]], "single"),  # an empty set is skipped
        ([["A_sentence_0", "B_sentence_0"]], "needs2"),
        ([["A_sentence_0", "B_cell_0_0_0"], ["C_sentence_1", "A_sentence_2"]], "needs2"),
        ([["A_sentence_0"], ["B_sentence_0"]], "either"),
        ([["A_sentence_0"], ["A_sentence_1", "B_sentence_1"]], "either"),
        ([], "none"),
        ([[]], "none"),
    ],
)
def test_claim_kind_is_the_rule_of_scaled_build_describe(
    scaled_build: Any, sets: list[list[str]], kind: str
) -> None:
    assert claim_kind(sets) == kind
    record = {"evidence": [{"content": s} for s in sets]}
    assert scaled_build.describe(record)[0] == kind


def test_claim_kind_gives_every_committed_scaled_claim_its_recorded_kind() -> None:
    with open(SCALED_CLAIMS, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert len(records) == 3000
    wrong = [
        r["id"] for r in records
        if claim_kind([s["elements"] for s in r["evidence_sets"]]) != r["kind"]
    ]
    assert wrong == []


# ---------------------------------------------------------------------------------------------
# ingest.resolve_element and evaluate.ground_truth_from_rows
# ---------------------------------------------------------------------------------------------

PAGE_IDS = {"Alpha": 1, "Café": 2}
SENTENCE_IDS = {(1, "sentence_0"): 10, (1, "item_0_1"): 11, (2, "sentence_3"): 12}


def test_resolve_element_resolves_page_and_text_unit() -> None:
    assert resolve_element("Alpha_sentence_0", PAGE_IDS, SENTENCE_IDS) == (
        "Alpha", "sentence", "sentence_0", 1, 10
    )
    assert resolve_element("Alpha_item_0_1", PAGE_IDS, SENTENCE_IDS) == (
        "Alpha", "item", "item_0_1", 1, 11
    )
    # the id is normalised to NFC before the title lookup, as the ingest stores titles
    nfd = unicodedata.normalize("NFD", "Café_sentence_3")
    assert nfd != "Café_sentence_3"
    assert resolve_element(nfd, PAGE_IDS, SENTENCE_IDS) == ("Café", "sentence", "sentence_3", 2, 12)


def test_resolve_element_gives_a_table_element_its_page_only() -> None:
    for element_id, element_type, key in (
        ("Alpha_cell_0_1_2", "cell", "cell_0_1_2"),
        ("Alpha_header_cell_0_0_1", "header_cell", "header_cell_0_0_1"),
        ("Alpha_table_caption_0", "table_caption", "table_caption_0"),
    ):
        assert resolve_element(element_id, PAGE_IDS, SENTENCE_IDS) == (
            "Alpha", element_type, key, 1, None
        )


def test_resolve_element_leaves_what_the_database_lacks_unresolved() -> None:
    # a page that is not in the database: neither page nor unit
    assert resolve_element("Delta_sentence_0", PAGE_IDS, SENTENCE_IDS) == (
        "Delta", "sentence", "sentence_0", None, None
    )
    # a page that is there without the sentence: the page only
    assert resolve_element("Alpha_sentence_9", PAGE_IDS, SENTENCE_IDS) == (
        "Alpha", "sentence", "sentence_9", 1, None
    )
    # the key of another page's sentence does not resolve on this one
    assert resolve_element("Café_sentence_0", PAGE_IDS, SENTENCE_IDS)[3:] == (2, None)


@pytest.mark.parametrize("element_id", ["Alpha", "Alpha_paragraph_0", "Alpha_sentence_", ""])
def test_resolve_element_refuses_a_malformed_id(element_id: str) -> None:
    with pytest.raises(ValueError, match="not a FEVEROUS element id"):
        resolve_element(element_id, PAGE_IDS, SENTENCE_IDS)


def test_ground_truth_from_rows_builds_the_claim_truths() -> None:
    claim_rows = [
        (9, "dev", "nine", "REFUTES"),  # kept in the order given, not sorted
        (4, "train", "four", "SUPPORTS"),
        (5, "train", "five", "NOT ENOUGH INFO"),
    ]
    evidence_rows = [
        # claim 4: a sentence-only set, a set with a cell, a set with an unresolved sentence,
        # and a set of a sentence and a list item
        (4, 0, "sentence", 1, "Alpha", 10),
        (4, 1, "cell", 2, "Beta", None),
        (4, 1, "sentence", 2, "Beta", 20),
        (4, 2, "sentence", 1, "Alpha", None),
        (4, 3, "sentence", 1, "Alpha", 10),
        (4, 3, "item", 1, "Alpha", 11),
        # claim 9: only a page that is not in the database
        (9, 0, "sentence", None, None, None),
    ]
    truths = ground_truth_from_rows(claim_rows, evidence_rows)
    assert truths == [
        ClaimTruth(9, "dev", "REFUTES", "nine", set(), [], set()),
        ClaimTruth(4, "train", "SUPPORTS", "four", {1, 2}, [{10}, {10, 11}], {"Alpha", "Beta"}),
        ClaimTruth(5, "train", "NOT ENOUGH INFO", "five", set(), [], set()),
    ]
    assert [t.eligible for t in truths] == [False, True, False]


# ---------------------------------------------------------------------------------------------
# the example rule
# ---------------------------------------------------------------------------------------------


def _offered(cid: int, label: str, sets: list[list[str]], *, eligible: bool = True) -> OfferedClaim:
    """Return an OfferedClaim whose evidence sets cite one sentence of each listed page (a cell
    when not ``eligible``); the kind is :func:`claim_kind` of those element ids."""
    element_type, key = ("sentence", "sentence_0") if eligible else ("cell", "cell_0_0_0")
    built = []
    n = 0
    for pages in sets:
        elements = []
        for page in pages:
            n += 1
            elements.append(
                GoldElement(f"{page}_{key}", page, element_type, key, n,
                            n if eligible else None, "text" if eligible else None)
            )
        built.append(tuple(elements))
    evidence_sets = tuple(tuple(e.element_id for e in s) for s in built)
    truth = ClaimTruth(
        cid, "train", label, f"claim {cid}", {e.page_id for s in built for e in s},
        [{e.sentence_id for e in s} for s in built] if eligible else [],
        {e.page_title for s in built for e in s},
    )
    claim = FileClaim(cid, "train", label, f"claim {cid}", evidence_sets, ("claims.jsonl",))
    return OfferedClaim(claim, truth, tuple(built), claim_kind(evidence_sets))


def _catalog(name: str, claims: list[OfferedClaim]) -> Catalog:
    return Catalog(name, 10, 100, ("claims.jsonl",), {c.claim_id: c for c in claims}, ())


def test_example_rank_is_the_salted_sha256_of_the_claim_id() -> None:
    assert example_rank(36003) == hashlib.sha256(
        f"{picker.EXAMPLE_SALT}:36003".encode()
    ).hexdigest()
    assert example_rank(1) != example_rank(2)


def test_pick_examples_takes_the_lowest_rank_per_label_under_the_rule() -> None:
    ids = sorted(range(1, 200), key=example_rank)  # ids[0] has the lowest rank
    s_demo_only, s_table, s_pick, s_later = ids[0:4]
    r_shares, r_pick, n_shares, n_pick = ids[4:8]
    two_demo_only, two_table, two_pick, two_later = ids[8:12]
    both = [
        _offered(s_table, "SUPPORTS", [["P0"]], eligible=False),  # no sentence-only set
        _offered(s_pick, "SUPPORTS", [["P1"]]),
        _offered(s_later, "SUPPORTS", [["P2"]]),
        _offered(r_shares, "REFUTES", [["P7"], ["P1"]]),  # cites the SUPPORTS example's page
        _offered(r_pick, "REFUTES", [["P3"]]),
        _offered(n_shares, "NOT ENOUGH INFO", [["P3"]]),  # cites the REFUTES example's page
        _offered(n_pick, "NOT ENOUGH INFO", [["P4"]]),
        _offered(two_later, "SUPPORTS", [["P5", "P6"]]),
    ]
    demo = _catalog(
        "demo",
        [
            _offered(s_demo_only, "SUPPORTS", [["P9"]]),  # not offered on S1
            _offered(two_demo_only, "REFUTES", [["P10", "P11"]]),  # needs2, but not on S1
            *both,
        ],
    )
    s1 = _catalog(
        "s1",
        [
            *both,
            _offered(two_table, "REFUTES", [["P8", "P9"]], eligible=False),
            _offered(two_pick, "REFUTES", [["P8", "P9"]]),  # offered on S1 only: allowed
        ],
    )
    assert [demo.claims[c].kind for c in (r_shares, two_later)] == ["either", "needs2"]
    assert pick_examples(demo, s1) == {
        "SUPPORTS": s_pick,
        "REFUTES": r_pick,
        "NOT ENOUGH INFO": n_pick,
        "two_page": two_pick,
    }


def test_pick_examples_refuses_a_group_without_a_candidate() -> None:
    supports = _offered(1, "SUPPORTS", [["P1"]])
    refutes = _offered(2, "REFUTES", [["P2"]])
    nei = _offered(3, "NOT ENOUGH INFO", [["P3"]])
    with pytest.raises(ValueError, match="no example candidate for SUPPORTS"):
        pick_examples(_catalog("demo", []), _catalog("s1", []))
    with pytest.raises(ValueError, match="no example candidate for NOT ENOUGH INFO"):
        pick_examples(_catalog("demo", [supports, refutes]), _catalog("s1", [supports, refutes]))
    full = [supports, refutes, nei]
    with pytest.raises(ValueError, match="no two-page example candidate"):
        pick_examples(_catalog("demo", full), _catalog("s1", full))


# ---------------------------------------------------------------------------------------------
# the fixed examples against the committed data
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_claims() -> dict[int, dict[str, Any]]:
    return {c["id"]: c for c in iter_claims(DEMO_CLAIMS)}


@pytest.fixture(scope="module")
def demo_pages() -> dict[str, dict[str, Any]]:
    return {p["title"]: p for p in iter_pages(DEMO_PAGES)}


def _sentence_only(evidence_set: list[str]) -> bool:
    return bool(evidence_set) and all(
        parse_element_id(e)[1] in TEXT_UNIT_TYPES for e in evidence_set
    )


def test_examples_have_the_fixed_parameters() -> None:
    by_key = {e.key: e for e in picker.EXAMPLES}
    assert list(by_key) == ["supported", "refuted", "nei", "filter", "free_text", "two_page"]
    for key, label in (("supported", "SUPPORTS"), ("refuted", "REFUTES"),
                       ("nei", "NOT ENOUGH INFO")):
        assert by_key[key].params == {
            "claim_id": str(picker.EXAMPLE_CLAIM_IDS[label]), "sentences": "1"
        }
    assert by_key["filter"].params == {
        "claim_id": str(picker.EXAMPLE_CLAIM_IDS["SUPPORTS"]),
        "max_words": str(picker.EXAMPLE_FILTER_MAX_WORDS),
        "sentences": "1",
    }
    assert by_key["free_text"].params == {"q": picker.EXAMPLE_FREE_TEXT}
    assert by_key["two_page"].params == {
        "claim_id": str(picker.EXAMPLE_CLAIM_IDS["two_page"]), "sentences": "1"
    }
    assert [e.key for e in picker.EXAMPLES if e.s1_only] == ["two_page"]
    # every example opens the page with parameters the API accepts
    for example in picker.EXAMPLES:
        assert set(example.params) <= webmod.API_PARAMETERS, example.key


def test_the_label_examples_are_demo_claims_with_a_sentence_only_set(
    demo_claims: dict[int, dict[str, Any]],
) -> None:
    for label in picker.LABELS:
        claim = demo_claims[picker.EXAMPLE_CLAIM_IDS[label]]
        assert claim["label"] == label
        assert any(_sentence_only(s["content"]) for s in claim["evidence"]), label


def test_the_filter_example_excludes_exactly_the_supported_claims_gold_page(
    demo_claims: dict[int, dict[str, Any]], demo_pages: dict[str, dict[str, Any]]
) -> None:
    page = demo_pages[picker.EXAMPLE_FILTER_PAGE]
    assert picker.EXAMPLE_FILTER_MAX_WORDS == page_stats(page)["n_words"] - 1
    claim = demo_claims[picker.EXAMPLE_CLAIM_IDS["SUPPORTS"]]
    gold_pages = {parse_element_id(e)[0] for s in claim["evidence"] for e in s["content"]}
    assert gold_pages == {picker.EXAMPLE_FILTER_PAGE}  # max_words removes the only gold page


def test_the_free_text_example_asks_about_a_demo_page(
    demo_pages: dict[str, dict[str, Any]],
) -> None:
    assert picker.EXAMPLE_FREE_TEXT_PAGE in demo_pages
    assert picker.EXAMPLE_FREE_TEXT_PAGE in picker.EXAMPLE_FREE_TEXT


def test_the_two_page_example_is_a_needs2_claim_of_the_scaled_file(
    demo_claims: dict[int, dict[str, Any]],
) -> None:
    cid = picker.EXAMPLE_CLAIM_IDS["two_page"]
    with open(SCALED_CLAIMS, encoding="utf-8") as handle:
        records = {r["id"]: r for r in (json.loads(line) for line in handle if line.strip())}
    record = records[cid]
    sets = [s["elements"] for s in record["evidence_sets"]]
    assert record["kind"] == "needs2" and claim_kind(sets) == "needs2"
    assert any(_sentence_only(s) for s in sets)
    assert cid not in demo_claims  # hence offered on S1 only


def test_read_claim_files_refuses_an_evidence_element_that_is_not_an_element_id(
    tmp_path: Path,
) -> None:
    """A malformed element id is reported when the file is read, not later as a server error."""
    record = _scaled_record(7, "SUPPORTS", "A.", [["A_sentence_0"]])
    record["evidence_sets"][0]["elements"] = ["A_title"]
    scaled = _write(tmp_path / "scaled.jsonl", [record])
    with pytest.raises(ClaimFileError, match="scaled.jsonl: claim 7: not a FEVEROUS element id"):
        read_claim_files([scaled])
