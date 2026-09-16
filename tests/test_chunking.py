"""Tests for wikilense.chunking: window filling, overlap, section boundaries, determinism."""

from __future__ import annotations

import pytest

from wikilense.chunking import Chunk, chunk_page, chunk_units, embedding_text
from wikilense.wikitext import TextUnit, parse_page


def _units(specs: list[tuple[int, int]]) -> list[TextUnit]:
    """Return units from ``(n_words, section_ordinal)`` specs; unit i has words ``w{i}_1 ...``."""
    units = []
    for i, (n_words, section) in enumerate(specs):
        text = " ".join(f"w{i}_{j}" for j in range(n_words))
        units.append(TextUnit(f"sentence_{i}", i, section, text, text))
    return units


def _keys(chunks: list[Chunk]) -> list[list[str]]:
    return [c.element_keys for c in chunks]


def test_fills_until_next_unit_would_exceed_max_words() -> None:
    chunks = chunk_units(_units([(5, 0), (5, 0), (5, 0), (5, 0)]), max_words=12, overlap_units=0)
    assert _keys(chunks) == [["sentence_0", "sentence_1"], ["sentence_2", "sentence_3"]]
    assert [c.n_words for c in chunks] == [10, 10]
    assert chunks[0].text == "w0_0 w0_1 w0_2 w0_3 w0_4 w1_0 w1_1 w1_2 w1_3 w1_4"
    assert chunks[0].n_words == len(chunks[0].text.split())


def test_max_words_boundary_exact_fit_and_one_over() -> None:
    exact = chunk_units(_units([(6, 0), (6, 0)]), max_words=12, overlap_units=0)
    assert _keys(exact) == [["sentence_0", "sentence_1"]]
    over = chunk_units(_units([(6, 0), (7, 0)]), max_words=12, overlap_units=0)
    assert _keys(over) == [["sentence_0"], ["sentence_1"]]
    single = chunk_units(_units([(12, 0)]), max_words=12, overlap_units=1)
    assert _keys(single) == [["sentence_0"]]


def test_long_unit_becomes_its_own_chunk() -> None:
    chunks = chunk_units(_units([(3, 0), (50, 0), (3, 0)]), max_words=10, overlap_units=1)
    assert _keys(chunks) == [["sentence_0"], ["sentence_1"], ["sentence_2"]]
    assert chunks[1].n_words == 50


def test_overlap_rule() -> None:
    # 4 units of 4 words, budget 8: [0,1] then start 1 unit before the end -> [1,2], [2,3]
    chunks = chunk_units(_units([(4, 0)] * 4), max_words=8, overlap_units=1)
    assert _keys(chunks) == [
        ["sentence_0", "sentence_1"],
        ["sentence_1", "sentence_2"],
        ["sentence_2", "sentence_3"],
    ]
    # overlap larger than the previous chunk's unit count: no overlap, no infinite loop
    chunks = chunk_units(_units([(4, 0)] * 4), max_words=8, overlap_units=2)
    assert _keys(chunks) == [["sentence_0", "sentence_1"], ["sentence_2", "sentence_3"]]
    # overlap 0 never repeats a unit
    chunks = chunk_units(_units([(4, 0)] * 5), max_words=8, overlap_units=0)
    assert _keys(chunks) == [
        ["sentence_0", "sentence_1"],
        ["sentence_2", "sentence_3"],
        ["sentence_4"],
    ]


def test_no_duplicate_or_subset_chunks() -> None:
    # the last chunk of a section would otherwise be repeated as an overlap-only chunk
    chunks = chunk_units(_units([(4, 0)] * 3), max_words=8, overlap_units=1)
    assert _keys(chunks) == [["sentence_0", "sentence_1"], ["sentence_1", "sentence_2"]]
    # an overlap unit that cannot take its successor would be a subset of the previous chunk
    chunks = chunk_units(_units([(2, 0), (6, 0), (6, 0), (2, 0)]), max_words=8, overlap_units=1)
    assert _keys(chunks) == [["sentence_0", "sentence_1"], ["sentence_2", "sentence_3"]]
    seen = {tuple(c.element_keys) for c in chunks}
    assert len(seen) == len(chunks)


def test_section_boundary_rule() -> None:
    chunks = chunk_units(_units([(2, 0), (2, 0), (2, 1), (2, 1), (2, 2)]), max_words=100)
    assert _keys(chunks) == [
        ["sentence_0", "sentence_1"],
        ["sentence_2", "sentence_3"],
        ["sentence_4"],
    ]
    assert [c.section_ordinal for c in chunks] == [0, 1, 2]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_empty_units_and_empty_input() -> None:
    assert chunk_units([]) == []
    units = _units([(0, 0), (3, 0), (0, 0)])
    chunks = chunk_units(units, max_words=10, overlap_units=1)
    assert _keys(chunks) == [["sentence_0", "sentence_1", "sentence_2"]]
    assert chunks[0].text == "w1_0 w1_1 w1_2" and chunks[0].n_words == 3
    assert chunk_units(_units([(0, 0), (0, 0)])) == []


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        chunk_units(_units([(1, 0)]), max_words=0)
    with pytest.raises(ValueError):
        chunk_units(_units([(1, 0)]), overlap_units=-1)


def test_determinism_and_purity() -> None:
    units = _units([(7, 0), (9, 0), (30, 0), (4, 1), (5, 1), (6, 1), (121, 2)])
    first = chunk_units(units, max_words=20, overlap_units=1)
    second = chunk_units(list(units), max_words=20, overlap_units=1)
    assert first == second
    assert [c.ordinal for c in first] == list(range(len(first)))
    assert units == _units([(7, 0), (9, 0), (30, 0), (4, 1), (5, 1), (6, 1), (121, 2)])


def test_chunk_page_uses_parsed_units() -> None:
    page = {
        "title": "T",
        "order": ["sentence_0", "section_0", "list_0"],
        "sentence_0": "One [[Two|two]] three.",
        "section_0": {"value": "H", "level": 2},
        "list_0": {
            "type": "unordered_list",
            "list": [{"id": "item_0_0", "value": "Four five.", "level": 0}],
        },
    }
    parsed = parse_page(page)
    chunks = chunk_page(parsed, 120, 1)
    assert chunks == chunk_units(parsed.units, max_words=120, overlap_units=1)
    assert [(c.section_ordinal, c.text) for c in chunks] == [
        (0, "One two three."),
        (1, "Four five."),
    ]
    assert chunk_page(parsed) == chunks


def test_embedding_text_with_and_without_path() -> None:
    assert embedding_text("Aare", "Course > Upper course", "Glaciers feed it.") == (
        "Aare > Course > Upper course: Glaciers feed it."
    )
    assert embedding_text("Aare", "", "The Aare is a river.") == "Aare: The Aare is a river."
