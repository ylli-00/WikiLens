"""Tests for wikilense.wikitext: section stack, text units, clean_text, links and page stats."""

from __future__ import annotations

from wikilense.wikitext import (
    HATNOTE_RE,
    ParsedPage,
    Section,
    TextUnit,
    clean_text,
    extract_links,
    is_hatnote,
    link_source,
    link_targets,
    normalise_title,
    page_stats,
    parse_page,
    section_path,
)


def _page() -> dict:
    """Return a small FEVEROUS-shaped page with sections at levels 2, 3, 2 and mixed elements."""
    return {
        "title": "Aare",
        "order": [
            "sentence_0",
            "table_0",
            "sentence_1",
            "section_0",
            "sentence_2",
            "list_0",
            "sentence_3",
            "section_1",
            "sentence_4",
            "section_2",
            "list_1",
        ],
        "sentence_0": "The Aare is a [[Tributary|tributary]] of the [[High_Rhine|High Rhine]].",
        "sentence_1": "It rises in the [[Bernese_Alps]].",
        "table_0": {
            "type": "infobox",
            "caption": "Facts about the [[Aare_(river)|Aare]]",
            "table": [
                [
                    {"id": "header_cell_0_0_0", "value": "Country", "is_header": True},
                    {
                        "id": "cell_0_0_1",
                        "value": "[[Switzerland|Switzerland]]",
                        "is_header": False,
                    },
                ]
            ],
        },
        "section_0": {"value": "Course", "level": 2},
        "sentence_2": "The river passes [[Bern|Bern]].",
        "list_0": {
            "type": "unordered_list",
            "list": [
                {"id": "item_0_0", "value": "[[Lake_Thun|Lake Thun]]", "level": 0},
                {"id": "item_0_1", "value": "[[Lake_Biel#Geography|Lake Biel]]", "level": 0},
            ],
        },
        "sentence_3": "It then flows north.",
        "section_1": {"value": "Upper course", "level": 3},
        "sentence_4": "Glaciers feed the [[Grimselsee|Grimselsee]].",
        "section_2": {"value": "History", "level": 2},
        "list_1": {
            "type": "ordered_list",
            "list": [{"id": "item_1_2", "value": "Latinized as Arula.", "level": 0}],
        },
    }


def test_section_stack_levels_2_3_2_paths() -> None:
    parsed = parse_page(_page())
    assert parsed.sections == [
        Section(0, "", 1, ""),
        Section(1, "Course", 2, "Course"),
        Section(2, "Upper course", 3, "Course > Upper course"),
        Section(3, "History", 2, "History"),
    ]


def test_section_stack_skipped_level_and_return() -> None:
    page = {
        "title": "T",
        "order": ["section_0", "section_1", "section_2", "section_3"],
        "section_0": {"value": "A", "level": 2},
        "section_1": {"value": "C", "level": 4},
        "section_2": {"value": "B", "level": 3},
        "section_3": {"value": "D", "level": 4},
    }
    paths = [s.path for s in parse_page(page).sections]
    assert paths == ["", "A", "A > C", "A > B", "A > B > D"]


def test_section_path_helper_leaves_out_lead() -> None:
    assert section_path([(1, "")]) == ""
    assert section_path([(1, ""), (2, "A"), (3, "B")]) == "A > B"


def test_lead_section_is_ordinal_zero() -> None:
    parsed = parse_page(_page())
    lead = parsed.sections[0]
    assert (lead.ordinal, lead.heading, lead.level, lead.path) == (0, "", 1, "")
    assert [u.section_ordinal for u in parsed.units[:2]] == [0, 0]


def test_units_in_page_order_with_list_items_interleaved() -> None:
    parsed = parse_page(_page())
    assert [u.element_key for u in parsed.units] == [
        "sentence_0",
        "sentence_1",
        "sentence_2",
        "item_0_0",
        "item_0_1",
        "sentence_3",
        "sentence_4",
        "item_1_2",
    ]
    assert [u.ordinal for u in parsed.units] == list(range(8))
    assert [u.section_ordinal for u in parsed.units] == [0, 0, 1, 1, 1, 1, 2, 3]
    unit = parsed.units[3]
    assert unit == TextUnit("item_0_0", 3, 1, "Lake Thun", "[[Lake_Thun|Lake Thun]]")
    assert parsed.units[4].text == "Lake Biel"


def test_parse_page_returns_parsed_page_with_nfc_title() -> None:
    page = _page()
    page["title"] = "Amélie"  # decomposed
    parsed = parse_page(page)
    assert isinstance(parsed, ParsedPage)
    assert parsed.title == "Amélie"
    assert parsed.stats == page_stats(page)
    assert parsed.links == extract_links(page)


def test_clean_text_piped_and_bare_links() -> None:
    assert clean_text("see [[Aare_(given_name)|Aare (given name)]].") == "see Aare (given name)."
    assert clean_text("in the [[Bernese_Alps]].") == "in the Bernese Alps."
    assert clean_text("[[Backspace|]]") == "Backspace"
    assert clean_text("a [[|]] b") == "a b"


def test_clean_text_fragments_kept_in_text_only_as_shown() -> None:
    assert clean_text("[[Absolute_magnitude#Solar_System|absolute magnitude]] H") == (
        "absolute magnitude H"
    )
    assert clean_text("[[#Patent_dispute|patent disputes]]") == "patent disputes"
    assert clean_text("[[Q27585#P856]]") == "Q27585#P856"


def test_clean_text_entities_and_whitespace() -> None:
    assert clean_text("Tom &amp; Jerry &quot;x&quot; &lt;3 &#162") == 'Tom & Jerry "x" <3 ¢'
    assert clean_text("  a\n b c\t d  ") == "a b c d"


def test_clean_text_broken_and_nested_brackets() -> None:
    # a link cut open at the end of the unit keeps its shown text, the continuation drops "]]"
    assert clean_text("The [[United_States_Department_of_Labor|U.S.") == "The U.S."
    assert clean_text("Department of Labor]] reported [Angola's] efforts.") == (
        "Department of Labor reported [Angola's] efforts."
    )
    assert clean_text(")|Oak Hill Cemetery]] in Washington.") == "Oak Hill Cemetery in Washington."
    # a nested opening bracket: the inner link wins, the stray "[[" goes
    assert clean_text("on [[marine science in the [[Tropics|tropics]].") == (
        "on marine science in the tropics."
    )
    # single brackets inside links and IPA that lost its closing bracket
    assert clean_text("like [[Benzo(a)pyrene|benzo[a]pyrene]].") == "like benzo[a]pyrene."
    assert clean_text("(German: [[Standard_German|[ˈaːrə]]) or") == "(German: [ˈaːrə]) or"
    assert clean_text("[[X#Allophony_of_[v]_and_[w]|Hindustani]], [[Y|H]]") == "Hindustani, H"
    assert clean_text("x ]] y [[ z") == "x y z"


def test_clean_text_multi_pipe_ref_template_and_quotes() -> None:
    assert clean_text("the [[Equinox|equinox (North)|vernal equinox]] was") == (
        "the vernal equinox was"
    )
    assert clean_text('educator<ref">M. Campbell (2001). p. 12.</ref>') == "educator"
    assert clean_text("given in </ref> Besides") == "given in Besides"
    assert (
        clean_text("with {{{1}}} and [[2011_CQ1|2011 CQ1]] with {{{1}}}")
        == "with and 2011 CQ1 with"
    )
    assert clean_text("{{Infobox x | name = J | type = [[P|P]] }} rest") == "rest"
    assert clean_text("golfer }}") == "golfer"
    assert (
        clean_text("the sets {{4, 5}, {10}} and {a, {a, b}}.")
        == "the sets {{4, 5}, {10}} and {a, {a, b}}."
    )
    assert (
        clean_text("1950: ''[[Prelude_to_Fame|Prelude to Fame]]'' based")
        == "1950: Prelude to Fame based"
    )
    assert clean_text("the D-stem Pa''el is formed.") == "the D-stem Pa''el is formed."


def test_normalise_title_and_link_targets() -> None:
    assert normalise_title("Amélie_(film)") == "Amélie (film)"
    assert link_targets("[[A_b#frag|x]] [[C]] [[#self|s]] [[|]] [[D|d") == ["A b", "C", "D"]
    assert link_targets("[[Oak_Hill_Cemetery_(Washington,") == []
    keys = ("sentence_3", "item_0_1", "cell_0_1_1", "header_cell_0_0_0")
    assert [link_source(k) for k in keys] == ["sentence", "list", "table", "table"]


def test_extract_links_from_all_three_sources() -> None:
    assert extract_links(_page()) == [
        ("Tributary", "sentence_0"),
        ("High Rhine", "sentence_0"),
        ("Switzerland", "cell_0_0_1"),
        ("Bernese Alps", "sentence_1"),
        ("Bern", "sentence_2"),
        ("Lake Thun", "item_0_0"),
        ("Lake Biel", "item_0_1"),
        ("Grimselsee", "sentence_4"),
    ]


def test_extract_links_keeps_duplicates_and_normalises() -> None:
    page = {
        "title": "T",
        "order": ["sentence_0"],
        "sentence_0": "[[Amélie_(film)|A]] and [[Amélie_(film)#Plot|B]] and [[X|x]]",
    }
    assert extract_links(page) == [
        ("Amélie (film)", "sentence_0"),
        ("Amélie (film)", "sentence_0"),
        ("X", "sentence_0"),
    ]


def test_page_stats_counts() -> None:
    expected_texts = [
        "The Aare is a tributary of the High Rhine.",
        "It rises in the Bernese Alps.",
        "The river passes Bern.",
        "Lake Thun",
        "Lake Biel",
        "It then flows north.",
        "Glaciers feed the Grimselsee.",
        "Latinized as Arula.",
    ]
    assert [u.text for u in parse_page(_page()).units] == expected_texts
    assert page_stats(_page()) == {
        "n_sentences": 5,
        "n_items": 3,
        "n_words": 34,  # 9 + 6 + 4 + 2 + 2 + 4 + 4 + 3
        "n_chars": 179,  # 42 + 29 + 22 + 9 + 9 + 20 + 29 + 19
        "n_sections": 3,
        "n_tables": 1,
        "n_lists": 2,
        "n_hatnotes": 0,
    }


HATNOTES = [
    "Main article: [[History_of_the_Aare|History of the Aare]]",
    "Main articles: [[Aare_Gorge|Aare Gorge]] and [[Lake_Thun|Lake Thun]]",
    "See also: [[List_of_rivers_of_Switzerland|List of rivers of Switzerland]]",
    "Further information: [[Bernese_Oberland|Bernese Oberland]]",
    "For other uses, see [[Aare_(disambiguation)|Aare (disambiguation)]].",
    "For the given name, see [[Aare_(given_name)|Aare (given name)]].",
    "For a more detailed discussion of the course, see [[Aare_Gorge|Aare Gorge]].",
    "Not to be confused with [[Aar_(Hesse)|Aar]].",
    "This article is about the river. For the surname, see [[Aare_(surname)|Aare (surname)]].",
    "This page is about the river.",
    '"Aar" redirects here.',
]
NOT_HATNOTES = [
    "For example, see the table below.",
    "For instance, see the map.",
    "main article: lower case is prose",
    "Further information on the treaty is scarce.",
    "See also the section on hydrology.",
    "The main article: a summary follows.",
    "Not to be confused, the mayor withdrew.",
    "This article is a stub.",
    'He said "the river" redirects here and there.',
    "",
]


def test_is_hatnote_matches_the_measured_openers_case_sensitively() -> None:
    for raw in HATNOTES:
        assert is_hatnote(clean_text(raw)), raw
    for text in NOT_HATNOTES:
        assert not is_hatnote(text), text
    assert HATNOTE_RE.pattern.startswith("^(?:")
    assert HATNOTE_RE.flags & 2 == 0  # re.IGNORECASE is not set


def test_hatnote_units_are_flagged_counted_and_kept_in_order() -> None:
    page = {
        "title": "Aare",
        "order": ["sentence_0", "section_0", "sentence_1", "sentence_2", "list_0"],
        "sentence_0": "For other uses, see [[Aare_(disambiguation)|Aare (disambiguation)]].",
        "section_0": {"value": "Course", "level": 2},
        "sentence_1": "Main article: [[Course_of_the_Aare|Course of the Aare]]",
        "sentence_2": "The river passes [[Bern|Bern]].",
        "list_0": {
            "type": "unordered_list",
            "list": [{"id": "item_0_0", "value": "See also: [[Rhine|Rhine]]", "level": 0}],
        },
    }
    parsed = parse_page(page)
    assert [(u.element_key, u.ordinal, u.is_hatnote) for u in parsed.units] == [
        ("sentence_0", 0, True),
        ("sentence_1", 1, True),
        ("sentence_2", 2, False),
        ("item_0_0", 3, True),
    ]
    assert parsed.units[1].text == "Main article: Course of the Aare"
    assert [u.chunkable for u in parsed.units] == [False, False, True, False]
    assert TextUnit("sentence_9", 9, 0, "", "").chunkable is False
    assert TextUnit("sentence_9", 9, 0, "Text.", "Text.").chunkable is True
    stats = page_stats(page)
    assert stats["n_hatnotes"] == 3
    assert stats["n_sentences"] == 3 and stats["n_items"] == 1
    assert stats["n_words"] == sum(len(u.text.split()) for u in parsed.units)  # hatnotes count
    assert parsed.stats == stats
    # the hatnote's link is still a link of the page
    assert ("Course of the Aare", "sentence_1") in parsed.links


def test_empty_page_and_missing_order_key() -> None:
    parsed = parse_page({"title": "Empty", "order": []})
    assert parsed.sections == [Section(0, "", 1, "")]
    assert parsed.units == [] and parsed.links == []
    assert parsed.stats["n_words"] == 0
    parsed = parse_page({"title": "T", "order": ["sentence_9", "sentence_0"], "sentence_0": "Hi."})
    assert [u.element_key for u in parsed.units] == ["sentence_0"]
