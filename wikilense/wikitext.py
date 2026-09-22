"""FEVEROUS page parsing: page order, section tree, text units, links and markup cleaning.

A page is the raw FEVEROUS page object (see docs/DESIGN.md, "Data contracts"). The parser walks
``order``, keeps a section stack keyed by heading level, and emits one ``TextUnit`` per sentence
and per list item. Tables and captions are counted in the page statistics only.

Hatnotes. Wikipedia's navigation lines ("Main article: X", "See also: Y", "For other uses, see
Z", ...) survive the FEVEROUS extraction as ordinary sentences, usually the first one of a
section. They carry no facts about the page, but a chunk that starts with one embeds much like
the page it points to, and with 8,868 chunks of the phase-1 corpus 788 (8.9%) started with
"Main article". ``HATNOTE_RE`` recognises them (``is_hatnote``); the parser marks the unit
(``TextUnit.is_hatnote``) and counts it in ``page_stats`` (``n_hatnotes``). A hatnote unit
stays a text unit, so it keeps its ordinal and becomes a ``sentence`` row at ingest (FEVEROUS
evidence ids still resolve), but ``chunking.chunk_units`` leaves it out of every chunk, exactly
like a whitespace-only unit. The openers were measured on ``wiki_000.jsonl`` (654,115 text
units on 9,996 pages): "Main article:" 4,545, "See also:" 1,636, "For other uses," 807,
"Further information:" 761, "Main articles:" 651, other "For ..., see ..." lines 1,462, "This
article is about" 530, '"X" redirects here' 484, "Not to be confused with" 261; together
11,137 units (1.70%), 9,030 of them the first unit of their section (the rest follow another
hatnote). "For example, ... see" is prose and is excluded (1 unit on the shard).
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

LEAD_LEVEL = 1
"""Heading level of the implicit lead section (ordinal 0, empty heading)."""

PATH_SEPARATOR = " > "
"""Joins the headings on the section stack into ``Section.path``."""

HATNOTE_RE = re.compile(
    r"^(?:"
    r"(?:Main articles?|See also|Further information|For other uses|Not to be confused with)[:,]"
    r"|For (?!example\b|instance\b)[^.]{0,80}?, see "
    r"|Not to be confused with "
    r"|This (?:article|page) is about "
    r'|"[^"]{1,120}" redirects here'
    r")"
)
"""Matches the cleaned text of a hatnote unit (case-sensitive; counts in the module docstring)."""


@dataclass(frozen=True)
class Section:
    """One section of a page; ordinal 0 is the lead with an empty heading and path."""

    ordinal: int
    heading: str
    level: int
    path: str


@dataclass(frozen=True)
class TextUnit:
    """One sentence or list item in page order, with its cleaned and raw text.

    ``is_hatnote`` is True when ``text`` matches ``HATNOTE_RE``. ``chunkable`` says whether the
    unit may be part of a chunk: it has text and is not a hatnote.
    """

    element_key: str
    ordinal: int
    section_ordinal: int
    text: str
    raw: str
    is_hatnote: bool = False

    @property
    def chunkable(self) -> bool:
        """Return True when the unit has text and is not a hatnote (it may be in a chunk)."""
        return bool(self.text) and not self.is_hatnote


@dataclass
class ParsedPage:
    """The parsed page: title (NFC), sections, text units, links and statistics."""

    title: str
    sections: list[Section]
    units: list[TextUnit]
    links: list[tuple[str, str]]
    stats: dict[str, int]


# A link target: no "|", no "[[" and no "]]" (single brackets do occur inside anchors).
_TARGET = r"(?:[^|\[\]]|\[(?!\[)|\](?!\]))*"
# Shown text: may contain "|" and single brackets, but no "[[" and no "]]".
_SHOWN = r"(?:[^\[\]]|\[(?!\[)|\](?!\]))*"
_LINK_RE = re.compile(r"\[\[(" + _TARGET + r")(?:\|(" + _SHOWN + r"))?\]\]")
# A link opened but never closed before the end of the unit (the sentence splitter cut it).
_OPEN_TAIL_RE = re.compile(r"\[\[(" + _TARGET + r")(?:\|(" + _SHOWN + r"))?$")
# The continuation of such a cut link at the start of the next unit: text up to the first "]]".
_CLOSE_HEAD_RE = re.compile(r"^(" + _SHOWN + r")\]\]")
_BRACKETS_RE = re.compile(r"\[\[|\]\]")
# <ref> elements that survived the FEVEROUS extraction, with or without their content.
_REF_RE = re.compile(r"<ref\b[^<>]*>.*?</ref>|<ref\b[^<>]*/>|</?ref\b[^<>]*>", re.DOTALL)
# Template remnants: unexpanded parameters ("{{{1}}}") and whole "{{Name | key = value}}" calls.
_TEMPLATE_PARAM_RE = re.compile(r"\{\{\{\d+\}\}\}")
_TEMPLATE_RE = re.compile(r"\{\{\s*[A-Za-z][^{}]*\|[^{}]*\}\}")
# Wikitext bold/italic quotes at a word edge; "Pa''el" style transliterations stay untouched.
_QUOTE_MARKUP_RE = re.compile(r"(?<!\w)'{2,}|'{2,}(?!\w)")


def _target_text(target: str) -> str:
    """Return the shown form of a bare link target: underscores become spaces."""
    return target.replace("_", " ")


def _shown_text(target: str, shown: str | None) -> str:
    """Return the text a link displays: the last non-empty "|" segment, else the target."""
    if shown is not None:
        for segment in reversed(shown.split("|")):
            segment = segment.strip()
            if segment:
                if segment.startswith("[") and "]" not in segment:
                    segment += "]"  # "[[X|[ipa]]]" lost its third bracket in the extraction
                return segment
    return _target_text(target)


def _replace_link(match: re.Match[str]) -> str:
    """Return the display text for one ``[[...]]`` match."""
    return _shown_text(match.group(1), match.group(2))


def _replace_close_head(match: re.Match[str]) -> str:
    """Return the text kept from a cut link's continuation ("shown]]" or "target|shown]]")."""
    return match.group(1).rsplit("|", 1)[-1]


def clean_text(raw: str) -> str:
    """Return the plain text of a FEVEROUS sentence, item or cell.

    Rules (docs/DESIGN.md plus the residual patterns measured on the shard): ``[[target|shown]]``
    becomes ``shown``, ``[[target]]`` becomes the target with underscores as spaces, links cut by
    the sentence splitter keep their shown text, remaining ``[[`` / ``]]`` are removed, ``<ref>``
    and template remnants and ``''`` quote markup are removed, HTML entities are decoded, and
    whitespace is collapsed and stripped.
    """
    text = _LINK_RE.sub(_replace_link, raw)
    text = _OPEN_TAIL_RE.sub(_replace_link, text)
    text = _CLOSE_HEAD_RE.sub(_replace_close_head, text)
    text = _BRACKETS_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _TEMPLATE_PARAM_RE.sub("", text)
    text = _TEMPLATE_RE.sub("", text)
    if "}}" in text and "{" not in text:
        text = text.replace("}}", "")
    text = _QUOTE_MARKUP_RE.sub("", text)
    text = html.unescape(text)
    return " ".join(text.split())


def is_hatnote(text: str) -> bool:
    """Return True when the cleaned text of a unit is a hatnote (``HATNOTE_RE`` matches)."""
    return HATNOTE_RE.match(text) is not None


def normalise_title(title: str) -> str:
    """Return a page title in Unicode NFC with underscores as spaces and whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFC", title.replace("_", " ")).split())


def link_targets(raw: str) -> list[str]:
    """Return the link targets in one raw text, in order, as titles (fragment dropped, NFC).

    Closed links and links cut open at the end of the unit (``[[target|shown`` with no ``]]``)
    count; a cut bare link has a truncated target and is skipped, as is a link whose target is
    empty after dropping the fragment (a self-link such as ``[[#Section|text]]``).
    """
    targets: list[str] = []
    for match in _LINK_RE.finditer(raw):
        targets.append(match.group(1))
    tail = _OPEN_TAIL_RE.search(_LINK_RE.sub("", raw))
    if tail is not None and tail.group(2) is not None:
        targets.append(tail.group(1))
    titles: list[str] = []
    for target in targets:
        title = normalise_title(target.split("#", 1)[0])
        if title:
            titles.append(title)
    return titles


def section_path(stack: list[tuple[int, str]]) -> str:
    """Return the path of the section on top of a stack of ``(level, heading)`` pairs.

    Empty headings (the lead) are left out, so the lead's path is ``""``.
    """
    return PATH_SEPARATOR.join(heading for _, heading in stack if heading)


def _element_texts(key: str, element: object) -> list[tuple[str, str]]:
    """Return the ``(element_key, raw_text)`` pairs that carry links in one page element."""
    if key.startswith("sentence_") and isinstance(element, str):
        return [(key, element)]
    if key.startswith("list_") and isinstance(element, dict):
        return [(item["id"], item["value"]) for item in element.get("list", [])]
    if key.startswith("table_") and isinstance(element, dict):
        return [(cell["id"], cell["value"]) for row in element.get("table", []) for cell in row]
    return []


def link_source(element_key: str) -> str:
    """Return ``"sentence"``, ``"list"`` or ``"table"`` for a link's source element key."""
    if element_key.startswith("sentence_"):
        return "sentence"
    if element_key.startswith("item_"):
        return "list"
    return "table"


def extract_links(page: dict) -> list[tuple[str, str]]:
    """Return ``(to_title, source_element_key)`` pairs from sentences, list items and table cells.

    Pairs are in page order and duplicates are kept. ``to_title`` is the target before ``|``
    with the ``#fragment`` dropped, underscores as spaces and NFC normalisation.
    """
    links: list[tuple[str, str]] = []
    for key in page.get("order", []):
        if key not in page:
            continue
        for element_key, raw in _element_texts(key, page[key]):
            for title in link_targets(raw):
                links.append((title, element_key))
    return links


def _unit(element_key: str, ordinal: int, section_ordinal: int, raw: str) -> TextUnit:
    """Return the ``TextUnit`` of one raw sentence or item: cleaned text and the hatnote flag."""
    text = clean_text(raw)
    return TextUnit(element_key, ordinal, section_ordinal, text, raw, is_hatnote(text))


def _walk(page: dict) -> tuple[list[Section], list[TextUnit], dict[str, int]]:
    """Return sections, text units and element counts from one walk over ``order``."""
    sections = [Section(0, "", LEAD_LEVEL, "")]
    stack: list[tuple[int, str]] = [(LEAD_LEVEL, "")]
    units: list[TextUnit] = []
    counts = {"n_sections": 0, "n_tables": 0, "n_lists": 0}
    for key in page.get("order", []):
        if key not in page:
            continue
        element = page[key]
        if key.startswith("section_"):
            level = int(element["level"])
            heading = clean_text(str(element["value"]))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            sections.append(Section(len(sections), heading, level, section_path(stack)))
            counts["n_sections"] += 1
        elif key.startswith("sentence_"):
            units.append(_unit(key, len(units), sections[-1].ordinal, str(element)))
        elif key.startswith("list_"):
            counts["n_lists"] += 1
            for item in element.get("list", []):
                units.append(
                    _unit(item["id"], len(units), sections[-1].ordinal, str(item["value"]))
                )
        elif key.startswith("table_"):
            counts["n_tables"] += 1
    return sections, units, counts


def _stats(units: list[TextUnit], counts: dict[str, int]) -> dict[str, int]:
    """Return the page statistics dict from the text units and element counts."""
    return {
        "n_sentences": sum(1 for u in units if u.element_key.startswith("sentence_")),
        "n_items": sum(1 for u in units if u.element_key.startswith("item_")),
        "n_words": sum(len(u.text.split()) for u in units),
        "n_chars": sum(len(u.text) for u in units),
        "n_sections": counts["n_sections"],
        "n_tables": counts["n_tables"],
        "n_lists": counts["n_lists"],
        "n_hatnotes": sum(1 for u in units if u.is_hatnote),
    }


def page_stats(page: dict) -> dict[str, int]:
    """Return ``n_sentences``, ``n_items``, ``n_words``, ``n_chars`` (over all cleaned text
    units, hatnotes included), ``n_sections`` (headings, the lead not counted), ``n_tables``,
    ``n_lists`` and ``n_hatnotes`` (text units that ``is_hatnote`` recognises)."""
    _, units, counts = _walk(page)
    return _stats(units, counts)


def parse_page(page: dict) -> ParsedPage:
    """Return the ``ParsedPage`` for one raw FEVEROUS page object."""
    sections, units, counts = _walk(page)
    return ParsedPage(
        title=unicodedata.normalize("NFC", str(page.get("title", ""))),
        sections=sections,
        units=units,
        links=extract_links(page),
        stats=_stats(units, counts),
    )
