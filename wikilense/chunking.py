"""Sentence-window chunker: consecutive text units of one section become one ``Chunk``.

``chunk_units`` is pure and deterministic. Word count is ``len(text.split())``. The rules are in
docs/DESIGN.md, "Chunking rules"; ``max_words`` and ``overlap_units`` were chosen with results/SUMMARY.md.

Only *chunkable* units take part (``TextUnit.chunkable``): a unit with no text after cleaning,
or a hatnote ("Main article: X", "See also: Y", ...; ``wikitext.HATNOTE_RE``), is in no chunk:
not in ``Chunk.element_keys``, not in ``Chunk.text`` and therefore in no ``chunk_sentence`` row.
Both kinds stay ``sentence`` rows at ingest, so evidence ids resolve. On the phase-1 corpus
(100 pages, 33,637 units, 372 empty and 1,316 hatnotes) the hatnote rule takes the chunk count
from 8,868 to 8,658 at 120 words and from 4,751 to 4,598 at 240 words; skipping the empty
units on their own changes no chunk count, they only ever added zero words.
"""

from __future__ import annotations

from dataclasses import dataclass

from wikilense.wikitext import ParsedPage, TextUnit

DEFAULT_MAX_WORDS = 240  # same value as config.DEFAULT_CHUNK_MAX_WORDS; see results/SUMMARY.md
"""The word budget of one chunk, chosen with results/SUMMARY.md (equal recall at equal retrieved
text as 60 or 120 words, with half the vectors of 120)."""

DEFAULT_OVERLAP_UNITS = 1
"""How many units the next chunk repeats from the end of the previous one (one sentence of overlap)."""


@dataclass
class Chunk:
    """One chunk of a page: the cleaned unit texts joined by single spaces."""

    ordinal: int
    section_ordinal: int
    element_keys: list[str]
    text: str
    n_words: int


def _section_runs(units: list[TextUnit]) -> list[list[TextUnit]]:
    """Return the chunkable units split into runs of consecutive units of one section.

    Units that are not chunkable (empty text or a hatnote) are left out before the split, so
    they never occupy a window slot or an overlap slot.
    """
    runs: list[list[TextUnit]] = []
    for unit in units:
        if not unit.chunkable:
            continue
        if runs and runs[-1][-1].section_ordinal == unit.section_ordinal:
            runs[-1].append(unit)
        else:
            runs.append([unit])
    return runs


def chunk_units(
    units: list[TextUnit],
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_units: int = DEFAULT_OVERLAP_UNITS,
) -> list[Chunk]:
    """Return the chunks of one page's text units, ordinals counting from 0.

    A chunk never crosses a section boundary. It is filled with consecutive chunkable units
    (``TextUnit.chunkable``: text present, not a hatnote; the others are in no chunk) until
    adding the next one would exceed ``max_words``; a single unit longer than ``max_words`` is a
    chunk on its own. The next chunk starts ``overlap_units`` units before the end of the
    previous one when that chunk has more than ``overlap_units`` units, else right after it. A
    chunk that would only repeat units of the previous chunk (nothing new to cover) is not
    emitted, and neither is a chunk without any words.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if overlap_units < 0:
        raise ValueError("overlap_units must not be negative")
    chunks: list[Chunk] = []
    for run in _section_runs(units):
        n = len(run)
        words = [len(unit.text.split()) for unit in run]
        start = 0
        covered = 0  # index after the last unit any emitted chunk of this run contains
        while start < n:
            end = start + 1
            total = words[start]
            while end < n and total + words[end] <= max_words:
                total += words[end]
                end += 1
            if end > covered and total > 0:
                window = run[start:end]
                chunks.append(
                    Chunk(
                        ordinal=len(chunks),
                        section_ordinal=window[0].section_ordinal,
                        element_keys=[unit.element_key for unit in window],
                        text=" ".join(unit.text for unit in window),
                        n_words=total,
                    )
                )
                covered = end
            if end >= n:
                break
            start = end - overlap_units if end - start > overlap_units else end
    return chunks


def embedding_text(title: str, section_path: str, text: str) -> str:
    """Return the text that is embedded: title and section path as a prefix (kept after measuring)."""
    if section_path:
        return f"{title} > {section_path}: {text}"
    return f"{title}: {text}"


def chunk_page(
    parsed: ParsedPage,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_units: int = DEFAULT_OVERLAP_UNITS,
) -> list[Chunk]:
    """Return ``chunk_units`` over the parsed page's text units."""
    return chunk_units(parsed.units, max_words=max_words, overlap_units=overlap_units)
