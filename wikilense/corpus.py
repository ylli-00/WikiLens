"""Corpus selection and readers for ``data/corpus`` (docs/DESIGN.md, "Corpus selection").

The corpus is a subset of one or more FEVEROUS Wikipedia shards plus the FEVEROUS claims whose
gold evidence lies entirely inside those pages:

1. A claim is *in-shard* when every page cited in every ``content`` id of every evidence set is
   a shard title (NFC). Claims without any content id are not in-shard (nothing to retrieve).
2. Evidence pages are the union of those pages.
3. Filler pages are link targets of the evidence pages that are shard titles and not evidence
   pages, ranked by the number of distinct evidence pages linking to them (descending), then
   title (ascending); up to ``n_fill``. When fewer exist, the rest is taken in shard order from
   pages with at least ``min_sentences`` sentences.

Standard library only. ``scripts/build_corpus.py`` writes the files; ``load_corpus`` reads them.
"""

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"

# ``<page title>_<type>_<numbers>``, anchored at the end so that a title containing an underscore
# still parses. The page group is non-greedy: with a greedy group ``X_header_cell_0_1_1`` would
# backtrack to page ``X_header`` and type ``cell`` (the shortest suffix match wins), whereas the
# non-greedy group makes ``header_cell`` win over ``cell``; the two only differ on that case.
ELEMENT_ID_RE = re.compile(
    r"^(?P<page>.+?)_(?P<type>sentence|cell|header_cell|table_caption|item)_(?P<nums>\d+(?:_\d+)*)$"
)

# ``[[target|shown]]`` or ``[[target]]``; the group holds everything between the brackets.
_LINK_RE = re.compile(r"\[\[([^\[\]]*)\]\]")


def nfc(title: str) -> str:
    """Return ``title`` normalised to Unicode NFC (the form of the shard titles)."""
    return unicodedata.normalize("NFC", title)


def parse_element_id(el: str) -> tuple[str, str, str]:
    """Return ``(page_title_nfc, element_type, key)`` for a FEVEROUS element id.

    ``key`` is the page-object key of the element (``sentence_7``, ``item_0_1``,
    ``cell_0_1_1``, ``header_cell_0_1_1``, ``table_caption_0``). Raises ``ValueError`` when
    ``el`` is not an element id.
    """
    match = ELEMENT_ID_RE.match(el)
    if match is None:
        raise ValueError(f"not a FEVEROUS element id: {el!r}")
    page, element_type, nums = match.group("page", "type", "nums")
    return nfc(page), element_type, f"{element_type}_{nums}"


def iter_pages(path: str | Path) -> Iterator[dict]:
    """Yield the page dicts of a shard or of ``pages.jsonl``, one per line, in file order."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_claims(path: str | Path, split: str | None = None) -> Iterator[dict]:
    """Yield the claim records of a FEVEROUS claim file or of ``claims.jsonl``, in file order.

    The header line (``id == ""``) is skipped. With ``split`` given (``"train"`` or ``"dev"``)
    every record gets ``record["split"] = split``; with ``split=None`` the record must already
    carry a ``split`` key (``claims.jsonl``), else ``ValueError`` is raised.
    """
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") == "":
                continue
            if split is not None:
                record["split"] = split
            elif "split" not in record:
                raise ValueError(f"claim {record.get('id')!r} in {path} has no 'split' key")
            yield record


def claim_pages(record: dict) -> set[str]:
    """Return the NFC titles of every page cited by a ``content`` id of any evidence set."""
    return {
        parse_element_id(element_id)[0]
        for evidence_set in record["evidence"]
        for element_id in evidence_set["content"]
    }


def _link_targets(page: dict) -> set[str]:
    """Return the distinct NFC link targets in a page's sentences, list items and table cells.

    Target = the text before ``|``, without a ``#fragment``, ``_`` replaced by a space, NFC.
    Empty targets (``[[#fragment|shown]]``) are dropped.
    """
    texts: list[str] = []
    for key, value in page.items():
        if key.startswith("sentence_"):
            texts.append(value)
        elif key.startswith("list_"):
            texts.extend(item["value"] for item in value["list"])
        elif key.startswith("table_"):
            texts.extend(cell["value"] for row in value["table"] for cell in row)
    targets: set[str] = set()
    for text in texts:
        for match in _LINK_RE.finditer(text):
            target = match.group(1).split("|", 1)[0].split("#", 1)[0].replace("_", " ")
            if target:
                targets.add(nfc(target))
    return targets


def _n_sentences(page: dict) -> int:
    """Return the number of ``sentence_N`` entries in the page's ``order``."""
    return sum(1 for key in page["order"] if key.startswith("sentence_"))


@dataclass(frozen=True)
class _PageInfo:
    """What the first shard pass keeps per page."""

    title: str
    n_sentences: int
    links: frozenset[str]


@dataclass(frozen=True)
class CorpusSelection:
    """Result of ``select_corpus``.

    ``claims`` are the in-shard claim records (with ``split``) in file order, train then dev;
    ``evidence_titles`` are sorted; ``filler_titles`` are in rank order (link-ranked pages first,
    then shard-order pages); ``filler_link_counts`` maps each link-ranked filler to the number of
    distinct evidence pages linking to it.
    """

    claims: list[dict]
    evidence_titles: list[str]
    filler_titles: list[str]
    filler_from_links: int
    filler_from_order: int
    filler_link_counts: dict[str, int] = field(default_factory=dict)

    @property
    def titles(self) -> set[str]:
        """Return the set of all selected page titles (evidence and filler)."""
        return set(self.evidence_titles) | set(self.filler_titles)


def _scan_shards(shard_paths: Iterable[str | Path]) -> list[_PageInfo]:
    """Return title, sentence count and link targets of every shard page, in shard order."""
    infos: list[_PageInfo] = []
    seen: set[str] = set()
    for path in shard_paths:
        for page in iter_pages(path):
            title = nfc(page["title"])
            if title in seen:
                raise ValueError(f"duplicate page title {title!r} in {path}")
            seen.add(title)
            infos.append(_PageInfo(title, _n_sentences(page), frozenset(_link_targets(page))))
    return infos


def select_corpus(
    shard_paths: Iterable[str | Path],
    train_path: str | Path,
    dev_path: str | Path,
    n_fill: int = 46,
    min_sentences: int = 20,
) -> CorpusSelection:
    """Return the corpus selection for the given shards and claim files (rule in the module doc).

    The shards are read once (titles, sentence counts, link targets); the claim files once.
    Deterministic: ties in the filler ranking are broken by title, the fallback follows shard
    order.
    """
    infos = _scan_shards(shard_paths)
    shard_titles = {info.title for info in infos}

    claims: list[dict] = []
    for split, path in (("train", train_path), ("dev", dev_path)):
        for record in iter_claims(path, split):
            pages = claim_pages(record)
            if pages and pages <= shard_titles:
                claims.append(record)
    evidence: set[str] = set()
    for record in claims:
        evidence |= claim_pages(record)

    link_counts: dict[str, int] = {}
    for info in infos:
        if info.title in evidence:
            for target in info.links:
                if target in shard_titles and target not in evidence:
                    link_counts[target] = link_counts.get(target, 0) + 1
    ranked = sorted(link_counts, key=lambda title: (-link_counts[title], title))
    from_links = ranked[: max(n_fill, 0)]

    chosen = evidence | set(from_links)
    from_order: list[str] = []
    for info in infos:
        if len(from_links) + len(from_order) >= n_fill:
            break
        if info.title not in chosen and info.n_sentences >= min_sentences:
            from_order.append(info.title)
            chosen.add(info.title)

    return CorpusSelection(
        claims=claims,
        evidence_titles=sorted(evidence),
        filler_titles=from_links + from_order,
        filler_from_links=len(from_links),
        filler_from_order=len(from_order),
        filler_link_counts={title: link_counts[title] for title in from_links},
    )


def load_corpus(corpus_dir: str | Path = DEFAULT_CORPUS_DIR) -> tuple[list[dict], list[dict]]:
    """Return ``(pages, claims)`` read from ``pages.jsonl`` and ``claims.jsonl`` in ``corpus_dir``.

    Pages are the raw FEVEROUS page objects in shard order; claims are the FEVEROUS records with
    ``split``. Raises ``FileNotFoundError`` when a file is missing.
    """
    directory = Path(corpus_dir)
    pages_path = directory / "pages.jsonl"
    claims_path = directory / "claims.jsonl"
    for path in (pages_path, claims_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run scripts/build_corpus.py first")
    return list(iter_pages(pages_path)), list(iter_claims(claims_path))
