"""The claims-file contract shared with ``scripts/scaled_build.py``, file hashing, and evidence
rendering.

One JSON object per line (``claims_3000.jsonl`` from ``scripts/scaled_build.py``; the probe
file from :mod:`wikilense.llm.probe` uses the same format):

- ``id`` (int), ``split``, ``label``, ``claim``;
- ``kind`` (``"single"`` or ``"needs2"``) and ``challenge`` (FEVEROUS's challenge tag);
- ``gold_pages``: every page cited by any evidence set, table sets included, NFC titles;
- ``evidence_sets``: ``[{elements, text_only, findable}]``, full FEVEROUS element ids with an NFC
  page part, for example ``"Algebraic logic_sentence_0"``;
- ``gold_set``: the index of the set used for the gold condition (the first findable text-only
  set), or null;
- ``gold_evidence``: ``[{element, page, text}]`` in the order of that set's content; ``text`` is
  the unit's cleaned text (``TextUnit.text`` from ``wikitext.parse_page``).

The runner reads only ``id``, ``label``, ``claim``, ``kind``, ``gold_set`` and ``gold_evidence``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wikilense.llm.prompts import LABELS

REQUIRED_FIELDS = ("id", "split", "label", "claim", "kind", "challenge", "gold_pages",
                   "evidence_sets", "gold_set", "gold_evidence")


class ClaimsError(ValueError):
    """A claims file that does not follow the contract."""


def file_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 of a file's bytes, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_claims(path: str | Path) -> list[dict[str, Any]]:
    """Return the claims of a contract file in file order, after checking every record.

    Raises :class:`ClaimsError` for a line that is not JSON, a missing field, a non-integer or
    duplicate id, an unknown label, or a ``gold_set`` that does not index ``evidence_sets``. A
    file that does not end in a newline is refused: it may still be being written.
    """
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ClaimsError(f"{path} does not end in a newline (still being written?)")
    claims: list[dict[str, Any]] = []
    seen: set[int] = set()
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClaimsError(f"{path} line {n}: not JSON ({exc})") from exc
        missing = [f for f in REQUIRED_FIELDS if f not in record]
        if missing:
            raise ClaimsError(f"{path} line {n}: missing {missing}")
        cid = record["id"]
        if not isinstance(cid, int) or isinstance(cid, bool):
            raise ClaimsError(f"{path} line {n}: id {cid!r} is not an integer")
        if cid in seen:
            raise ClaimsError(f"{path} line {n}: duplicate id {cid}")
        seen.add(cid)
        if record["label"] not in LABELS:
            raise ClaimsError(f"{path} line {n}: unknown label {record['label']!r}")
        gold_set = record["gold_set"]
        if gold_set is not None and not (
            isinstance(gold_set, int) and 0 <= gold_set < len(record["evidence_sets"])
        ):
            raise ClaimsError(f"{path} line {n}: gold_set {gold_set!r} is not a set index")
        claims.append(record)
    if not claims:
        raise ClaimsError(f"{path} holds no claims")
    return claims


def chunk_header(title: str, section_path: str) -> str:
    """Return ``"Title > Section path"``, or the title alone for the lead section."""
    return f"{title} > {section_path}" if section_path else title


def render_retrieved(hits: list[dict[str, Any]]) -> str:
    """Return the evidence block of the ``retrieved`` condition.

    Each chunk in rank order, numbered from 1, with its page title and section path on the first
    line, then its text exactly as stored (``chunk.text``). Nothing is merged or dropped, so the
    LLM sees exactly the top k.
    """
    blocks = [
        f"[{rank}] {chunk_header(hit['title'], hit['section_path'])}\n{hit['text']}"
        for rank, hit in enumerate(hits, 1)
    ]
    return "\n\n".join(blocks)


def render_gold(gold_evidence: list[dict[str, Any]]) -> str:
    """Return the evidence block of the ``gold`` condition.

    The units in the order of the gold set's content, one per line. Consecutive units of one
    page form one numbered block headed by the page title (the contract carries no section
    path for gold units). Raises :class:`ClaimsError` for an empty list.
    """
    if not gold_evidence:
        raise ClaimsError("no gold evidence to render")
    blocks: list[tuple[str, list[str]]] = []
    for unit in gold_evidence:
        page, text = str(unit["page"]), str(unit["text"])
        if blocks and blocks[-1][0] == page:
            blocks[-1][1].append(text)
        else:
            blocks.append((page, [text]))
    return "\n\n".join(
        f"[{n}] {page}\n" + "\n".join(lines) for n, (page, lines) in enumerate(blocks, 1)
    )
