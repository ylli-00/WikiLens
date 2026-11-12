"""The probe claims file: a few claims of the 75-claim corpus in the claims-file contract.

The probes need claims whose gold pages are in the ``wikilense`` database (the 100-page corpus),
or ``retrieved`` would only show unrelated text, so they come from ``data/corpus/claims.jsonl``,
not from random FEVEROUS claims. Per label, ``per_label`` claims with a text-only evidence set
are drawn with a fixed seed. The gold text is each unit's ``TextUnit.text`` from
``wikitext.parse_page`` over ``data/corpus/pages.jsonl`` (already NFC). ``findable`` is null: it
needs the chunks, which the probes do not compute.

Rules, as in chat B's pool script (``reviews/SCALING_PLAN_2026-11-11_claim_pool.py``):
- text-only set: every element a ``sentence`` or ``item`` (and here: every unit resolves to a
  unit with text on a corpus page);
- kind: ``single`` when every set cites one page and it is the same page, ``needs2`` when every
  set cites two or more pages, ``either`` otherwise (the 75 claims are all ``single``);
- ``gold_set``: the first text-only set whose units all resolve.

The output is interleaved by label (SUPPORTS, REFUTES, NOT ENOUGH INFO, SUPPORTS, ...), so the
first three claims are one per label.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from wikilense.corpus import nfc, parse_element_id
from wikilense.llm.claims import file_sha256
from wikilense.llm.prompts import LABELS
from wikilense.wikitext import parse_page

TEXT_TYPES = frozenset({"sentence", "item"})
DEFAULT_SEED = 20261112


def load_units(pages_path: Path) -> dict[str, dict[str, Any]]:
    """Return ``{NFC title: {element_key: TextUnit}}`` for every page of a pages file."""
    pages: dict[str, dict[str, Any]] = {}
    with pages_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                parsed = parse_page(json.loads(line))
                pages[parsed.title] = {unit.element_key: unit for unit in parsed.units}
    return pages


def describe_claim(record: dict[str, Any], units: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return one corpus claim in the claims-file contract."""
    evidence_sets: list[dict[str, Any]] = []
    parsed_sets: list[list[tuple[str, str, str]]] = []
    for evidence_set in record["evidence"]:
        parsed = [parse_element_id(el) for el in evidence_set["content"]]
        parsed_sets.append(parsed)
        evidence_sets.append({
            "elements": [f"{page}_{key}" for page, _, key in parsed],
            "text_only": bool(parsed) and all(t in TEXT_TYPES for _, t, _ in parsed),
            "findable": None,
        })
    nonempty = [s for s in parsed_sets if s]
    pages_per_set = [{page for page, _, _ in s} for s in nonempty]
    all_pages = set().union(*pages_per_set) if pages_per_set else set()
    if nonempty and all(len(p) >= 2 for p in pages_per_set):
        kind = "needs2"
    elif len(all_pages) == 1:
        kind = "single"
    else:
        kind = "either"
    gold_set: int | None = None
    gold_evidence: list[dict[str, Any]] = []
    for index, (meta, parsed) in enumerate(zip(evidence_sets, parsed_sets, strict=True)):
        if not meta["text_only"]:
            continue
        found = [units.get(page, {}).get(key) for page, _, key in parsed]
        if all(unit is not None and unit.text for unit in found):
            gold_set = index
            gold_evidence = [{"element": f"{page}_{key}", "page": page, "text": unit.text}
                             for (page, _, key), unit in zip(parsed, found, strict=True)]
            break
    return {
        "id": int(record["id"]),
        "split": record.get("split"),
        "label": record["label"],
        "claim": record["claim"],
        "kind": kind,
        "challenge": record.get("challenge"),
        "gold_pages": sorted(nfc(p) for p in all_pages),
        "evidence_sets": evidence_sets,
        "gold_set": gold_set,
        "gold_evidence": gold_evidence,
    }


def build_probe_claims(claims_path: Path, pages_path: Path, per_label: int = 3,
                       seed: int = DEFAULT_SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (the probe records, interleaved by label; a description with seed and counts)."""
    units = load_units(pages_path)
    with claims_path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    described = [describe_claim(r, units) for r in records]
    rng = random.Random(seed)
    picked: dict[str, list[dict[str, Any]]] = {}
    eligible_counts: dict[str, int] = {}
    for label in LABELS:
        eligible = sorted((c for c in described if c["label"] == label and c["gold_set"] is not None),
                          key=lambda c: c["id"])
        eligible_counts[label] = len(eligible)
        if len(eligible) < per_label:
            raise ValueError(f"only {len(eligible)} {label} claims with a text-only set")
        picked[label] = rng.sample(eligible, per_label)
    ordered = [picked[label][i] for i in range(per_label) for label in LABELS]
    info = {
        "seed": seed,
        "per_label": per_label,
        "source_claims": str(claims_path),
        "source_claims_sha256": file_sha256(claims_path),
        "source_pages": str(pages_path),
        "source_pages_sha256": file_sha256(pages_path),
        "eligible_per_label": eligible_counts,
        "ids": [c["id"] for c in ordered],
        "rule": "per label, random.Random(seed).sample(eligible sorted by id, per_label); "
                "eligible = a text-only set whose units all resolve on a corpus page",
    }
    return ordered, info


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSON lines (UTF-8, one object per line, newline-terminated)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:      # never overwrite: start a new file
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
