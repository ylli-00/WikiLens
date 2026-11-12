"""Helpers for the LLM runner's tests: a fake binary on disk, claims files, a fake retriever."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from wikilense.llm.prompts import LABELS, PROMPTS_DIR

FAKE = Path(__file__).resolve().parent / "fake_claude.py"


def make_fake_binary(folder: Path) -> Path:
    """Write an executable ``claude`` wrapper that runs the fake with this interpreter."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "claude"
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n', encoding="utf-8")
    path.chmod(0o755)
    return path


def claim_record(cid: int, text: str, label: str | None = None) -> dict[str, Any]:
    """Return one claim in the claims-file contract with a one-unit gold set."""
    return {
        "id": cid, "split": "train", "label": label or LABELS[cid % 3], "claim": text,
        "kind": "single", "challenge": "Other", "gold_pages": [f"Page {cid}"],
        "evidence_sets": [{"elements": [f"Page {cid}_sentence_0"], "text_only": True,
                           "findable": None}],
        "gold_set": 0,
        "gold_evidence": [{"element": f"Page {cid}_sentence_0", "page": f"Page {cid}",
                           "text": f"Gold sentence of claim {cid}."}],
    }


def write_claims(path: Path, texts: list[str]) -> Path:
    """Write a claims file with ids 1..n and the given claim texts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for n, text in enumerate(texts, 1):
            fh.write(json.dumps(claim_record(n, text)) + "\n")
    return path


def copy_prompt(prompts_dir: Path, new_id: str, source_id: str = "generic_probe_v0") -> Path:
    """Copy a prompt set under a new id (to test a prompt that is not a generic probe)."""
    target = prompts_dir / new_id
    shutil.copytree(PROMPTS_DIR / source_id, target)
    spec = json.loads((target / "prompt.json").read_text(encoding="utf-8"))
    spec["prompt_id"] = new_id
    (target / "prompt.json").write_text(json.dumps(spec), encoding="utf-8")
    return target


class FakeRetriever:
    """Stands in for :class:`wikilense.llm.retrieval.Retriever`: fixed hits, no database."""

    def __init__(self, database: str = "wikilense", k: int = 5) -> None:
        self.database, self.k = database, k
        self.calls: list[list[int]] = []

    def describe(self) -> dict[str, Any]:
        return {"database": self.database, "k": self.k, "strategy": "inline",
                "ef_search": 100, "embedding_model": "BAAI/bge-small-en-v1.5",
                "embedding_revision": "test", "device": "cpu", "cache_dir": "/nonexistent"}

    def count_chunks(self) -> int:
        return 42

    def hits_for(self, claim_id: int) -> list[dict[str, Any]]:
        return [{"rank": r, "chunk_id": claim_id * 100 + r, "page_id": claim_id,
                 "title": f"Page {claim_id}", "section_path": "History" if r % 2 else "",
                 "chunk_ordinal": r, "n_words": 10, "distance": 0.1 * r,
                 "text": f"Chunk {r} of claim {claim_id}."} for r in range(1, self.k + 1)]

    def retrieve_many(self, claims: list[tuple[int, str]]) -> dict[int, list[dict[str, Any]]]:
        self.calls.append([cid for cid, _ in claims])
        return {cid: self.hits_for(cid) for cid, _ in claims}

    def _read_cache(self, claim_id: int, claim_text: str) -> list[dict[str, Any]] | None:
        return self.hits_for(claim_id)
