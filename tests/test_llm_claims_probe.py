"""The claims-file contract, evidence rendering, prompt sets and the probe file built from the
75-claim corpus (reads data/corpus only; no database)."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from wikilense.llm.claims import ClaimsError, load_claims, render_gold, render_retrieved
from wikilense.llm.probe import build_probe_claims, load_units, write_jsonl
from wikilense.llm.prompts import PromptError, load_prompt_set
from wikilense.llm.testing.fixtures import claim_record, copy_prompt

REPO = Path(__file__).resolve().parents[1]
CLAIMS = REPO / "data" / "corpus" / "claims.jsonl"
PAGES = REPO / "data" / "corpus" / "pages.jsonl"


def test_load_claims_refuses_a_half_written_file_and_bad_records(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(claim_record(1, "a")))                 # no final newline
    with pytest.raises(ClaimsError, match="newline"):
        load_claims(path)
    path.write_text(json.dumps(claim_record(1, "a")) + "\n" + json.dumps(claim_record(1, "b")) + "\n")
    with pytest.raises(ClaimsError, match="duplicate"):
        load_claims(path)
    bad = claim_record(2, "c") | {"label": "MAYBE"}
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ClaimsError, match="label"):
        load_claims(path)


def test_render_retrieved_and_gold():
    hits = [{"title": "Paris", "section_path": "History > Middle Ages", "text": "t1"},
            {"title": "Paris", "section_path": "", "text": "t2"}]
    assert render_retrieved(hits) == "[1] Paris > History > Middle Ages\nt1\n\n[2] Paris\nt2"
    gold = [{"page": "A", "text": "a1"}, {"page": "A", "text": "a2"}, {"page": "B", "text": "b1"}]
    assert render_gold(gold) == "[1] A\na1\na2\n\n[2] B\nb1"


def test_prompt_hash_changes_with_any_template_and_checks_the_schema(tmp_path):
    base = load_prompt_set("generic_probe_v0")
    folder = copy_prompt(tmp_path, "p_v1")
    first = load_prompt_set("p_v1", tmp_path)
    (folder / "user_gold.txt").write_text("CLAIM: {{claim}}\n")
    assert load_prompt_set("p_v1", tmp_path).prompt_hash != first.prompt_hash
    assert first.prompt_hash != base.prompt_hash          # the id is part of the hash
    (folder / "schema.json").write_text(json.dumps(
        {"type": "object", "properties": {"label": {"type": "string"},
                                          "why": {"type": "string"}}}))
    with pytest.raises(PromptError, match="one top-level property"):
        load_prompt_set("p_v1", tmp_path)


def test_render_user_refuses_a_missing_field():
    with pytest.raises(PromptError):
        load_prompt_set("generic_probe_v0").render_user("retrieved", {"claim": "x"})


def test_probe_file_has_three_claims_per_label_with_their_gold_text(tmp_path):
    records, info = build_probe_claims(CLAIMS, PAGES, per_label=3, seed=20261112)
    assert info["eligible_per_label"] == {"SUPPORTS": 4, "REFUTES": 53, "NOT ENOUGH INFO": 9}
    assert [r["label"] for r in records[:3]] == ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    assert len(records) == 9 and len({r["id"] for r in records}) == 9
    again, _ = build_probe_claims(CLAIMS, PAGES, per_label=3, seed=20261112)
    assert [r["id"] for r in again] == [r["id"] for r in records]   # the seed fixes the pick
    units = load_units(PAGES)
    for record in records:
        gold_set = record["evidence_sets"][record["gold_set"]]
        assert gold_set["text_only"] and gold_set["findable"] is None
        assert [u["element"] for u in record["gold_evidence"]] == gold_set["elements"]
        for unit in record["gold_evidence"]:
            key = unit["element"][len(unit["page"]) + 1:]
            assert unit["text"] == units[unit["page"]][key].text and unit["text"]
            assert unicodedata.is_normalized("NFC", unit["page"])
        assert set(record["gold_pages"]) >= {u["page"] for u in record["gold_evidence"]}
        assert record["kind"] in ("single", "needs2", "either")
    path = tmp_path / "probe.jsonl"
    write_jsonl(path, records)
    assert len(load_claims(path)) == 9
    with pytest.raises(FileExistsError):
        write_jsonl(path, records)                                  # never overwritten
