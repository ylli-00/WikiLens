"""The LLM runner's item guard: generic probe prompts never exceed 50 items per model, even
with --full, and any run above 50 items per model needs --full. No network, no database."""

from __future__ import annotations

import json

import pytest

from wikilense.llm.backend import CallResult
from wikilense.llm.guard import ITEM_LIMIT, GuardError, check_item_budget, items_per_model
from wikilense.llm.prompts import load_prompt_set
from wikilense.llm.runner import EXIT_CONFIG, RunConfig, Runner, item_key
from wikilense.llm.testing.fixtures import FakeRetriever, copy_prompt, write_claims


class NoCallBackend:
    """A backend that fails the test if a call is ever made."""

    name = "no_call"

    def describe(self):
        return {"version": "test"}

    def preflight(self):
        return {}

    def call(self, request, stop):  # pragma: no cover - must never run
        raise AssertionError("the guard must stop the run before any call")

    def kill_all(self):
        pass


def test_generic_probe_prompt_over_the_limit_is_refused_even_with_full():
    with pytest.raises(GuardError, match="even with --full"):
        check_item_budget("generic_probe_v0", {"claude-haiku-5-5": ITEM_LIMIT + 1}, full=True)
    with pytest.raises(GuardError):
        check_item_budget("generic_probe_v7", {"claude-haiku-5-5": ITEM_LIMIT + 1}, full=False)


def test_generic_probe_prompt_at_the_limit_passes():
    check_item_budget("generic_probe_v0", {"claude-haiku-5-5": ITEM_LIMIT}, full=False)


def test_any_prompt_over_the_limit_needs_full():
    with pytest.raises(GuardError, match="needs --full"):
        check_item_budget("real_prompt_v1", {"claude-sonnet-5-5": ITEM_LIMIT + 1}, full=False)
    check_item_budget("real_prompt_v1", {"claude-sonnet-5-5": 9000}, full=True)


def test_counts_are_per_model_and_union_planned_with_done():
    planned = [("m1", 1, "gold"), ("m1", 2, "gold"), ("m2", 1, "gold")]
    done = [("m1", 1, "gold"), ("m1", 3, "retrieved")]
    assert items_per_model(planned, done) == {"m1": 3, "m2": 1}


def _config(tmp_path, prompt, n_claims, limit=None, full=False):
    claims = write_claims(tmp_path / "claims.jsonl", [f"claim {i}" for i in range(n_claims)])
    return RunConfig(run_dir=tmp_path / "runs" / "r1", claims_path=claims, prompt=prompt,
                     models={"haiku": "claude-haiku-5-5"}, effort="low",
                     conditions=["no_retrieval", "retrieved", "gold"], limit_claims=limit,
                     workers=1, full=full, min_free_mb=0)


def test_runner_stops_a_generic_probe_run_over_50_items_before_any_call(tmp_path):
    prompt = load_prompt_set("generic_probe_v0")
    cfg = _config(tmp_path, prompt, n_claims=17, full=True)       # 17 x 3 = 51 > 50
    assert Runner(cfg, NoCallBackend(), FakeRetriever()).run() == EXIT_CONFIG
    assert not (cfg.run_dir / "results.jsonl").exists()


def test_runner_needs_full_for_a_real_prompt_over_50_items(tmp_path):
    copy_prompt(tmp_path / "prompts", "real_prompt_v1")
    prompt = load_prompt_set("real_prompt_v1", tmp_path / "prompts")
    cfg = _config(tmp_path, prompt, n_claims=17)
    assert Runner(cfg, NoCallBackend(), FakeRetriever()).run() == EXIT_CONFIG
    cfg.full = True
    cfg.dry_run = True
    assert Runner(cfg, NoCallBackend(), FakeRetriever()).run() == 0


def test_runner_counts_items_already_done_in_a_resumed_folder(tmp_path):
    """16 claims x 3 = 48 planned; 3 more already done in the folder make 51 > 50."""
    prompt = load_prompt_set("generic_probe_v0")
    cfg = _config(tmp_path, prompt, n_claims=17, limit=16)
    cfg.run_dir.mkdir(parents=True)
    res = CallResult("ok", label="SUPPORTS", parse_status="ok")
    with (cfg.run_dir / "results.jsonl").open("w") as fh:
        for condition in ("no_retrieval", "retrieved", "gold"):
            fh.write(json.dumps({
                "key": item_key(17, condition, "claude-haiku-5-5", "low", prompt.prompt_hash),
                "claim_id": 17, "condition": condition, "model_requested": "claude-haiku-5-5",
                "status": res.status}) + "\n")
    assert Runner(cfg, NoCallBackend(), FakeRetriever()).run() == EXIT_CONFIG
    cfg.limit_claims = 15                                          # 45 + 3 = 48: allowed
    cfg.dry_run = True
    assert Runner(cfg, NoCallBackend(), FakeRetriever()).run() == 0
