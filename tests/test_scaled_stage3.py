"""scripts/scaled_stage3.py without the network or a database: the resumable fetch, the rows a
load writes (the ingest's category numbering) and the database guard."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone

import pytest

from wikilense.config import REPO_ROOT
from wikilense.corpus import PageMeta

SCRIPT = REPO_ROOT / "scripts" / "scaled_stage3.py"
spec = importlib.util.spec_from_file_location("scaled_stage3", SCRIPT)
stage3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage3)


class FakeApi:
    requests = 0


def fake_fetch(api, page):
    if page["title"].startswith("BAD"):
        raise OSError("network down")
    api.requests += 2
    return {"title": page["title"], "wiki_title": page["title"], "method": "last_before_end",
            "revision_id": 7, "last_edited": "2020-11-15T08:34:15Z", "categories": ["C"]}


def test_fetch_appends_and_resumes(tmp_path):
    out = tmp_path / "meta.jsonl"
    pages = [{"title": t} for t in ("A", "B", "C")]
    assert stage3.run_fetch(pages, out, FakeApi(), fake_fetch, limit=2)["fetched"] == 2
    counts = stage3.run_fetch(pages, out, FakeApi(), fake_fetch)
    assert counts == {"skipped": 2, "fetched": 1, "errors": 0}
    assert [json.loads(line)["title"] for line in out.read_text().splitlines()] == ["A", "B", "C"]


def test_a_failed_page_is_logged_and_tried_again(tmp_path):
    out = tmp_path / "meta.jsonl"
    pages = [{"title": "A"}, {"title": "BAD 1"}, {"title": "B"}]
    assert stage3.run_fetch(pages, out, FakeApi(), fake_fetch)["errors"] == 1
    errors = (tmp_path / "meta.jsonl.errors.jsonl").read_text().splitlines()
    assert json.loads(errors[0])["title"] == "BAD 1"
    assert stage3.run_fetch(pages, out, FakeApi(), fake_fetch)["errors"] == 1   # asked again
    assert len(out.read_text().splitlines()) == 2


def test_many_failures_in_a_row_stop_the_fetch(tmp_path):
    pages = [{"title": f"BAD {n}"} for n in range(stage3.MAX_ERRORS_IN_A_ROW + 5)]
    with pytest.raises(stage3.Stage3Error):
        stage3.run_fetch(pages, tmp_path / "meta.jsonl", FakeApi(), fake_fetch)


def test_a_torn_last_line_is_repaired(tmp_path):
    out = tmp_path / "meta.jsonl"
    out.write_text('{"title": "A"}\n{"title": "B"')            # a crash mid-write
    assert stage3.done_titles(out) == {"A"}
    stage3.run_fetch([{"title": "B"}], out, FakeApi(), fake_fetch)
    assert [json.loads(line)["title"] for line in out.read_text().splitlines()[2:]] == ["B"]


def meta(cats, revision=5, edited=True):
    when = datetime(2020, 11, 15, 8, 34, 15, tzinfo=timezone.utc) if edited else None
    return PageMeta(title="t", revision_id=revision, last_edited=when, categories=tuple(cats))


def test_rows_number_categories_in_first_seen_order():
    pages = [(1, "P1"), (2, "P2"), (3, "P3")]
    records = {"P2": meta(["B", "A"]), "P1": meta(["A"]), "Other": meta(["Z"])}
    updates, categories, links = stage3.meta_rows(pages, records)
    assert categories == [(1, "A"), (2, "B")]
    assert links == [(1, 1), (2, 2), (2, 1)]
    naive_utc = datetime(2020, 11, 15, 8, 34, 15, tzinfo=timezone.utc).replace(tzinfo=None)
    assert updates[0] == (5, naive_utc, 1)                       # naive UTC, as the ingest binds
    want = stage3.expected_counts(pages, records)
    assert want == {"pages": 3, "pages_with_revision": 2, "pages_with_last_edited": 2,
                    "categories": 2, "page_category_rows": 3, "pages_with_a_category": 2}


@pytest.mark.parametrize("name", ["wikilense", "wikilense_test", "wikilense_s2", "other"])
def test_only_the_two_scaled_databases(name):
    with pytest.raises(stage3.Stage3Error):
        stage3.check_database(name)
    assert stage3.check_database("wikilense_s1") == "wikilense_s1"
