"""Stage 3 of the scaled experiment: categories and edit dates for the pages of S0 and S1.

With them, all four metadata filters (category, length, edited since, linked from) work on the
scaled databases as on the 100-page one. Subcommands:

    fetch   for every page of a corpus folder's pages.jsonl, the December 2020 Wikipedia revision,
            its timestamp and its categories, with ``scripts/fetch_page_meta.py``'s own
            ``fetch_page`` and its polite API client (one request at a time, at most four per
            second, a 5 s pause after a slow request, maxlag, Retry-After). One record is
            appended and fsynced per page, so a stop loses at most one page: re-running the same
            command skips the titles already in the output. A page whose fetch fails is written
            to ``<out>.errors.jsonl`` (not to the output) and is tried again by the next run;
            ``MAX_ERRORS_IN_A_ROW`` failures in a row stop the run (the network is probably down).
    load    write a fetched file into a loaded database: ``page.revision_id``,
            ``page.last_edited``, ``category`` and ``page_category``, the same rows the ingest
            writes when the corpus folder holds ``page_meta.jsonl`` (category ids 1, 2, ... in
            the order the pages first name them, pages in ``page_id`` order). One transaction:
            the three tables are cleared and refilled, so a second load gives the same rows.
            Only ``wikilense_s0`` or ``wikilense_s1``; never the 100-page ``wikilense`` and never
            a ``*_test`` database.
    verify  the database's metadata counts against the file.

S1 holds every page of S0, so S1's pages are fetched once and the same file is loaded into both
databases (``load`` uses the records whose title is a page of that database).

Usage (from the repository root):
    .venv/bin/python scripts/scaled_stage3.py fetch --pages <scaled_data>/S1/pages.jsonl \
        --out <scaled_data>/stage3/page_meta_S1.jsonl
    .venv/bin/python scripts/scaled_stage3.py load --database wikilense_s1 \
        --meta <scaled_data>/stage3/page_meta_S1.jsonl
    .venv/bin/python scripts/scaled_stage3.py verify --database wikilense_s1 --meta <same file>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wikilense.config import REPO_ROOT, load_settings
from wikilense.corpus import load_page_meta, nfc

ALLOWED_DATABASES = ("wikilense_s0", "wikilense_s1")
MAX_ERRORS_IN_A_ROW = 20
PROGRESS_EVERY = 50
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_page_meta.py"


class Stage3Error(RuntimeError):
    """Stage 3 cannot run as asked."""


def log(message: str) -> None:
    """Print one time-stamped line to stderr."""
    print(f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S} {message}", file=sys.stderr, flush=True)


def fetch_module() -> Any:
    """Return ``scripts/fetch_page_meta.py`` as a module (its Api and fetch_page are reused)."""
    spec = importlib.util.spec_from_file_location("fetch_page_meta", FETCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check_database(name: str) -> str:
    """Return ``name`` when it is one of ALLOWED_DATABASES; else raise (the 100-page and test DBs first)."""
    if name == "wikilense" or name.endswith("_test") or name not in ALLOWED_DATABASES:
        raise Stage3Error(f"refusing database {name!r}: only {ALLOWED_DATABASES}")
    return name


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------
def done_titles(out: Path) -> set[str]:
    """Return the NFC titles already in the output file (a torn last line is ignored)."""
    titles: set[str] = set()
    if not out.exists():
        return titles
    with out.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                titles.add(nfc(json.loads(line)["title"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return titles


def append_line(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line, flushed and fsynced; a torn last line gets its newline first."""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            torn = fh.read(1) != b"\n"
        if torn:
            with path.open("ab") as fh:
                fh.write(b"\n")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_fetch(pages: Iterable[dict[str, Any]], out: Path, api: Any, fetch_page: Any,
              limit: int | None = None, clock: Any = time.monotonic) -> dict[str, int]:
    """Fetch every page whose title is not in ``out`` yet; return the counts of this run."""
    out.parent.mkdir(parents=True, exist_ok=True)
    errors_path = out.with_name(out.name + ".errors.jsonl")
    done = done_titles(out)
    counts = {"skipped": 0, "fetched": 0, "errors": 0}
    in_a_row = 0
    started = clock()
    for page in pages:
        title = nfc(page["title"])
        if title in done:
            counts["skipped"] += 1
            continue
        if limit is not None and counts["fetched"] + counts["errors"] >= limit:
            break
        try:
            record = fetch_page(api, page)
        except Exception as exc:
            counts["errors"] += 1
            in_a_row += 1
            append_line(errors_path, {"title": title, "error": f"{type(exc).__name__}: {exc}",
                                      "ts": datetime.now(timezone.utc).isoformat()})
            log(f"ERROR {title!r}: {type(exc).__name__}: {exc}")
            if in_a_row >= MAX_ERRORS_IN_A_ROW:
                raise Stage3Error(f"{in_a_row} pages in a row failed; stopping") from exc
            continue
        in_a_row = 0
        append_line(out, record)
        done.add(title)
        counts["fetched"] += 1
        if counts["fetched"] % PROGRESS_EVERY == 0:
            elapsed = clock() - started
            log(f"{len(done)} pages done ({counts['fetched']} this run, {counts['errors']} "
                f"errors), {api.requests} requests, {elapsed / 60:.1f} min, "
                f"{elapsed / counts['fetched']:.2f} s/page")
    return counts


# --------------------------------------------------------------------------------------
# load / verify
# --------------------------------------------------------------------------------------
def meta_rows(pages: list[tuple[int, str]], meta: dict[str, Any]) -> tuple[
        list[tuple[Any, Any, int]], list[tuple[int, str]], list[tuple[int, int]]]:
    """Return the rows to write: (revision_id, last_edited, page_id) updates, category rows and
    page_category rows. ``pages`` is (page_id, title) in page_id order; ``meta`` maps NFC title to
    ``corpus.PageMeta``. Category ids are 1, 2, ... in first-seen order, as the ingest assigns
    them; a page without a record keeps NULLs and no categories."""
    updates: list[tuple[Any, Any, int]] = []
    category_ids: dict[str, int] = {}
    links: list[tuple[int, int]] = []
    for page_id, title in pages:
        record = meta.get(nfc(title))
        if record is None:
            continue
        edited = record.last_edited.replace(tzinfo=None) if record.last_edited else None
        updates.append((record.revision_id, edited, page_id))
        for name in record.categories:
            if name not in category_ids:
                category_ids[name] = len(category_ids) + 1
            links.append((page_id, category_ids[name]))
    categories = [(cid, name) for name, cid in category_ids.items()]
    return updates, categories, links


def db_pages(conn: Any) -> list[tuple[int, str]]:
    """Return (page_id, title) of every page, in page_id order."""
    with conn.cursor() as cur:
        cur.execute("SELECT page_id, title FROM page ORDER BY page_id")
        return [(int(r[0]), str(r[1])) for r in cur.fetchall()]


def run_load(conn: Any, meta: dict[str, Any]) -> dict[str, int]:
    """Clear and refill the metadata of one database in one transaction; return the counts."""
    pages = db_pages(conn)
    titles = {nfc(t) for _, t in pages}
    updates, categories, links = meta_rows(pages, meta)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM page_category")
            cur.execute("DELETE FROM category")
            cur.execute("UPDATE page SET revision_id = NULL, last_edited = NULL")
            cur.executemany("UPDATE page SET revision_id = %s, last_edited = %s "
                            "WHERE page_id = %s", updates)
            cur.executemany("INSERT INTO category (category_id, name) VALUES (%s, %s)", categories)
            cur.executemany("INSERT INTO page_category (page_id, category_id) VALUES (%s, %s)",
                            links)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"pages_in_db": len(pages), "records_in_file": len(meta),
            "records_used": len(updates), "records_not_in_db": len(set(meta) - titles),
            "pages_without_record": len(pages) - len(updates),
            "pages_with_revision": sum(1 for u in updates if u[0] is not None),
            "categories": len(categories), "page_category_rows": len(links)}


def db_counts(conn: Any) -> dict[str, int]:
    """Return the metadata counts of one database."""
    sql = {
        "pages": "SELECT COUNT(*) FROM page",
        "pages_with_revision": "SELECT COUNT(*) FROM page WHERE revision_id IS NOT NULL",
        "pages_with_last_edited": "SELECT COUNT(*) FROM page WHERE last_edited IS NOT NULL",
        "categories": "SELECT COUNT(*) FROM category",
        "page_category_rows": "SELECT COUNT(*) FROM page_category",
        "pages_with_a_category": "SELECT COUNT(DISTINCT page_id) FROM page_category",
    }
    result = {}
    with conn.cursor() as cur:
        for name, statement in sql.items():
            cur.execute(statement)
            result[name] = int(cur.fetchone()[0])
    return result


def expected_counts(pages: list[tuple[int, str]], meta: dict[str, Any]) -> dict[str, int]:
    """Return what :func:`db_counts` must read after a load of ``meta``."""
    updates, categories, links = meta_rows(pages, meta)
    return {"pages": len(pages),
            "pages_with_revision": sum(1 for u in updates if u[0] is not None),
            "pages_with_last_edited": sum(1 for u in updates if u[1] is not None),
            "categories": len(categories), "page_category_rows": len(links),
            "pages_with_a_category": len({p for p, _ in links})}


# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="fetch the metadata of every page (resumable)")
    f.add_argument("--pages", required=True, type=Path, help="a corpus folder's pages.jsonl")
    f.add_argument("--out", required=True, type=Path, help="page_meta JSONL (appended)")
    f.add_argument("--limit", type=int, default=None, help="at most N new pages (a trial)")
    for name in ("load", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--database", required=True, choices=ALLOWED_DATABASES)
        s.add_argument("--meta", required=True, type=Path, help="the fetched page_meta JSONL")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "fetch":
        from wikilense.corpus import iter_pages

        module = fetch_module()
        api = module.Api()
        log(f"fetch {args.pages} -> {args.out}")
        try:
            counts = run_fetch(iter_pages(args.pages), args.out, api, module.fetch_page,
                               limit=args.limit)
        except Stage3Error as exc:
            log(f"STOPPED: {exc}")
            return 3
        log(f"done: {counts}, {api.requests} API requests this run, "
            f"{len(done_titles(args.out))} pages in {args.out}")
        return 0
    from wikilense import db

    database = check_database(args.database)
    meta = load_page_meta(args.meta)
    conn = db.connect(load_settings(), database=database)
    try:
        if args.command == "load":
            counts = run_load(conn, meta)
            log(f"load {database}: {counts}")
            print(json.dumps({"database": database, **counts}, indent=1))
            return 0
        got = db_counts(conn)
        want = expected_counts(db_pages(conn), meta)
        ok = got == want
        print(json.dumps({"database": database, "ok": ok, "db": got, "expected": want}, indent=1))
        return 0 if ok else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
