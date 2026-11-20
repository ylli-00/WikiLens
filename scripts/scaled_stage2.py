"""Stage 2 of the scaled experiment: load the corpus folders S0 and S1 into their own databases.

Subcommands, in order:

    create-dbs  CREATE DATABASE wikilense_s0 / wikilense_s1 and GRANT them to 'wikilense'@'%'
                (root, through ``docker exec``; the root password reaches the client only through
                the MYSQL_PWD environment variable, never a command line)
    embed       embed every distinct chunk text of a corpus folder (S1 holds S0's) with
                ``embedding.Embedder()`` (the default model at its pinned revision), in parts of
                PART_SIZE texts saved under --vectors-dir as they finish (a stop loses one part)
    ingest      ``ingest.run_ingest`` into WIKILENSE_DB_NAME with an embedder that returns the
                saved vectors by chunk text; then the graph size (root, information_schema)
    verify      the database's counts against the folder's corpora.json numbers

    WIKILENSE_DB_NAME=wikilense_s1 .venv/bin/python scripts/scaled_stage2.py ingest \
        --corpus-dir <scaled_data>/S1 --vectors-dir <scaled_data>/vectors --out <json>

The chunk texts are made exactly as the ingest makes them (``parse_page``, ``chunk_page`` with the
settings' chunk parameters, ``embedding_text`` with the title prefix); ``embed`` checks their
SHA-256 against the one ``scripts/scaled_build.py corpora`` recorded. ``ingest`` and ``verify``
stop unless WIKILENSE_DB_NAME is wikilense_s0 or wikilense_s1: never the 100-page database
``wikilense`` and never a ``*_test`` database (the ingest drops every table of its target).

``--allow-check-database`` (``create-dbs``, ``ingest``, ``verify``) also accepts
``wikilense_s1_check``, a scratch database for testing the S1 load from the release file
(``make load-s1 S1_CHECK=1``) without touching ``wikilense_s1``; ``create-dbs`` then creates only
that database. Without the option the name is refused like any other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from dotenv import dotenv_values

from wikilense.chunking import chunk_page, embedding_text
from wikilense.config import DEFAULT_ENV_FILE, REPO_ROOT, Settings, load_settings
from wikilense.corpus import iter_pages
from wikilense.wikitext import parse_page

ALLOWED_DATABASES = ("wikilense_s0", "wikilense_s1")
CONTAINER = "wikilense-mariadb"
ROOT_PASSWORD_VARIABLE = "WIKILENSE_DB_ROOT_PASSWORD"
CREATE_STATEMENTS = (
    "CREATE DATABASE wikilense_s0",
    "CREATE DATABASE wikilense_s1",
    "GRANT ALL PRIVILEGES ON wikilense_s0.* TO 'wikilense'@'%'",
    "GRANT ALL PRIVILEGES ON wikilense_s1.* TO 'wikilense'@'%'",
)
SHOW_DATABASES_SQL = "SHOW DATABASES LIKE 'wikilense_s_'"
#: The scratch database of a test load of S1, accepted only with --allow-check-database.
CHECK_DATABASE = "wikilense_s1_check"
CHECK_CREATE_STATEMENTS = (
    "CREATE DATABASE wikilense_s1_check",
    "GRANT ALL PRIVILEGES ON wikilense_s1_check.* TO 'wikilense'@'%'",
)
#: The hidden InnoDB tablespace(s) of the vector index, as scripts/run_experiments.py reads them.
HIDDEN_TABLESPACE_SQL = (
    "SELECT name, file_size FROM information_schema.innodb_sys_tablespaces "
    "WHERE name LIKE CONCAT(DATABASE(), '/chunk#i#%')"
)
COUNT_SQL = {
    "pages": "SELECT COUNT(*) FROM page",
    "chunks": "SELECT COUNT(*) FROM chunk",
    "chunk_words": "SELECT COALESCE(SUM(n_words), 0) FROM chunk",
    "claims": "SELECT COUNT(*) FROM claim",
    "evidence_ids": "SELECT COUNT(*) FROM claim_evidence",
    "evidence_with_page": "SELECT COUNT(*) FROM claim_evidence WHERE page_id IS NOT NULL",
    "evidence_with_unit": "SELECT COUNT(*) FROM claim_evidence WHERE sentence_id IS NOT NULL",
}
PART_SIZE = 10_000
BATCH_SIZE = 64


class Stage2Error(RuntimeError):
    """A refused database name, a missing vector, or a mismatch; the message says which."""


def log(message: str) -> None:
    """Print one progress line, flushed."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def allowed_databases(allow_check: bool = False) -> tuple[str, ...]:
    """Return ALLOWED_DATABASES, plus CHECK_DATABASE when ``allow_check``."""
    return ALLOWED_DATABASES + ((CHECK_DATABASE,) if allow_check else ())


def check_database(settings: Settings, allow_check: bool = False) -> str:
    """Return ``settings.db_name`` when it is one of ``allowed_databases(allow_check)``; else raise.

    The 100-page database ``wikilense`` and every ``*_test`` database are refused by name first.
    """
    name = settings.db_name
    allowed = allowed_databases(allow_check)
    if name == "wikilense" or name.endswith("_test") or name not in allowed:
        raise Stage2Error(
            f"refusing database {name!r}: set WIKILENSE_DB_NAME to one of {allowed}"
        )
    return name


def folder_suffix(name: str) -> str:
    """Return the corpus folder (lower case) that belongs in database ``name``: ``s1`` for
    ``wikilense_s1`` and for CHECK_DATABASE, ``s0`` for ``wikilense_s0``."""
    return "s1" if name == CHECK_DATABASE else name.rsplit("_", 1)[1]


def chunk_texts(pages_path: Path, settings: Settings) -> Iterator[str]:
    """Yield the embedding text of every chunk, in the order the ingest embeds them."""
    for page in iter_pages(pages_path):
        parsed = parse_page(page)
        for ch in chunk_page(parsed, max_words=settings.chunk_max_words,
                             overlap_units=settings.chunk_overlap_units):
            yield embedding_text(parsed.title, parsed.sections[ch.section_ordinal].path, ch.text)


def texts_digest(texts: Sequence[str]) -> str:
    """Return the SHA-256 over every text plus a newline (scaled_build's definition)."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update((text + "\n").encode("utf-8"))
    return digest.hexdigest()


def distinct(texts: Sequence[str]) -> list[str]:
    """Return the texts without repeats, first occurrence order."""
    return list(dict.fromkeys(texts))


def part_paths(vectors_dir: Path, index: int) -> tuple[Path, Path]:
    """Return (vectors .npy, texts .json) of part ``index``."""
    return vectors_dir / f"part_{index:03d}.npy", vectors_dir / f"part_{index:03d}.json"


def part_is_done(vectors_dir: Path, index: int, texts: Sequence[str], dim: int) -> bool:
    """Return True when part ``index`` exists and holds exactly ``texts`` with dim columns."""
    npy, meta = part_paths(vectors_dir, index)
    if not (npy.is_file() and meta.is_file()):
        return False
    info = json.loads(meta.read_text(encoding="utf-8"))
    if info.get("texts_sha256") != texts_digest(texts) or info.get("n") != len(texts):
        return False
    shape = np.load(npy, mmap_mode="r").shape
    return shape == (len(texts), dim)


def save_part(vectors_dir: Path, index: int, texts: Sequence[str], vectors: np.ndarray,
              model: dict[str, str]) -> None:
    """Save one part: the vectors, then its texts and metadata (the .json marks it complete)."""
    npy, meta = part_paths(vectors_dir, index)
    tmp = npy.with_name(npy.name + ".part")
    with open(tmp, "wb") as handle:
        np.save(handle, np.ascontiguousarray(vectors, dtype=np.float32))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, npy)
    tmp_meta = meta.with_name(meta.name + ".part")
    tmp_meta.write_text(json.dumps({"n": len(texts), "texts_sha256": texts_digest(texts),
                                    "texts": list(texts), **model}, ensure_ascii=False),
                        encoding="utf-8")
    os.replace(tmp_meta, meta)


def load_vectors(vectors_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return ``{text: vector}`` from every saved part and the model metadata they share."""
    cache: dict[str, np.ndarray] = {}
    model: dict[str, str] = {}
    for meta in sorted(vectors_dir.glob("part_*.json")):
        info = json.loads(meta.read_text(encoding="utf-8"))
        vectors = np.load(meta.with_suffix(".npy"))
        if len(info["texts"]) != len(vectors) or texts_digest(info["texts"]) != info["texts_sha256"]:
            raise Stage2Error(f"{meta.name} does not match its vectors")
        this = {"model_name": info["model_name"], "revision_label": info["revision_label"]}
        if model and this != model:
            raise Stage2Error(f"{meta.name} was made with {this}, the others with {model}")
        model = this
        for text, vector in zip(info["texts"], vectors, strict=True):
            cache[text] = vector
    return cache, model


class CachedEmbedder:
    """An ingest embedder that returns saved vectors by text, in the order asked.

    ``model_name`` and ``revision_label`` are the real embedder's, so ``ingest_meta`` records the
    model that made the vectors.
    """

    def __init__(self, cache: dict[str, np.ndarray], model_name: str, revision_label: str) -> None:
        if not cache:
            raise Stage2Error("no saved vectors")
        self._cache = cache
        self.model_name = model_name
        self.revision_label = revision_label
        self._dim = len(next(iter(cache.values())))

    @property
    def dim(self) -> int:
        """Return the vector dimension."""
        return self._dim

    def embed_passages(self, texts: Sequence[str], batch_size: int = 64,
                       show_progress: bool = False) -> np.ndarray:
        """Return the saved vectors of ``texts`` as one float32 array; raise for a missing text."""
        missing = [t for t in texts if t not in self._cache]
        if missing:
            raise Stage2Error(f"{len(missing)} chunk texts have no saved vector, e.g. {missing[0][:80]!r}")
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts]).astype(np.float32, copy=False)


def root_sql(sql: str, database: str | None = None,
             allow_check: bool = False) -> list[list[str]]:
    """Run one statement as root in the container; rows as lists of strings."""
    password = os.environ.get(ROOT_PASSWORD_VARIABLE) or dotenv_values(DEFAULT_ENV_FILE).get(
        ROOT_PASSWORD_VARIABLE)
    if not password:
        raise Stage2Error(f"{ROOT_PASSWORD_VARIABLE} is not set in .env or the environment")
    if database is not None and database not in allowed_databases(allow_check):
        raise Stage2Error(f"root_sql on {database!r} is not allowed")
    proc = subprocess.run(
        ["docker", "exec", "-e", "MYSQL_PWD", CONTAINER, "mariadb", "-uroot", "--batch",
         "--skip-column-names", *([database] if database else []), "-e", sql],
        env={**os.environ, "MYSQL_PWD": password}, cwd=REPO_ROOT,
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise Stage2Error(f"root statement failed ({proc.returncode}): {proc.stderr.strip()[:300]}")
    return [line.split("\t") for line in proc.stdout.splitlines() if line]


# ---- subcommands ------------------------------------------------------------------------------


def run_create(args: argparse.Namespace) -> int:
    """Create the two databases and grant them, unless they exist (with
    --allow-check-database: only CHECK_DATABASE)."""
    if args.allow_check_database:
        if CHECK_DATABASE in {row[0] for row in root_sql("SHOW DATABASES")}:
            log(f"already present: {CHECK_DATABASE}; nothing created")
            return 0
        for statement in CHECK_CREATE_STATEMENTS:
            root_sql(statement)
            log(statement)
        return 0
    existing = {row[0] for row in root_sql(SHOW_DATABASES_SQL)}
    if existing & set(ALLOWED_DATABASES):
        log(f"already present: {sorted(existing)}; nothing created")
        return 0 if set(ALLOWED_DATABASES) <= existing else 2
    for statement in CREATE_STATEMENTS:
        root_sql(statement)
        log(statement)
    log(f"now present: {sorted(row[0] for row in root_sql(SHOW_DATABASES_SQL))}")
    return 0


def run_embed(args: argparse.Namespace) -> int:
    """Embed the folder's distinct chunk texts in saved parts."""
    from wikilense.embedding import Embedder

    settings = load_settings()
    folder = Path(args.corpus_dir)
    texts = list(chunk_texts(folder / "pages.jsonl", settings))
    expected = json.loads(Path(args.corpora_json).read_text(encoding="utf-8"))[folder.name]
    if texts_digest(texts) != expected["embedding_text_sha256"] or len(texts) != expected["chunks"]:
        raise Stage2Error("the chunk texts differ from the ones scaled_build recorded")
    unique = distinct(texts)
    log(f"{folder.name}: {len(texts):,} chunks, {len(unique):,} distinct texts; digest matches")
    embedder = Embedder()
    model = {"model_name": embedder.model_name, "revision_label": embedder.revision_label}
    vectors_dir = Path(args.vectors_dir)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done_now = 0
    for index, offset in enumerate(range(0, len(unique), PART_SIZE)):
        part = unique[offset: offset + PART_SIZE]
        if part_is_done(vectors_dir, index, part, embedder.dim):
            log(f"part {index}: already saved ({len(part):,})")
            continue
        t0 = time.perf_counter()
        vectors = embedder.embed_passages(part, batch_size=BATCH_SIZE)
        save_part(vectors_dir, index, part, vectors, model)
        done_now += len(part)
        log(f"part {index}: {len(part):,} texts in {time.perf_counter() - t0:.1f} s "
            f"({len(part) / (time.perf_counter() - t0):.0f}/s), device {embedder.device}")
    summary = {"folder": folder.name, "chunks": len(texts), "distinct_texts": len(unique),
               "embedded_now": done_now, "seconds": round(time.perf_counter() - start, 1),
               "device": embedder.device, **model, "part_size": PART_SIZE, "batch_size": BATCH_SIZE}
    (vectors_dir / "embed_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"embed done: {summary}")
    return 0


def db_counts(settings: Settings) -> dict[str, int]:
    """Return the COUNT_SQL numbers of the settings' database."""
    from wikilense import db

    conn = db.connect(settings)
    try:
        with conn.cursor() as cur:
            result = {}
            for name, sql in COUNT_SQL.items():
                cur.execute(sql)
                result[name] = int(cur.fetchone()[0])
            return result
    finally:
        conn.close()


def run_ingest(args: argparse.Namespace) -> int:
    """Ingest the folder into WIKILENSE_DB_NAME with the saved vectors; record the report."""
    from wikilense.ingest import run_ingest as ingest

    settings = load_settings()
    name = check_database(settings, args.allow_check_database)
    folder = Path(args.corpus_dir)
    if folder.name.lower() != folder_suffix(name):
        raise Stage2Error(f"folder {folder.name} does not belong in database {name}")
    cache, model = load_vectors(Path(args.vectors_dir))
    log(f"{len(cache):,} saved vectors ({model}); ingest {folder} into {name}")
    embedder = CachedEmbedder(cache, model["model_name"], model["revision_label"])
    start = time.perf_counter()
    report = ingest(settings, corpus_dir=folder, reset=True, embedder=embedder,
                    batch_size=BATCH_SIZE, use_prefix=True, progress=False)
    wall = time.perf_counter() - start
    graph = root_sql(HIDDEN_TABLESPACE_SQL, database=name, allow_check=args.allow_check_database)
    result = {"database": name, "folder": folder.name, "wall_seconds": round(wall, 1),
              "report": report.counts(), "seconds": {k: round(v, 2) for k, v in report.seconds.items()},
              "graph_tablespaces": [{"name": r[0], "file_size": int(r[1])} for r in graph],
              "graph_bytes": sum(int(r[1]) for r in graph), "embedding": model}
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    log(f"ingest done: {result['report']['n_chunks']:,} chunks, graph {result['graph_bytes']:,} B, "
        f"{wall:.0f} s; seconds {result['seconds']}")
    return 0


def run_verify(args: argparse.Namespace) -> int:
    """Compare the database's counts with corpora.json; exit 1 on a difference."""
    settings = load_settings()
    name = check_database(settings, args.allow_check_database)
    folder = Path(args.corpus_dir)
    want = json.loads(Path(args.corpora_json).read_text(encoding="utf-8"))[folder.name]
    got = db_counts(settings)
    expected = {"pages": want["pages"], "chunks": want["chunks"], "chunk_words": want["chunk_words"],
                "claims": want["claims"], "evidence_ids": want["evidence"]["evidence_ids"],
                "evidence_with_page": want["evidence"]["with_a_page"],
                "evidence_with_unit": want["evidence"]["with_a_text_unit"]}
    diffs = {k: (got[k], v) for k, v in expected.items() if got[k] != v}
    log(f"{name}: {got}")
    if diffs:
        log(f"MISMATCH (database, expected): {diffs}")
        return 1
    log(f"{name}: every count matches corpora.json")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="step", required=True)
    check_help = f"also accept {CHECK_DATABASE}, the scratch database of a test load"
    sub.add_parser("create-dbs").add_argument(
        "--allow-check-database", action="store_true",
        help=f"create only {CHECK_DATABASE}, the scratch database of a test load")
    for step in ("embed", "ingest", "verify"):
        p = sub.add_parser(step)
        if step in ("ingest", "verify"):
            p.add_argument("--allow-check-database", action="store_true", help=check_help)
        p.add_argument("--corpus-dir", required=True, type=Path)
        if step in ("embed", "ingest"):
            p.add_argument("--vectors-dir", required=True, type=Path)
        if step in ("embed", "verify"):
            p.add_argument("--corpora-json", required=True, type=Path)
        if step == "ingest":
            p.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one step; return its exit status (Stage2Error -> 2)."""
    args = parse_args(argv)
    steps = {"create-dbs": run_create, "embed": run_embed, "ingest": run_ingest,
             "verify": run_verify}
    try:
        return steps[args.step](args)
    except Stage2Error as exc:
        log(f"STOP: {exc}")
        return 2




if __name__ == "__main__":
    sys.exit(main())
