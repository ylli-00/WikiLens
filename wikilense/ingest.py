"""Ingest pipeline: corpus files -> page, section, sentence, link, claim tables -> chunks -> embeddings.

``run_ingest`` is the single entry point (docs/DESIGN.md, "Ingest (ingest.py), contract for
phase 2"). It runs six steps against ``settings.db_name`` and commits each one when it completes:

1. ``db.apply_schema`` (``reset=True`` drops the tables first). The test database
   (``settings.test_db_name``) is refused.
2. Every page is parsed (``wikitext.parse_page``, ``chunking.chunk_page``) and its ``page``,
   ``section`` (lead included), ``sentence`` (every text unit, whitespace-only ones too, so that
   evidence ids resolve) and ``link`` rows are written; a link target longer than
   ``MAX_TITLE_LENGTH`` characters is skipped and counted. Chunks stay in memory with the text
   that will be embedded (``chunking.embedding_text`` when ``use_prefix``).
3. The embedder's dimension is checked against ``settings.vector_dim`` (which must equal the
   literal in ``sql/schema.sql``), the chunk texts are embedded with ``embed_passages`` and the
   ``chunk`` rows (``db.vec_param`` bytes) and the ``chunk_sentence`` map are written. A chunk is
   mapped to the units whose text it carries: whitespace-only units belong to no chunk.
4. ``link.to_page_id`` is resolved with one ``UPDATE link JOIN page``.
5. ``claim`` and ``claim_evidence`` are written; ``page_id`` is resolved by title and
   ``sentence_id`` by ``(page_id, element_key)`` for sentences and list items (cells and captions
   keep ``sentence_id`` NULL).
6. ``ingest_meta`` records the model, chunk and index parameters, the SHA-256 of both corpus
   files, the server version and the time.

Every INSERT goes through ``db.insert_rows`` in batches of at most ``INSERT_BATCH_ROWS`` rows.
Progress bars (tqdm) are shown only with ``progress=True``; each stage logs one summary line
through the ``logging`` module.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pymysql
from tqdm import tqdm

from wikilense import __version__, db
from wikilense.chunking import chunk_page, embedding_text
from wikilense.config import Settings
from wikilense.corpus import DEFAULT_CORPUS_DIR, iter_claims, iter_pages, parse_element_id
from wikilense.wikitext import parse_page

logger = logging.getLogger(__name__)

INSERT_BATCH_ROWS = 1000
"""Largest number of rows handed to one ``db.insert_rows`` call."""

MAX_TITLE_LENGTH = 255
"""Length of ``link.to_title`` (VARCHAR(255)); longer targets are skipped and counted."""

INDEX_DISTANCE = "cosine"
"""The DISTANCE of the VECTOR INDEX in sql/schema.sql, recorded in ``ingest_meta``."""

STAGES: tuple[str, ...] = ("schema", "parse", "embed", "load", "resolve", "total")
"""Keys of ``IngestReport.seconds``: parse = parse_page + chunk_page; embed = model dimension
check (loads the model) + embed_passages; load = every INSERT (plus file hashing);
resolve = the link UPDATE and the evidence id lookups; total = the whole run."""

META_KEYS: tuple[str, ...] = (
    "embedding_model",
    "embedding_dim",
    "embedding_prefix",
    "chunk_max_words",
    "chunk_overlap_units",
    "index_m",
    "index_distance",
    "corpus_dir",
    "corpus_pages_sha256",
    "corpus_claims_sha256",
    "mariadb_version",
    "wikilense_version",
    "ingested_at",
)
"""The keys written to ``ingest_meta``."""

#: SQL statements of the pipeline besides the INSERTs that ``db.insert_rows`` builds.
RESOLVE_LINKS_SQL = (
    "UPDATE link JOIN page ON page.title = link.to_title SET link.to_page_id = page.page_id"
)
VERSION_SQL = "SELECT VERSION()"
CLEAR_META_SQL = "DELETE FROM ingest_meta"

# Explicit primary keys are assigned in Python (parents before children, ids consecutive per
# table) so that foreign keys are known without reading AUTO_INCREMENT values back. Each run
# starts after the largest existing id; with reset=True every table is empty and ids start at 1.
_ID_COLUMNS: dict[str, str] = {
    "page": "page_id",
    "section": "section_id",
    "sentence": "sentence_id",
    "chunk": "chunk_id",
}
_NEXT_ID_SQL: dict[str, str] = {
    table: f"SELECT COALESCE(MAX(`{column}`), 0) + 1 FROM `{table}`"
    for table, column in _ID_COLUMNS.items()
}

#: Column lists of the INSERTs, in the order the row tuples are built (subsets of
#: db.SCHEMA_COLUMNS; link_id is left to AUTO_INCREMENT because nothing references it).
_INSERT_COLUMNS: dict[str, tuple[str, ...]] = {
    "page": db.SCHEMA_COLUMNS["page"],
    "section": db.SCHEMA_COLUMNS["section"],
    "sentence": db.SCHEMA_COLUMNS["sentence"],
    "chunk": db.SCHEMA_COLUMNS["chunk"],
    "chunk_sentence": db.SCHEMA_COLUMNS["chunk_sentence"],
    "link": ("from_page_id", "to_title", "to_page_id", "source_element"),
    "claim": db.SCHEMA_COLUMNS["claim"],
    "claim_evidence": db.SCHEMA_COLUMNS["claim_evidence"],
    "ingest_meta": db.SCHEMA_COLUMNS["ingest_meta"],
}

_TEXT_UNIT_TYPES = frozenset({"sentence", "item"})


class IngestError(RuntimeError):
    """The ingest refused to run or found inconsistent inputs; the message says why."""


class EmbedderLike(Protocol):
    """What ``run_ingest`` needs from an embedder (``embedding.Embedder`` or a test double)."""

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""

    def embed_passages(
        self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dim)`` with unit rows."""


@dataclass
class IngestReport:
    """Counts and per-stage seconds of one ``run_ingest`` call.

    ``n_sentences`` counts every text unit written to ``sentence`` (whitespace-only units
    included; those are ``n_units_empty``). ``n_links`` counts the link rows written;
    ``n_links_skipped`` the targets over ``MAX_TITLE_LENGTH`` characters that were not written;
    ``n_links_resolved`` the rows whose target is a corpus page. ``n_evidence`` counts every
    element id of every evidence set; the two ``*_resolved`` counts the rows with a ``page_id``
    and with a ``sentence_id``. ``seconds`` has the keys of ``STAGES``.
    """

    n_pages: int = 0
    n_sections: int = 0
    n_sentences: int = 0
    n_units_empty: int = 0
    n_chunks: int = 0
    n_links: int = 0
    n_links_resolved: int = 0
    n_links_skipped: int = 0
    n_claims: int = 0
    n_evidence: int = 0
    n_evidence_page_resolved: int = 0
    n_evidence_sentence_resolved: int = 0
    seconds: dict[str, float] = field(default_factory=lambda: dict.fromkeys(STAGES, 0.0))

    def counts(self) -> dict[str, int]:
        """Return the twelve counts as a dict (everything except ``seconds``)."""
        return {
            name: value
            for name, value in self.__dict__.items()
            if name != "seconds" and isinstance(value, int)
        }


@dataclass
class _PendingChunk:
    """A chunk parsed in step 2 and waiting for its embedding in step 3."""

    chunk_id: int
    page_id: int
    section_id: int
    ordinal: int
    text: str
    n_words: int
    sentence_ids: list[int]
    embedding_text: str


class _Timer:
    """Accumulates wall-clock seconds per stage name."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = dict.fromkeys(STAGES, 0.0)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Add the time spent inside the block to ``seconds[name]``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start


class _Loader:
    """Buffers rows per table and writes them with ``db.insert_rows`` in capped batches.

    ``flush`` writes the buffers in schema order (parents first), so a row is always inserted
    after the rows it references; ``add`` flushes everything as soon as one buffer is full.
    Nothing is committed here.
    """

    def __init__(self, conn: pymysql.Connection, batch_rows: int = INSERT_BATCH_ROWS) -> None:
        if batch_rows < 1:
            raise ValueError(f"batch_rows must be at least 1, got {batch_rows}")
        self._conn = conn
        self._batch_rows = batch_rows
        self._buffers: dict[str, list[tuple[Any, ...]]] = {table: [] for table in _INSERT_COLUMNS}
        self.inserted: dict[str, int] = dict.fromkeys(_INSERT_COLUMNS, 0)

    def add(self, table: str, row: Sequence[Any]) -> None:
        """Queue one row (values in ``_INSERT_COLUMNS[table]`` order); flush when a buffer is full."""
        buffer = self._buffers[table]
        buffer.append(tuple(row))
        if len(buffer) >= self._batch_rows:
            self.flush()

    def flush(self) -> None:
        """Write every buffered row, parents first, in batches of at most ``batch_rows``."""
        for table in db.table_names():
            rows = self._buffers.get(table)
            if not rows:
                continue
            columns = _INSERT_COLUMNS[table]
            for start in range(0, len(rows), self._batch_rows):
                batch = rows[start : start + self._batch_rows]
                self.inserted[table] += db.insert_rows(self._conn, table, columns, batch)
            rows.clear()


def sha256_of_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_paths(corpus_dir: str | Path) -> tuple[Path, Path]:
    """Return ``(pages.jsonl, claims.jsonl)`` under ``corpus_dir``.

    Raises ``FileNotFoundError`` naming the missing file.
    """
    directory = Path(corpus_dir)
    pages_path = directory / "pages.jsonl"
    claims_path = directory / "claims.jsonl"
    for path in (pages_path, claims_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run scripts/build_corpus.py first")
    return pages_path, claims_path


def _count_lines(path: Path) -> int:
    """Return the number of non-blank lines of a file (the tqdm total)."""
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _next_id(conn: pymysql.Connection, table: str) -> int:
    """Return the largest primary key of ``table`` plus one (1 on an empty table)."""
    with conn.cursor() as cur:
        cur.execute(_NEXT_ID_SQL[table])
        return int(cur.fetchone()[0])


def _server_version(conn: pymysql.Connection) -> str:
    """Return the server's ``VERSION()`` string."""
    with conn.cursor() as cur:
        cur.execute(VERSION_SQL)
        return str(cur.fetchone()[0])


class _Ingest:
    """One run of the pipeline; ``run`` executes the six steps in order."""

    def __init__(
        self,
        settings: Settings,
        conn: pymysql.Connection,
        pages_path: Path,
        claims_path: Path,
        reset: bool,
        embedder: EmbedderLike | None,
        batch_size: int,
        use_prefix: bool,
        progress: bool,
    ) -> None:
        self.settings = settings
        self.conn = conn
        self.pages_path = pages_path
        self.claims_path = claims_path
        self.reset = reset
        self.embedder = embedder
        self.batch_size = batch_size
        self.use_prefix = use_prefix
        self.progress = progress
        self.report = IngestReport()
        self.timer = _Timer()
        self.loader = _Loader(conn)
        self.page_ids: dict[str, int] = {}  # NFC title -> page_id
        self.sentence_ids: dict[tuple[int, str], int] = {}  # (page_id, element_key) -> id
        self.chunks: list[_PendingChunk] = []
        self.model_name = settings.embedding_model
        self.next_id: dict[str, int] = {}

    def run(self) -> IngestReport:
        """Execute the six steps and return the report."""
        start = time.perf_counter()
        self._apply_schema()
        self._load_pages()
        self._embed_and_load_chunks()
        self._resolve_links()
        self._load_claims()
        self._write_meta()
        self.timer.seconds["total"] = time.perf_counter() - start
        self.report.seconds = dict(self.timer.seconds)
        logger.info(
            "ingest done: %d pages, %d chunks, %d claims in %.1f s",
            self.report.n_pages,
            self.report.n_chunks,
            self.report.n_claims,
            self.report.seconds["total"],
        )
        return self.report

    # step 1 ------------------------------------------------------------------------------------

    def _apply_schema(self) -> None:
        """Apply (or reset and apply) the schema and read the next free ids."""
        with self.timer.stage("schema"):
            n_statements = db.apply_schema(self.conn, reset=self.reset)
            self.next_id = {table: _next_id(self.conn, table) for table in _ID_COLUMNS}
        logger.info(
            "schema: %d statements applied (reset=%s) in %.2f s",
            n_statements,
            self.reset,
            self.timer.seconds["schema"],
        )

    # step 2 ------------------------------------------------------------------------------------

    def _load_pages(self) -> None:
        """Parse every page and write page, section, sentence and link rows; keep the chunks."""
        report = self.report
        total = _count_lines(self.pages_path)
        pages = tqdm(
            iter_pages(self.pages_path),
            total=total,
            disable=not self.progress,
            unit="page",
            desc="parse",
        )
        for page in pages:
            with self.timer.stage("parse"):
                parsed = parse_page(page)
                chunks = chunk_page(
                    parsed,
                    max_words=self.settings.chunk_max_words,
                    overlap_units=self.settings.chunk_overlap_units,
                )
            if parsed.title in self.page_ids:
                raise IngestError(
                    f"duplicate page title {parsed.title!r} in {self.pages_path}; "
                    "page titles must be unique"
                )
            with self.timer.stage("load"):
                page_id = self.next_id["page"]
                self.next_id["page"] += 1
                self.page_ids[parsed.title] = page_id
                stats = parsed.stats
                self.loader.add(
                    "page",
                    (
                        page_id,
                        parsed.title,
                        stats["n_sentences"],
                        stats["n_items"],
                        stats["n_words"],
                        stats["n_chars"],
                        stats["n_sections"],
                        stats["n_tables"],
                        stats["n_lists"],
                    ),
                )
                section_base = self.next_id["section"]
                self.next_id["section"] += len(parsed.sections)
                for section in parsed.sections:
                    self.loader.add(
                        "section",
                        (
                            section_base + section.ordinal,
                            page_id,
                            section.ordinal,
                            section.heading,
                            section.level,
                            section.path,
                        ),
                    )
                unit_ids: dict[str, int] = {}
                empty_keys: set[str] = set()
                for unit in parsed.units:
                    sentence_id = self.next_id["sentence"]
                    self.next_id["sentence"] += 1
                    unit_ids[unit.element_key] = sentence_id
                    self.sentence_ids[(page_id, unit.element_key)] = sentence_id
                    if not unit.text:
                        empty_keys.add(unit.element_key)
                    self.loader.add(
                        "sentence",
                        (
                            sentence_id,
                            page_id,
                            section_base + unit.section_ordinal,
                            unit.element_key,
                            unit.ordinal,
                            unit.text,
                        ),
                    )
                for to_title, source_element in parsed.links:
                    if len(to_title) > MAX_TITLE_LENGTH:
                        report.n_links_skipped += 1
                        continue
                    self.loader.add("link", (page_id, to_title, None, source_element))
                    report.n_links += 1
                for chunk in chunks:
                    section = parsed.sections[chunk.section_ordinal]
                    chunk_id = self.next_id["chunk"]
                    self.next_id["chunk"] += 1
                    self.chunks.append(
                        _PendingChunk(
                            chunk_id=chunk_id,
                            page_id=page_id,
                            section_id=section_base + chunk.section_ordinal,
                            ordinal=chunk.ordinal,
                            text=chunk.text,
                            n_words=chunk.n_words,
                            sentence_ids=[
                                unit_ids[key] for key in chunk.element_keys if key not in empty_keys
                            ],
                            embedding_text=(
                                embedding_text(parsed.title, section.path, chunk.text)
                                if self.use_prefix
                                else chunk.text
                            ),
                        )
                    )
                report.n_pages += 1
                report.n_sections += len(parsed.sections)
                report.n_sentences += len(parsed.units)
                report.n_units_empty += len(empty_keys)
        with self.timer.stage("load"):
            self.loader.flush()
            self.conn.commit()
        report.n_chunks = len(self.chunks)
        logger.info(
            "pages: %d pages, %d sections, %d sentences (%d empty), %d links (%d skipped), "
            "%d chunks; parse %.2f s, load %.2f s",
            report.n_pages,
            report.n_sections,
            report.n_sentences,
            report.n_units_empty,
            report.n_links,
            report.n_links_skipped,
            report.n_chunks,
            self.timer.seconds["parse"],
            self.timer.seconds["load"],
        )

    # step 3 ------------------------------------------------------------------------------------

    def _embed_and_load_chunks(self) -> None:
        """Check the model dimension, embed the chunk texts, write chunk and chunk_sentence."""
        with self.timer.stage("embed"):
            if self.embedder is None:
                from wikilense.embedding import Embedder

                self.embedder = Embedder(self.settings.embedding_model)
            self.model_name = str(getattr(self.embedder, "model_name", self.model_name))
            dim = int(self.embedder.dim)
            if dim != self.settings.vector_dim:
                raise IngestError(
                    f"embedding model {self.model_name!r} has dimension {dim}, but the schema "
                    f"and WIKILENSE_VECTOR_DIM expect {self.settings.vector_dim}; no chunk "
                    "rows were written"
                )
            texts = [chunk.embedding_text for chunk in self.chunks]
            vectors = self.embedder.embed_passages(
                texts, batch_size=self.batch_size, show_progress=self.progress
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(texts), dim):
                raise IngestError(
                    f"embedder returned shape {vectors.shape}, expected {(len(texts), dim)}"
                )
        logger.info(
            "embed: %d chunk texts with %s (dim %d, prefix=%s, batch_size %d) in %.2f s",
            len(texts),
            self.model_name,
            dim,
            self.use_prefix,
            self.batch_size,
            self.timer.seconds["embed"],
        )
        load_before = self.timer.seconds["load"]
        with self.timer.stage("load"):
            chunks = tqdm(
                self.chunks, disable=not self.progress, unit="chunk", desc="load chunks"
            )
            for index, chunk in enumerate(chunks):
                self.loader.add(
                    "chunk",
                    (
                        chunk.chunk_id,
                        chunk.page_id,
                        chunk.section_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.n_words,
                        db.vec_param(vectors[index]),
                    ),
                )
                for sentence_id in chunk.sentence_ids:
                    self.loader.add("chunk_sentence", (chunk.chunk_id, sentence_id))
            self.loader.flush()
            self.conn.commit()
        logger.info(
            "chunks: %d chunk rows and %d chunk_sentence rows written in %.2f s",
            self.loader.inserted["chunk"],
            self.loader.inserted["chunk_sentence"],
            self.timer.seconds["load"] - load_before,
        )

    # step 4 ------------------------------------------------------------------------------------

    def _resolve_links(self) -> None:
        """Set ``link.to_page_id`` for every link whose target title is a corpus page."""
        with self.timer.stage("resolve"):
            with self.conn.cursor() as cur:
                cur.execute(RESOLVE_LINKS_SQL)
                self.report.n_links_resolved = int(cur.rowcount)
            self.conn.commit()
        logger.info(
            "links: %d of %d resolved to corpus pages in %.2f s",
            self.report.n_links_resolved,
            self.report.n_links,
            self.timer.seconds["resolve"],
        )

    # step 5 ------------------------------------------------------------------------------------

    def _load_claims(self) -> None:
        """Write claim and claim_evidence rows with page_id and sentence_id resolved."""
        report = self.report
        for record in iter_claims(self.claims_path):
            claim_id = int(record["id"])
            with self.timer.stage("load"):
                self.loader.add(
                    "claim",
                    (
                        claim_id,
                        record["split"],
                        record["claim"],
                        record["label"],
                        str(record.get("challenge") or ""),
                    ),
                )
            for set_index, evidence_set in enumerate(record["evidence"]):
                for position, element_id in enumerate(evidence_set["content"]):
                    with self.timer.stage("resolve"):
                        try:
                            title, element_type, key = parse_element_id(element_id)
                        except ValueError as exc:
                            raise IngestError(f"claim {claim_id}: {exc}") from exc
                        page_id = self.page_ids.get(title)
                        sentence_id = None
                        if page_id is not None and element_type in _TEXT_UNIT_TYPES:
                            sentence_id = self.sentence_ids.get((page_id, key))
                    with self.timer.stage("load"):
                        self.loader.add(
                            "claim_evidence",
                            (
                                claim_id,
                                set_index,
                                position,
                                element_id,
                                title,
                                element_type,
                                page_id,
                                sentence_id,
                            ),
                        )
                    report.n_evidence += 1
                    report.n_evidence_page_resolved += int(page_id is not None)
                    report.n_evidence_sentence_resolved += int(sentence_id is not None)
            report.n_claims += 1
        with self.timer.stage("load"):
            self.loader.flush()
            self.conn.commit()
        logger.info(
            "claims: %d claims, %d evidence ids (%d with a corpus page, %d with a text unit)",
            report.n_claims,
            report.n_evidence,
            report.n_evidence_page_resolved,
            report.n_evidence_sentence_resolved,
        )

    # step 6 ------------------------------------------------------------------------------------

    def _write_meta(self) -> None:
        """Replace the ingest_meta rows with this run's parameters and corpus digests."""
        settings = self.settings
        with self.timer.stage("load"):
            values = {
                "embedding_model": self.model_name,
                "embedding_dim": str(settings.vector_dim),
                "embedding_prefix": "true" if self.use_prefix else "false",
                "chunk_max_words": str(settings.chunk_max_words),
                "chunk_overlap_units": str(settings.chunk_overlap_units),
                "index_m": str(settings.index_m),
                "index_distance": INDEX_DISTANCE,
                "corpus_dir": str(self.pages_path.parent.resolve()),
                "corpus_pages_sha256": sha256_of_file(self.pages_path),
                "corpus_claims_sha256": sha256_of_file(self.claims_path),
                "mariadb_version": _server_version(self.conn),
                "wikilense_version": __version__,
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            with self.conn.cursor() as cur:
                cur.execute(CLEAR_META_SQL)
            for key in META_KEYS:
                self.loader.add("ingest_meta", (key, values[key]))
            self.loader.flush()
            self.conn.commit()
        logger.info(
            "meta: %d ingest_meta rows written (model %s, corpus %s)",
            len(META_KEYS),
            self.model_name,
            values["corpus_dir"],
        )


def run_ingest(
    settings: Settings,
    corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
    reset: bool = True,
    embedder: EmbedderLike | None = None,
    batch_size: int = 64,
    use_prefix: bool = True,
    progress: bool = True,
) -> IngestReport:
    """Ingest ``corpus_dir`` into ``settings.db_name`` and return the ``IngestReport``.

    ``reset=True`` drops and recreates every table first; ``reset=False`` only creates missing
    tables, so it is for a database prepared with ``init-db`` (re-ingesting on top of existing
    rows fails on the unique page title). ``embedder`` defaults to
    ``embedding.Embedder(settings.embedding_model)``; ``batch_size`` is the embedding batch;
    ``use_prefix`` embeds ``"title > section path: text"`` instead of the bare chunk text;
    ``progress`` shows tqdm bars. Each of the six steps (module doc) is committed when it
    completes.

    Raises ``IngestError`` when ``settings.db_name`` is the test database, when
    ``settings.vector_dim`` differs from the literal in ``sql/schema.sql``, when the embedder's
    dimension differs from ``settings.vector_dim`` (before any chunk row is written), on a
    duplicate page title or a malformed evidence id; ``FileNotFoundError`` when a corpus file is
    missing; ``ValueError`` for ``batch_size < 1``.
    """
    if settings.db_name == settings.test_db_name:
        raise IngestError(
            f"refusing to ingest into {settings.db_name!r}: it is the test database "
            "(WIKILENSE_TEST_DB_NAME); point WIKILENSE_DB_NAME at the corpus database"
        )
    if settings.vector_dim != db.VECTOR_DIM:
        raise IngestError(
            f"WIKILENSE_VECTOR_DIM is {settings.vector_dim}, but sql/schema.sql declares "
            f"VECTOR({db.VECTOR_DIM}); change the schema and db.VECTOR_DIM together"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    pages_path, claims_path = corpus_paths(corpus_dir)
    conn = db.connect(settings)
    try:
        ingest = _Ingest(
            settings,
            conn,
            pages_path,
            claims_path,
            reset=reset,
            embedder=embedder,
            batch_size=batch_size,
            use_prefix=use_prefix,
            progress=progress,
        )
        try:
            return ingest.run()
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()
