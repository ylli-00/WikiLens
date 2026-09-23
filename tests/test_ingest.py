"""Tests for wikilense.ingest.

The ``db`` tests ingest a three-page synthetic corpus (FEVEROUS page format, written to
``tmp_path``) into the *test* database with a deterministic fake embedder, so no model is
downloaded. ``run_ingest`` refuses ``settings.test_db_name``, so the tests hand it Settings whose
``db_name`` is the test database and whose ``test_db_name`` is a different, unused name
(``dataclasses.replace``); the pure refusal test checks the guard itself. The expected counts
below are counted by hand from the pages and claims; the chunk map is compared with
``chunk_page`` on the same pages.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pymysql
import pytest

from wikilense import db as dbmod
from wikilense.chunking import chunk_page, embedding_text
from wikilense.config import REPO_ROOT, Settings
from wikilense.ingest import (
    ANALYZE_TABLES,
    INSERT_BATCH_ROWS,
    META_KEYS,
    STAGES,
    IngestError,
    IngestReport,
    corpus_dir_label,
    run_ingest,
)
from wikilense.wikitext import HATNOTE_RE, parse_page

DIM = dbmod.VECTOR_DIM
CHUNK_MAX_WORDS = 12  # small, so that the synthetic pages give several overlapping chunks
CHUNK_OVERLAP_UNITS = 1

# ---------------------------------------------------------------------------------------------
# synthetic corpus
# ---------------------------------------------------------------------------------------------

LONG_TARGET = "x" * 300  # a link target over the 255-character to_title column: skipped


def _cell(cell_id: str, value: str, is_header: bool = False) -> dict:
    """Return one FEVEROUS table cell."""
    return {"id": cell_id, "value": value, "is_header": is_header, "row_span": 1, "column_span": 1}


PAGES: list[dict] = [
    {
        "title": "Alpha City",
        "order": [
            "sentence_0",
            "sentence_1",
            "section_0",
            "sentence_6",
            "sentence_2",
            "sentence_3",
            "list_0",
            "section_1",
            "sentence_4",
            "table_0",
            "sentence_5",
        ],
        "sentence_0": (
            "Alpha City is a city on the [[Beta_River]] near [[Gamma Mountain]] "
            "and [[Delta Town|Delta]]."
        ),
        "sentence_1": "It has about fifty thousand inhabitants.",
        "section_0": {"value": "History", "level": 2},
        # a hatnote: a sentence row (evidence ids resolve), in no chunk
        "sentence_6": "Main article: [[History of Alpha City|History of Alpha City]]",
        "sentence_2": "The city was founded in 1200.",
        "sentence_3": "It grew quickly in the 1800s.",
        "list_0": {
            "list": [
                {"id": "item_0_0", "value": "1200: founded", "level": 1},
                {"id": "item_0_1", "value": "1850: railway to [[Beta River]]", "level": 1},
            ],
            "type": "unordered",
        },
        "section_1": {"value": "Economy", "level": 2},
        "sentence_4": f"Trade with [[{LONG_TARGET}|far away]] is important.",
        "table_0": {
            "table": [
                [_cell("header_cell_0_0_0", "Sector", True), _cell("header_cell_0_0_1", "Share", True)],
                [_cell("cell_0_1_0", "Fishing on the [[Beta River|river]]"), _cell("cell_0_1_1", "40%")],
            ],
            "type": "general",
            "caption": "Economy by sector",
        },
        "sentence_5": "   ",  # whitespace-only: a sentence row, but in no chunk
    },
    {
        "title": "Beta River",
        "order": ["sentence_0", "sentence_1", "section_0", "sentence_2"],
        "sentence_0": "The Beta River flows past [[Alpha City]].",
        "sentence_1": "It is 120 km long.",
        "section_0": {"value": "Course", "level": 2},
        "sentence_2": "The river rises on [[Gamma Mountain]] and ends in the [[Omega Sea|sea]].",
    },
    {
        "title": "Gamma Mountain",
        "order": ["sentence_0", "section_0", "section_1", "sentence_1", "list_0", "sentence_2"],
        "sentence_0": "Gamma Mountain is the highest peak of the region.",
        "section_0": {"value": "Geology", "level": 2},
        "section_1": {"value": "Rock types", "level": 3},
        "sentence_1": "The summit is granite.",
        "list_0": {"list": [{"id": "item_0_0", "value": "granite", "level": 1}], "type": "unordered"},
        "sentence_2": "Basalt occurs lower down.",
    },
]

CLAIMS: list[dict] = [
    {
        "id": 101,
        "claim": "Alpha City has fifty thousand inhabitants and got its railway in 1850.",
        "label": "SUPPORTS",
        "evidence": [
            {
                "content": ["Alpha City_sentence_1", "Alpha City_item_0_1"],
                "context": {"Alpha City_sentence_1": ["Alpha City_title"]},
            },
            {"content": ["Alpha City_cell_0_1_0", "Alpha City_header_cell_0_0_0"], "context": {}},
        ],
        "annotator_operations": [],
        "challenge": "Numerical Reasoning",
        "split": "dev",
    },
    {
        "id": 102,
        "claim": "The Beta River rises on a basalt mountain.",
        "label": "REFUTES",
        "evidence": [
            {
                "content": [
                    "Beta River_sentence_2",
                    "Gamma Mountain_item_0_0",
                    "Missing Page_sentence_0",
                    "Alpha City_table_caption_0",
                    "Alpha City_sentence_5",
                    "Alpha City_sentence_6",
                ],
                "context": {},
            }
        ],
        "annotator_operations": [],
        "challenge": "Other",
        "split": "train",
    },
]

# Counted by hand from PAGES and CLAIMS (chunks with CHUNK_MAX_WORDS=12, overlap 1: Alpha City
# 6, Beta River 2, Gamma Mountain 2, the hatnote and the whitespace-only unit in none; links:
# Alpha City 6 written + 1 skipped, Beta River 3).
EXPECTED_COUNTS: dict[str, int] = {
    "n_pages": 3,
    "n_sections": 8,  # 3 + 2 + 3, lead sections included
    "n_sentences": 16,  # 9 + 3 + 4 text units, the whitespace-only one and the hatnote included
    "n_units_empty": 1,
    "n_units_hatnote": 1,
    "n_chunks": 10,
    "n_links": 9,
    "n_links_resolved": 6,  # Delta Town, Omega Sea and History of Alpha City are not corpus pages
    "n_links_skipped": 1,
    "n_claims": 2,
    "n_evidence": 10,
    "n_evidence_page_resolved": 9,  # all but Missing Page
    "n_evidence_sentence_resolved": 6,  # sentence_1, item_0_1, sentence_2, item_0_0, 5 and 6
}

EXPECTED_LINKS: list[tuple[str, str, str, bool]] = [
    ("Alpha City", "Beta River", "sentence_0", True),
    ("Alpha City", "Gamma Mountain", "sentence_0", True),
    ("Alpha City", "Delta Town", "sentence_0", False),
    ("Alpha City", "History of Alpha City", "sentence_6", False),
    ("Alpha City", "Beta River", "item_0_1", True),
    ("Alpha City", "Beta River", "cell_0_1_0", True),
    ("Beta River", "Alpha City", "sentence_0", True),
    ("Beta River", "Gamma Mountain", "sentence_2", True),
    ("Beta River", "Omega Sea", "sentence_2", False),
]

# (claim_id, evidence_set, position, element_type, resolved page title, resolved element_key)
EXPECTED_EVIDENCE: list[tuple[int, int, int, str, str | None, str | None]] = [
    (101, 0, 0, "sentence", "Alpha City", "sentence_1"),
    (101, 0, 1, "item", "Alpha City", "item_0_1"),
    (101, 1, 0, "cell", "Alpha City", None),
    (101, 1, 1, "header_cell", "Alpha City", None),
    (102, 0, 0, "sentence", "Beta River", "sentence_2"),
    (102, 0, 1, "item", "Gamma Mountain", "item_0_0"),
    (102, 0, 2, "sentence", None, None),
    (102, 0, 3, "table_caption", "Alpha City", None),
    (102, 0, 4, "sentence", "Alpha City", "sentence_5"),
    (102, 0, 5, "sentence", "Alpha City", "sentence_6"),  # the hatnote resolves
]


class FakeEmbedder:
    """Deterministic stand-in for ``embedding.Embedder``: one unit vector per text.

    The vector is seeded from the SHA-256 of the text, so equal texts give equal vectors and
    the embeddings can be recomputed in a test. ``calls`` records ``(n_texts, batch_size)``.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.model_name = "fake-embedder"
        self.calls: list[tuple[int, int]] = []

    def embed_passages(
        self, texts: list[str], batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        """Return unit-length float32 vectors, one per text."""
        self.calls.append((len(texts), batch_size))
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dim)
            out[i] = v / np.linalg.norm(v)
        return out


def _write_corpus(directory: Path, pages: list[dict], claims: list[dict]) -> Path:
    """Write pages.jsonl and claims.jsonl into ``directory`` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "pages.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(page, ensure_ascii=False) + "\n" for page in pages)
    with open(directory / "claims.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(claim, ensure_ascii=False) + "\n" for claim in claims)
    return directory


def _rows(conn: pymysql.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    """Return every row of a parameterised query."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _scalar(conn: pymysql.Connection, sql: str, params: tuple = ()) -> int:
    """Return the first column of the first row of a query as an int."""
    return int(_rows(conn, sql, params)[0][0])


def _table_counts(conn: pymysql.Connection) -> dict[str, int]:
    """Return the row count of every schema table."""
    counts = {}
    for table in dbmod.table_names():
        counts[table] = _scalar(conn, f"SELECT COUNT(*) FROM `{table}`")  # fixed allowlist
    return counts


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def ingest_settings(settings: Settings) -> Settings:
    """Settings that ingest into the test database with the small chunk parameters.

    ``db_name`` becomes the test database and ``test_db_name`` a different, unused name, so
    that the refusal guard in ``run_ingest`` lets the test database through and the real
    corpus database is never touched.
    """
    return replace(
        settings,
        db_name=settings.test_db_name,
        test_db_name=settings.test_db_name + "_unused",
        chunk_max_words=CHUNK_MAX_WORDS,
        chunk_overlap_units=CHUNK_OVERLAP_UNITS,
    )


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """The synthetic corpus written to a temporary directory."""
    return _write_corpus(tmp_path / "corpus", PAGES, CLAIMS)


# ---------------------------------------------------------------------------------------------
# pure tests
# ---------------------------------------------------------------------------------------------


def test_refuses_the_test_database() -> None:
    same = Settings(db_password="x", db_name="wikilense_test", test_db_name="wikilense_test")
    with pytest.raises(IngestError, match="test database"):
        run_ingest(same, corpus_dir="/nonexistent", embedder=FakeEmbedder(), progress=False)


def test_refuses_a_vector_dim_that_disagrees_with_the_schema() -> None:
    settings = Settings(db_password="x", db_name="a", test_db_name="b", vector_dim=DIM + 1)
    with pytest.raises(IngestError, match=f"VECTOR\\({DIM}\\)"):
        run_ingest(settings, corpus_dir="/nonexistent", embedder=FakeEmbedder(), progress=False)


def test_missing_corpus_file_is_named(tmp_path: Path) -> None:
    settings = Settings(db_password="x", db_name="a", test_db_name="b")
    (tmp_path / "pages.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="claims.jsonl"):
        run_ingest(settings, corpus_dir=tmp_path, embedder=FakeEmbedder(), progress=False)


class NoDatabase:
    """``db.connect`` replacement for the pure tests: reaching the database is the failure."""

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("the database must not be touched")


@pytest.mark.parametrize(
    ("change", "variable"),
    [({"chunk_max_words": 0}, "WIKILENSE_CHUNK_MAX_WORDS"),
     ({"chunk_overlap_units": -1}, "WIKILENSE_CHUNK_OVERLAP_UNITS")],
)
def test_bad_chunk_settings_are_refused_before_the_database_is_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict, variable: str
) -> None:
    monkeypatch.setattr(dbmod, "connect", NoDatabase())
    settings = replace(Settings(db_password="x", db_name="a", test_db_name="b"), **change)
    corpus = _write_corpus(tmp_path / "corpus", PAGES, CLAIMS)
    with pytest.raises(IngestError, match=variable):
        run_ingest(settings, corpus_dir=corpus, embedder=FakeEmbedder(), progress=False)


class DeadConnection:
    """A connection whose socket is gone: rollback raises like PyMySQL's (InterfaceError 0)."""

    def __init__(self, is_open: bool) -> None:
        self.open = is_open
        self.closed = False

    def rollback(self) -> None:
        raise pymysql.err.InterfaceError(0, "")

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("is_open", [False, True])
@pytest.mark.parametrize("original", [KeyboardInterrupt(), RuntimeError("the real error")])
def test_a_failed_rollback_does_not_replace_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_open: bool, original: BaseException
) -> None:
    from wikilense import ingest as ingestmod

    conn = DeadConnection(is_open)
    monkeypatch.setattr(dbmod, "connect", lambda settings: conn)

    def fail(self: object) -> None:
        raise original

    monkeypatch.setattr(ingestmod._Ingest, "run", fail)
    settings = Settings(db_password="x", db_name="a", test_db_name="b")
    corpus = _write_corpus(tmp_path / "corpus", PAGES, CLAIMS)
    with pytest.raises(type(original)) as raised:
        run_ingest(settings, corpus_dir=corpus, embedder=FakeEmbedder(), progress=False)
    assert raised.value is original and conn.closed


class AnalyzeConnection:
    """Answers ANALYZE TABLE with the given (table, op, msg_type, msg_text) rows."""

    def __init__(self, rows: list[tuple[str, str, str, str]]) -> None:
        self.rows = rows
        self.executed: list[str] = []
        self.commits = 0

    def cursor(self) -> closing:  # a context manager whose target is this object
        return closing(self)

    def close(self) -> None:
        pass

    def execute(self, sql: str) -> None:
        self.executed.append(sql)

    def fetchall(self) -> list[tuple[str, str, str, str]]:
        return self.rows

    def commit(self) -> None:
        self.commits += 1


def test_an_analyze_error_row_is_an_ingest_error(tmp_path: Path) -> None:
    from wikilense import ingest as ingestmod

    settings = Settings(db_password="x", db_name="a", test_db_name="b")
    ok = AnalyzeConnection([("a.chunk", "analyze", "status", "OK")])
    run = ingestmod._Ingest(settings, ok, tmp_path, tmp_path, True, FakeEmbedder(), 64, True, False)
    run._analyze_tables()
    assert ok.executed == [f"ANALYZE TABLE `{t}`" for t in ANALYZE_TABLES] and ok.commits == 1
    bad = AnalyzeConnection([("a.chunk", "analyze", "Error", "Table 'a.chunk' is marked as crashed")])
    run = ingestmod._Ingest(settings, bad, tmp_path, tmp_path, True, FakeEmbedder(), 64, True, False)
    with pytest.raises(IngestError, match="ANALYZE TABLE chunk failed: Table 'a.chunk' is marked"):
        run._analyze_tables()


def test_schema_literal_is_the_index_m_constant() -> None:
    """sql/schema.sql is the only place M is set; db.VECTOR_INDEX_M names it for the code."""
    script = dbmod.SCHEMA_PATH.read_text(encoding="utf-8")
    assert f"VECTOR INDEX (embedding) M={dbmod.VECTOR_INDEX_M} DISTANCE=cosine" in script
    assert not hasattr(Settings(db_password="x"), "index_m")  # not a setting: it did nothing


def test_report_counts_and_stages() -> None:
    report = IngestReport()
    assert set(report.counts()) == set(EXPECTED_COUNTS)
    assert set(report.seconds) == set(STAGES)
    assert {"parse", "embed", "load", "resolve", "analyze"} <= set(STAGES)
    assert ANALYZE_TABLES == ("chunk", "page", "section", "link")


def test_corpus_dir_label_is_relative_inside_the_repository(tmp_path: Path) -> None:
    assert corpus_dir_label(REPO_ROOT / "data" / "corpus") == "data/corpus"
    assert corpus_dir_label(REPO_ROOT / "data" / ".." / "data" / "corpus") == "data/corpus"
    assert corpus_dir_label(str(REPO_ROOT)) == "."
    outside = tmp_path / "corpus"
    assert corpus_dir_label(outside) == str(outside.resolve())
    assert not corpus_dir_label(REPO_ROOT / "data" / "corpus").startswith("/")


# ---------------------------------------------------------------------------------------------
# database tests (fake embedder, no model)
# ---------------------------------------------------------------------------------------------


@pytest.mark.db
def test_run_ingest_on_the_synthetic_corpus(ingest_settings: Settings, corpus_dir: Path) -> None:
    embedder = FakeEmbedder()
    report = run_ingest(ingest_settings, corpus_dir=corpus_dir, embedder=embedder, progress=False)

    assert report.counts() == EXPECTED_COUNTS
    assert set(report.seconds) == set(STAGES)
    assert all(value >= 0.0 for value in report.seconds.values())
    assert report.seconds["total"] >= report.seconds["parse"]
    assert report.seconds["analyze"] > 0.0
    assert embedder.calls == [(EXPECTED_COUNTS["n_chunks"], 64)]

    with closing(dbmod.connect(ingest_settings)) as conn:
        _check_tables(conn)
        _check_chunk_map(conn, embedder)
        _check_links(conn)
        _check_evidence(conn)
        _check_meta(conn, corpus_dir, report)
        _check_statistics(conn)
        first_counts = _table_counts(conn)

    # A second run with reset=True rebuilds everything and gives the same counts.
    second = run_ingest(ingest_settings, corpus_dir=corpus_dir, embedder=FakeEmbedder(), progress=False)
    assert second.counts() == report.counts()
    with closing(dbmod.connect(ingest_settings)) as conn:
        assert _table_counts(conn) == first_counts
        assert _scalar(conn, "SELECT MIN(page_id) FROM page") == 1


def _check_tables(conn: pymysql.Connection) -> None:
    """Row counts of the nine tables against the hand counts."""
    counts = _table_counts(conn)
    assert counts["page"] == 3
    assert counts["section"] == EXPECTED_COUNTS["n_sections"]
    assert counts["sentence"] == EXPECTED_COUNTS["n_sentences"]
    assert counts["chunk"] == EXPECTED_COUNTS["n_chunks"]
    assert counts["link"] == EXPECTED_COUNTS["n_links"]
    assert counts["claim"] == 2
    assert counts["claim_evidence"] == EXPECTED_COUNTS["n_evidence"]
    assert counts["ingest_meta"] == len(META_KEYS)
    # page statistics and the section tree of Alpha City
    assert _rows(
        conn, "SELECT n_sentences, n_items, n_sections, n_tables, n_lists FROM page WHERE title = %s",
        ("Alpha City",),
    ) == [(7, 2, 2, 1, 1)]
    assert _rows(
        conn,
        "SELECT s.ordinal, s.heading, s.level, s.path FROM section s JOIN page p USING (page_id) "
        "WHERE p.title = %s ORDER BY s.ordinal",
        ("Gamma Mountain",),
    ) == [(0, "", 1, ""), (1, "Geology", 2, "Geology"), (2, "Rock types", 3, "Geology > Rock types")]
    # the hatnote and the whitespace-only unit are sentence rows (the hatnote with its cleaned
    # text and its page-order ordinal), and the only two units in no chunk
    unmapped = _rows(
        conn,
        "SELECT p.title, s.element_key, s.ordinal, s.text FROM sentence s "
        "JOIN page p USING (page_id) LEFT JOIN chunk_sentence cs USING (sentence_id) "
        "WHERE cs.chunk_id IS NULL ORDER BY s.ordinal",
    )
    assert unmapped == [
        ("Alpha City", "sentence_6", 2, "Main article: History of Alpha City"),
        ("Alpha City", "sentence_5", 8, ""),
    ]
    assert _scalar(conn, "SELECT COUNT(*) FROM chunk WHERE text LIKE %s", ("Main article%",)) == 0


def _check_chunk_map(conn: pymysql.Connection, embedder: FakeEmbedder) -> None:
    """chunk rows and the chunk_sentence map against chunk_page() on the same pages."""
    for page in PAGES:
        parsed = parse_page(page)
        expected = chunk_page(parsed, max_words=CHUNK_MAX_WORDS, overlap_units=CHUNK_OVERLAP_UNITS)
        stored = _rows(
            conn,
            "SELECT c.chunk_id, c.ordinal, c.text, c.n_words, sec.ordinal FROM chunk c "
            "JOIN page p USING (page_id) JOIN section sec ON sec.section_id = c.section_id "
            "WHERE p.title = %s ORDER BY c.ordinal",
            (parsed.title,),
        )
        assert [row[1:] for row in stored] == [
            (c.ordinal, c.text, c.n_words, c.section_ordinal) for c in expected
        ]
        for (chunk_id, *_), chunk in zip(stored, expected):
            keys = [
                row[0]
                for row in _rows(
                    conn,
                    "SELECT s.element_key FROM chunk_sentence cs JOIN sentence s USING (sentence_id) "
                    "WHERE cs.chunk_id = %s ORDER BY s.ordinal",
                    (chunk_id,),
                )
            ]
            assert keys == chunk.element_keys
            assert all(u.chunkable for u in parsed.units if u.element_key in keys)
            # the stored vector is the fake embedding of "title > path: text" (prefix on)
            section_path = parsed.sections[chunk.section_ordinal].path
            text = embedding_text(parsed.title, section_path, chunk.text)
            raw = _rows(conn, "SELECT embedding FROM chunk WHERE chunk_id = %s", (chunk_id,))[0][0]
            np.testing.assert_allclose(
                dbmod.vec_from_bytes(raw), embedder.embed_passages([text])[0], rtol=0, atol=1e-6
            )
    # a unit shared by two overlapping chunks appears twice in the map
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM chunk_sentence cs JOIN sentence s USING (sentence_id) "
        "JOIN page p ON p.page_id = s.page_id WHERE p.title = %s AND s.element_key = %s",
        ("Alpha City", "sentence_3"),
    ) == 2


def _check_links(conn: pymysql.Connection) -> None:
    """Every link row with its source and whether the target resolved to a corpus page."""
    rows = _rows(
        conn,
        "SELECT f.title, l.to_title, l.source_element, t.title FROM link l "
        "JOIN page f ON f.page_id = l.from_page_id LEFT JOIN page t ON t.page_id = l.to_page_id "
        "ORDER BY l.link_id",
    )
    assert [(a, b, c, d is not None) for a, b, c, d in rows] == EXPECTED_LINKS
    assert all(d is None or d == b for _, b, _, d in rows)
    assert not any(len(b) > 255 for _, b, _, _ in rows)


def _check_evidence(conn: pymysql.Connection) -> None:
    """claim rows and claim_evidence resolution for sentence, item, cell and caption ids."""
    assert _rows(conn, "SELECT claim_id, split, label, challenge FROM claim ORDER BY claim_id") == [
        (101, "dev", "SUPPORTS", "Numerical Reasoning"),
        (102, "train", "REFUTES", "Other"),
    ]
    rows = _rows(
        conn,
        "SELECT e.claim_id, e.evidence_set, e.position, e.element_type, p.title, s.element_key "
        "FROM claim_evidence e LEFT JOIN page p ON p.page_id = e.page_id "
        "LEFT JOIN sentence s ON s.sentence_id = e.sentence_id "
        "ORDER BY e.claim_id, e.evidence_set, e.position",
    )
    assert rows == EXPECTED_EVIDENCE
    assert _rows(
        conn, "SELECT element_id, page_title FROM claim_evidence WHERE claim_id = 102 AND position = 2"
    ) == [("Missing Page_sentence_0", "Missing Page")]


def _check_meta(conn: pymysql.Connection, corpus_dir: Path, report: IngestReport) -> None:
    """ingest_meta holds every key with the values of this run."""
    meta = dict(_rows(conn, "SELECT `key`, `value` FROM ingest_meta"))
    assert set(meta) == set(META_KEYS)
    assert meta["embedding_model"] == "fake-embedder"
    assert meta["embedding_revision"] == "unpinned"  # the fake has no revision_label
    assert meta["embedding_dim"] == str(DIM)
    assert meta["embedding_prefix"] == "true"
    assert meta["chunk_max_words"] == str(CHUNK_MAX_WORDS)
    assert meta["chunk_overlap_units"] == str(CHUNK_OVERLAP_UNITS)
    assert meta["index_distance"] == "cosine"
    assert meta["hatnote_pattern"] == HATNOTE_RE.pattern
    assert meta["n_units_hatnote"] == "1"
    assert meta["analyze_tables"] == "chunk,page,section,link"
    assert float(meta["analyze_seconds"]) == pytest.approx(report.seconds["analyze"], abs=0.001)
    assert float(meta["analyze_seconds"]) > 0.0
    # the temporary corpus lies outside the repository, so the path stays absolute
    assert meta["corpus_dir"] == str(corpus_dir.resolve())
    for name in ("pages", "claims"):
        digest = hashlib.sha256((corpus_dir / f"{name}.jsonl").read_bytes()).hexdigest()
        assert meta[f"corpus_{name}_sha256"] == digest
    assert meta["mariadb_version"].startswith("11.")
    assert meta["index_m"] == str(dbmod.VECTOR_INDEX_M)  # read from the index itself
    assert datetime.fromisoformat(meta["ingested_at"]).utcoffset() is not None
    assert meta["wikilense_version"]


def _check_statistics(conn: pymysql.Connection) -> None:
    """ANALYZE TABLE ran: the primary-key cardinality of every analysed table is its row count.

    InnoDB's persistent statistics are exact for tables this small (every leaf page is sampled),
    and they are only refreshed by ANALYZE TABLE or by the background recalculation, which is
    not guaranteed to have run right after the bulk insert.
    """
    for table in ANALYZE_TABLES:
        cardinality = _scalar(
            conn,
            "SELECT CARDINALITY FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE() "
            "AND TABLE_NAME = %s AND INDEX_NAME = 'PRIMARY' AND SEQ_IN_INDEX = 1",
            (table,),
        )
        assert cardinality == _table_counts(conn)[table], table


@pytest.mark.db
def test_wrong_embedder_dim_is_refused_before_the_previous_ingest_is_dropped(
    ingest_settings: Settings, corpus_dir: Path
) -> None:
    run_ingest(ingest_settings, corpus_dir=corpus_dir, embedder=FakeEmbedder(), progress=False)
    with closing(dbmod.connect(ingest_settings)) as conn:
        before = _table_counts(conn)
    with pytest.raises(IngestError, match=f"dimension {DIM + 16}.*expect {DIM}"):
        run_ingest(ingest_settings, corpus_dir=corpus_dir, embedder=FakeEmbedder(DIM + 16), progress=False)
    with closing(dbmod.connect(ingest_settings)) as conn:
        assert _table_counts(conn) == before  # nothing was dropped
    assert before["chunk"] > 0 and before["ingest_meta"] == len(META_KEYS)


@pytest.mark.db
def test_no_reset_refuses_a_database_that_already_holds_an_ingest(
    ingest_settings: Settings, corpus_dir: Path
) -> None:
    run_ingest(ingest_settings, corpus_dir=corpus_dir, embedder=FakeEmbedder(), progress=False)
    with closing(dbmod.connect(ingest_settings)) as conn:
        before = _table_counts(conn)
    with pytest.raises(IngestError, match="already holds 3 pages"):
        run_ingest(ingest_settings, corpus_dir=corpus_dir, embedder=FakeEmbedder(), reset=False,
                   progress=False)
    with closing(dbmod.connect(ingest_settings)) as conn:
        assert _table_counts(conn) == before


@pytest.mark.db
def test_no_reset_ingests_into_an_empty_schema_and_records_the_live_index_m(
    ingest_settings: Settings, corpus_dir: Path
) -> None:
    """``init-db`` then ``ingest --no-reset``; the index was rebuilt with another M in between,
    and ingest_meta records the M the index has, not a setting."""
    with closing(dbmod.connect(ingest_settings)) as conn:
        dbmod.apply_schema(conn, reset=True)
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE chunk DROP INDEX embedding")
            cur.execute("ALTER TABLE chunk ADD VECTOR INDEX embedding (embedding) M=8 DISTANCE=cosine")
    report = run_ingest(ingest_settings, corpus_dir=corpus_dir, embedder=FakeEmbedder(),
                        reset=False, progress=False)
    assert report.counts() == EXPECTED_COUNTS
    with closing(dbmod.connect(ingest_settings)) as conn:
        meta = dict(_rows(conn, "SELECT `key`, `value` FROM ingest_meta"))
        assert dbmod.vector_index_info(conn) == {"index_m": 8, "index_distance": "cosine"}
    assert meta["index_m"] == "8" and meta["index_distance"] == "cosine"


@pytest.mark.db
def test_duplicate_page_title_and_malformed_evidence_id_are_ingest_errors(
    ingest_settings: Settings, tmp_path: Path
) -> None:
    twice = _write_corpus(tmp_path / "twice", [PAGES[0], PAGES[0]], [])
    with pytest.raises(IngestError, match="duplicate page title 'Alpha City'"):
        run_ingest(ingest_settings, corpus_dir=twice, embedder=FakeEmbedder(), progress=False)
    bad_claim = {**CLAIMS[0], "id": 999, "evidence": [{"content": ["no element id"], "context": {}}]}
    bad = _write_corpus(tmp_path / "bad", PAGES, [bad_claim])
    with pytest.raises(IngestError, match="claim 999: not a FEVEROUS element id"):
        run_ingest(ingest_settings, corpus_dir=bad, embedder=FakeEmbedder(), progress=False)


@pytest.mark.db
def test_no_prefix_embeds_the_bare_chunk_text(ingest_settings: Settings, corpus_dir: Path) -> None:
    embedder = FakeEmbedder()
    run_ingest(ingest_settings, corpus_dir=corpus_dir, embedder=embedder, use_prefix=False, progress=False)
    with closing(dbmod.connect(ingest_settings)) as conn:
        text, raw = _rows(conn, "SELECT text, embedding FROM chunk ORDER BY chunk_id LIMIT 1")[0]
        assert dict(_rows(conn, "SELECT `key`, `value` FROM ingest_meta"))["embedding_prefix"] == "false"
    np.testing.assert_allclose(dbmod.vec_from_bytes(raw), embedder.embed_passages([text])[0], atol=1e-6)


@pytest.mark.db
def test_inserts_are_batched_at_most_1000_rows(
    ingest_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_sentences = 2 * INSERT_BATCH_ROWS + 345
    page = {
        "title": "Long Page",
        "order": [f"sentence_{i}" for i in range(n_sentences)],
        **{f"sentence_{i}": f"Sentence number {i} links to [[Long Page]]." for i in range(n_sentences)},
    }
    corpus = _write_corpus(tmp_path / "long", [page], [])
    batches: list[tuple[str, int]] = []
    real_insert_rows = dbmod.insert_rows

    def recording_insert_rows(conn, table, columns, rows):
        rows = list(rows)
        batches.append((table, len(rows)))
        return real_insert_rows(conn, table, columns, rows)

    monkeypatch.setattr(dbmod, "insert_rows", recording_insert_rows)
    report = run_ingest(ingest_settings, corpus_dir=corpus, embedder=FakeEmbedder(), progress=False)

    assert report.n_sentences == n_sentences and report.n_links == n_sentences
    assert report.n_links_resolved == n_sentences  # every link points at the page itself
    assert max(size for _, size in batches) <= INSERT_BATCH_ROWS
    per_table: dict[str, int] = {}
    for table, size in batches:
        per_table[table] = per_table.get(table, 0) + size
    assert per_table["sentence"] == n_sentences and per_table["link"] == n_sentences
    assert per_table["chunk"] == report.n_chunks
    assert sum(1 for table, _ in batches if table == "sentence") >= 3
