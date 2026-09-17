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
from wikilense.config import Settings
from wikilense.ingest import (
    INSERT_BATCH_ROWS,
    META_KEYS,
    STAGES,
    IngestError,
    IngestReport,
    run_ingest,
)
from wikilense.wikitext import parse_page

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
# 6, Beta River 2, Gamma Mountain 2; links: Alpha City 5 written + 1 skipped, Beta River 3).
EXPECTED_COUNTS: dict[str, int] = {
    "n_pages": 3,
    "n_sections": 8,  # 3 + 2 + 3, lead sections included
    "n_sentences": 15,  # 8 + 3 + 4 text units, the whitespace-only one included
    "n_units_empty": 1,
    "n_chunks": 10,
    "n_links": 8,
    "n_links_resolved": 6,  # Delta Town and Omega Sea are not corpus pages
    "n_links_skipped": 1,
    "n_claims": 2,
    "n_evidence": 9,
    "n_evidence_page_resolved": 8,  # all but Missing Page
    "n_evidence_sentence_resolved": 5,  # sentence_1, item_0_1, sentence_2, item_0_0, sentence_5
}

EXPECTED_LINKS: list[tuple[str, str, str, bool]] = [
    ("Alpha City", "Beta River", "sentence_0", True),
    ("Alpha City", "Gamma Mountain", "sentence_0", True),
    ("Alpha City", "Delta Town", "sentence_0", False),
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


def test_report_counts_and_stages() -> None:
    report = IngestReport()
    assert set(report.counts()) == set(EXPECTED_COUNTS)
    assert set(report.seconds) == set(STAGES)
    assert {"parse", "embed", "load", "resolve"} <= set(STAGES)


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
    assert embedder.calls == [(EXPECTED_COUNTS["n_chunks"], 64)]

    with closing(dbmod.connect(ingest_settings)) as conn:
        _check_tables(conn)
        _check_chunk_map(conn, embedder)
        _check_links(conn)
        _check_evidence(conn)
        _check_meta(conn, corpus_dir)
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
    ) == [(6, 2, 2, 1, 1)]
    assert _rows(
        conn,
        "SELECT s.ordinal, s.heading, s.level, s.path FROM section s JOIN page p USING (page_id) "
        "WHERE p.title = %s ORDER BY s.ordinal",
        ("Gamma Mountain",),
    ) == [(0, "", 1, ""), (1, "Geology", 2, "Geology"), (2, "Rock types", 3, "Geology > Rock types")]
    # the whitespace-only unit is a sentence row with empty text, and the only one in no chunk
    unmapped = _rows(
        conn,
        "SELECT p.title, s.element_key, s.text FROM sentence s JOIN page p USING (page_id) "
        "LEFT JOIN chunk_sentence cs USING (sentence_id) WHERE cs.chunk_id IS NULL",
    )
    assert unmapped == [("Alpha City", "sentence_5", "")]


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
            non_empty = {u.element_key for u in parsed.units if u.text}
            assert keys == [k for k in chunk.element_keys if k in non_empty]
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


def _check_meta(conn: pymysql.Connection, corpus_dir: Path) -> None:
    """ingest_meta holds every key with the values of this run."""
    meta = dict(_rows(conn, "SELECT `key`, `value` FROM ingest_meta"))
    assert set(meta) == set(META_KEYS)
    assert meta["embedding_model"] == "fake-embedder"
    assert meta["embedding_dim"] == str(DIM)
    assert meta["embedding_prefix"] == "true"
    assert meta["chunk_max_words"] == str(CHUNK_MAX_WORDS)
    assert meta["chunk_overlap_units"] == str(CHUNK_OVERLAP_UNITS)
    assert meta["index_distance"] == "cosine"
    assert meta["corpus_dir"] == str(corpus_dir.resolve())
    for name in ("pages", "claims"):
        digest = hashlib.sha256((corpus_dir / f"{name}.jsonl").read_bytes()).hexdigest()
        assert meta[f"corpus_{name}_sha256"] == digest
    assert meta["mariadb_version"].startswith("11.")
    assert meta["index_m"].isdigit()
    assert datetime.fromisoformat(meta["ingested_at"]).utcoffset() is not None
    assert meta["wikilense_version"]


@pytest.mark.db
def test_wrong_embedder_dim_raises_before_any_chunk_row(
    ingest_settings: Settings, corpus_dir: Path
) -> None:
    with pytest.raises(IngestError, match=f"dimension {DIM + 16}.*expect {DIM}"):
        run_ingest(ingest_settings, corpus_dir=corpus_dir, embedder=FakeEmbedder(DIM + 16), progress=False)
    with closing(dbmod.connect(ingest_settings)) as conn:
        counts = _table_counts(conn)
    assert counts["chunk"] == 0 and counts["chunk_sentence"] == 0
    assert counts["page"] == 3  # step 2 was committed before the check
    assert counts["claim"] == 0 and counts["ingest_meta"] == 0


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
