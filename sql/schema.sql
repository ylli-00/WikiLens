-- WikiLense data model for MariaDB 11.8 (VECTOR column, VECTOR INDEX, VEC_DISTANCE_COSINE).
--
-- wikilense.db.apply_schema() runs this file statement by statement; the tables are created in
-- dependency order (parents first) and dropped in the reverse order on reset. Every table is
-- InnoDB with utf8mb4. Title columns use utf8mb4_bin so that a title matches exactly (case- and
-- accent-sensitive, as Wikipedia titles are); other text keeps the server's default
-- case-insensitive collation so that a heading filter such as `heading LIKE 'history%'` matches
-- "History". Rows that belong to a page (sections, sentences, chunks, outgoing links) are
-- deleted with it (ON DELETE CASCADE); references that only resolve a title to a corpus page
-- become NULL again when that page goes (ON DELETE SET NULL).
--
-- The vector dimension 384 (BAAI/bge-small-en-v1.5) and the index parameter M=16 were chosen with the experiments in results/SUMMARY.md
-- (docs/DESIGN.md); the ingest checks the model dimension against this literal.

-- page: one row per Wikipedia page in the corpus, with the size statistics that the length
-- filters use (n_words is indexed for "only pages longer than N words").
CREATE TABLE IF NOT EXISTS page (
    page_id     INT UNSIGNED NOT NULL AUTO_INCREMENT,
    title       VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,
    n_sentences INT UNSIGNED NOT NULL,
    n_items     INT UNSIGNED NOT NULL,
    n_words     INT UNSIGNED NOT NULL,
    n_chars     INT UNSIGNED NOT NULL,
    n_sections  INT UNSIGNED NOT NULL,
    n_tables    INT UNSIGNED NOT NULL,
    n_lists     INT UNSIGNED NOT NULL,
    PRIMARY KEY (page_id),
    UNIQUE KEY uq_page_title (title),
    KEY ix_page_n_words (n_words)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- section: the section tree of a page, flattened in page order. Ordinal 0 is the lead
-- (heading '', level 1, path ''); path is the headings from the top of the tree joined by ' > '.
-- heading is indexed for the "chunks under a heading like X" filter.
CREATE TABLE IF NOT EXISTS section (
    section_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
    page_id    INT UNSIGNED NOT NULL,
    ordinal    INT UNSIGNED NOT NULL,
    heading    VARCHAR(255) NOT NULL,
    level      TINYINT UNSIGNED NOT NULL,
    path       TEXT NOT NULL,
    PRIMARY KEY (section_id),
    UNIQUE KEY uq_section_page_ordinal (page_id, ordinal),
    KEY ix_section_heading (heading),
    CONSTRAINT fk_section_page FOREIGN KEY (page_id)
        REFERENCES page (page_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- sentence: one row per text unit, a sentence (element_key 'sentence_N') or a list item
-- ('item_N_M'), in page order with the cleaned text. This is the unit of FEVEROUS gold
-- evidence, so recall is measured against these rows.
CREATE TABLE IF NOT EXISTS sentence (
    sentence_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
    page_id     INT UNSIGNED NOT NULL,
    section_id  INT UNSIGNED NOT NULL,
    element_key VARCHAR(32) NOT NULL,
    ordinal     INT UNSIGNED NOT NULL,
    text        TEXT NOT NULL,
    PRIMARY KEY (sentence_id),
    UNIQUE KEY uq_sentence_page_key (page_id, element_key),
    KEY ix_sentence_section (section_id),
    CONSTRAINT fk_sentence_page FOREIGN KEY (page_id)
        REFERENCES page (page_id) ON DELETE CASCADE,
    CONSTRAINT fk_sentence_section FOREIGN KEY (section_id)
        REFERENCES section (section_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- chunk: the retrieval unit. Consecutive text units of one section (at most
-- WIKILENSE_CHUNK_MAX_WORDS words, never across a section boundary) with the embedding of
-- "title > section path: text" stored in a VECTOR(384) column. The HNSW vector index
-- (M=16, cosine distance) serves ORDER BY VEC_DISTANCE_COSINE(embedding, ?) LIMIT n.
-- M=16 (server default 6): with M=6 the HNSW search misses gold pages that an exact ranking finds
-- unless mhnsw_ef_search is raised to 200+; with M=16 it matches the exact ranking at the default
-- ef_search 20 for the same sub-millisecond latency (build 2.6 s vs 0.7 s on 8,868 chunks).
-- The FULLTEXT index on text serves MATCH(text) AGAINST (?) for the hybrid ``rrf`` search
-- strategy (vector top-N and keyword top-N fused by reciprocal rank fusion in one statement).
CREATE TABLE IF NOT EXISTS chunk (
    chunk_id   INT UNSIGNED NOT NULL AUTO_INCREMENT,
    page_id    INT UNSIGNED NOT NULL,
    section_id INT UNSIGNED NOT NULL,
    ordinal    INT UNSIGNED NOT NULL,
    text       TEXT NOT NULL,
    n_words    INT UNSIGNED NOT NULL,
    embedding  VECTOR(384) NOT NULL,
    PRIMARY KEY (chunk_id),
    UNIQUE KEY uq_chunk_page_ordinal (page_id, ordinal),
    KEY ix_chunk_section (section_id),
    FULLTEXT KEY ft_chunk_text (text),
    VECTOR INDEX (embedding) M=16 DISTANCE=cosine,
    CONSTRAINT fk_chunk_page FOREIGN KEY (page_id)
        REFERENCES page (page_id) ON DELETE CASCADE,
    CONSTRAINT fk_chunk_section FOREIGN KEY (section_id)
        REFERENCES section (section_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- chunk_sentence: which text units each chunk contains. Neighbouring chunks overlap by
-- WIKILENSE_CHUNK_OVERLAP_UNITS units, so a sentence can belong to two chunks. This map joins a
-- retrieved chunk back to its sentences and to the gold evidence.
CREATE TABLE IF NOT EXISTS chunk_sentence (
    chunk_id    INT UNSIGNED NOT NULL,
    sentence_id INT UNSIGNED NOT NULL,
    PRIMARY KEY (chunk_id, sentence_id),
    KEY ix_chunk_sentence_sentence (sentence_id),
    CONSTRAINT fk_chunk_sentence_chunk FOREIGN KEY (chunk_id)
        REFERENCES chunk (chunk_id) ON DELETE CASCADE,
    CONSTRAINT fk_chunk_sentence_sentence FOREIGN KEY (sentence_id)
        REFERENCES sentence (sentence_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- link: every [[target|shown]] wiki link of a page (from sentences, list items and table
-- cells), one row per occurrence, with the element key it came from. to_page_id is set when
-- the target title is a corpus page, which makes "chunks of pages linked from X" a join;
-- links to pages outside the corpus keep only to_title.
CREATE TABLE IF NOT EXISTS link (
    link_id        INT UNSIGNED NOT NULL AUTO_INCREMENT,
    from_page_id   INT UNSIGNED NOT NULL,
    to_title       VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,
    to_page_id     INT UNSIGNED NULL,
    source_element VARCHAR(32) NOT NULL,
    PRIMARY KEY (link_id),
    KEY ix_link_from_page (from_page_id),
    KEY ix_link_to_page (to_page_id),
    KEY ix_link_to_title (to_title),
    CONSTRAINT fk_link_from_page FOREIGN KEY (from_page_id)
        REFERENCES page (page_id) ON DELETE CASCADE,
    CONSTRAINT fk_link_to_page FOREIGN KEY (to_page_id)
        REFERENCES page (page_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- claim: a FEVEROUS claim used as a query. claim_id is the FEVEROUS id; split, label and the
-- annotator's challenge tag are kept so that results can be sliced by them.
CREATE TABLE IF NOT EXISTS claim (
    claim_id  INT UNSIGNED NOT NULL,
    split     ENUM('train', 'dev') NOT NULL,
    text      TEXT NOT NULL,
    label     ENUM('SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO') NOT NULL,
    challenge VARCHAR(64) NOT NULL,
    PRIMARY KEY (claim_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- claim_evidence: the gold evidence of a claim, one row per element id of every alternative
-- evidence set (evidence_set = index of the set, position = index inside its content list).
-- page_id and sentence_id are resolved when the page and the text unit are in the corpus;
-- cells and captions resolve to the page only.
CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id     INT UNSIGNED NOT NULL,
    evidence_set TINYINT UNSIGNED NOT NULL,
    position     SMALLINT UNSIGNED NOT NULL,
    element_id   VARCHAR(300) NOT NULL,
    page_title   VARCHAR(255) COLLATE utf8mb4_bin NOT NULL,
    element_type ENUM('sentence', 'cell', 'header_cell', 'table_caption', 'item') NOT NULL,
    page_id      INT UNSIGNED NULL,
    sentence_id  INT UNSIGNED NULL,
    PRIMARY KEY (claim_id, evidence_set, position),
    KEY ix_claim_evidence_page (page_id),
    KEY ix_claim_evidence_sentence (sentence_id),
    CONSTRAINT fk_claim_evidence_claim FOREIGN KEY (claim_id)
        REFERENCES claim (claim_id) ON DELETE CASCADE,
    CONSTRAINT fk_claim_evidence_page FOREIGN KEY (page_id)
        REFERENCES page (page_id) ON DELETE SET NULL,
    CONSTRAINT fk_claim_evidence_sentence FOREIGN KEY (sentence_id)
        REFERENCES sentence (sentence_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ingest_meta: how the corpus was ingested, as key/value pairs (embedding model and dimension,
-- chunk parameters, index M, corpus SHA-256, ingested_at), so that every result can be tied to
-- the exact settings that produced it.
CREATE TABLE IF NOT EXISTS ingest_meta (
    `key`   VARCHAR(64) NOT NULL,
    `value` TEXT NOT NULL,
    PRIMARY KEY (`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
