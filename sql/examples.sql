-- WikiLense: the query shapes, for the mariadb client alone (no Python, no model).
-- The "query vector" is the stored embedding of chunk 1 (the lead of the page "Aare"), so every
-- statement runs after `wikilense ingest`:  mariadb -h 127.0.0.1 -u wikilense -p wikilense < sql/examples.sql
-- Cosine distance is 0 for the chunk itself, so it comes first.

-- 1. The bare index query: the HNSW search proper, sub-millisecond, accuracy set by mhnsw_ef_search.
SET SESSION mhnsw_ef_search = 100;
SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
FROM chunk
ORDER BY distance
LIMIT 5;

-- 2. Predicates and joins in the same statement: pages longer than 5,000 words, sections whose
--    heading contains "History". On the ingested corpus EXPLAIN shows chunk with key = embedding (the vector index) and
--    page / section joined by primary key.
SELECT chunk.chunk_id, page.title, section.path,
       VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
FROM chunk
  STRAIGHT_JOIN page ON page.page_id = chunk.page_id
  STRAIGHT_JOIN section ON section.section_id = chunk.section_id
WHERE page.n_words >= 5000 AND section.heading LIKE '%History%'
ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1))
LIMIT 5;

EXPLAIN
SELECT chunk.chunk_id, page.title, section.path,
       VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
FROM chunk
  STRAIGHT_JOIN page ON page.page_id = chunk.page_id
  STRAIGHT_JOIN section ON section.section_id = chunk.section_id
WHERE page.n_words >= 5000 AND section.heading LIKE '%History%'
ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1))
LIMIT 5;

-- 3. The link graph as a predicate: only pages that "Aare" links to.
SELECT chunk.chunk_id, page.title,
       VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
FROM chunk
  STRAIGHT_JOIN page ON page.page_id = chunk.page_id
  STRAIGHT_JOIN (SELECT DISTINCT link.to_page_id AS page_id
                 FROM link JOIN page AS src ON src.page_id = link.from_page_id
                 WHERE src.title = 'Aare' AND link.to_page_id IS NOT NULL) AS linked
                ON linked.page_id = chunk.page_id
ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1))
LIMIT 5;

-- 4. Bounded over-fetch: the index returns 50 candidates, the outer query filters and re-limits.
SELECT chunk.chunk_id, page.title, knn.distance
FROM (SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
      FROM chunk ORDER BY distance LIMIT 50) AS knn
JOIN chunk ON chunk.chunk_id = knn.chunk_id
JOIN page ON page.page_id = chunk.page_id
WHERE page.n_words >= 5000
ORDER BY knn.distance, chunk.chunk_id
LIMIT 5;

-- 5. Hybrid: the vector top-50 and the FULLTEXT top-50 fused by reciprocal rank fusion.
WITH vec AS (SELECT knn.chunk_id, ROW_NUMBER() OVER (ORDER BY knn.distance, knn.chunk_id) AS rnk
             FROM (SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
                   FROM chunk ORDER BY distance LIMIT 50) AS knn),
     ft AS (SELECT m.chunk_id, ROW_NUMBER() OVER (ORDER BY m.relevance DESC, m.chunk_id) AS rnk
            FROM (SELECT chunk_id, MATCH(text) AGAINST ('Aare river Switzerland' IN NATURAL LANGUAGE MODE) AS relevance
                  FROM chunk WHERE MATCH(text) AGAINST ('Aare river Switzerland' IN NATURAL LANGUAGE MODE)
                  ORDER BY relevance DESC, chunk_id LIMIT 50) AS m),
     fused AS (SELECT r.chunk_id, SUM(CAST(1 AS DOUBLE) / (60 + r.rnk)) AS score
               FROM (SELECT chunk_id, rnk FROM vec UNION ALL SELECT chunk_id, rnk FROM ft) AS r
               GROUP BY r.chunk_id)
SELECT chunk.chunk_id, page.title, fused.score
FROM fused JOIN chunk ON chunk.chunk_id = fused.chunk_id JOIN page ON page.page_id = chunk.page_id
ORDER BY fused.score DESC, chunk.chunk_id
LIMIT 5;

-- 6. The join back to the sentences a chunk was built from (the evaluation's evidence match).
SELECT s.element_key, s.text
FROM chunk_sentence AS cs JOIN sentence AS s ON s.sentence_id = cs.sentence_id
WHERE cs.chunk_id = 1
ORDER BY s.ordinal;

-- 7. The vector as text, and the server's vector settings.
SELECT chunk_id, LENGTH(embedding) AS bytes, LEFT(VEC_ToText(embedding), 60) AS first_values FROM chunk WHERE chunk_id = 1;
SHOW VARIABLES LIKE 'mhnsw%';

-- 8. The vectors are rows of the transaction. Inside one, chunk 2 gets chunk 1's vector: the same
--    session's search sees chunk 2 at distance 0 at once; ROLLBACK puts the old vector back, and
--    the index with it (chunk 2 returns to 0.1467). Another session sees neither change until
--    COMMIT (tests/test_db.py, test_vector_index_follows_the_transaction). The derived table
--    "one" lets the UPDATE read the table it writes.
START TRANSACTION;
UPDATE chunk SET embedding = (SELECT e FROM (SELECT embedding AS e FROM chunk WHERE chunk_id = 1) AS one)
WHERE chunk_id = 2;
SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
FROM chunk ORDER BY distance LIMIT 3;
ROLLBACK;
SELECT chunk_id, VEC_DISTANCE_COSINE(embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
FROM chunk ORDER BY distance LIMIT 3;

-- 9. The brief's four predicates in one statement: only pages in the category "Member states of
--    the United Nations", longer than 1,000 words, last edited (in the revision the corpus holds)
--    since 1 December 2020, and linked from the page "Asia"; joined back to page and section.
--    On the ingested corpus EXPLAIN shows chunk read through the vector index (type index, key embedding) with the two
--    filter tables materialised (<derivedN>). The DISTINCT keeps them materialised: merged into
--    the join they make MariaDB scan chunk in full. Which filters keep the index is measured in
--    results/planner_choices.md.
SELECT chunk.chunk_id, page.title, section.path, page.last_edited,
       VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
FROM chunk
  STRAIGHT_JOIN page ON page.page_id = chunk.page_id
  STRAIGHT_JOIN section ON section.section_id = chunk.section_id
  STRAIGHT_JOIN (SELECT DISTINCT link.to_page_id AS page_id
                 FROM link JOIN page AS src ON src.page_id = link.from_page_id
                 WHERE src.title = 'Asia' AND link.to_page_id IS NOT NULL) AS linked
                ON linked.page_id = chunk.page_id
  STRAIGHT_JOIN (SELECT DISTINCT page_category.page_id
                 FROM page_category JOIN category ON category.category_id = page_category.category_id
                 WHERE category.name = 'Member states of the United Nations') AS in_category
                ON in_category.page_id = chunk.page_id
WHERE page.n_words >= 1000 AND page.last_edited >= '2020-12-01'
ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1))
LIMIT 5;

EXPLAIN
SELECT chunk.chunk_id, page.title, section.path, page.last_edited,
       VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1)) AS distance
FROM chunk
  STRAIGHT_JOIN page ON page.page_id = chunk.page_id
  STRAIGHT_JOIN section ON section.section_id = chunk.section_id
  STRAIGHT_JOIN (SELECT DISTINCT link.to_page_id AS page_id
                 FROM link JOIN page AS src ON src.page_id = link.from_page_id
                 WHERE src.title = 'Asia' AND link.to_page_id IS NOT NULL) AS linked
                ON linked.page_id = chunk.page_id
  STRAIGHT_JOIN (SELECT DISTINCT page_category.page_id
                 FROM page_category JOIN category ON category.category_id = page_category.category_id
                 WHERE category.name = 'Member states of the United Nations') AS in_category
                ON in_category.page_id = chunk.page_id
WHERE page.n_words >= 1000 AND page.last_edited >= '2020-12-01'
ORDER BY VEC_DISTANCE_COSINE(chunk.embedding, (SELECT embedding FROM chunk WHERE chunk_id = 1))
LIMIT 5;

-- 10. The page metadata is plain relational data: the revision, edit date and categories of a
--     page (data/corpus/page_meta.jsonl, fetched by scripts/fetch_page_meta.py).
SELECT page.title, page.revision_id, page.last_edited, category.name AS category
FROM page JOIN page_category USING (page_id) JOIN category USING (category_id)
WHERE page.title = 'Aare'
ORDER BY category.name;
