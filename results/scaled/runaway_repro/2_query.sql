-- Run after the table is loaded and the server restarted, in a NEW connection, from outside the
-- server's container. Max_memory_used is the peak memory of this connection so far.
SET @q = (SELECT v FROM qv);
SET SESSION mhnsw_ef_search = 100;
SET SESSION max_statement_time = 60;   -- ends a statement after 60 s; on a large table a 2 GiB server runs out of memory before that

-- 1. a plain k-NN search
SELECT id FROM t ORDER BY VEC_DISTANCE_COSINE(v, @q) LIMIT 10;
SHOW SESSION STATUS LIKE 'Max_memory_used';

-- 2. a distance threshold that exactly 3 rows meet: halfway between the 3rd and 4th exact distance
SELECT (MIN(d) + MAX(d)) / 2 INTO @t3 FROM
  (SELECT VEC_DISTANCE_COSINE(v, @q) AS d FROM t IGNORE INDEX (v) ORDER BY d LIMIT 2, 2) x;
SELECT id FROM t IGNORE INDEX (v) WHERE VEC_DISTANCE_COSINE(v, @q) < @t3 ORDER BY VEC_DISTANCE_COSINE(v, @q) LIMIT 10;
SHOW SESSION STATUS LIKE 'Max_memory_used';

-- 3. the same statement through the vector index, three times (expected: the same 3 ids, in the same order)
SELECT id FROM t WHERE VEC_DISTANCE_COSINE(v, @q) < @t3 ORDER BY VEC_DISTANCE_COSINE(v, @q) LIMIT 10;
SHOW SESSION STATUS LIKE 'Max_memory_used';
SELECT id FROM t WHERE VEC_DISTANCE_COSINE(v, @q) < @t3 ORDER BY VEC_DISTANCE_COSINE(v, @q) LIMIT 10;
SHOW SESSION STATUS LIKE 'Max_memory_used';
SELECT id FROM t WHERE VEC_DISTANCE_COSINE(v, @q) < @t3 ORDER BY VEC_DISTANCE_COSINE(v, @q) LIMIT 10;
SHOW SESSION STATUS LIKE 'Max_memory_used';

-- 4. an ordinary filter that exactly 5 rows pass (expected: those 5 ids)
SELECT id FROM t IGNORE INDEX (v) WHERE w < 5 ORDER BY VEC_DISTANCE_COSINE(v, @q) LIMIT 10;
SELECT id FROM t WHERE w < 5 ORDER BY VEC_DISTANCE_COSINE(v, @q) LIMIT 10;
SHOW SESSION STATUS LIKE 'Max_memory_used';
SHOW SESSION STATUS LIKE 'Memory_used';
