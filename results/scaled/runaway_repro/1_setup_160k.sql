-- Self-contained setup: 160,000 pseudo-random vectors of 384 dimensions, made in SQL with MD5 (deterministic).
-- Needs the Sequence engine (on by default).
SET SESSION group_concat_max_len = 1048576;
DROP TABLE IF EXISTS t, qv;
CREATE TABLE t (id INT UNSIGNED NOT NULL PRIMARY KEY, w INT NOT NULL, v VECTOR(384) NOT NULL,
  VECTOR INDEX (v) M=16 DISTANCE=cosine) ENGINE=InnoDB;
-- w is a permutation of 0..159999, so exactly k rows have w < k
INSERT INTO t (id, w, v)
SELECT r.seq, r.seq * 7919 % 160000,
  VEC_FromText(CONCAT('[', (SELECT GROUP_CONCAT(ROUND(CONV(LEFT(MD5(CONCAT(r.seq, ':', d.seq)), 8), 16, 10) / 4294967296 - 0.5, 6) ORDER BY d.seq) FROM seq_1_to_384 d), ']'))
FROM seq_1_to_160000 r;
-- the query vector: one more vector made the same way, not in the table
CREATE TABLE qv (v VECTOR(384) NOT NULL);
INSERT INTO qv SELECT VEC_FromText(CONCAT('[', (SELECT GROUP_CONCAT(ROUND(CONV(LEFT(MD5(CONCAT(160001, ':', d.seq)), 8), 16, 10) / 4294967296 - 0.5, 6) ORDER BY d.seq) FROM seq_1_to_384 d), ']'));
SELECT COUNT(*) AS rows_in_t FROM t;
