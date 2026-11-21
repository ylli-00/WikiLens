# A vector search that fewer rows pass than LIMIT, on 11.8.9, 12.3.3 and 13.0.2

**Conclusion.**
- On MariaDB 11.8.9, a search that fewer rows pass than its `LIMIT` can keep searching until the server runs out of memory.
- On 12.3.3 and 13.0.2, the same statements on the same data peaked at 552 MiB or less, and nothing was killed.
- On 11.8.9, a short `max_statement_time` stops such a search: 10 s did, and under 60 s the server was killed first.

A later reproduction in SQL only, with the connection's own memory, is in [runaway_repro](../runaway_repro/runaway_repro.md).

## Setup

Throwaway containers of the official images, each started with `--memory 2g --memory-swap 2g --cpus 4 --mhnsw-max-cache-size=536870912`; never the project's database server.

**The table**
```sql
t (id INT PRIMARY KEY, w INT NOT NULL, v VECTOR(384) NOT NULL, VECTOR INDEX (v) M=16 DISTANCE=cosine)
```
- **v** holds the 132,359 S1 chunk vectors from the release file's `vectors/part_*.npy`.
- **w** holds random integers.
- The rows were inserted in one transaction.

**The query.** One claim's query vector (bge-small-en-v1.5), `mhnsw_ef_search` 100, `LIMIT` 10, `max_statement_time` 60.

**The cases** (`capprobe.py`, a fresh connection per case unless "repeat"):
- `threshold_n3`: `WHERE VEC_DISTANCE_COSINE(v, q) < t`, with `t` halfway between the 3rd and 4th exact distance, so 3 rows qualify.
- `threshold_n10`, `threshold_n30`: the same with 10 and 30 rows qualifying.
- `predicate_w_5`: `WHERE w < x`, with `x` one more than the 5th smallest `w`: at least 5 rows pass. On the table of S1 vectors 6 pass, because the 5th and 6th smallest `w` are equal.
- `predicate_w_100`: the same for the 100th smallest `w` (102 rows pass on the table of S1 vectors).

Memory is the container's cgroup `memory.current`, sampled every 0.1 s. Each log file's lines are in the order they ran; a `marker` line starts a new setup.

**The files.**
- `capprobe_q.npy`: three claims' query vectors (claim ids 195, 199 and 342, the first three of the 200-claim sample, seed 20261119). The 11.8.9 cases used the first; the 12.3.3 and 13.0.2 runs used all three.
- The two scripts were tidied for the linter after the runs (exception types, imports and file handling). What they do is unchanged.

## 11.8.9 (`capprobe_results.jsonl`)

**Real vectors, after four other statements**
- `threshold_n30` and `threshold_n10`: 10 rows in 0.01 s.
- `threshold_n3`: 3 rows in 10.25 s, peak 842 MiB.

**After a restart, the first run was fast and the next run was killed**
- `threshold_n3` repeated on one connection:
  - run 1: 3 rows in 1.51 s;
  - run 2: lost connection after 20.27 s at the 2 GB cap, and the kernel killed `mariadbd` (OOM).
- `threshold_n3` on fresh connections:
  - run 1: 3 rows in 1.71 s;
  - run 2: killed after 21.74 s at 2,048 MiB.
- After `predicate_w_100` (2 of 10 rows, 1.46 s), `predicate_w_5` was killed after 21.95 s at 2,048 MiB.
- An earlier repeat on one connection was also killed, but its runs were not logged.

**Random unit vectors, 132,359 rows**
- Five of the cases (threshold n = 3, 10 and 30, and the two `w` filters) took 3.29 to 16.55 s.
- They peaked at 748 to 1,311 MiB and came back short (0 to 3 rows where an exact answer has 3 to 10, except `predicate_w_100` with 10).
- The memory was given back, and nothing was killed.

**With 20,000 random vectors:** at most 0.36 s and 351 MiB.

## 11.8.9 with a time limit (`tl_check_results.jsonl`, `tl_check.py`)

Same data, loaded with the project's own schema into a throwaway container, with `max_statement_time` = 10:
- `threshold_n3` gave 3 rows in 1.99 s (peak 579 MiB).
- It was then stopped at 10.01 s and 10.00 s with error 1969 (peaks 1,307 and 1,545 MiB).
- The server stayed up.

## 12.3.3 (`probe12_results.jsonl`) and 13.0.2 (`probe13_results.jsonl`)

Same data and cases:
- **`threshold_n3`:**
  - 12.3.3: 3 rows in 6.85 s on its first run, then 0.81 to 1.69 s;
  - 13.0.2: 3 rows in 5.47 s, then 0.63 to 1.16 s;
  - 4 repeats on one connection each, plus three query vectors.
- **`predicate_w_5`:** 0 or 1 of the 6 passing rows, in 0.59 to 1.49 s.
- **Memory:** peaks of 481 to 552 MiB. Nothing was killed.
