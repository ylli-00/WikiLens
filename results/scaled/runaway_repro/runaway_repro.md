# A filtered vector search that takes the server's memory: a reproduction in SQL only

**Conclusion.**
- On MariaDB 11.8.9, one `SELECT ... WHERE ... ORDER BY VEC_DISTANCE_COSINE(...) LIMIT 10` whose `WHERE` few rows pass took 709 MB of the connection's memory on 80,000 generated vectors, where a plain search took 0.13 MB. On 160,000 vectors the server was killed in a 2 GiB container, in 2 of 2 runs.
- On 12.3.3 and 13.0.2 the same statements took 42 to 150 MB, and nothing was killed.
- On all three versions such a search can come back with fewer rows than pass.

This repeats [the version probe](../version_probe/version_probe.md) with a script that needs no data file, so that it can go with a bug report to MariaDB. Measured on 2026-11-21.

## Setup

Throwaway containers of the official images, never the project's database server:
```
docker run -d --memory 2g --memory-swap 2g --cpus 2 -p 127.0.0.1:3306:3306 -e MARIADB_ROOT_PASSWORD=... -e MARIADB_DATABASE=test mariadb:11.8.9 --mhnsw-max-cache-size=536870912
```

**The table.** `t (id INT UNSIGNED PRIMARY KEY, w INT NOT NULL, v VECTOR(384) NOT NULL, VECTOR INDEX (v) M=16 DISTANCE=cosine)`, and `qv` with one query vector.
- **Generated data:** [1_setup_80k.sql](1_setup_80k.sql) and [1_setup_160k.sql](1_setup_160k.sql) build both tables in SQL. Each vector component is 32 bits of `MD5('<row>:<dimension>')`, so the data is the same on every run.
- **Real data:** [load_real.py](load_real.py) loads the first rows of the S1 vectors (`vectors/part_*.npy` of the release file) and [query_vector.json](query_vector.json), one claim's query vector.
- **w** is a permutation of 0 to n - 1, so exactly 5 rows have `w < 5`.

**The script.** [2_query.sql](2_query.sql), with `mhnsw_ef_search` 100 and `max_statement_time` 60:
1. a plain search, `LIMIT` 10;
2. a distance threshold halfway between the 3rd and 4th exact distance, so 3 rows qualify, answered without the index (`IGNORE INDEX`);
3. the same statement through the index, three times;
4. `WHERE w < 5`, without and through the index.

`SHOW SESSION STATUS LIKE 'Max_memory_used'` after each step gives the connection's peak memory so far (MB = 10^6 bytes).

**The procedure.** Build the table, restart the server, then run the script twice, each time in a new connection, statement by statement from outside the container. The files in [outputs/](outputs/) hold every statement with its time and result, then the run's time and the container's peak memory (`memory.current`). A killed run ends with the kernel's line.

The CPU was limited to half its clock speed, so the times are longer than on an unthrottled machine. The memory figures do not depend on that.

## Generated vectors

**160,000 rows** (run 1 / run 2):

| Step | 11.8.9 | 12.3.3 |
|---|---|---|
| 1. plain search | 0.13 MB | 0.17 / 0.20 MB |
| 3. threshold through the index, first execution | 20.7 / 21.3 s, 719 / 642 MB, 1 of the 3 rows | 14.2 / 1.7 s, 150 / 93 MB, 0 / 1 of the 3 rows |
| 3. second execution | connection lost after 22.4 / 22.5 s: the kernel killed `mariadbd` at the container's 2 GiB | 1.9 / 1.7 s |
| container peak | 2,041 / 2,040 MiB | 528 / 483 MiB |

For run 2 on 11.8.9 the server had just been started again after the kill. The 60 s time limit did not prevent the kill: the memory ran out first.

**80,000 rows** (run 1 / run 2):

| Step | 11.8.9 | 12.3.3 |
|---|---|---|
| 1. plain search | 0.13 / 0.17 MB | 0.17 MB |
| 3. threshold through the index, peak | 564 MB after the first execution, then 709 MB / 709 MB | 133 / 101 MB |
| 3. time per execution | 7.5 to 9.8 s | 5.0 s for the first, then 1.2 to 1.5 s |
| 3. rows returned (3 qualify) | 0 | 0 |
| 4. `w < 5` through the index (5 pass) | 0 rows, 7.6 / 7.5 s | 0 rows, 1.3 / 1.1 s |
| container peak | 974 MiB | 418 / 389 MiB |

After each script `Memory_used` was back at 0.1 MB: the memory belongs to the statement and is given back when it ends.

## Real vectors

**80,000 rows** (run 1 / run 2):

| Step | 11.8.9 | 12.3.3 | 13.0.2 |
|---|---|---|---|
| 1. plain search | 0.13 / 0.17 MB | 0.13 MB | 0.13 MB |
| 3. threshold through the index, peak | 224 MB after the first execution, then 984 MB / 984 MB | 84 / 58 MB | 42 / 42 MB |
| 3. time per execution | 4.7 s for the first, then 10.3 to 11.2 s | 3.2 s for the first, then 0.5 to 1.2 s | 2.5 s for the first, then 0.3 to 0.7 s |
| 3. rows returned (3 qualify) | 3 | 3 | 3 |
| 4. `w < 5` through the index (5 pass) | 3 rows | 2 rows | 0 rows |
| container peak | 1,228 / 1,237 MiB | 380 MiB | 335 MiB |

**132,359 rows, 11.8.9.** The first threshold execution returned its 3 rows in 11.5 / 10.3 s with 454 / 361 MB. The second lost the connection after 24.4 / 24.7 s: `mariadbd` was killed at the 2 GiB in both runs.

## What this adds to the version probe

- **The connection's own memory.** The growth shows in `Max_memory_used`, which by MariaDB's source counts the connection's own allocations and not the shared index cache, and it is gone when the statement ends.
- **No data file is needed.** Vectors generated in SQL show it, and at 160,000 rows they end in a kill.
- **12.3.3 and 13.0.2 are not flat.** They took 42 to 150 MB for a statement whose plain form takes 0.13 to 0.20 MB.
- **A 60 s time limit is too long** for a 2 GiB server at these sizes. The version probe's 10 s limit was short enough.
- **Not tested:** 11.8.10 and 12.3.4, released on 2026-11-16, had no Docker image yet. 13.0.2 ran the real 80,000 rows only. Each table had one query vector.
