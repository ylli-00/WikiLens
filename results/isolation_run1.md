# The vector index, isolation levels, and a reader that leaves the index stale

Generated 2026-11-11T04:13:17+00:00 by `scripts/measure_index_behaviour.py isolation` on MariaDB 11.8.9-MariaDB-ubu2404 (default isolation level REPEATABLE-READ, innodb_snapshot_isolation 1). A scratch table in the test database: 1,000 random unit vectors (seed 7), `VECTOR INDEX (embedding) M=16 DISTANCE=cosine`, `mhnsw_ef_search` 100, k = 10. A writer adds 5 rows that are nearer to the query than every other row, so a search that sees them returns all 5 among its top 10. "Index" is the ordinary statement; "scan" is the same statement with `IGNORE INDEX (embedding)`, which reads the table.

## What a reader sees of another connection's new rows

Reader A opens a transaction at the given level and searches. Writer B inserts the 5 rows. Each cell: new rows A finds through the index / by a scan, over 5 trials on fresh tables (one number when every trial agreed).

| A's isolation level | B has not committed | B has committed, A's transaction still open | A's next transaction | B's insert |
|---|---|---|---|---|
| READ UNCOMMITTED | 5 / 5 | 5 / 5 | 5 / 5 | went through |
| READ COMMITTED | 0 / 0 | 5 / 5 | 5 / 5 | went through |
| REPEATABLE READ | 0 / 0 | 0 / 0 | 0 / 5 | went through |
| SERIALIZABLE | 0 / 0 | 0 / 0 | 0 / 0 | blocked: Lock wait timeout exceeded; try restarting transaction (error 1205) |

## The sequence after which the index is stale

1. A, at REPEATABLE READ, opens a transaction and searches the index.
2. B inserts the 5 rows and commits.
3. A searches again, still inside its old snapshot (correctly without the new rows).
4. A commits and disconnects.
5. A new connection searches.

At step 5 the table holds the new rows and a scan returns them. "Stale" below means the index search of the new connection returned fewer of them than the scan.

| Sequence | Stale at step 5 | New rows found through the index, per trial |
|---|---|---|
| as above | 10 of 10 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| control: A is at READ COMMITTED | 0 of 5 | [5, 5, 5, 5, 5] |
| control: A skips step 3 | 0 of 5 | [5, 5, 5, 5, 5] |
| control: A commits but stays connected | 5 of 5 | [0, 0, 0, 0, 0] |

Once stale, it stays stale for later connections: three more new connections found [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] new rows (one list per stale table). What cleared it:

- the next committed write to the table: after one more row was inserted and committed, a new connection found [6, 6, 6] of 6 new rows through the index;
- `FLUSH TABLES isolation_probe` (as root): afterwards a new connection found [5, 5, 5] of 5 new rows through the index.

So the index on disk is right and the copy the server keeps in memory is not.

## The same sequence with a DELETE

B deletes the three rows nearest to the query instead of inserting. A's snapshot keeps seeing them, as it should. Afterwards, for a new connection:

| Trial | Rows the index returns for that query | Rows a scan returns | Of 20 other queries, how many got no row (fewest rows any got) | Rows in the table |
|---|---|---|---|---|
| 1 | 0 | 10 | 0 (6) | 997 |
| 2 | 0 | 10 | 0 (6) | 997 |
| 3 | 0 | 10 | 0 (6) | 997 |
| 4 | 0 | 10 | 0 (6) | 997 |
| 5 | 0 | 10 | 0 (6) | 997 |
| 6 | 0 | 10 | 0 (6) | 997 |
| 7 | 0 | 10 | 0 (6) | 997 |
| 8 | 0 | 10 | 0 (6) | 997 |
| 9 | 0 | 10 | 0 (6) | 997 |
| 10 | 0 | 10 | 0 (6) | 997 |

After `FLUSH TABLES` the index returned 10 rows for that query again.

## What follows for an application

- Search at READ COMMITTED or in autocommit mode, or end the transaction before searching again, when other connections write to the table. PyMySQL opens connections with autocommit off, so a connection that has run one `SELECT` is inside a REPEATABLE READ transaction until it commits, rolls back or closes.
- WikiLense itself is not affected by this in its measurements: nothing writes to `chunk` while a search runs, and `wikilense ingest` recreates the tables.
- We found no MariaDB ticket that describes this (searched on 2026-11-11) and have not reported it. It was measured on one server version only.
