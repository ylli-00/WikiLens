# The filtered statement with its join order fixed, and left to the optimizer

Generated 2026-11-20T03:51:38+00:00 by `scripts/measure_index_behaviour.py join_order` on MariaDB 11.8.9-MariaDB-ubu2404: 43,134 chunks, index M = 16, a random sample of 200 of the 3,000 corpus claims as queries (`random.Random(20261119)`), k = 5, `mhnsw_ef_search` 100. "As written" is the statement `wikilense query` runs: `STRAIGHT_JOIN`, `chunk` first. "Free" is the same text with every `STRAIGHT_JOIN` replaced by `JOIN`, so the optimizer picks the order. "Plan" is the first table of `EXPLAIN` and how `chunk` is read; `index embedding` is the vector index. "Rows read" is the median over the claims of the `chunk` rows read (`r_loops` times `r_rows` in `ANALYZE FORMAT=JSON`). "Same rows / order" counts the claims, of 200, whose hit list holds the same chunks, and the same chunks in the same order, as the same statement under `IGNORE INDEX (embedding)`, which reads and sorts every chunk. "p50 ms" is the median time of the statement and the fetch of its rows (without the Python sort that `wikilense` adds) when the form runs alone: one untimed pass over the claims, then 1 timed passes. Each form was timed alone only, the two forms one after the other.

| Filter | Chunks passing | As written: plan | Rows read | p50 ms | Same rows / order | Free: plan | Rows read | p50 ms | Same rows / order |
|---|---|---|---|---|---|---|---|---|---|
| none | 43,134 | chunk first; chunk index embedding | 5 | 16.16 | 200 / 200 | chunk first; chunk index embedding | 5 | 16.11 | 200 / 200 |
| length_50pct | 21,570 | chunk first; chunk index embedding | 16 | 16.50 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 21570 | 98.72 | 200 / 200 |
| length_5pct | 2,166 | chunk first; chunk index embedding | 168 | 16.94 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 2166 | 8.46 | 200 / 200 |
| length_0.5pct | 256 | chunk first; chunk ALL no key | 43134 | 307.35 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 256 | 1.51 | 200 / 200 |
| length_0.05pct | 139 | chunk first; chunk ALL no key | 43134 | 296.55 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 139 | 1.07 | 200 / 200 |
| edited_50pct | 21,578 | chunk first; chunk index embedding | 13 | 16.33 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 21578 | 91.69 | 200 / 200 |
| edited_5pct | 2,124 | chunk first; chunk index embedding | 130 | 16.96 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 2124 | 8.20 | 200 / 200 |
| edited_0.5pct | 219 | chunk first; chunk index embedding | 1552 | 24.97 | 198 / 198 | page first; chunk ref uq_chunk_page_ordinal | 219 | 1.42 | 200 / 200 |
| edited_0.05pct | 34 | chunk first; chunk ALL no key | 43134 | 275.00 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 34 | 0.72 | 200 / 200 |
| category_50pct | 5,081 | chunk first; chunk index embedding | 84 | 17.21 | 199 / 199 | <derived2> first; chunk ref uq_chunk_page_ordinal | 5081 | 19.83 | 200 / 200 |
| category_5pct | 1,857 | chunk first; chunk index embedding | 174 | 18.01 | 200 / 200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 1857 | 7.00 | 200 / 200 |
| category_0.5pct | 215 | chunk first; chunk index embedding | 4482 | 52.76 | 192 / 189 | <derived2> first; chunk ref uq_chunk_page_ordinal | 215 | 1.39 | 200 / 200 |
| category_0.05pct | 22 | chunk first; chunk index embedding | 12133 | 145.22 | 200 / 200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 22 | 0.82 | 200 / 200 |
| linked_50pct | 1,629 | chunk first; chunk index embedding | 212 | 19.34 | 199 / 198 | <derived2> first; chunk ref uq_chunk_page_ordinal | 1629 | 6.57 | 200 / 200 |
| linked_0.5pct | 216 | chunk first; chunk index embedding | 2098 | 36.33 | 198 / 198 | <derived2> first; chunk ref uq_chunk_page_ordinal | 216 | 1.51 | 200 / 200 |
| linked_0.05pct | 22 | chunk first; chunk index embedding | 13358 | 164.88 | 200 / 200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 22 | 0.91 | 200 / 200 |
| four_filters | 824 | chunk first; chunk index embedding | 446 | 21.45 | 199 / 198 | <derived3> first; chunk ref uq_chunk_page_ordinal | 824 | 3.94 | 200 / 200 |

How to read it:

- As written, `chunk` is always the first table. Under 14 of the 17 filters the server walks the vector index; under the other 3 it reads every chunk and sorts.
- Left free, the optimizer starts from another table under 16 of the 17 filters: `page`, or the derived table of the link or category filter. It then reads `chunk` through `uq_chunk_page_ordinal`, as many rows as pass the filter, and the vector index is not in that plan. The JSON file holds every `EXPLAIN` row of the first claim for both forms (`explain_first_claim`).
- The plan moved and the median got faster by more than a factor of 1.5, as written to free: `length_5pct` 16.94 to 8.46 ms, `length_0.5pct` 307.35 to 1.51 ms, `length_0.05pct` 296.55 to 1.07 ms, `edited_5pct` 16.96 to 8.20 ms, `edited_0.5pct` 24.97 to 1.42 ms, `edited_0.05pct` 275.00 to 0.72 ms, `category_5pct` 18.01 to 7.00 ms, `category_0.5pct` 52.76 to 1.39 ms, `category_0.05pct` 145.22 to 0.82 ms, `linked_50pct` 19.34 to 6.57 ms, `linked_0.5pct` 36.33 to 1.51 ms, `linked_0.05pct` 164.88 to 0.91 ms, `four_filters` 21.45 to 3.94 ms.
- The plan moved and it got slower by more than that: `length_50pct` 16.50 to 98.72 ms, `edited_50pct` 16.33 to 91.69 ms.
- The plan moved and the time stayed within that factor: `category_50pct` 17.21 to 19.83 ms.
- Same chunks in another order than the scan. The gap is the largest difference between the distances the two lists report at a position where they name different chunks (`category_0.5pct` as written: 3 of 200 claims, 4 of them with rows not in ascending order of the distance the statement reports, largest gap 9.2e-03; `linked_50pct` as written: 1 of 200 claims, 1 of them with rows not in ascending order of the distance the statement reports, largest gap 3.2e-06; `four_filters` as written: 1 of 200 claims, 1 of them with rows not in ascending order of the distance the statement reports, largest gap 3.2e-06). `wikilense` sorts the rows by the reported distance and the chunk id afterwards.

## The derived tables without DISTINCT

The link and category filters join a `SELECT DISTINCT` derived table. The same statement as written, with every `SELECT DISTINCT` turned into `SELECT`, for the 200 claims: the plan, how many plans still hold a materialised derived table, and how many hit lists name the same chunk more than once.

| Filter | Plan | Plans with a derived table | Lists with a chunk twice |
|---|---|---|---|
| category_50pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| category_5pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| category_0.5pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| category_0.05pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| linked_50pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| linked_0.5pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| linked_0.05pct | chunk first; chunk index embedding (200/200) | 0/200 | 200/200 |
| four_filters | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |

Without `DISTINCT` MariaDB merges the derived table into the join. Whether the vector index survives is in the plan column; a list with a chunk twice means the merged join returned one chunk once per matching row of the filter table.

Filters:

- `none`: {}
- `length_50pct`: {"min_words": 3359}
- `length_5pct`: {"min_words": 13542}
- `length_0.5pct`: {"min_words": 20112}
- `length_0.05pct`: {"min_words": 23132}
- `edited_50pct`: {"edited_since": "2020-12-18 02:38:12"}
- `edited_5pct`: {"edited_since": "2020-12-31 01:41:51"}
- `edited_0.5pct`: {"edited_since": "2020-12-31 21:03:02"}
- `edited_0.05pct`: {"edited_since": "2020-12-31 23:58:53"}
- `category_50pct`: {"category": "Living people"}
- `category_5pct`: {"category": "Articles containing video clips"}
- `category_0.5pct`: {"category": "American film producers"}
- `category_0.05pct`: {"category": "1608 births"}
- `linked_50pct`: {"linked_from": "List of countries by maternal mortality ratio"}
- `linked_0.5pct`: {"linked_from": "ABC Records"}
- `linked_0.05pct`: {"linked_from": "30th New York State Legislature"}
- `four_filters`: {"min_words": 3359, "linked_from": "List of countries by maternal mortality ratio", "category": "Member states of the United Nations", "edited_since": "2020-12-18 02:38:12"}
