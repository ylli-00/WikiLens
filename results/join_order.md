# The filtered statement with its join order fixed, and left to the optimizer

Generated 2026-11-11T08:26:05+00:00 by `scripts/measure_index_behaviour.py join_order` on MariaDB 11.8.9-MariaDB-ubu2404: 4,598 chunks, index M = 16, the 75 corpus claims as queries, k = 10, `mhnsw_ef_search` 100. "As written" is the statement `wikilense query` runs: `STRAIGHT_JOIN`, `chunk` first. "Free" is the same text with every `STRAIGHT_JOIN` replaced by `JOIN`, so the optimizer picks the order. "Plan" is the first table of `EXPLAIN` and how `chunk` is read; `index embedding` is the vector index. "Rows read" is the median over the claims of the `chunk` rows read (`r_loops` times `r_rows` in `ANALYZE FORMAT=JSON`). "Same rows / order" counts the claims, of 75, whose hit list holds the same chunks, and the same chunks in the same order, as the same statement under `IGNORE INDEX (embedding)`, which reads and sorts every chunk. "p50 ms" is the median time of the statement and the fetch of its rows (without the Python sort that `wikilense` adds) when the form runs alone: one untimed pass over the claims, then 3 timed passes. In brackets is the same median when the two forms take turns query by query. Taking turns guards against the machine drifting between two blocks, but a form that reads thousands of rows slows down the other form's next run, so the bracketed time of the faster form is the larger one.

| Filter | Chunks passing | As written: plan | Rows read | p50 ms (taking turns) | Same rows / order | Free: plan | Rows read | p50 ms (taking turns) | Same rows / order |
|---|---|---|---|---|---|---|---|---|---|
| none | 4,598 | chunk first; chunk index embedding | 10 | 2.11 (2.19) | 75 / 75 | chunk first; chunk index embedding | 10 | 2.09 (2.19) | 75 / 75 |
| min_words_1000 | 4,544 | chunk first; chunk index embedding | 10 | 2.08 (2.09) | 75 / 75 | chunk first; chunk index embedding | 10 | 2.31 (2.10) | 75 / 75 |
| min_words_5000 | 3,891 | chunk first; chunk index embedding | 17 | 2.11 (2.61) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 3891 | 10.24 (10.37) | 75 / 75 |
| max_words_2000 | 89 | chunk first; chunk index embedding | 601 | 3.29 (3.20) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 89 | 0.77 (0.96) | 75 / 75 |
| max_words_500 | 25 | chunk first; chunk ALL no key | 4598 | 4.76 (4.72) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 25 | 0.50 (0.73) | 75 / 75 |
| heading_history | 117 | chunk first; chunk index embedding | 315 | 2.99 (3.02) | 75 / 75 | chunk first; chunk index embedding | 315 | 2.93 (3.03) | 75 / 75 |
| linked_from | 661 | chunk first; chunk index embedding | 121 | 3.13 (3.17) | 75 / 75 | <derived2> first; chunk ref uq_chunk_page_ordinal | 661 | 2.58 (2.66) | 75 / 75 |
| category | 695 | chunk first; chunk index embedding | 108 | 2.41 (3.09) | 75 / 75 | <derived2> first; chunk ref uq_chunk_page_ordinal | 695 | 2.34 (2.62) | 75 / 75 |
| edited_since_2020_12_01 | 4,525 | chunk first; chunk index embedding | 10 | 2.54 (2.60) | 75 / 75 | chunk first; chunk index embedding | 10 | 2.61 (2.60) | 75 / 75 |
| edited_before_2020_12_15 | 778 | chunk first; chunk index embedding | 67 | 2.86 (2.81) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 778 | 2.63 (2.79) | 75 / 75 |
| edited_before_2020_06_01 | 4 | chunk first; chunk ALL no key | 4598 | 3.59 (3.31) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 4 | 1.09 (0.84) | 75 / 75 |
| four_filters | 377 | chunk first; chunk index embedding | 184 | 3.36 (3.47) | 75 / 73 | <derived2> first; chunk ref uq_chunk_page_ordinal | 377 | 2.07 (2.12) | 75 / 75 |

How to read it:

- As written, `chunk` is always the first table. Under 10 of the 12 filters the server walks the vector index; under the other 2 it reads every chunk and sorts.
- Left free, the optimizer starts from another table under 8 of the 12 filters: `page`, or the derived table of the link or category filter. It then reads `chunk` through `uq_chunk_page_ordinal`, as many rows as pass the filter, and the vector index is not in that plan. The JSON file holds every `EXPLAIN` row of the first claim for both forms (`explain_first_claim`).
- The plan moved and the median got faster by more than a factor of 1.5, as written to free: `max_words_2000` 3.29 to 0.77 ms, `max_words_500` 4.76 to 0.50 ms, `edited_before_2020_06_01` 3.59 to 1.09 ms, `four_filters` 3.36 to 2.07 ms.
- The plan moved and it got slower by more than that: `min_words_5000` 2.11 to 10.24 ms.
- The plan moved and the time stayed within that factor: `linked_from` 3.13 to 2.58 ms, `category` 2.41 to 2.34 ms, `edited_before_2020_12_15` 2.86 to 2.63 ms.
- Same chunks in another order than the scan. The gap is the largest difference between the distances the two lists report at a position where they name different chunks (`four_filters` as written: 2 of 75 claims, 2 of them with rows not in ascending order of the distance the statement reports, largest gap 3.9e-06). `wikilense` sorts the rows by the reported distance and the chunk id afterwards.

## The derived tables without DISTINCT

The link and category filters join a `SELECT DISTINCT` derived table. The same statement as written, with every `SELECT DISTINCT` turned into `SELECT`, for the 75 claims: the plan, how many plans still hold a materialised derived table, and how many hit lists name the same chunk more than once.

| Filter | Plan | Plans with a derived table | Lists with a chunk twice |
|---|---|---|---|
| linked_from | chunk first; chunk index embedding (75/75) | 0/75 | 75/75 |
| category | chunk first; chunk index embedding (75/75) | 0/75 | 0/75 |
| four_filters | chunk first; chunk index embedding (75/75) | 0/75 | 75/75 |

Without `DISTINCT` MariaDB merges the derived table into the join. Whether the vector index survives is in the plan column; a list with a chunk twice means the merged join returned one chunk once per matching row of the filter table.

Filters:

- `none`: {}
- `min_words_1000`: {"min_words": 1000}
- `min_words_5000`: {"min_words": 5000}
- `max_words_2000`: {"max_words": 2000}
- `max_words_500`: {"max_words": 500}
- `heading_history`: {"heading_like": "%History%"}
- `linked_from`: {"linked_from": "Asia"}
- `category`: {"category": "Member states of the United Nations"}
- `edited_since_2020_12_01`: {"edited_since": "2020-12-01"}
- `edited_before_2020_12_15`: {"edited_before": "2020-12-15"}
- `edited_before_2020_06_01`: {"edited_before": "2020-06-01"}
- `four_filters`: {"min_words": 1000, "linked_from": "Asia", "category": "Member states of the United Nations", "edited_since": "2020-12-01"}
