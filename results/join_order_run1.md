# The filtered statement with its join order fixed, and left to the optimizer

Generated 2026-11-11T08:17:13+00:00 by `scripts/measure_index_behaviour.py join_order` on MariaDB 11.8.9-MariaDB-ubu2404: 4,598 chunks, index M = 16, the 75 corpus claims as queries, k = 10, `mhnsw_ef_search` 100. "As written" is the statement `wikilense query` runs: `STRAIGHT_JOIN`, `chunk` first. "Free" is the same text with every `STRAIGHT_JOIN` replaced by `JOIN`, so the optimizer picks the order. "Plan" is the first table of `EXPLAIN` and how `chunk` is read; `index embedding` is the vector index. "Rows read" is the median over the claims of the `chunk` rows read (`r_loops` times `r_rows` in `ANALYZE FORMAT=JSON`). "Same rows / order" counts the claims, of 75, whose hit list holds the same chunks, and the same chunks in the same order, as the same statement under `IGNORE INDEX (embedding)`, which reads and sorts every chunk. "p50 ms" is the median time of the statement and the fetch of its rows (without the Python sort that `wikilense` adds) when the form runs alone: one untimed pass over the claims, then 3 timed passes. In brackets is the same median when the two forms take turns query by query. Taking turns guards against the machine drifting between two blocks, but a form that reads thousands of rows slows down the other form's next run, so the bracketed time of the faster form is the larger one.

| Filter | Chunks passing | As written: plan | Rows read | p50 ms (taking turns) | Same rows / order | Free: plan | Rows read | p50 ms (taking turns) | Same rows / order |
|---|---|---|---|---|---|---|---|---|---|
| none | 4,598 | chunk first; chunk index embedding | 10 | 2.16 (2.14) | 75 / 75 | chunk first; chunk index embedding | 10 | 2.14 (2.16) | 75 / 75 |
| min_words_1000 | 4,544 | chunk first; chunk index embedding | 10 | 2.60 (2.64) | 75 / 75 | chunk first; chunk index embedding | 10 | 2.35 (2.64) | 75 / 75 |
| min_words_5000 | 3,891 | chunk first; chunk index embedding | 17 | 2.58 (2.94) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 3891 | 10.61 (10.79) | 75 / 75 |
| max_words_2000 | 89 | chunk first; chunk index embedding | 601 | 3.30 (3.11) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 89 | 0.77 (0.95) | 75 / 75 |
| max_words_500 | 25 | chunk first; chunk ALL no key | 4598 | 4.94 (4.95) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 25 | 0.54 (0.90) | 75 / 75 |
| heading_history | 117 | chunk first; chunk index embedding | 315 | 3.05 (2.98) | 75 / 75 | chunk first; chunk index embedding | 315 | 3.10 (2.97) | 75 / 75 |
| linked_from | 661 | chunk first; chunk index embedding | 121 | 2.83 (2.88) | 75 / 75 | <derived2> first; chunk ref uq_chunk_page_ordinal | 661 | 2.27 (2.41) | 75 / 75 |
| category | 695 | chunk first; chunk index embedding | 108 | 2.91 (2.98) | 75 / 75 | <derived2> first; chunk ref uq_chunk_page_ordinal | 695 | 2.48 (2.55) | 75 / 75 |
| edited_since_2020_12_01 | 4,525 | chunk first; chunk index embedding | 10 | 2.37 (2.49) | 75 / 75 | chunk first; chunk index embedding | 10 | 2.44 (2.47) | 75 / 75 |
| edited_before_2020_12_15 | 778 | chunk first; chunk index embedding | 67 | 2.76 (2.88) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 778 | 2.74 (2.82) | 75 / 75 |
| edited_before_2020_06_01 | 4 | chunk first; chunk ALL no key | 4598 | 3.23 (3.25) | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 4 | 0.52 (0.72) | 75 / 75 |
| four_filters | 377 | chunk first; chunk index embedding | 184 | 3.35 (3.60) | 75 / 73 | <derived2> first; chunk ref uq_chunk_page_ordinal | 377 | 1.83 (2.17) | 75 / 75 |

How to read it:

- As written, `chunk` is always the first table. Under 10 of the 12 filters the server walks the vector index; under the other 2 it reads every chunk and sorts.
- Left free, the optimizer starts from another table under 8 of the 12 filters: `page`, or the derived table of the link or category filter. It then reads `chunk` through `uq_chunk_page_ordinal`, as many rows as pass the filter, and the vector index is not in that plan. The JSON file holds every `EXPLAIN` row of the first claim for both forms (`explain_first_claim`).
- The plan moved and the median got faster by more than a factor of 1.5, as written to free: `max_words_2000` 3.30 to 0.77 ms, `max_words_500` 4.94 to 0.54 ms, `edited_before_2020_06_01` 3.23 to 0.52 ms, `four_filters` 3.35 to 1.83 ms.
- The plan moved and it got slower by more than that: `min_words_5000` 2.58 to 10.61 ms.
- The plan moved and the time stayed within that factor: `linked_from` 2.83 to 2.27 ms, `category` 2.91 to 2.48 ms, `edited_before_2020_12_15` 2.76 to 2.74 ms.
- Same chunks in another order than the scan. The gap is the largest difference between the distances the two lists report at a position where they name different chunks (`four_filters` as written: 2 of 75 claims, 2 of them with rows not in ascending order of the distance the statement reports, largest gap 3.9e-06). `wikilense` sorts the rows by the reported distance and the chunk id afterwards.

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
