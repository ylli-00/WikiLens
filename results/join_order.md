# The filtered statement with its join order fixed, and left to the optimizer

Generated 2026-11-11T04:29:29+00:00 by `scripts/measure_index_behaviour.py join_order` on MariaDB 11.8.9-MariaDB-ubu2404: 4,598 chunks, index M = 16, the 75 corpus claims as queries, k = 10, `mhnsw_ef_search` 100, 3 timed passes after one untimed pass, the two forms taking turns query by query. "As written" is the statement `wikilense query` runs: `STRAIGHT_JOIN`, `chunk` first. "Free" is the same text with every `STRAIGHT_JOIN` replaced by `JOIN`, so the optimizer picks the order. "Plan" is the first table of `EXPLAIN` and how `chunk` is read; `index embedding` is the vector index. "Rows read" is the median over the claims of the `chunk` rows read (`r_loops` times `r_rows` in `ANALYZE FORMAT=JSON`). "Same rows / order" counts the claims, of 75, whose hit list holds the same chunks, and the same chunks in the same order, as the same statement under `IGNORE INDEX (embedding)`, which reads and sorts every chunk. The times are the statement and the fetch of its rows, without the Python sort that `wikilense` adds.

| Filter | Chunks passing | As written: plan | Rows read | p50 ms | Same rows / order | Free: plan | Rows read | p50 ms | Same rows / order |
|---|---|---|---|---|---|---|---|---|---|
| none | 4,598 | chunk first; chunk index embedding | 10 | 2.05 | 75 / 75 | chunk first; chunk index embedding | 10 | 2.04 | 75 / 75 |
| min_words_1000 | 4,544 | chunk first; chunk index embedding | 10 | 2.18 | 75 / 75 | chunk first; chunk index embedding | 10 | 2.13 | 75 / 75 |
| min_words_5000 | 3,891 | chunk first; chunk index embedding | 17 | 2.92 | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 3891 | 10.44 | 75 / 75 |
| max_words_2000 | 89 | chunk first; chunk index embedding | 601 | 3.06 | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 89 | 0.91 | 75 / 75 |
| max_words_500 | 25 | chunk first; chunk ALL no key | 4598 | 4.59 | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 25 | 0.63 | 75 / 75 |
| heading_history | 117 | chunk first; chunk index embedding | 315 | 2.94 | 75 / 75 | chunk first; chunk index embedding | 315 | 2.94 | 75 / 75 |
| linked_from | 661 | chunk first; chunk index embedding | 121 | 3.12 | 75 / 75 | <derived2> first; chunk ref uq_chunk_page_ordinal | 661 | 2.58 | 75 / 75 |
| category | 695 | chunk first; chunk index embedding | 108 | 3.00 | 75 / 75 | <derived2> first; chunk ref uq_chunk_page_ordinal | 695 | 2.58 | 75 / 75 |
| edited_since_2020_12_01 | 4,525 | chunk first; chunk index embedding | 10 | 2.52 | 75 / 75 | chunk first; chunk index embedding | 10 | 2.54 | 75 / 75 |
| edited_before_2020_12_15 | 778 | chunk first; chunk index embedding | 67 | 2.68 | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 778 | 2.64 | 75 / 75 |
| edited_before_2020_06_01 | 4 | chunk first; chunk ALL no key | 4598 | 3.15 | 75 / 75 | page first; chunk ref uq_chunk_page_ordinal | 4 | 0.67 | 75 / 75 |
| four_filters | 377 | chunk first; chunk index embedding | 184 | 3.42 | 75 / 73 | <derived2> first; chunk ref uq_chunk_page_ordinal | 377 | 2.10 | 75 / 75 |

How to read it:

- As written, `chunk` is always the first table. Under 10 of the 12 filters the server walks the vector index; under the other 2 it reads every chunk and sorts.
- Left free, the optimizer starts from another table under 8 of the 12 filters: the filter's own index or derived table. It then reads only the chunks that pass, through `uq_chunk_page_ordinal`, and sorts them. The vector index is not used in that plan.
- The plan moved and the median got faster by more than a factor of 1.25, as written to free: `max_words_2000` 3.06 to 0.91 ms, `max_words_500` 4.59 to 0.63 ms, `edited_before_2020_06_01` 3.15 to 0.67 ms, `four_filters` 3.42 to 2.10 ms.
- The plan moved and it got slower by more than that: `min_words_5000` 2.92 to 10.44 ms.
- The plan moved and the time stayed within that factor: `linked_from` 3.12 to 2.58 ms, `category` 3.00 to 2.58 ms, `edited_before_2020_12_15` 2.68 to 2.64 ms.
- Same chunks in another order: the server returned two neighbours the other way round than the distances it reports for them, which differ by the gap given here (`four_filters` as written: 2 of 75 claims, largest gap 3.9e-06). `wikilense` sorts the rows by the reported distance and the chunk id afterwards.

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
