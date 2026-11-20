# The filtered statement with its join order fixed, and left to the optimizer

Generated 2026-11-20T06:25:30+00:00 by `scripts/measure_index_behaviour.py join_order` on MariaDB 11.8.9-MariaDB-ubu2404: 132,359 chunks, index M = 16, a random sample of 200 of the 3,000 corpus claims as queries (`random.Random(20261119)`), k = 5, `mhnsw_ef_search` 100. "As written" is the statement `wikilense query` runs: `STRAIGHT_JOIN`, `chunk` first. "Free" is the same text with every `STRAIGHT_JOIN` replaced by `JOIN`, so the optimizer picks the order. "Plan" is the first table of `EXPLAIN` and how `chunk` is read; `index embedding` is the vector index. "Rows read" is the median over the claims of the `chunk` rows read (`r_loops` times `r_rows` in `ANALYZE FORMAT=JSON`). "Same rows / order" counts the claims, of 200, whose hit list holds the same chunks, and the same chunks in the same order, as the same statement under `IGNORE INDEX (embedding)`, which reads and sorts every chunk. "p50 ms" is the median time of the statement and the fetch of its rows (without the Python sort that `wikilense` adds) when the form runs alone: one untimed pass over the claims, then 1 timed passes. Each form was timed alone only, the two forms one after the other.

| Filter | Chunks passing | As written: plan | Rows read | p50 ms | Same rows / order | Free: plan | Rows read | p50 ms | Same rows / order |
|---|---|---|---|---|---|---|---|---|---|
| none | 132,359 | chunk first; chunk index embedding | 5 | 26.77 | 200 / 200 | chunk first; chunk index embedding | 5 | 26.65 | 200 / 200 |
| length_50pct | 66,171 | chunk first; chunk index embedding | 17 | 26.68 | 200 / 199 | page first; chunk ref uq_chunk_page_ordinal | 66171 | 1566.76 | 200 / 200 |
| length_5pct | 6,616 | chunk first; chunk index embedding | 158 | 37.09 | 200 / 199 | page first; chunk ref uq_chunk_page_ordinal | 6616 | 26.39 | 200 / 200 |
| length_0.5pct | 626 | chunk first; chunk ALL no key | 132359 | 359.63 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 626 | 2.82 | 200 / 200 |
| length_0.05pct | 139 | chunk first; chunk ALL no key | 132359 | 335.33 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 139 | 1.06 | 200 / 200 |
| edited_50pct | 66,208 | chunk first; chunk index embedding | 13 | 26.40 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 66208 | 1689.04 | 200 / 200 |
| edited_5pct | 6,599 | chunk first; chunk index embedding | 132 | 34.41 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 6599 | 25.84 | 200 / 200 |
| edited_0.5pct | 660 | chunk first; chunk index embedding | 1924 | 129.95 | 200 / 199 | page first; chunk ref uq_chunk_page_ordinal | 660 | 3.04 | 200 / 200 |
| edited_0.05pct | 77 | chunk first; chunk ALL no key | 132359 | 336.89 | 200 / 200 | page first; chunk ref uq_chunk_page_ordinal | 77 | 0.86 | 200 / 200 |
| category_50pct | 9,267 | chunk first; chunk index embedding | 154 | 38.78 | 197 / 197 | <derived2> first; chunk ref uq_chunk_page_ordinal | 9267 | 36.12 | 200 / 200 |
| category_5pct | 7,313 | chunk first; chunk index embedding | 202 | 40.27 | 200 / 200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 7313 | 29.64 | 200 / 200 |
| category_0.5pct | 660 | chunk first; chunk index embedding | 6148 | 323.29 | 175 / 169 | <derived2> first; chunk ref uq_chunk_page_ordinal | 660 | 2.95 | 200 / 200 |
| category_0.05pct | 66 | chunk first; chunk index embedding | 19656 | 1387.73 | 198 / 193 | <derived2> first; chunk ref uq_chunk_page_ordinal | 66 | 0.99 | 200 / 200 |
| linked_50pct | 8,203 | chunk first; chunk index embedding | 170 | 40.06 | 200 / 200 | <derived2> first; chunk ref uq_chunk_page_ordinal | 8203 | 34.13 | 200 / 200 |
| linked_5pct | 6,698 | chunk first; chunk index embedding | 121 | 38.15 | 200 / 199 | <derived2> first; chunk ref uq_chunk_page_ordinal | 6698 | 27.97 | 200 / 200 |
| linked_0.5pct | 662 | chunk first; chunk index embedding | 1888 | 142.39 | 193 / 193 | <derived2> first; chunk ref uq_chunk_page_ordinal | 662 | 3.13 | 200 / 200 |
| linked_0.05pct | 66 | chunk first; chunk index embedding | 19656 | 1385.72 | 198 / 193 | <derived2> first; chunk ref uq_chunk_page_ordinal | 66 | 0.96 | 200 / 200 |
| four_filters | 3,107 | chunk first; chunk index embedding | 440 | 54.47 | 197 / 197 | <derived3> first; chunk ref uq_chunk_page_ordinal | 3107 | 12.69 | 200 / 200 |

How to read it:

- As written, `chunk` is always the first table. Under 15 of the 18 filters the server walks the vector index; under the other 3 it reads every chunk and sorts.
- Left free, the optimizer starts from another table under 17 of the 18 filters: `page`, or the derived table of the link or category filter. It then reads `chunk` through `uq_chunk_page_ordinal`, as many rows as pass the filter, and the vector index is not in that plan. The JSON file holds every `EXPLAIN` row of the first claim for both forms (`explain_first_claim`).
- The plan moved and the median got faster by more than a factor of 1.5, as written to free: `length_0.5pct` 359.63 to 2.82 ms, `length_0.05pct` 335.33 to 1.06 ms, `edited_0.5pct` 129.95 to 3.04 ms, `edited_0.05pct` 336.89 to 0.86 ms, `category_0.5pct` 323.29 to 2.95 ms, `category_0.05pct` 1387.73 to 0.99 ms, `linked_0.5pct` 142.39 to 3.13 ms, `linked_0.05pct` 1385.72 to 0.96 ms, `four_filters` 54.47 to 12.69 ms.
- The plan moved and it got slower by more than that: `length_50pct` 26.68 to 1566.76 ms, `edited_50pct` 26.40 to 1689.04 ms.
- The plan moved and the time stayed within that factor: `length_5pct` 37.09 to 26.39 ms, `edited_5pct` 34.41 to 25.84 ms, `category_50pct` 38.78 to 36.12 ms, `category_5pct` 40.27 to 29.64 ms, `linked_50pct` 40.06 to 34.13 ms, `linked_5pct` 38.15 to 27.97 ms.
- Same chunks in another order than the scan. The gap is the largest difference between the distances the two lists report at a position where they name different chunks (`length_50pct` as written: 1 of 200 claims, 1 of them with rows not in ascending order of the distance the statement reports, largest gap 2.0e-06; `length_5pct` as written: 1 of 200 claims, 1 of them with rows not in ascending order of the distance the statement reports, largest gap 6.7e-07; `edited_0.5pct` as written: 1 of 200 claims, 1 of them with rows not in ascending order of the distance the statement reports, largest gap 4.5e-06; `category_0.5pct` as written: 6 of 200 claims, 7 of them with rows not in ascending order of the distance the statement reports, largest gap 2.6e-02; `category_0.05pct` as written: 5 of 200 claims, 5 of them with rows not in ascending order of the distance the statement reports, largest gap 9.4e-03; `linked_5pct` as written: 1 of 200 claims, 1 of them with rows not in ascending order of the distance the statement reports, largest gap 2.0e-06; `linked_0.05pct` as written: 5 of 200 claims, 5 of them with rows not in ascending order of the distance the statement reports, largest gap 9.4e-03). `wikilense` sorts the rows by the reported distance and the chunk id afterwards.

## The derived tables without DISTINCT

The link and category filters join a `SELECT DISTINCT` derived table. The same statement as written, with every `SELECT DISTINCT` turned into `SELECT`, for the 200 claims: the plan, how many plans still hold a materialised derived table, and how many hit lists name the same chunk more than once.

| Filter | Plan | Plans with a derived table | Lists with a chunk twice |
|---|---|---|---|
| category_50pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| category_5pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| category_0.5pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| category_0.05pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| linked_50pct | chunk first; chunk index embedding (200/200) | 0/200 | 24/200 |
| linked_5pct | chunk first; chunk index embedding (200/200) | 0/200 | 135/200 |
| linked_0.5pct | chunk first; chunk index embedding (200/200) | 0/200 | 138/200 |
| linked_0.05pct | chunk first; chunk index embedding (200/200) | 0/200 | 0/200 |
| four_filters | chunk first; chunk index embedding (200/200) | 0/200 | 147/200 |

Without `DISTINCT` MariaDB merges the derived table into the join. Whether the vector index survives is in the plan column; a list with a chunk twice means the merged join returned one chunk once per matching row of the filter table.

Filters:

- `none`: {}
- `length_50pct`: {"min_words": 5214}
- `length_5pct`: {"min_words": 14929}
- `length_0.5pct`: {"min_words": 20112}
- `length_0.05pct`: {"min_words": 23132}
- `edited_50pct`: {"edited_since": "2020-12-21 15:34:08"}
- `edited_5pct`: {"edited_since": "2020-12-31 08:42:11"}
- `edited_0.5pct`: {"edited_since": "2020-12-31 22:30:26"}
- `edited_0.05pct`: {"edited_since": "2020-12-31 23:59:17"}
- `category_50pct`: {"category": "Living people"}
- `category_5pct`: {"category": "Member states of the United Nations"}
- `category_0.5pct`: {"category": "American male film actors"}
- `category_0.05pct`: {"category": "1821 establishments in North America"}
- `linked_50pct`: {"linked_from": "List of sovereign states"}
- `linked_5pct`: {"linked_from": "United Kingdom"}
- `linked_0.5pct`: {"linked_from": "Jean-Paul Sartre"}
- `linked_0.05pct`: {"linked_from": "1930 Nicaraguan parliamentary election"}
- `four_filters`: {"min_words": 5214, "linked_from": "Legality of cannabis", "category": "Member states of the United Nations", "edited_since": "2020-12-21 15:34:08"}
