# planner_choices

Created 2026-10-21T22:33:41+00:00 by `scripts/run_experiments.py planner` on MariaDB 11.8.9-MariaDB-ubu2404, 4598 chunks. For every filter below and every claim embedding, the inline statement (`search(..., strategy="inline")`, k 10, ef_search 100) is explained (EXPLAIN: does the chunk row read through the vector index, type `index`, key `embedding`?), analysed (ANALYZE: chunk rows actually read) and timed (3 repeats). "chunks passing" counts the chunks the filter keeps; "short results" the queries that got fewer than min(k, passing) hits.

| filter | chunks passing | plans using the vector index | chunk access (type / key) | chunk rows read, p50 | SQL ms p50 / p95 | short results |
|---|---|---|---|---|---|---|
| none | 4598 | 75/75 | index / embedding | 10 (max 10) | 2.28 / 2.97 | 0 |
| min_words_1000 | 4544 | 75/75 | index / embedding | 10 (max 19) | 2.21 / 3.64 | 0 |
| min_words_5000 | 3891 | 75/75 | index / embedding | 17 (max 75) | 2.26 / 2.91 | 0 |
| max_words_2000 | 89 | 75/75 | index / embedding | 601 (max 1661) | 2.90 / 4.17 | 0 |
| max_words_500 | 25 | 0/75 | ALL / None | 4598 (max 4598) | 5.19 / 7.06 | 0 |
| heading_history | 117 | 75/75 | index / embedding | 315 (max 1015) | 2.99 / 4.13 | 0 |
| linked_from | 661 | 75/75 | index / embedding | 121 (max 492) | 3.08 / 4.03 | 0 |
| category | 695 | 75/75 | index / embedding | 108 (max 420) | 2.80 / 3.37 | 0 |
| edited_since_2020_12_01 | 4525 | 75/75 | index / embedding | 10 (max 25) | 2.41 / 3.41 | 0 |
| edited_before_2020_12_15 | 778 | 75/75 | index / embedding | 67 (max 491) | 2.52 / 3.54 | 0 |
| edited_before_2020_06_01 | 4 | 0/75 | ALL / None | 4598 (max 4598) | 3.10 / 3.99 | 0 |
| four_filters | 377 | 75/75 | index / embedding | 184 (max 888) | 3.14 / 4.24 | 0 |

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
