# Filter values on wikilense_s0

Written 2026-11-19T19:24:06+00:00 by `scripts/scaled_filters.py`: 43,134 chunks on 3,851 pages. Rule (the script's docstring has it in full): per filter, the value whose passing chunks are nearest, on a log scale, to 50%, 5%, 0.5%, 0.05% of the chunk table; `four_filters` sets length and edit date at their 50% values and the category and linking page whose pair keeps the most chunks with them.

| Case | Target share | Filter value | Chunks passing | Share |
|---|---|---|---|---|
| none | - | no filter | 43,134 | 100.000% |
| length_50pct | 50% | min_words = 3359 | 21,570 | 50.007% |
| length_5pct | 5% | min_words = 13542 | 2,166 | 5.022% |
| length_0.5pct | 0.5% | min_words = 20112 | 256 | 0.594% |
| length_0.05pct | 0.05% | min_words = 23132 | 139 | 0.322% |
| edited_50pct | 50% | edited_since = 2020-12-18T02:38:12 | 21,578 | 50.026% |
| edited_5pct | 5% | edited_since = 2020-12-31T01:41:51 | 2,124 | 4.924% |
| edited_0.5pct | 0.5% | edited_since = 2020-12-31T21:03:02 | 219 | 0.508% |
| edited_0.05pct | 0.05% | edited_since = 2020-12-31T23:58:53 | 34 | 0.079% |
| category_50pct | 50% | category = Living people | 5,081 | 11.780% |
| category_5pct | 5% | category = Articles containing video clips | 1,857 | 4.305% |
| category_0.5pct | 0.5% | category = American film producers | 215 | 0.498% |
| category_0.05pct | 0.05% | category = 1608 births | 22 | 0.051% |
| linked_50pct | 50% | linked_from = List of countries by maternal mortality ratio | 1,629 | 3.777% |
| linked_0.5pct | 0.5% | linked_from = ABC Records | 216 | 0.501% |
| linked_0.05pct | 0.05% | linked_from = 30th New York State Legislature | 22 | 0.051% |
| four_filters | - | category = Member states of the United Nations; min_words = 3359; edited_since = 2020-12-18T02:38:12; linked_from = List of countries by maternal mortality ratio | 824 | 1.910% |
