# Filter values on wikilense_s1

Written 2026-11-19T19:24:06+00:00 by `scripts/scaled_filters.py`: 132,359 chunks on 7,686 pages. Rule (the script's docstring has it in full): per filter, the value whose passing chunks are nearest, on a log scale, to 50%, 5%, 0.5%, 0.05% of the chunk table; `four_filters` sets length and edit date at their 50% values and the category and linking page whose pair keeps the most chunks with them.

| Case | Target share | Filter value | Chunks passing | Share |
|---|---|---|---|---|
| none | - | no filter | 132,359 | 100.000% |
| length_50pct | 50% | min_words = 5214 | 66,171 | 49.994% |
| length_5pct | 5% | min_words = 14929 | 6,616 | 4.998% |
| length_0.5pct | 0.5% | min_words = 20112 | 626 | 0.473% |
| length_0.05pct | 0.05% | min_words = 23132 | 139 | 0.105% |
| edited_50pct | 50% | edited_since = 2020-12-21T15:34:08 | 66,208 | 50.021% |
| edited_5pct | 5% | edited_since = 2020-12-31T08:42:11 | 6,599 | 4.986% |
| edited_0.5pct | 0.5% | edited_since = 2020-12-31T22:30:26 | 660 | 0.499% |
| edited_0.05pct | 0.05% | edited_since = 2020-12-31T23:59:17 | 77 | 0.058% |
| category_50pct | 50% | category = Living people | 9,267 | 7.001% |
| category_5pct | 5% | category = Member states of the United Nations | 7,313 | 5.525% |
| category_0.5pct | 0.5% | category = American male film actors | 660 | 0.499% |
| category_0.05pct | 0.05% | category = 1821 establishments in North America | 66 | 0.050% |
| linked_50pct | 50% | linked_from = List of sovereign states | 8,203 | 6.197% |
| linked_5pct | 5% | linked_from = United Kingdom | 6,698 | 5.060% |
| linked_0.5pct | 0.5% | linked_from = Jean-Paul Sartre | 662 | 0.500% |
| linked_0.05pct | 0.05% | linked_from = 1930 Nicaraguan parliamentary election | 66 | 0.050% |
| four_filters | - | category = Member states of the United Nations; min_words = 5214; edited_since = 2020-12-21T15:34:08; linked_from = Legality of cannabis | 3,107 | 2.347% |
