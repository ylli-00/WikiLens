# FEVEROUS-style evidence score, `wikilense_s1`

Written by `scripts/scaled_sentence_score.py` on 2026-11-15T03:06:36+01:00. Top 10 chunks per claim (`search.search`, strategy `none`, ef_search 100); the units of those chunks ranked by embedding against the claim; best 5 sentences + 25 list items; found = one complete sentence-only gold set among them. Eligible = claims with a sentence-only set.

- distinct units embedded: 169106 in 114.3 s (cuda); candidates per claim: mean 68.7 sentences + items

| claims | eligible | complete set in top 10 chunks (ours) | 5 + 25 by embedding (FEVEROUS-style) | 5 + 25 in chunk order |
|---|---|---|---|---|
| all (3000) | 3000 | 0.825 (2476/3000) | 0.529 (1588/3000) | 0.397 (1190/3000) |
| SUPPORTS / single (700) | 700 | 0.981 (687/700) | 0.739 (517/700) | 0.537 (376/700) |
| SUPPORTS / needs2 (300) | 300 | 0.737 (221/300) | 0.260 (78/300) | 0.073 (22/300) |
| REFUTES / single (700) | 700 | 0.974 (682/700) | 0.747 (523/700) | 0.657 (460/700) |
| REFUTES / needs2 (300) | 300 | 0.407 (122/300) | 0.080 (24/300) | 0.053 (16/300) |
| NOT ENOUGH INFO / single (700) | 700 | 0.890 (623/700) | 0.593 (415/700) | 0.443 (310/700) |
| NOT ENOUGH INFO / needs2 (300) | 300 | 0.470 (141/300) | 0.103 (31/300) | 0.020 (6/300) |

Rates are over the eligible claims of each row. The overall row is the raw count over the 3,000 claims (not reweighted to the FEVEROUS population).
