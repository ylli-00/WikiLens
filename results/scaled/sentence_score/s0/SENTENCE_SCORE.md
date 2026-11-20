# FEVEROUS-style evidence score, `wikilense_s0`

Written by `scripts/scaled_sentence_score.py` on 2026-11-15T03:10:38+01:00. Top 10 chunks per claim (`search.search`, strategy `none`, ef_search 100); the units of those chunks ranked by embedding against the claim; best 5 sentences + 25 list items; found = one complete sentence-only gold set among them. Eligible = claims with a sentence-only set.

- distinct units embedded: 133007 in 90.9 s (cuda); candidates per claim: mean 66.2 sentences + items

| claims | eligible | complete set in top 10 chunks (ours) | 5 + 25 by embedding (FEVEROUS-style) | 5 + 25 in chunk order |
|---|---|---|---|---|
| all (3000) | 3000 | 0.853 (2558/3000) | 0.546 (1637/3000) | 0.413 (1239/3000) |
| SUPPORTS / single (700) | 700 | 0.984 (689/700) | 0.750 (525/700) | 0.541 (379/700) |
| SUPPORTS / needs2 (300) | 300 | 0.787 (236/300) | 0.280 (84/300) | 0.087 (26/300) |
| REFUTES / single (700) | 700 | 0.986 (690/700) | 0.769 (538/700) | 0.677 (474/700) |
| REFUTES / needs2 (300) | 300 | 0.467 (140/300) | 0.090 (27/300) | 0.070 (21/300) |
| NOT ENOUGH INFO / single (700) | 700 | 0.914 (640/700) | 0.616 (431/700) | 0.473 (331/700) |
| NOT ENOUGH INFO / needs2 (300) | 300 | 0.543 (163/300) | 0.107 (32/300) | 0.027 (8/300) |

Rates are over the eligible claims of each row. The overall row is the raw count over the 3,000 claims (not reweighted to the FEVEROUS population).
