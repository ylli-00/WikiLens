# Rule D distractor pages (scaled experiment)

Written by `scripts/scaled_build.py report` from `distractors_D.json` and `corpora.json`
(measured by the `distractors` and `corpora` steps). The list is `distractors_D.jsonl`.

## Rule

Per gold article, count its links (`wikitext.extract_links`, duplicates kept, page order);
its favourite is the most-linked title, ties broken by the first mention. Skip gold pages,
titles another article already took, and titles missing from the FEVEROUS DB; then take
the next favourite. Gold articles in `sorted(gold)` order shuffled with `random.Random(7)`;
one page per gold article with a usable link, no cap.

## Results

- Gold articles walked: 3,851 (missing from the DB: 0).
- Rule-D pages: **3,835**.
- Gold articles without a usable link: 16.
- Skips during the walk: gold page 551, already taken 1,123, missing from the DB 790.
- Distinct titles looked up in the DB: 4,492, of which 657 missing.
- Links per gold article: median 35, mean 92.84, max 7,039.
- Chunks per rule-D page: median 15, mean 23.27, p90 58, max 196; total 89,225; pages without any chunk 0.

Which favourite was taken (0 = the most-linked title; a higher rank means the titles
before it were skipped):

| Rank | Pages |
|---|---|
| 0 | 2,567 |
| 1 | 748 |
| 2 | 279 |
| 3 | 103 |
| 4 | 52 |
| 5 | 25 |
| 6 | 27 |
| 7 | 12 |
| 8 | 7 |
| 9 | 6 |
| 10 | 6 |
| 11 | 1 |
| 13 | 1 |
| 16 | 1 |

How often the gold article links to the page it took (10 = 10 or more):

| Times linked | Pages |
|---|---|
| 1 | 811 |
| 2 | 1,161 |
| 3 | 797 |
| 4 | 389 |
| 5 | 193 |
| 6 | 147 |
| 7 | 84 |
| 8 | 54 |
| 9 | 35 |
| 10 | 164 |

Gold articles without a usable link:

| Gold article | Links | Distinct titles | Skipped |
|---|---|---|---|
| RAPGEF1 | 3 | 3 | {"gold": 1, "taken": 1, "missing": 1} |
| RAD9A | 3 | 3 | {"gold": 1, "taken": 1, "missing": 1} |
| List of West Bengal districts ranked by literacy rate | 5 | 4 | {"missing": 2, "taken": 2} |
| Vertical resistance | 2 | 2 | {"missing": 2} |
| Hongze | 2 | 2 | {"missing": 2} |
| Teldenia alba | 11 | 10 | {"taken": 7, "gold": 3} |
| Yang Tianyong | 2 | 1 | {"missing": 1} |
| Xigong District | 6 | 6 | {"gold": 1, "taken": 3, "missing": 2} |
| Francesca Melandri | 3 | 3 | {"gold": 3} |
| Eddie Blay | 4 | 4 | {"gold": 4} |
| RAD17 | 3 | 3 | {"gold": 1, "taken": 1, "missing": 1} |
| Chiman Saparaiya | 2 | 2 | {"taken": 1, "missing": 1} |
| DAB2 | 3 | 3 | {"gold": 1, "taken": 1, "missing": 1} |
| Talk, Talk, Talk | 1 | 1 | {"gold": 1} |
| Ditrigona candida | 12 | 11 | {"taken": 8, "gold": 3} |
| Czechoslovak Ocean Shipping | 2 | 2 | {"taken": 1, "gold": 1} |

## Corpus sizes

| Folder | Pages | Gold pages | Rule-D pages | Chunks |
|---|---|---|---|---|
| S0 | 3,851 | 3,851 | 0 | 43,134 |
| S1 | 7,686 | 3,851 | 3,835 | 132,359 |
