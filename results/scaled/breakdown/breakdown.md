# Scaled run: recall by claim group and challenge tag, and reweighted to FEVEROUS

Written 2026-11-19T21:39:55+00:00 by `scripts/scaled_breakdown.py` from the per-claim files of the runs named below (no database). A claim counts at k when its rank in the run's LIMIT 20 search is at most k; that reproduces the run's own counts per k. "Complete set in top 10 chunks" and "FEVEROUS-style" are the two evidence scores of `scripts/scaled_sentence_score.py`.

Inputs: claims: `data/scaled/claims_3000.jsonl`; FEVEROUS table: `analysis/feverous/feverous_claims.csv`; eval S0: `results/scaled/eval/s0_full_run1.json`; eval S1: `results/scaled/eval/s1_full_run1.json`; sentences S0: `results/scaled/sentence_score/s0/claims.jsonl`; sentences S1: `results/scaled/sentence_score/s1/claims.jsonl`.

## Weights: FEVEROUS's mix of the six groups

From `analysis/feverous/feverous_claims.csv` (79,181 train + dev claims). A claim is in (label, one article) when every evidence set lies on one page (`max_pages` = 1), in (label, two or more) when every set spans two or more pages (`min_pages` >= 2); claims with sets of both kinds are in no group. `all` counts every such claim; `text` only those with an evidence set of sentences only or list items only. Reweighted = sum over the groups of weight x the group's measured rate.

| Group | Claims here | FEVEROUS, all | weight | FEVEROUS, text | weight |
|---|---|---|---|---|---|
| SUPPORTS, one article | 700 | 33,981 | 0.4310 | 13,988 | 0.4057 |
| SUPPORTS, two or more | 300 | 11,529 | 0.1462 | 4,067 | 0.1180 |
| REFUTES, one article | 700 | 29,100 | 0.3691 | 14,453 | 0.4192 |
| REFUTES, two or more | 300 | 1,494 | 0.0189 | 584 | 0.0169 |
| NOT ENOUGH INFO, one article | 700 | 2,059 | 0.0261 | 1,125 | 0.0326 |
| NOT ENOUGH INFO, two or more | 300 | 677 | 0.0086 | 263 | 0.0076 |
| total | 3000 | 78,840 | 1 | 34,480 | 1 |

## S0

### Six groups

| Group | n | article@1 | article@3 | article@5 | article@10 | article@20 | evidence@1 | evidence@3 | evidence@5 | evidence@10 | evidence@20 | complete_set_top10_chunks | feverous_style |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SUPPORTS, one article | 700 | 0.993 | 0.994 | 0.994 | 0.996 | 0.997 | 0.661 | 0.931 | 0.960 | 0.984 | 0.993 | 0.984 | 0.750 |
| SUPPORTS, two or more | 300 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.463 | 0.633 | 0.787 | 0.883 | 0.787 | 0.280 |
| REFUTES, one article | 700 | 0.971 | 0.989 | 0.993 | 0.993 | 0.996 | 0.717 | 0.924 | 0.956 | 0.986 | 0.986 | 0.986 | 0.769 |
| REFUTES, two or more | 300 | 0.980 | 0.990 | 0.993 | 1.000 | 1.000 | 0.000 | 0.213 | 0.327 | 0.467 | 0.560 | 0.467 | 0.090 |
| NOT ENOUGH INFO, one article | 700 | 0.914 | 0.941 | 0.950 | 0.954 | 0.967 | 0.531 | 0.787 | 0.866 | 0.914 | 0.944 | 0.914 | 0.616 |
| NOT ENOUGH INFO, two or more | 300 | 0.957 | 0.977 | 0.990 | 0.997 | 0.997 | 0.000 | 0.183 | 0.353 | 0.543 | 0.663 | 0.543 | 0.107 |
| all claims, as measured | 3000 | 0.965 | 0.979 | 0.984 | 0.986 | 0.990 | 0.446 | 0.703 | 0.780 | 0.853 | 0.893 | 0.853 | 0.546 |
| reweighted to FEVEROUS (all) | - | 0.983 | 0.991 | 0.993 | 0.994 | 0.996 | 0.564 | 0.837 | 0.891 | 0.940 | 0.962 | 0.940 | 0.667 |
| reweighted to FEVEROUS (text) | - | 0.981 | 0.991 | 0.993 | 0.994 | 0.996 | 0.586 | 0.851 | 0.901 | 0.947 | 0.966 | 0.947 | 0.682 |

### Challenge tags (k = 5 for the recall columns)

| Challenge tag | n | article@5 | evidence@5 | complete_set_top10_chunks | feverous_style |
|---|---|---|---|---|---|
| Other | 1906 | 0.991 | 0.887 | 0.929 | 0.669 |
| Multi-hop Reasoning | 709 | 0.997 | 0.518 | 0.673 | 0.223 |
| Entity Disambiguation | 151 | 0.921 | 0.669 | 0.748 | 0.444 |
| Search terms not in claim | 131 | 0.885 | 0.733 | 0.786 | 0.519 |
| Numerical Reasoning | 71 | 0.972 | 0.789 | 0.873 | 0.606 |
| Combining Tables and Text | 32 | 1.000 | 0.969 | 1.000 | 0.812 |

## S1

### Six groups

| Group | n | article@1 | article@3 | article@5 | article@10 | article@20 | evidence@1 | evidence@3 | evidence@5 | evidence@10 | evidence@20 | complete_set_top10_chunks | feverous_style |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SUPPORTS, one article | 700 | 0.984 | 0.994 | 0.994 | 0.996 | 0.997 | 0.654 | 0.921 | 0.956 | 0.981 | 0.991 | 0.981 | 0.739 |
| SUPPORTS, two or more | 300 | 0.977 | 0.997 | 1.000 | 1.000 | 1.000 | 0.000 | 0.410 | 0.553 | 0.737 | 0.837 | 0.737 | 0.260 |
| REFUTES, one article | 700 | 0.944 | 0.979 | 0.986 | 0.989 | 0.991 | 0.697 | 0.911 | 0.940 | 0.977 | 0.983 | 0.974 | 0.747 |
| REFUTES, two or more | 300 | 0.947 | 0.983 | 0.990 | 0.993 | 0.997 | 0.000 | 0.167 | 0.287 | 0.407 | 0.503 | 0.407 | 0.080 |
| NOT ENOUGH INFO, one article | 700 | 0.876 | 0.926 | 0.936 | 0.946 | 0.953 | 0.510 | 0.761 | 0.833 | 0.891 | 0.921 | 0.890 | 0.593 |
| NOT ENOUGH INFO, two or more | 300 | 0.913 | 0.960 | 0.983 | 0.993 | 0.997 | 0.000 | 0.177 | 0.303 | 0.470 | 0.600 | 0.470 | 0.103 |
| all claims, as measured | 3000 | 0.938 | 0.970 | 0.978 | 0.982 | 0.986 | 0.434 | 0.681 | 0.751 | 0.826 | 0.870 | 0.825 | 0.529 |
| reweighted to FEVEROUS (all) | - | 0.964 | 0.987 | 0.990 | 0.992 | 0.994 | 0.553 | 0.818 | 0.870 | 0.926 | 0.951 | 0.925 | 0.650 |
| reweighted to FEVEROUS (text) | - | 0.962 | 0.985 | 0.989 | 0.992 | 0.994 | 0.574 | 0.833 | 0.881 | 0.934 | 0.956 | 0.933 | 0.665 |

### Challenge tags (k = 5 for the recall columns)

| Challenge tag | n | article@5 | evidence@5 | complete_set_top10_chunks | feverous_style |
|---|---|---|---|---|---|
| Other | 1906 | 0.986 | 0.868 | 0.910 | 0.652 |
| Multi-hop Reasoning | 709 | 0.993 | 0.468 | 0.622 | 0.210 |
| Entity Disambiguation | 151 | 0.881 | 0.616 | 0.709 | 0.411 |
| Search terms not in claim | 131 | 0.885 | 0.687 | 0.763 | 0.504 |
| Numerical Reasoning | 71 | 0.972 | 0.746 | 0.859 | 0.606 |
| Combining Tables and Text | 32 | 1.000 | 0.969 | 1.000 | 0.812 |
