# Scaled run: recall by claim group and challenge tag, and reweighted to FEVEROUS

Written 2026-11-20T16:01:51+00:00 by `scripts/scaled_breakdown.py` from the per-claim files of the runs named below (no database). A claim counts at k when its rank in the run's LIMIT 20 search is at most k; that reproduces the run's own counts per k. "Complete set in top 10 chunks" and "FEVEROUS-style" are the two evidence scores of `scripts/scaled_sentence_score.py`.

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

## Label check and equal-weight figures

Does the search do worse on some labels than on others? All three labels stay in, as everywhere in this file. Step 1 compares, per database, metric and kind, every pair of labels with Newcombe's 95% interval (from the two Wilson intervals). Within a kind every label has 700 or 300 claims, so a difference there is a label effect, not a mix effect. A metric differs by label when any pair's interval excludes 0, in either kind. Step 2, only for those metrics: the equal-weight figure of a group is the mean of its SUPPORTS, REFUTES and NOT ENOUGH INFO rates, a third each, so that the group's label mix does not move it; a group without claims of some label has none ("not computable"). Its interval is a paired bootstrap (10,000 resamples, seed 20261120, the claims resampled within each label and tag, or label, tag and kind). Tags with fewer than 100 claims get their numbers but no conclusions.

### Step 1: the three labels within each kind (k = 5)

| Database | Metric | Kind | Pair | First | Second | Difference (95% interval) | Excludes 0 |
|---|---|---|---|---|---|---|---|
| S0 | article@5 | one article | S − R | 0.994 (0.985–0.998) | 0.993 (0.983–0.997) | 0.001 (-0.008 to 0.012) | no |
| S0 | article@5 | one article | S − NEI | 0.994 (0.985–0.998) | 0.950 (0.931–0.964) | 0.044 (0.028 to 0.063) | yes |
| S0 | article@5 | one article | R − NEI | 0.993 (0.983–0.997) | 0.950 (0.931–0.964) | 0.043 (0.026 to 0.062) | yes |
| S0 | article@5 | two or more | S − R | 1.000 (0.987–1.000) | 0.993 (0.976–0.998) | 0.007 (-0.007 to 0.024) | no |
| S0 | article@5 | two or more | S − NEI | 1.000 (0.987–1.000) | 0.990 (0.971–0.997) | 0.010 (-0.004 to 0.029) | no |
| S0 | article@5 | two or more | R − NEI | 0.993 (0.976–0.998) | 0.990 (0.971–0.997) | 0.003 (-0.015 to 0.023) | no |
| S0 | evidence@5 | one article | S − R | 0.960 (0.943–0.972) | 0.956 (0.938–0.969) | 0.004 (-0.017 to 0.026) | no |
| S0 | evidence@5 | one article | S − NEI | 0.960 (0.943–0.972) | 0.866 (0.838–0.889) | 0.094 (0.065 to 0.124) | yes |
| S0 | evidence@5 | one article | R − NEI | 0.956 (0.938–0.969) | 0.866 (0.838–0.889) | 0.090 (0.061 to 0.120) | yes |
| S0 | evidence@5 | two or more | S − R | 0.633 (0.577–0.686) | 0.327 (0.276–0.382) | 0.307 (0.228 to 0.380) | yes |
| S0 | evidence@5 | two or more | S − NEI | 0.633 (0.577–0.686) | 0.353 (0.301–0.409) | 0.280 (0.201 to 0.354) | yes |
| S0 | evidence@5 | two or more | R − NEI | 0.327 (0.276–0.382) | 0.353 (0.301–0.409) | -0.027 (-0.102 to 0.049) | no |
| S0 | feverous_style | one article | S − R | 0.750 (0.717–0.781) | 0.769 (0.736–0.798) | -0.019 (-0.063 to 0.026) | no |
| S0 | feverous_style | one article | S − NEI | 0.750 (0.717–0.781) | 0.616 (0.579–0.651) | 0.134 (0.086 to 0.182) | yes |
| S0 | feverous_style | one article | R − NEI | 0.769 (0.736–0.798) | 0.616 (0.579–0.651) | 0.153 (0.105 to 0.200) | yes |
| S0 | feverous_style | two or more | S − R | 0.280 (0.232–0.333) | 0.090 (0.063–0.128) | 0.190 (0.129 to 0.250) | yes |
| S0 | feverous_style | two or more | S − NEI | 0.280 (0.232–0.333) | 0.107 (0.077–0.147) | 0.173 (0.111 to 0.235) | yes |
| S0 | feverous_style | two or more | R − NEI | 0.090 (0.063–0.128) | 0.107 (0.077–0.147) | -0.017 (-0.065 to 0.032) | no |
| S1 | article@5 | one article | S − R | 0.994 (0.985–0.998) | 0.986 (0.974–0.992) | 0.009 (-0.002 to 0.021) | no |
| S1 | article@5 | one article | S − NEI | 0.994 (0.985–0.998) | 0.936 (0.915–0.952) | 0.059 (0.040 to 0.080) | yes |
| S1 | article@5 | one article | R − NEI | 0.986 (0.974–0.992) | 0.936 (0.915–0.952) | 0.050 (0.030 to 0.072) | yes |
| S1 | article@5 | two or more | S − R | 1.000 (0.987–1.000) | 0.990 (0.971–0.997) | 0.010 (-0.004 to 0.029) | no |
| S1 | article@5 | two or more | S − NEI | 1.000 (0.987–1.000) | 0.983 (0.962–0.993) | 0.017 (0.001 to 0.038) | yes |
| S1 | article@5 | two or more | R − NEI | 0.990 (0.971–0.997) | 0.983 (0.962–0.993) | 0.007 (-0.015 to 0.029) | no |
| S1 | evidence@5 | one article | S − R | 0.956 (0.938–0.969) | 0.940 (0.920–0.955) | 0.016 (-0.008 to 0.040) | no |
| S1 | evidence@5 | one article | S − NEI | 0.956 (0.938–0.969) | 0.833 (0.803–0.859) | 0.123 (0.091 to 0.155) | yes |
| S1 | evidence@5 | one article | R − NEI | 0.940 (0.920–0.955) | 0.833 (0.803–0.859) | 0.107 (0.074 to 0.140) | yes |
| S1 | evidence@5 | two or more | S − R | 0.553 (0.497–0.609) | 0.287 (0.238–0.340) | 0.267 (0.189 to 0.340) | yes |
| S1 | evidence@5 | two or more | S − NEI | 0.553 (0.497–0.609) | 0.303 (0.254–0.358) | 0.250 (0.172 to 0.324) | yes |
| S1 | evidence@5 | two or more | R − NEI | 0.287 (0.238–0.340) | 0.303 (0.254–0.358) | -0.017 (-0.089 to 0.056) | no |
| S1 | feverous_style | one article | S − R | 0.739 (0.705–0.770) | 0.747 (0.714–0.778) | -0.009 (-0.054 to 0.037) | no |
| S1 | feverous_style | one article | S − NEI | 0.739 (0.705–0.770) | 0.593 (0.556–0.629) | 0.146 (0.096 to 0.194) | yes |
| S1 | feverous_style | one article | R − NEI | 0.747 (0.714–0.778) | 0.593 (0.556–0.629) | 0.154 (0.105 to 0.202) | yes |
| S1 | feverous_style | two or more | S − R | 0.260 (0.214–0.312) | 0.080 (0.054–0.116) | 0.180 (0.121 to 0.238) | yes |
| S1 | feverous_style | two or more | S − NEI | 0.260 (0.214–0.312) | 0.103 (0.074–0.143) | 0.157 (0.096 to 0.217) | yes |
| S1 | feverous_style | two or more | R − NEI | 0.080 (0.054–0.116) | 0.103 (0.074–0.143) | -0.023 (-0.071 to 0.023) | no |

S0: differs by label: article@5, evidence@5, feverous_style. No metric is left unweighted.

S1: differs by label: article@5, evidence@5, feverous_style. No metric is left unweighted.

### Step 2, S0: equal-weight figures per challenge tag (k = 5)

| Challenge tag | n (S / R / NEI) | article@5 | article@5, equal weight | evidence@5 | evidence@5, equal weight | feverous_style | feverous_style, equal weight |
|---|---|---|---|---|---|---|---|
| Other | 1906 (602 / 709 / 595) | 0.991 | 0.990 (0.985–0.994) | 0.887 | 0.887 (0.873–0.901) | 0.669 | 0.668 (0.646–0.689) |
| Multi-hop Reasoning | 709 (319 / 169 / 221) | 0.997 | 0.997 (0.992–1.000) | 0.518 | 0.482 (0.446–0.518) | 0.223 | 0.197 (0.169–0.225) |
| Entity Disambiguation | 151 (34 / 46 / 71) | 0.921 | 0.941 (0.908–0.972) | 0.669 | 0.722 (0.658–0.784) | 0.444 | 0.475 (0.394–0.557) |
| Search terms not in claim | 131 (18 / 15 / 98) | 0.885 | 0.930 (0.874–0.969) | 0.733 | 0.809 (0.723–0.883) | 0.519 | 0.609 (0.498–0.711) |
| Numerical Reasoning (fewer than 100) | 71 (6 / 51 / 14) | 0.972 | 0.938 (0.827–1.000) | 0.789 | 0.818 (0.679–0.924) | 0.606 | 0.584 (0.420–0.742) |
| Combining Tables and Text (fewer than 100) | 32 (21 / 10 / 1) | 1.000 | 1.000 (1.000–1.000) | 0.969 | 0.984 (0.952–1.000) | 0.812 | 0.552 (0.452–0.635) |

### Step 2, S0: Multi-hop Reasoning against Other

Plain: the share of the tag's claims, Wilson interval, gap with Newcombe's interval. Equal weight: bootstrap. The gaps are Multi-hop Reasoning minus Other.

| Metric | Claims | Multi-hop Reasoning, n (S / R / NEI) | Other, n | Multi-hop Reasoning | Other | Gap | Multi-hop Reasoning, equal weight | Other, equal weight | Gap, equal weight |
|---|---|---|---|---|---|---|---|---|---|
| article@5 | all | 709 (319 / 169 / 221) | 1906 (602 / 709 / 595) | 0.997 (0.990–0.999) | 0.991 (0.985–0.994) | 0.007 (-0.002 to 0.012) | 0.997 (0.992–1.000) | 0.990 (0.985–0.994) | 0.007 (0.000 to 0.013) |
| article@5 | one article | 112 (53 / 8 / 51) | 1704 (574 / 614 / 516) | 1.000 (0.967–1.000) | 0.989 (0.983–0.993) | 0.011 (-0.023 to 0.017) | 1.000 (1.000–1.000) | 0.989 (0.984–0.994) | 0.011 (0.006 to 0.016) |
| article@5 | two or more | 597 (266 / 161 / 170) | 202 (28 / 95 / 79) | 0.997 (0.988–0.999) | 1.000 (0.981–1.000) | -0.003 (-0.012 to 0.015) | 0.996 (0.990–1.000) | 1.000 (1.000–1.000) | -0.004 (-0.010 to 0.000) |
| evidence@5 | all | 709 (319 / 169 / 221) | 1906 (602 / 709 / 595) | 0.518 (0.481–0.554) | 0.887 (0.872–0.900) | -0.369 (-0.408 to -0.329) | 0.482 (0.446–0.518) | 0.887 (0.873–0.901) | -0.405 (-0.444 to -0.366) |
| evidence@5 | one article | 112 (53 / 8 / 51) | 1704 (574 / 614 / 516) | 0.830 (0.750–0.889) | 0.945 (0.934–0.955) | -0.115 (-0.196 to -0.056) | 0.878 (0.827–0.924) | 0.944 (0.932–0.954) | -0.065 (-0.117 to -0.016) |
| evidence@5 | two or more | 597 (266 / 161 / 170) | 202 (28 / 95 / 79) | 0.459 (0.419–0.499) | 0.391 (0.326–0.460) | 0.068 (-0.011 to 0.144) | 0.425 (0.387–0.465) | 0.456 (0.382–0.530) | -0.030 (-0.114 to 0.054) |
| feverous_style | all | 709 (319 / 169 / 221) | 1906 (602 / 709 / 595) | 0.223 (0.194–0.255) | 0.669 (0.647–0.690) | -0.446 (-0.482 to -0.408) | 0.197 (0.169–0.225) | 0.668 (0.646–0.689) | -0.471 (-0.506 to -0.436) |
| feverous_style | one article | 112 (53 / 8 / 51) | 1704 (574 / 614 / 516) | 0.500 (0.409–0.591) | 0.734 (0.713–0.755) | -0.234 (-0.327 to -0.141) | 0.464 (0.336–0.598) | 0.730 (0.709–0.751) | -0.266 (-0.395 to -0.132) |
| feverous_style | two or more | 597 (266 / 161 / 170) | 202 (28 / 95 / 79) | 0.171 (0.143–0.203) | 0.119 (0.081–0.171) | 0.052 (-0.007 to 0.102) | 0.149 (0.122–0.176) | 0.142 (0.087–0.203) | 0.007 (-0.059 to 0.069) |

### Step 2, S1: equal-weight figures per challenge tag (k = 5)

| Challenge tag | n (S / R / NEI) | article@5 | article@5, equal weight | evidence@5 | evidence@5, equal weight | feverous_style | feverous_style, equal weight |
|---|---|---|---|---|---|---|---|
| Other | 1906 (602 / 709 / 595) | 0.986 | 0.986 (0.980–0.991) | 0.868 | 0.868 (0.853–0.883) | 0.652 | 0.651 (0.629–0.672) |
| Multi-hop Reasoning | 709 (319 / 169 / 221) | 0.993 | 0.992 (0.984–0.998) | 0.468 | 0.435 (0.400–0.471) | 0.210 | 0.186 (0.159–0.213) |
| Entity Disambiguation | 151 (34 / 46 / 71) | 0.881 | 0.905 (0.860–0.945) | 0.616 | 0.672 (0.602–0.739) | 0.411 | 0.447 (0.367–0.527) |
| Search terms not in claim | 131 (18 / 15 / 98) | 0.885 | 0.930 (0.874–0.969) | 0.687 | 0.770 (0.678–0.852) | 0.504 | 0.583 (0.474–0.688) |
| Numerical Reasoning (fewer than 100) | 71 (6 / 51 / 14) | 0.972 | 0.938 (0.827–1.000) | 0.746 | 0.799 (0.659–0.904) | 0.606 | 0.584 (0.420–0.742) |
| Combining Tables and Text (fewer than 100) | 32 (21 / 10 / 1) | 1.000 | 1.000 (1.000–1.000) | 0.969 | 0.984 (0.952–1.000) | 0.812 | 0.552 (0.452–0.635) |

### Step 2, S1: Multi-hop Reasoning against Other

Plain: the share of the tag's claims, Wilson interval, gap with Newcombe's interval. Equal weight: bootstrap. The gaps are Multi-hop Reasoning minus Other.

| Metric | Claims | Multi-hop Reasoning, n (S / R / NEI) | Other, n | Multi-hop Reasoning | Other | Gap | Multi-hop Reasoning, equal weight | Other, equal weight | Gap, equal weight |
|---|---|---|---|---|---|---|---|---|---|
| article@5 | all | 709 (319 / 169 / 221) | 1906 (602 / 709 / 595) | 0.993 (0.984–0.997) | 0.986 (0.979–0.990) | 0.007 (-0.003 to 0.015) | 0.992 (0.984–0.998) | 0.986 (0.980–0.991) | 0.006 (-0.003 to 0.015) |
| article@5 | one article | 112 (53 / 8 / 51) | 1704 (574 / 614 / 516) | 0.991 (0.951–0.998) | 0.984 (0.977–0.989) | 0.007 (-0.033 to 0.017) | 0.993 (0.980–1.000) | 0.984 (0.977–0.989) | 0.010 (-0.006 to 0.021) |
| article@5 | two or more | 597 (266 / 161 / 170) | 202 (28 / 95 / 79) | 0.993 (0.983–0.997) | 1.000 (0.981–1.000) | -0.007 (-0.017 to 0.012) | 0.992 (0.984–0.998) | 1.000 (1.000–1.000) | -0.008 (-0.016 to -0.002) |
| evidence@5 | all | 709 (319 / 169 / 221) | 1906 (602 / 709 / 595) | 0.468 (0.432–0.505) | 0.868 (0.852–0.882) | -0.400 (-0.439 to -0.359) | 0.435 (0.400–0.471) | 0.868 (0.853–0.883) | -0.433 (-0.472 to -0.395) |
| evidence@5 | one article | 112 (53 / 8 / 51) | 1704 (574 / 614 / 516) | 0.812 (0.730–0.874) | 0.930 (0.916–0.941) | -0.117 (-0.200 to -0.054) | 0.865 (0.813–0.916) | 0.927 (0.915–0.939) | -0.062 (-0.116 to -0.011) |
| evidence@5 | two or more | 597 (266 / 161 / 170) | 202 (28 / 95 / 79) | 0.404 (0.365–0.444) | 0.347 (0.284–0.414) | 0.057 (-0.021 to 0.131) | 0.374 (0.336–0.411) | 0.389 (0.315–0.468) | -0.016 (-0.103 to 0.069) |
| feverous_style | all | 709 (319 / 169 / 221) | 1906 (602 / 709 / 595) | 0.210 (0.182–0.242) | 0.652 (0.630–0.673) | -0.441 (-0.477 to -0.403) | 0.186 (0.159–0.213) | 0.651 (0.629–0.672) | -0.464 (-0.499 to -0.430) |
| feverous_style | one article | 112 (53 / 8 / 51) | 1704 (574 / 614 / 516) | 0.482 (0.392–0.574) | 0.716 (0.694–0.737) | -0.234 (-0.327 to -0.140) | 0.451 (0.324–0.583) | 0.712 (0.691–0.733) | -0.261 (-0.389 to -0.128) |
| feverous_style | two or more | 597 (266 / 161 / 170) | 202 (28 / 95 / 79) | 0.159 (0.132–0.191) | 0.109 (0.073–0.159) | 0.050 (-0.007 to 0.098) | 0.139 (0.113–0.165) | 0.134 (0.079–0.193) | 0.005 (-0.061 to 0.066) |
