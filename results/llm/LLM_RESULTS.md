# LLM fact-checker: results

Written by `scripts/llm_results.py stats` from `per_claim.csv`, `retrieval_top5.csv` and `runs.json` in this folder, the weights in `results/scaled/breakdown/breakdown.json` and the database timings named in the cost section. Claude Haiku 5.5, effort high, the same 3,000 FEVEROUS claims in four conditions. A claim is right when the answer equals the FEVEROUS label. Percentages are rounded to one decimal, half up. Intervals are 95%: Wilson for an accuracy; for a paired difference of accuracies (b − c)/n ± 1.96 · √(b + c − (b − c)²/n) / n with McNemar's exact two-sided p; for F1 and the reweighted accuracy a paired bootstrap with 10,000 resamples (seeds 20261120 for the 3,000 claims, 20261121 for the 2,000 supported/refuted claims). No correction for multiple comparisons.

## 1. Three classes (3,000 claims)

| condition | accuracy (95% interval) | macro F1 (interval) | recall S / R / NEI | NEI answers | NEI answers right |
|---|---|---|---|---|---|
| no passages | 1,356 / 3,000 = 45.2% (43.4–47.0) | 44.7 (42.9–46.5) | 30.5 / 40.0 / 65.1 | 1,765 | 651 = 36.9% |
| web | 1,839 / 3,000 = 61.3% (59.5–63.0) | 56.8 (55.1–58.5) | 80.0 / 83.8 / 20.1 | 308 | 201 = 65.3% |
| RAG | 2,127 / 3,000 = 70.9% (69.2–72.5) | 69.6 (68.0–71.2) | 85.0 / 82.2 / 45.5 | 606 | 455 = 75.1% |
| gold | 2,186 / 3,000 = 72.9% (71.2–74.4) | 72.1 (70.5–73.7) | 85.7 / 80.0 / 52.9 | 755 | 529 = 70.1% |

RAG gains 771 right claims over no passages, of the 830 that gold gains: 92.9%.

On NOT ENOUGH INFO claims, the share where the model picked a side: no passages 34.9%; web 79.9%; RAG 54.5%; gold 47.1%.

### Confusion matrices

Rows: the FEVEROUS label; columns: the answer (SUPPORTS / REFUTES / NOT ENOUGH INFO).

| label | no passages | web | RAG | gold |
|---|---|---|---|---|
| SUPPORTS | 305 / 85 / 610 | 800 / 148 / 52 | 850 / 99 / 51 | 857 / 80 / 63 |
| REFUTES | 96 / 400 / 504 | 107 / 838 / 55 | 78 / 822 / 100 | 37 / 800 / 163 |
| NOT ENOUGH INFO | 186 / 163 / 651 | 411 / 388 / 201 | 254 / 291 / 455 | 226 / 245 / 529 |

### Paired comparisons on three classes

| pair | only the first right (b) | only the second right (c) | difference, points | 95% interval | McNemar p |
|---|---|---|---|---|---|
| RAG vs web | 451 | 163 | 9.6 | 8.0 to 11.2 | 3.5e-32 |
| RAG vs gold | 170 | 229 | -2.0 | -3.3 to -0.7 | 0.0036 |
| gold vs web | 530 | 183 | 11.6 | 9.9 to 13.3 | 5.5e-40 |
| RAG vs no passages | 1,087 | 316 | 25.7 | 23.4 to 28.0 | 3.6e-99 |
| gold vs no passages | 1,095 | 265 | 27.7 | 25.5 to 29.9 | 5.6e-120 |
| web vs no passages | 998 | 515 | 16.1 | 13.6 to 18.6 | 7.5e-36 |

### One article or two or more

| claims | no passages | web | RAG | gold |
|---|---|---|---|---|
| three classes, one article (2,100) | 44.9% | 63.0% | 72.8% | 74.4% |
| three classes, two or more (900) | 46.0% | 57.4% | 66.6% | 69.3% |
| supported/refuted, one article (1,400) | 33.5% | 84.3% | 85.4% | 83.9% |
| supported/refuted, two or more (600) | 39.3% | 76.3% | 79.3% | 80.3% |

### The six groups (label × one article or two or more)

| group | n | no passages | web | RAG | gold |
|---|---|---|---|---|---|
| SUPPORTS, one article | 700 | 27.7% | 80.3% | 84.7% | 85.1% |
| SUPPORTS, two or more | 300 | 37.0% | 79.3% | 85.7% | 87.0% |
| REFUTES, one article | 700 | 39.3% | 88.3% | 86.1% | 82.7% |
| REFUTES, two or more | 300 | 41.7% | 73.3% | 73.0% | 73.7% |
| NOT ENOUGH INFO, one article | 700 | 67.6% | 20.3% | 47.4% | 55.3% |
| NOT ENOUGH INFO, two or more | 300 | 59.3% | 19.7% | 41.0% | 47.3% |

### Reweighted to FEVEROUS's mix of the six groups

Weights from `results/scaled/breakdown/breakdown.json` (`all`: every FEVEROUS train + dev claim in a group; `text`: those with a text-only evidence set). Accuracy = sum over the groups of weight × the group's accuracy.

| weights | no passages | web | RAG | gold |
|---|---|---|---|---|
| all | 34.9% (32.8–37.0) | 80.9% (79.2–82.6) | 83.8% (82.2–85.4) | 83.2% (81.5–84.8) |
| text | 35.4% (33.3–37.6) | 81.0% (79.3–82.6) | 83.7% (82.0–85.2) | 82.9% (81.2–84.5) |

| difference, weights `all` | point | interval |
|---|---|---|
| RAG − web | 2.9 | 1.3 to 4.5 |
| RAG − gold | 0.6 | -0.7 to 2.0 |
| gold − web | 2.3 | 0.6 to 4.0 |
| RAG − no passages | 48.9 | 46.5 to 51.2 |
| gold − no passages | 48.3 | 45.9 to 50.6 |
| web − no passages | 46.0 | 43.5 to 48.3 |

## 2. Supported and refuted claims only (2,000 claims)

A NOT ENOUGH INFO answer counts as wrong; no claim is dropped because of an answer.

| condition | accuracy (95% interval) | picked a side | right when it picked a side |
|---|---|---|---|
| no passages | 705 / 2,000 = 35.3% (33.2–37.4) | 886 = 44.3% | 705 / 886 = 79.6% (76.8–82.1) |
| web | 1,638 / 2,000 = 81.9% (80.2–83.5) | 1,893 = 94.7% | 1,638 / 1,893 = 86.5% (84.9–88.0) |
| RAG | 1,672 / 2,000 = 83.6% (81.9–85.2) | 1,849 = 92.5% | 1,672 / 1,849 = 90.4% (89.0–91.7) |
| gold | 1,657 / 2,000 = 82.9% (81.1–84.4) | 1,774 = 88.7% | 1,657 / 1,774 = 93.4% (92.2–94.5) |

### Paired comparisons

| pair | only the first right (b) | only the second right (c) | difference, points | 95% interval | McNemar p |
|---|---|---|---|---|---|
| RAG vs web | 145 | 111 | 1.7 | 0.1 to 3.3 | 0.039 |
| RAG vs gold | 102 | 87 | 0.8 | -0.6 to 2.1 | 0.31 |
| gold vs web | 161 | 142 | 1.0 | -0.8 to 2.7 | 0.30 |
| RAG vs no passages | 998 | 31 | 48.4 | 46.0 to 50.7 | 6.7e-251 |
| gold vs no passages | 989 | 37 | 47.6 | 45.3 to 49.9 | 2.8e-241 |
| web vs no passages | 976 | 43 | 46.7 | 44.3 to 49.0 | 5.6e-231 |

### Precision, recall and F1 per label

| condition | SUPPORTS P / R / F1 (F1 interval) | REFUTES P / R / F1 (F1 interval) |
|---|---|---|
| no passages | 76.1 / 30.5 / 43.5 (40.3–46.8) | 82.5 / 40.0 / 53.9 (50.7–56.9) |
| web | 88.2 / 80.0 / 83.9 (82.1–85.6) | 85.0 / 83.8 / 84.4 (82.6–86.1) |
| RAG | 91.6 / 85.0 / 88.2 (86.6–89.7) | 89.3 / 82.2 / 85.6 (83.8–87.2) |
| gold | 95.9 / 85.7 / 90.5 (89.0–91.8) | 90.9 / 80.0 / 85.1 (83.3–86.7) |

| F1 difference | SUPPORTS (interval) | REFUTES (interval) |
|---|---|---|
| RAG − web | 4.3 (2.8 to 5.9) | 1.2 (-0.2 to 2.6) |
| RAG − gold | -2.3 (-3.6 to -1.0) | 0.5 (-0.8 to 1.8) |
| gold − web | 6.6 (4.9 to 8.3) | 0.7 (-0.9 to 2.4) |
| RAG − no passages | 44.6 (41.4 to 47.8) | 31.7 (28.7 to 34.8) |
| gold − no passages | 47.0 (43.7 to 50.3) | 31.2 (28.3 to 34.2) |
| web − no passages | 40.4 (36.9 to 43.7) | 30.5 (27.6 to 33.6) |

## 3. RAG accuracy and what retrieval found

Pages of the five RAG passages against the claim's `gold_pages` (a gold page) and against every page of the evidence set shown in the gold condition (the whole gold set).

| claims | split | share of claims | accuracy when yes | accuracy when no |
|---|---|---|---|---|
| all 3,000, three classes | a gold page in the top 5 | 2,943 = 98.1% | 2,094 / 2,943 = 71.2% (69.5–72.8) | 33 / 57 = 57.9% (45.0–69.8) |
| all 3,000, three classes | the whole gold set in the top 5 | 2,534 = 84.5% | 1,818 / 2,534 = 71.7% (70.0–73.5) | 309 / 466 = 66.3% (61.9–70.5) |
| 2,000 supported/refuted | a gold page in the top 5 | 1,987 = 99.4% | 1,668 / 1,987 = 83.9% (82.3–85.5) | 4 / 13 = 30.8% (12.7–57.6) |
| 2,000 supported/refuted | the whole gold set in the top 5 | 1,719 = 86.0% | 1,458 / 1,719 = 84.8% (83.0–86.4) | 214 / 281 = 76.2% (70.8–80.8) |

## 4. Cost and time

From the run rows (the final attempt of each claim). The LLM times are observed medians of calls through the subscription, five at a time, not a controlled timing.

| condition | calls | median s per claim | × RAG | p95 s | prompt tokens | median thinking tokens | output tokens |
|---|---|---|---|---|---|---|---|
| no passages | 3,000 | 4.2 | 1.8 | 8.3 | 5,069,111 | 636 | 2,656,432 |
| web | 3,000 | 18.2 | 7.8 | 46.1 | 165,580,936 | 2,405 | 13,957,898 |
| RAG | 3,000 | 2.3 | 1.0 | 5.7 | 9,766,910 | 227 | 1,480,127 |
| gold | 3,000 | 2.6 | 1.1 | 6.2 | 5,532,364 | 276 | 1,613,496 |

Usage windows over each run folder (readings of the whole account, so other use at the same time is included; per window, the highest reading minus the first, summed over the windows a run spanned):

| run | folder | conditions | calls | CLI | five-hour used (windows) | seven-day used |
|---|---|---|---|---|---|---|
| original | `haiku_high_fact_check_v1_noret_gold` | no passages, gold | 6,000 | 2.1.322 (Claude Code) | 0.11 (1) | 0.00 |
| original | `haiku_high_fact_check_v1_retrieved` | RAG | 3,000 | 2.1.322 (Claude Code) | 0.05 (1) | 0.01 |
| original | `haiku_high_fact_check_web_v1_web` | web | 3,000 | 2.1.322 (Claude Code) | 0.60 (3) | 0.09 |
| r1 | `haiku_high_fact_check_v1_noret_gold_rep450_r1` | no passages, gold | 900 | 2.1.322 (Claude Code) | 0.06 (1) | 0.01 |
| r1 | `haiku_high_fact_check_v1_retrieved_rep450_r1` | RAG | 450 | 2.1.322 (Claude Code) | 0.02 (1) | 0.00 |
| r1 | `haiku_high_fact_check_web_v1_web_rep450_r1` | web | 450 | 2.1.322 (Claude Code) | 0.11 (2) | 0.01 |
| r2 | `haiku_high_fact_check_v1_noret_gold_rep450_r2` | no passages, gold | 900 | 2.1.322 (Claude Code) | 0.01 (1) | 0.00 |
| r2 | `haiku_high_fact_check_v1_retrieved_rep450_r2` | RAG | 450 | 2.1.322 (Claude Code) | 0.00 (1) | 0.00 |
| r2 | `haiku_high_fact_check_web_v1_web_rep450_r2` | web | 450 | 2.1.322 (Claude Code) | 0.09 (1) | 0.01 |

Retrieval's share of a RAG call: the joined `inline` statement on S1 at k = 5 without a filter, median 27.301 ms (`results/scaled/run1/s1/separate_index.json`), plus one claim embedding, median 6.07 ms on cuda (`results/scaled/eval/s1_full_run1.json`), against the median RAG call of 2.319 s: 1.4%. The database times were measured with the CPU capped at 50% and the GPU at 2,100 MHz. The RAG run embedded its claims on the CPU; the stated embedding time was measured on cuda.

## Tool use of the web condition and the earlier analyses

Web condition: 3,965 WebSearch and 4,265 WebFetch calls over 3,000 claims, 4 fetches denied by the deny list, median 4 turns per call, 228 claims answered without a tool call.

No passages: 1,763 of its 1,765 NOT ENOUGH INFO justifications contain the words "no evidence" and 1,036 contain "cannot confirm" (case-insensitive substring of the justification).

Web audit (2026-11-16T06:02:24+01:00): 3,000 rows audited, 35 flagged (kept), 4 denied calls; 3,744 of the 4,265 fetches opened en.wikipedia.org.

NOT ENOUGH INFO audit: 100 reviewed cases, rated by two reviewer agents each (100 rated by both), who agreed on the main reason in 94 and on whether the text settles the claim in 93; 9 settled by a third agent.

| condition | cases | the FEVEROUS label looks doubtful | passages settle the claim (supports / refutes) |
|---|---|---|---|
| gold | 40 | 16 | 7 / 8 |
| RAG | 40 | 16 | 6 / 8 |
| no passages | 20 | 4 | 0 / 4 |

## 5. Run-to-run variance (450 repeated claims)

f = share of claims that switched between right and wrong across two runs; rerun standard deviation √(f / 2n), times √2 for a difference between two conditions.

| condition | claims | runs | identical labels | switched (f) | SD n = 2,000 | SD difference | SD n = 3,000 | SD difference |
|---|---|---|---|---|---|---|---|---|
| no passages | 450 | original-r1 | 388 = 86.2% | 49 = 10.9% | 0.5 | 0.7 | 0.4 | 0.6 |
| no passages | 450 | original-r2 | 396 = 88.0% | 45 = 10.0% | 0.5 | 0.7 | 0.4 | 0.6 |
| no passages | 450 | r1-r2 | 398 = 88.4% | 38 = 8.4% | 0.5 | 0.6 | 0.4 | 0.5 |
| no passages | 300 S/R | original-r1 | 255 = 85.0% | 33 = 11.0% | 0.5 | 0.7 | 0.4 | 0.6 |
| no passages | 300 S/R | original-r2 | 260 = 86.7% | 32 = 10.7% | 0.5 | 0.7 | 0.4 | 0.6 |
| no passages | 300 S/R | r1-r2 | 268 = 89.3% | 19 = 6.3% | 0.4 | 0.6 | 0.3 | 0.5 |
| web | 450 | original-r1 | 406 = 90.2% | 35 = 7.8% | 0.4 | 0.6 | 0.4 | 0.5 |
| web | 450 | original-r2 | 400 = 88.9% | 39 = 8.7% | 0.5 | 0.7 | 0.4 | 0.5 |
| web | 450 | r1-r2 | 400 = 88.9% | 40 = 8.9% | 0.5 | 0.7 | 0.4 | 0.5 |
| web | 300 S/R | original-r1 | 278 = 92.7% | 17 = 5.7% | 0.4 | 0.5 | 0.3 | 0.4 |
| web | 300 S/R | original-r2 | 273 = 91.0% | 21 = 7.0% | 0.4 | 0.6 | 0.3 | 0.5 |
| web | 300 S/R | r1-r2 | 272 = 90.7% | 24 = 8.0% | 0.4 | 0.6 | 0.4 | 0.5 |
| RAG | 450 | original-r1 | 427 = 94.9% | 20 = 4.4% | 0.3 | 0.5 | 0.3 | 0.4 |
| RAG | 450 | original-r2 | 429 = 95.3% | 15 = 3.3% | 0.3 | 0.4 | 0.2 | 0.3 |
| RAG | 450 | r1-r2 | 420 = 93.3% | 25 = 5.6% | 0.4 | 0.5 | 0.3 | 0.4 |
| RAG | 300 S/R | original-r1 | 287 = 95.7% | 11 = 3.7% | 0.3 | 0.4 | 0.2 | 0.3 |
| RAG | 300 S/R | original-r2 | 290 = 96.7% | 6 = 2.0% | 0.2 | 0.3 | 0.2 | 0.3 |
| RAG | 300 S/R | r1-r2 | 283 = 94.3% | 15 = 5.0% | 0.4 | 0.5 | 0.3 | 0.4 |
| gold | 450 | original-r1 | 410 = 91.1% | 38 = 8.4% | 0.5 | 0.6 | 0.4 | 0.5 |
| gold | 450 | original-r2 | 421 = 93.6% | 26 = 5.8% | 0.4 | 0.5 | 0.3 | 0.4 |
| gold | 450 | r1-r2 | 418 = 92.9% | 32 = 7.1% | 0.4 | 0.6 | 0.3 | 0.5 |
| gold | 300 S/R | original-r1 | 278 = 92.7% | 20 = 6.7% | 0.4 | 0.6 | 0.3 | 0.5 |
| gold | 300 S/R | original-r2 | 287 = 95.7% | 10 = 3.3% | 0.3 | 0.4 | 0.2 | 0.3 |
| gold | 300 S/R | r1-r2 | 284 = 94.7% | 16 = 5.3% | 0.4 | 0.5 | 0.3 | 0.4 |

| condition | claims | original | r1 | r2 |
|---|---|---|---|---|
| no passages | 450 | 48.0% | 44.7% | 46.0% |
| no passages | 300 S/R | 37.0% | 32.7% | 34.3% |
| web | 450 | 61.6% | 62.7% | 61.8% |
| web | 300 S/R | 79.0% | 81.3% | 80.7% |
| RAG | 450 | 69.6% | 69.1% | 70.7% |
| RAG | 300 S/R | 83.0% | 82.0% | 83.7% |
| gold | 450 | 72.0% | 72.4% | 72.9% |
| gold | 300 S/R | 83.3% | 80.0% | 82.0% |

Accuracy differences between two runs of the same condition, claim by claim (b: right only in the first run, c: right only in the second):

| condition | claims | runs | b | c | difference, points | 95% interval | McNemar p |
|---|---|---|---|---|---|---|---|
| no passages | 450 | original-r1 | 32 | 17 | 3.3 | 0.3 to 6.4 | 0.044 |
| no passages | 450 | original-r2 | 27 | 18 | 2.0 | -0.9 to 4.9 | 0.23 |
| no passages | 450 | r1-r2 | 16 | 22 | -1.3 | -4.0 to 1.3 | 0.42 |
| no passages | 300 S/R | original-r1 | 23 | 10 | 4.3 | 0.6 to 8.1 | 0.035 |
| no passages | 300 S/R | original-r2 | 20 | 12 | 2.7 | -1.0 to 6.4 | 0.22 |
| no passages | 300 S/R | r1-r2 | 7 | 12 | -1.7 | -4.5 to 1.2 | 0.36 |
| web | 450 | original-r1 | 15 | 20 | -1.1 | -3.7 to 1.5 | 0.50 |
| web | 450 | original-r2 | 19 | 20 | -0.2 | -2.9 to 2.5 | 1.0 |
| web | 450 | r1-r2 | 22 | 18 | 0.9 | -1.9 to 3.6 | 0.64 |
| web | 300 S/R | original-r1 | 5 | 12 | -2.3 | -5.0 to 0.3 | 0.14 |
| web | 300 S/R | original-r2 | 8 | 13 | -1.7 | -4.7 to 1.3 | 0.38 |
| web | 300 S/R | r1-r2 | 13 | 11 | 0.7 | -2.5 to 3.9 | 0.84 |
| RAG | 450 | original-r1 | 11 | 9 | 0.4 | -1.5 to 2.4 | 0.82 |
| RAG | 450 | original-r2 | 5 | 10 | -1.1 | -2.8 to 0.6 | 0.30 |
| RAG | 450 | r1-r2 | 9 | 16 | -1.6 | -3.7 to 0.6 | 0.23 |
| RAG | 300 S/R | original-r1 | 7 | 4 | 1.0 | -1.2 to 3.2 | 0.55 |
| RAG | 300 S/R | original-r2 | 2 | 4 | -0.7 | -2.3 to 0.9 | 0.69 |
| RAG | 300 S/R | r1-r2 | 5 | 10 | -1.7 | -4.2 to 0.9 | 0.30 |
| gold | 450 | original-r1 | 18 | 20 | -0.4 | -3.1 to 2.2 | 0.87 |
| gold | 450 | original-r2 | 11 | 15 | -0.9 | -3.1 to 1.3 | 0.56 |
| gold | 450 | r1-r2 | 15 | 17 | -0.4 | -2.9 to 2.0 | 0.86 |
| gold | 300 S/R | original-r1 | 15 | 5 | 3.3 | 0.4 to 6.2 | 0.041 |
| gold | 300 S/R | original-r2 | 7 | 3 | 1.3 | -0.7 to 3.4 | 0.34 |
| gold | 300 S/R | r1-r2 | 5 | 11 | -2.0 | -4.6 to 0.6 | 0.21 |

Two rerun SDs of a difference for each paired comparison between conditions: the larger SD of the two conditions over the three run pairs, from the 450 claims for the 3,000-claim comparisons and from their 300 supported/refuted claims for the 2,000-claim ones.

| claims | pair | difference, points | level | 2 rerun SDs | from | gap below 2 SDs |
|---|---|---|---|---|---|---|
| 3,000 | RAG vs web | 9.6 | no | 1.1 | web r1-r2 | no |
| 3,000 | RAG vs gold | -2.0 | no | 1.1 | gold original-r1 | no |
| 3,000 | gold vs web | 11.6 | no | 1.1 | web r1-r2 | no |
| 3,000 | RAG vs no passages | 25.7 | no | 1.2 | no passages original-r1 | no |
| 3,000 | gold vs no passages | 27.7 | no | 1.2 | no passages original-r1 | no |
| 3,000 | web vs no passages | 16.1 | no | 1.2 | no passages original-r1 | no |
| 2,000 S/R | RAG vs web | 1.7 | no | 1.3 | web r1-r2 | no |
| 2,000 S/R | RAG vs gold | 0.8 | yes | 1.2 | gold original-r1 | yes |
| 2,000 S/R | gold vs web | 1.0 | yes | 1.3 | web r1-r2 | yes |
| 2,000 S/R | RAG vs no passages | 48.4 | no | 1.5 | no passages original-r1 | no |
| 2,000 S/R | gold vs no passages | 47.6 | no | 1.5 | no passages original-r1 | no |
| 2,000 S/R | web vs no passages | 46.7 | no | 1.5 | no passages original-r1 | no |

User texts of the repeats against the original rows: r1/no_retrieval 0 of 450 differ; r1/gold 0 of 450 differ; r1/retrieved 0 of 450 differ; r1/web 0 of 450 differ; r2/no_retrieval 0 of 450 differ; r2/gold 0 of 450 differ; r2/retrieved 0 of 450 differ; r2/web 0 of 450 differ.

## 6. Multi-hop claims: supported and refuted claims, equal weight

The 2,000 supported and refuted claims of the original run; NOT ENOUGH INFO claims are left out of this section. A group's equal-weight figure is the mean of its supported and its refuted figure, half each, so that a group's label mix does not move it. The two kinds hold 700 + 700 and 300 + 300 claims, so for them the equal-weight figure is the plain one. Intervals are 95%: Wilson for one label's accuracy; Newcombe's, from the two Wilson intervals, for a gap between two sets of claims of one label; the paired formula with McNemar's exact p for a gap between two conditions on the same claims of one label; a paired bootstrap (10,000 resamples, seed 20261121, the claims resampled within each label and group) for every equal-weight figure, gap and change of a gap. Gaps are in percentage points. Tags with fewer than 100 supported/refuted claims get their numbers but no conclusions. The challenge tags come from `data/scaled/claims_3000.jsonl` (SHA-256 checked).

Two meanings of "multi-hop": **two or more articles** (`kind` `needs2`: every evidence set spans two or more pages), 600 claims; the **Multi-hop Reasoning** challenge tag (the verifying annotator's view of the main difficulty; the pick did not balance on it), 488 claims. 427 claims are both (266 supported, 161 refuted).

The repeat runs of section 5 hold 210 of these supported/refuted claims on one article and 90 on two or more, so this section uses the original run only, and its subgroup gaps have no rerun SD of their own.

### One article against two or more

| condition | claims | SUPPORTS | REFUTES | equal weight |
|---|---|---|---|---|
| no passages | one article (700 + 700) | 27.7% (24.5–31.1) | 39.3% (35.7–43.0) | 33.5% (31.1–35.9) |
| no passages | two or more (300 + 300) | 37.0% (31.7–42.6) | 41.7% (36.2–47.3) | 39.3% (35.5–43.2) |
| web | one article (700 + 700) | 80.3% (77.2–83.1) | 88.3% (85.7–90.5) | 84.3% (82.4–86.1) |
| web | two or more (300 + 300) | 79.3% (74.4–83.5) | 73.3% (68.1–78.0) | 76.3% (72.8–79.7) |
| RAG | one article (700 + 700) | 84.7% (81.9–87.2) | 86.1% (83.4–88.5) | 85.4% (83.6–87.2) |
| RAG | two or more (300 + 300) | 85.7% (81.2–89.2) | 73.0% (67.7–77.7) | 79.3% (76.0–82.5) |
| gold | one article (700 + 700) | 85.1% (82.3–87.6) | 82.7% (79.7–85.3) | 83.9% (81.9–85.8) |
| gold | two or more (300 + 300) | 87.0% (82.7–90.3) | 73.7% (68.4–78.3) | 80.3% (77.2–83.3) |

Gap, one article minus two or more:

| condition | SUPPORTS (Newcombe) | REFUTES (Newcombe) | equal weight (bootstrap) |
|---|---|---|---|
| no passages | -9.3 (-15.7 to -3.0) | -2.4 (-9.1 to 4.2) | -5.8 (-10.5 to -1.3) |
| web | 1.0 (-4.3 to 6.6) | 15.0 (9.6 to 20.7) | 8.0 (4.2 to 11.9) |
| RAG | -1.0 (-5.5 to 4.1) | 13.1 (7.7 to 18.9) | 6.1 (2.4 to 9.9) |
| gold | -1.9 (-6.2 to 3.1) | 9.0 (3.5 to 14.9) | 3.6 (-0.0 to 7.3) |

Paired gaps within each kind (RAG minus the other condition, on the same claims):

| pair | claims | SUPPORTS (paired formula, McNemar p) | REFUTES (paired formula, McNemar p) | equal weight (bootstrap) |
|---|---|---|---|---|
| RAG − no passages | one article | 57.0 (53.1 to 60.9), p 5.8e-108 | 46.9 (43.0 to 50.8), p 2.3e-86 | 51.9 (49.2 to 54.6) |
| RAG − no passages | two or more | 48.7 (42.6 to 54.7), p 1.6e-38 | 31.3 (25.5 to 37.1), p 1.8e-22 | 40.0 (35.8 to 44.2) |
| RAG − gold | one article | -0.4 (-2.6 to 1.8), p 0.80 | 3.4 (1.3 to 5.6), p 0.0027 | 1.5 (-0.0 to 3.1) |
| RAG − gold | two or more | -1.3 (-4.9 to 2.2), p 0.58 | -0.7 (-4.7 to 3.4), p 0.87 | -1.0 (-3.7 to 1.7) |
| RAG − web | one article | 4.4 (1.6 to 7.3), p 0.0032 | -2.1 (-4.3 to 0.0), p 0.072 | 1.1 (-0.7 to 2.9) |
| RAG − web | two or more | 6.3 (2.1 to 10.6), p 0.0054 | -0.3 (-4.8 to 4.1), p 1.0 | 3.0 (0.0 to 6.0) |

Change of each paired gap from one article to two or more (the gap on two or more minus the gap on one article; bootstrap):

| pair | SUPPORTS | REFUTES | equal weight |
|---|---|---|---|
| RAG − no passages | -8.3 (-15.7 to -1.0) | -15.5 (-22.6 to -8.6) | -11.9 (-16.9 to -6.9) |
| RAG − gold | -0.9 (-5.2 to 3.2) | -4.1 (-8.7 to 0.4) | -2.5 (-5.6 to 0.6) |
| RAG − web | 1.9 (-3.0 to 7.0) | 1.8 (-3.2 to 6.9) | 1.9 (-1.6 to 5.5) |

### What the five RAG passages covered, by kind

Pages, not sentences: a claim counts for "the whole gold set" when every page of the evidence set shown in the gold condition is among the pages of the five passages. Whether the passages held the gold sentences themselves was not measured.

| claims | measure | SUPPORTS | REFUTES | equal weight |
|---|---|---|---|---|
| one article | a gold page in the top 5 (share of claims) | 696 / 700 = 99.4% (98.5–99.8) | 694 / 700 = 99.1% (98.1–99.6) | 99.3% (98.8–99.7) |
| one article | the whole gold set in the top 5 (share of claims) | 696 / 700 = 99.4% (98.5–99.8) | 694 / 700 = 99.1% (98.1–99.6) | 99.3% (98.8–99.7) |
| one article | RAG accuracy when the whole gold set is in the top 5 | 593 / 696 = 85.2% (82.4–87.6) | 601 / 694 = 86.6% (83.9–88.9) | 85.9% (84.0–87.7) |
| one article | RAG accuracy when it is not | 0 / 4 = 0.0% (0.0–49.0) | 2 / 6 = 33.3% (9.7–70.0) | 16.7% (0.0–33.3) |
| one article | RAG accuracy, gold set in minus not in (points; Newcombe, bootstrap) | 85.2 (36.1 to 87.6) | 53.3 (16.5 to 77.0) | 69.2 (51.5 to 86.4) |
| two or more | a gold page in the top 5 (share of claims) | 300 / 300 = 100.0% (98.7–100.0) | 297 / 300 = 99.0% (97.1–99.7) | 99.5% (98.8–100.0) |
| two or more | the whole gold set in the top 5 (share of claims) | 198 / 300 = 66.0% (60.5–71.1) | 131 / 300 = 43.7% (38.2–49.3) | 54.8% (51.0–58.8) |
| two or more | RAG accuracy when the whole gold set is in the top 5 | 174 / 198 = 87.9% (82.6–91.7) | 90 / 131 = 68.7% (60.3–76.0) | 78.3% (73.8–82.9) |
| two or more | RAG accuracy when it is not | 83 / 102 = 81.4% (72.7–87.7) | 129 / 169 = 76.3% (69.4–82.1) | 78.9% (73.7–83.8) |
| two or more | RAG accuracy, gold set in minus not in (points; Newcombe, bootstrap) | 6.5 (-1.8 to 16.0) | -7.6 (-17.8 to 2.5) | -0.6 (-7.3 to 6.2) |

### The challenge tags

| tag | claims (S + R) | SUPPORTS: one article / two or more | REFUTES: one article / two or more | |
|---|---|---|---|---|
| Other | 1,311 (602 + 709) | 574 / 28 | 614 / 95 |  |
| Multi-hop Reasoning | 488 (319 + 169) | 53 / 266 | 8 / 161 |  |
| Entity Disambiguation | 80 (34 + 46) | 31 / 3 | 26 / 20 | fewer than 100: numbers only |
| Search terms not in claim | 33 (18 + 15) | 16 / 2 | 9 / 6 | fewer than 100: numbers only |
| Numerical Reasoning | 57 (6 + 51) | 6 / 0 | 34 / 17 | fewer than 100: numbers only |
| Combining Tables and Text | 31 (21 + 10) | 20 / 1 | 9 / 1 | fewer than 100: numbers only |

Equal-weight accuracy per tag (bootstrap interval):

| tag | no passages | web | RAG | gold |
|---|---|---|---|---|
| Other | 34.0% (31.5–36.5) | 83.1% (81.0–85.1) | 85.4% (83.4–87.3) | 83.5% (81.4–85.5) |
| Multi-hop Reasoning | 38.7% (34.2–43.3) | 77.8% (73.9–81.6) | 79.4% (75.5–83.2) | 81.4% (77.7–85.0) |
| Entity Disambiguation | 32.9% (22.6–43.5) | 77.0% (67.6–85.8) | 77.7% (68.6–86.5) | 77.0% (67.5–85.8) |
| Search terms not in claim | 27.8% (12.8–43.3) | 81.1% (67.2–93.9) | 87.8% (75.6–97.2) | 75.6% (60.0–90.0) |
| Numerical Reasoning | 27.0% (13.7–45.1) | 72.1% (53.4–90.7) | 83.8% (66.2–96.1) | 86.8% (69.1–98.0) |
| Combining Tables and Text | 49.0% (29.8–66.4) | 80.5% (65.5–95.0) | 68.3% (51.0–85.5) | 70.7% (53.3–87.9) |

Accuracy per tag and label, SUPPORTS / REFUTES (Wilson intervals in the JSON):

| tag | no passages | web | RAG | gold |
|---|---|---|---|---|
| Other | 27.7 / 40.2 | 80.2 / 85.9 | 86.5 / 84.2 | 85.9 / 81.1 |
| Multi-hop Reasoning | 35.4 / 42.0 | 79.3 / 76.3 | 83.7 / 75.1 | 87.1 / 75.7 |
| Entity Disambiguation | 35.3 / 30.4 | 82.4 / 71.7 | 79.4 / 76.1 | 82.4 / 71.7 |
| Search terms not in claim | 22.2 / 33.3 | 88.9 / 73.3 | 88.9 / 86.7 | 77.8 / 73.3 |
| Numerical Reasoning | 16.7 / 37.3 | 50.0 / 94.1 | 83.3 / 84.3 | 83.3 / 90.2 |
| Combining Tables and Text | 38.1 / 60.0 | 81.0 / 80.0 | 66.7 / 70.0 | 71.4 / 70.0 |

### Multi-hop Reasoning against Other

Per condition: SUPPORTS / REFUTES accuracy and the equal-weight accuracy of each tag, overall and within each kind, and the gap Multi-hop Reasoning minus Other.

| claims (Multi-hop Reasoning S + R; Other S + R) | condition | Multi-hop Reasoning | Other | gap SUPPORTS / REFUTES (Newcombe) | gap, equal weight (bootstrap) |
|---|---|---|---|---|---|
| all (319 + 169; 602 + 709) | no passages | 35.4 / 42.0; 38.7% (34.2–43.3) | 27.7 / 40.2; 34.0% (31.5–36.5) | 7.7 (1.4 to 14.1) / 1.8 (-6.2 to 10.1) | 4.7 (-0.5 to 9.9) |
| all (319 + 169; 602 + 709) | web | 79.3 / 76.3; 77.8% (73.9–81.6) | 80.2 / 85.9; 83.1% (81.0–85.1) | -0.9 (-6.6 to 4.4) / -9.6 (-16.9 to -3.2) | -5.2 (-9.7 to -0.9) |
| all (319 + 169; 602 + 709) | RAG | 83.7 / 75.1; 79.4% (75.5–83.2) | 86.5 / 84.2; 85.4% (83.4–87.3) | -2.8 (-7.9 to 1.9) / -9.1 (-16.5 to -2.5) | -6.0 (-10.2 to -1.6) |
| all (319 + 169; 602 + 709) | gold | 87.1 / 75.7; 81.4% (77.7–85.0) | 85.9 / 81.1; 83.5% (81.4–85.5) | 1.3 (-3.6 to 5.7) / -5.4 (-12.9 to 1.2) | -2.0 (-6.2 to 2.1) |
| one article (53 + 8; 574 + 614) | no passages | 28.3 / 12.5; 20.4% (9.4–34.8) | 27.5 / 40.6; 34.0% (31.3–36.7) | 0.8 (-10.2 to 14.5) / -28.1 (-39.0 to 6.7) | -13.6 (-24.9 to 1.1) |
| one article (53 + 8; 574 + 614) | web | 77.4 / 100.0; 88.7% (83.0–94.3) | 80.7 / 87.9; 84.3% (82.2–86.3) | -3.3 (-16.5 to 6.5) / 12.1 (-20.5 to 14.9) | 4.4 (-1.8 to 10.1) |
| one article (53 + 8; 574 + 614) | RAG | 73.6 / 100.0; 86.8% (80.2–92.5) | 86.8 / 85.8; 86.3% (84.3–88.2) | -13.2 (-26.6 to -2.8) / 14.2 (-18.4 to 17.2) | 0.5 (-6.1 to 6.5) |
| one article (53 + 8; 574 + 614) | gold | 88.7 / 100.0; 94.3% (89.6–98.1) | 85.7 / 82.7; 84.2% (82.1–86.2) | 3.0 (-8.6 to 9.7) / 17.3 (-15.3 to 20.5) | 10.1 (5.1 to 14.5) |
| two or more (266 + 161; 28 + 95) | no passages | 36.8 / 43.5; 40.2% (35.4–45.1) | 32.1 / 37.9; 35.0% (25.2–45.2) | 4.7 (-14.6 to 20.1) / 5.6 (-6.9 to 17.5) | 5.1 (-6.1 to 16.0) |
| two or more (266 + 161; 28 + 95) | web | 79.7 / 75.2; 77.4% (73.2–81.5) | 71.4 / 72.6; 72.0% (62.3–81.1) | 8.3 (-6.0 to 27.3) / 2.5 (-8.2 to 14.0) | 5.4 (-4.6 to 16.0) |
| two or more (266 + 161; 28 + 95) | RAG | 85.7 / 73.9; 79.8% (75.8–83.7) | 82.1 / 73.7; 77.9% (68.9–85.7) | 3.6 (-7.5 to 21.7) / 0.2 (-10.5 to 11.7) | 1.9 (-7.0 to 11.6) |
| two or more (266 + 161; 28 + 95) | gold | 86.8 / 74.5; 80.7% (76.7–84.5) | 89.3 / 70.5; 79.9% (72.2–86.7) | -2.4 (-10.8 to 14.4) / 4.0 (-7.0 to 15.6) | 0.8 (-7.3 to 9.2) |

RAG minus gold within each tag, on the same claims, and the difference between the two tags' gaps (Multi-hop Reasoning minus Other; bootstrap):

| tag | SUPPORTS (paired formula, McNemar p) | REFUTES (paired formula, McNemar p) | equal weight (bootstrap) |
|---|---|---|---|
| Multi-hop Reasoning | -3.4 (-7.0 to 0.1), p 0.080 | -0.6 (-6.2 to 5.0), p 1.0 | -2.0 (-5.3 to 1.2) |
| Other | 0.7 (-1.5 to 2.9), p 0.66 | 3.1 (1.0 to 5.2), p 0.0062 | 1.9 (0.3 to 3.4) |
| difference, Multi-hop Reasoning − Other | -4.1 (-8.2 to -0.0) | -3.7 (-9.6 to 2.3) | -3.9 (-7.5 to -0.2) |
