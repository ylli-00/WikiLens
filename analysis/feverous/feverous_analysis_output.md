# FEVEROUS analysis output

## Example record (first train record with non-empty evidence)

```
    keys: evidence (list), id (int), claim (str), label (str), annotator_operations (list), challenge (str)
    id = 24435; label = 'REFUTES'; challenge = 'Numerical Reasoning'
    claim = 'Michael Folivi competed with ten teams from 2016 to 2021, appearing in 54 games and making seven goals in total.'
    evidence: list of 2 set(s); set keys = ['content', 'context']
    evidence[0].content (list of 5 str) = ['Michael Folivi_cell_1_2_0', 'Michael Folivi_cell_1_7_0', 'Michael Folivi_cell_1_8_0', 'Michael Folivi_cell_1_9_0', 'Michael Folivi_cell_1_12_0']
    evidence[0].context (dict str -> list[str]); e.g. 'Michael Folivi_cell_1_2_0' -> ['Michael Folivi_title', 'Michael Folivi_section_4', 'Michael Folivi_header_cell_1_0_0']
    annotator_operations: list of 19 dicts; first 3 = [{"operation": "start", "value": "start", "time": "0"}, {"operation": "Now on", "value": "?search=", "time": "0.78"}, {"operation": "search", "value": "Michael Folivi", "time": "78.101"}]
```

## Format checks

| check (count of violations) | train | dev | train+dev | examples (claim id, ...) |
|---|---|---|---|---|
| extra_record_keys | 0 | 0 | 0 |  |
| extra_evidence_set_keys | 0 | 0 | 0 |  |
| empty_content_set | 0 | 0 | 0 |  |
| duplicate_id_within_set | 0 | 0 | 0 |  |
| context_keys_differ_from_content | 0 | 0 | 0 |  |
| duplicate_evidence_set_in_claim | 2 | 0 | 2 | [31013, 38669] |
| split_rule_disagrees_with_regex | 0 | 0 | 0 |  |
| unexpected_id_arity | 0 | 0 | 0 |  |
| context_title_page_differs | 2 | 0 | 2 | [(81268, 'bupropion_sentence_0', 'Bupropion_title'), (93433, 'bupropion_sentence_1', 'Bupropion_title')] |
| context_section_page_differs | 0 | 0 | 0 |  |

Checks that raise instead of counting: header line, JSON syntax, required record keys, label values, evidence is a list of objects with content (list) and context (object), annotator_operations is a list of objects, every element id matches `<page>_<type>_<numbers>`.

Record and evidence-set key sets:

| split | object | keys | count |
|---|---|---|---|
| dev | record | annotator_operations,challenge,claim,evidence,id,label | 7,890 |
| train | record | annotator_operations,challenge,claim,evidence,id,label | 71,291 |
| dev | evidence set | content,context | 8,681 |
| train | evidence set | content,context | 77,492 |

Unicode form of page ids parsed from evidence (a join against an NFC-normalised Wikipedia table needs normalising):

| split | distinct pages | non-ASCII | non-ASCII and not NFC | ids removed by NFC | ids removed by NFC+casefold | NFC+casefold collision groups | examples of non-NFC ids (shown NFC) |
|---|---|---|---|---|---|---|---|
| train | 38,043 | 2,880 | 1,864 | 0 | 1 | [['Dog Boy', 'Dog boy']] | ['1906 Argentine Primera División', '1915 Argentine Primera División'] |
| dev | 4,314 | 262 | 154 | 0 | 0 | [] | ['1936–37 Cupa României', '2005–06 Süper Lig'] |
| train+dev | 41,974 | 3,130 | 2,011 | 0 | 1 | [['Dog Boy', 'Dog boy']] | ['1906 Argentine Primera División', '1915 Argentine Primera División'] |

annotator_operations, per claim:

| claims ... | train | dev |
|---|---|---|
| claims | 71,291 | 7,890 |
| claims not ending with finish | 0 | 0 |
| claims whose time goes backwards | 2 | 0 |
| claims with 2+ finish operations | 694 | 83 |
| claims with 2+ start operations | 0 | 0 |
| claims with a time > 1e+06 after the first finish | 694 | 83 |
| claims with operations after the first finish | 694 | 83 |
| claims without a finish operation | 0 | 0 |
| claims without a start operation | 68 | 77 |

annotator_operations, per operation entry:

| operation entries ... | train | dev |
|---|---|---|
| entries with keys {operation, time, value} | 1,134,590 | 119,460 |
| entries with time of type str | 1,134,590 | 119,460 |
| entries with value of type str | 1,134,590 | 119,460 |

| operation | train | dev |
|---|---|---|
| Highlighting | 478,346 | 48,676 |
| Now on | 224,150 | 18,605 |
| search | 93,380 | 11,119 |
| Highlighting deleted | 87,694 | 10,207 |
| finish | 72,018 | 7,978 |
| start | 71,223 | 7,813 |
| hyperlink | 49,323 | 6,271 |
| Page search | 35,341 | 5,542 |
| back-button-clicked | 22,631 | 3,192 |
| page-search-reset | 484 | 57 |

Claim ids: train 71,291 distinct, dev 7,890 distinct, shared between splits 0.

Repeated claim texts (exact string; and normalised = NFKC + casefold + all non-word characters removed):

| match | groups of 2+ claims | within train | within dev | across splits |
|---|---|---|---|---|
| exact | 0 | 0 | 0 | 0 |
| normalised | 8 | 7 | 1 | 0 |

| normalised-text group | labels | same label | same evidence sets |
|---|---|---|---|
| dev 1004, dev 4144 | SUPPORTS, SUPPORTS | True | False |
| train 7037, train 55294 | SUPPORTS, SUPPORTS | True | True |
| train 8483, train 26984 | SUPPORTS, SUPPORTS | True | True |
| train 16429, train 49215 | SUPPORTS, SUPPORTS | True | False |
| train 18829, train 73401 | SUPPORTS, SUPPORTS | True | True |
| train 20543, train 61460 | REFUTES, REFUTES | True | True |
| train 23097, train 72735 | REFUTES, REFUTES | True | False |
| train 42464, train 45234 | REFUTES, REFUTES | True | False |

Released records with id 31027 or evidence on page 'Days of War' (lookup for the paper's QA-interface screenshot):

| split | id | label | challenge | claim | evidence sets (content) | finish ops |
|---|---|---|---|---|---|---|
| train | 45610 | SUPPORTS | Other | 'Days of War is a multiplayer first-person shooter video game developed and published by Driven Arts, The game is set to fully release for Microsoft Windows, in 2020.' | [['Days of War_sentence_0', 'Days of War_sentence_2']] | 1 |
| train | 46874 | SUPPORTS | Combining Tables and Text | 'Days of War is a first-person shooter game with a multiplayer  mode, it is developed by driven art.' | [['Days of War_sentence_0', 'Days of War_cell_0_6_1']] | 1 |
| train | 23427 | SUPPORTS | Other | 'Days of War is a multiplayer first-person shooter video game developed and published by Driven Arts, It was released for Microsoft Windows through Steam Early Access on January 26, 2017.' | [['Days of War_sentence_0', 'Days of War_sentence_1']] | 1 |
| train | 31027 | REFUTES | Other | 'The 1st Canadian Division, an operational command and control formation of the Canadian Joint Operations Command, was raised shortly before the outbreak of World War I and started training upon the arrival in the Untied Kingdom in 1914.' | [['1st Canadian Division_sentence_0', '1st Canadian Division_sentence_8', '1st Canadian Division_sentence_9']] | 1 |

## Basic counts vs. paper (arXiv:2106.05707 v3: Table 2 claims, labels and evidence sets; Table 5 verification challenges)

| split | count | paper | files | diff | match |
|---|---|---|---|---|---|
| train | claims | 71,291 | 71,291 | +0 | yes |
| train | SUPPORTS | 41,835 | 41,835 | +0 | yes |
| train | REFUTES | 27,215 | 27,215 | +0 | yes |
| train | NOT ENOUGH INFO | 2,241 | 2,241 | +0 | yes |
| train | Multi-hop Reasoning | 11,624 | 11,624 | +0 | yes |
| train | Numerical Reasoning | 7,214 | 7,214 | +0 | yes |
| train | Entity Disambiguation | 1,353 | 1,353 | +0 | yes |
| train | Combining Tables and Text | 10,083 | 10,083 | +0 | yes |
| train | Search terms not in claim | 824 | 824 | +0 | yes |
| train | Other | 40,193 | 40,193 | +0 | yes |
| train | sets: sentences only | 31,607 | 31,607 | +0 | yes |
| train | sets: cells only (tables/lists, no sentence) | 25,020 | 25,020 | +0 | yes |
| train | sets: sentences + cells | 20,865 | 20,865 | +0 | yes |
| dev | claims | 7,890 | 7,890 | +0 | yes |
| dev | SUPPORTS | 3,908 | 3,908 | +0 | yes |
| dev | REFUTES | 3,481 | 3,481 | +0 | yes |
| dev | NOT ENOUGH INFO | 501 | 501 | +0 | yes |
| dev | Multi-hop Reasoning | 1,281 | 1,281 | +0 | yes |
| dev | Numerical Reasoning | 873 | 873 | +0 | yes |
| dev | Entity Disambiguation | 201 | 201 | +0 | yes |
| dev | Combining Tables and Text | 1,035 | 1,035 | +0 | yes |
| dev | Search terms not in claim | 131 | 131 | +0 | yes |
| dev | Other | 4,369 | 4,369 | +0 | yes |
| dev | sets: sentences only | 3,745 | 3,745 | +0 | yes |
| dev | sets: cells only (tables/lists, no sentence) | 2,738 | 2,738 | +0 | yes |
| dev | sets: sentences + cells | 2,468 | 2,198 | -270 | NO |

Rows starting with 'sets:' count evidence sets, not claims. To follow the paper's grouping, sets made of table and list elements without a sentence count as 'cells only' (they are 'mixed' in the modality table).

## Evidence sets per claim

| split | 1 set(s) | 2 set(s) | 3 set(s) |
|---|---|---|---|
| train | 65,764 (92.2%) | 4,853 (6.8%) | 674 (0.9%) |
| dev | 7,181 (91.0%) | 627 (7.9%) | 82 (1.0%) |
| train+dev | 72,945 (92.1%) | 5,480 (6.9%) | 756 (1.0%) |

## Element types in evidence content (occurrences)

| split | sentence | cell | header_cell | table_caption | item | total | distinct pages cited | max pages in one set |
|---|---|---|---|---|---|---|---|---|
| train | 100,044 (28.6%) | 230,261 (65.9%) | 12,142 (3.5%) | 3,515 (1.0%) | 3,594 (1.0%) | 349,556 | 38,043 | 12 |
| dev | 11,099 (33.1%) | 20,598 (61.3%) | 1,293 (3.9%) | 400 (1.2%) | 191 (0.6%) | 33,581 | 4,314 | 6 |
| train+dev | 111,143 (29.0%) | 250,859 (65.5%) | 13,435 (3.5%) | 3,915 (1.0%) | 3,785 (1.0%) | 383,137 | 41,974 | 12 |

## Multi-article claims (strict: min_pages >= 2; lenient: max_pages >= 2)

| split | label | claims | strict multi-article | lenient multi-article |
|---|---|---|---|---|
| train | ALL | 71,291 | 12,420 (17.42%) | 12,726 (17.85%) |
| train | SUPPORTS | 41,835 | 10,551 (25.22%) | 10,764 (25.73%) |
| train | REFUTES | 27,215 | 1,310 (4.81%) | 1,399 (5.14%) |
| train | NOT ENOUGH INFO | 2,241 | 559 (24.94%) | 563 (25.12%) |
| train | SUPPORTS+REFUTES (excl. NEI) | 69,050 | 11,861 (17.18%) | 12,163 (17.61%) |
| dev | ALL | 7,890 | 1,280 (16.22%) | 1,315 (16.67%) |
| dev | SUPPORTS | 3,908 | 978 (25.03%) | 998 (25.54%) |
| dev | REFUTES | 3,481 | 184 (5.29%) | 197 (5.66%) |
| dev | NOT ENOUGH INFO | 501 | 118 (23.55%) | 120 (23.95%) |
| dev | SUPPORTS+REFUTES (excl. NEI) | 7,389 | 1,162 (15.73%) | 1,195 (16.17%) |
| train+dev | ALL | 79,181 | 13,700 (17.30%) | 14,041 (17.73%) |
| train+dev | SUPPORTS | 45,743 | 11,529 (25.20%) | 11,762 (25.71%) |
| train+dev | REFUTES | 30,696 | 1,494 (4.87%) | 1,596 (5.20%) |
| train+dev | NOT ENOUGH INFO | 2,742 | 677 (24.69%) | 683 (24.91%) |
| train+dev | SUPPORTS+REFUTES (excl. NEI) | 76,439 | 13,023 (17.04%) | 13,358 (17.48%) |

## Pages per evidence set (share of evidence sets)

| split | label | 1 | 2 | 3 | 4 | 5+ | sets |
|---|---|---|---|---|---|---|---|
| train | ALL | 64,418 (83.1%) | 12,049 (15.5%) | 822 (1.1%) | 135 (0.2%) | 68 (0.1%) | 77,492 |
| train | SUPPORTS | 33,333 (75.1%) | 10,352 (23.3%) | 613 (1.4%) | 83 (0.2%) | 30 (0.1%) | 44,411 |
| train | REFUTES | 29,375 (95.3%) | 1,281 (4.2%) | 117 (0.4%) | 23 (0.1%) | 12 (0.0%) | 30,808 |
| train | NOT ENOUGH INFO | 1,710 (75.2%) | 416 (18.3%) | 92 (4.0%) | 29 (1.3%) | 26 (1.1%) | 2,273 |
| dev | ALL | 7,320 (84.3%) | 1,218 (14.0%) | 121 (1.4%) | 18 (0.2%) | 4 (0.0%) | 8,681 |
| dev | SUPPORTS | 3,211 (75.6%) | 957 (22.5%) | 73 (1.7%) | 6 (0.1%) | 2 (0.0%) | 4,249 |
| dev | REFUTES | 3,717 (94.8%) | 167 (4.3%) | 26 (0.7%) | 8 (0.2%) | 1 (0.0%) | 3,919 |
| dev | NOT ENOUGH INFO | 392 (76.4%) | 94 (18.3%) | 22 (4.3%) | 4 (0.8%) | 1 (0.2%) | 513 |
| train+dev | ALL | 71,738 (83.2%) | 13,267 (15.4%) | 943 (1.1%) | 153 (0.2%) | 72 (0.1%) | 86,173 |
| train+dev | SUPPORTS | 36,544 (75.1%) | 11,309 (23.2%) | 686 (1.4%) | 89 (0.2%) | 32 (0.1%) | 48,660 |
| train+dev | REFUTES | 33,092 (95.3%) | 1,448 (4.2%) | 143 (0.4%) | 31 (0.1%) | 13 (0.0%) | 34,727 |
| train+dev | NOT ENOUGH INFO | 2,102 (75.4%) | 510 (18.3%) | 114 (4.1%) | 33 (1.2%) | 27 (1.0%) | 2,786 |

## Evidence modality per set (share of evidence sets)

| split | label | sentence-only | table-only | list-only | mixed | sets |
|---|---|---|---|---|---|---|
| train | ALL | 31,607 (40.8%) | 24,649 (31.8%) | 215 (0.3%) | 21,021 (27.1%) | 77,492 |
| train | SUPPORTS | 16,718 (37.6%) | 10,120 (22.8%) | 35 (0.1%) | 17,538 (39.5%) | 44,411 |
| train | REFUTES | 13,819 (44.9%) | 14,136 (45.9%) | 160 (0.5%) | 2,693 (8.7%) | 30,808 |
| train | NOT ENOUGH INFO | 1,070 (47.1%) | 393 (17.3%) | 20 (0.9%) | 790 (34.8%) | 2,273 |
| dev | ALL | 3,745 (43.1%) | 2,691 (31.0%) | 25 (0.3%) | 2,220 (25.6%) | 8,681 |
| dev | SUPPORTS | 1,612 (37.9%) | 942 (22.2%) | 6 (0.1%) | 1,689 (39.8%) | 4,249 |
| dev | REFUTES | 1,830 (46.7%) | 1,670 (42.6%) | 14 (0.4%) | 405 (10.3%) | 3,919 |
| dev | NOT ENOUGH INFO | 303 (59.1%) | 79 (15.4%) | 5 (1.0%) | 126 (24.6%) | 513 |
| train+dev | ALL | 35,352 (41.0%) | 27,340 (31.7%) | 240 (0.3%) | 23,241 (27.0%) | 86,173 |
| train+dev | SUPPORTS | 18,330 (37.7%) | 11,062 (22.7%) | 41 (0.1%) | 19,227 (39.5%) | 48,660 |
| train+dev | REFUTES | 15,649 (45.1%) | 15,806 (45.5%) | 174 (0.5%) | 3,098 (8.9%) | 34,727 |
| train+dev | NOT ENOUGH INFO | 1,373 (49.3%) | 472 (16.9%) | 25 (0.9%) | 916 (32.9%) | 2,786 |

## Composition of 'mixed' sets (share of mixed sets; table = cell/header_cell/table_caption, list = item)

| split | label | list+sentence | list+sentence+table | list+table | sentence+table | sets |
|---|---|---|---|---|---|---|
| train | ALL | 531 (2.5%) | 344 (1.6%) | 156 (0.7%) | 19,990 (95.1%) | 21,021 |
| train | SUPPORTS | 310 (1.8%) | 248 (1.4%) | 104 (0.6%) | 16,876 (96.2%) | 17,538 |
| train | REFUTES | 129 (4.8%) | 49 (1.8%) | 39 (1.4%) | 2,476 (91.9%) | 2,693 |
| train | NOT ENOUGH INFO | 92 (11.6%) | 47 (5.9%) | 13 (1.6%) | 638 (80.8%) | 790 |
| dev | ALL | 49 (2.2%) | 33 (1.5%) | 22 (1.0%) | 2,116 (95.3%) | 2,220 |
| dev | SUPPORTS | 28 (1.7%) | 22 (1.3%) | 11 (0.7%) | 1,628 (96.4%) | 1,689 |
| dev | REFUTES | 11 (2.7%) | 7 (1.7%) | 9 (2.2%) | 378 (93.3%) | 405 |
| dev | NOT ENOUGH INFO | 10 (7.9%) | 4 (3.2%) | 2 (1.6%) | 110 (87.3%) | 126 |
| train+dev | ALL | 580 (2.5%) | 377 (1.6%) | 178 (0.8%) | 22,106 (95.1%) | 23,241 |
| train+dev | SUPPORTS | 338 (1.8%) | 270 (1.4%) | 115 (0.6%) | 18,504 (96.2%) | 19,227 |
| train+dev | REFUTES | 140 (4.5%) | 56 (1.8%) | 48 (1.5%) | 2,854 (92.1%) | 3,098 |
| train+dev | NOT ENOUGH INFO | 102 (11.1%) | 51 (5.6%) | 15 (1.6%) | 748 (81.7%) | 916 |

## Challenge tag vs. multi-article

The released `challenge` field is the verification challenge chosen by the verifying annotator: its train and dev counts equal the paper's Table 5 'Verification Challenges' rows (see above).

### train

| challenge tag | claims | strict multi-article | lenient multi-article |
|---|---|---|---|
| Combining Tables and Text | 10,083 | 1,271 (12.6%) | 1,312 (13.0%) |
| Entity Disambiguation | 1,353 | 238 (17.6%) | 242 (17.9%) |
| Multi-hop Reasoning | 11,624 | 9,145 (78.7%) | 9,274 (79.8%) |
| Numerical Reasoning | 7,214 | 509 (7.1%) | 527 (7.3%) |
| Other | 40,193 | 1,121 (2.8%) | 1,230 (3.1%) |
| Search terms not in claim | 824 | 136 (16.5%) | 141 (17.1%) |
| **all** | 71,291 | 12,420 (17.4%) | 12,726 (17.9%) |

Strict multi-article by label (train; cells are strict / claims):

| challenge tag | ALL | SUPPORTS | REFUTES | NOT ENOUGH INFO |
|---|---|---|---|---|
| Combining Tables and Text | 1,271 / 10,083 (12.6%) | 1,120 / 8,751 (12.8%) | 129 / 1,149 (11.2%) | 22 / 183 (12.0%) |
| Entity Disambiguation | 238 / 1,353 (17.6%) | 147 / 763 (19.3%) | 48 / 471 (10.2%) | 43 / 119 (36.1%) |
| Multi-hop Reasoning | 9,145 / 11,624 (78.7%) | 8,188 / 9,978 (82.1%) | 632 / 1,198 (52.8%) | 325 / 448 (72.5%) |
| Numerical Reasoning | 509 / 7,214 (7.1%) | 353 / 3,061 (11.5%) | 144 / 4,078 (3.5%) | 12 / 75 (16.0%) |
| Other | 1,121 / 40,193 (2.8%) | 660 / 18,890 (3.5%) | 338 / 20,043 (1.7%) | 123 / 1,260 (9.8%) |
| Search terms not in claim | 136 / 824 (16.5%) | 83 / 392 (21.2%) | 19 / 276 (6.9%) | 34 / 156 (21.8%) |
| **all** | 12,420 / 71,291 (17.4%) | 10,551 / 41,835 (25.2%) | 1,310 / 27,215 (4.8%) | 559 / 2,241 (24.9%) |

'Multi-hop Reasoning' tag as a predictor of multi-article evidence (train):

| target | tag & multi (TP) | tag & not multi (FP) | no tag & multi (FN) | neither (TN) | precision (share of tagged) | recall (share of multi covered by tag) |
|---|---|---|---|---|---|---|
| strict | 9,145 | 2,479 | 3,275 | 56,392 | 78.7% | 73.6% |
| lenient | 9,274 | 2,350 | 3,452 | 56,215 | 79.8% | 72.9% |

### dev

| challenge tag | claims | strict multi-article | lenient multi-article |
|---|---|---|---|
| Combining Tables and Text | 1,035 | 106 (10.2%) | 111 (10.7%) |
| Entity Disambiguation | 201 | 27 (13.4%) | 31 (15.4%) |
| Multi-hop Reasoning | 1,281 | 962 (75.1%) | 973 (76.0%) |
| Numerical Reasoning | 873 | 53 (6.1%) | 58 (6.6%) |
| Other | 4,369 | 118 (2.7%) | 128 (2.9%) |
| Search terms not in claim | 131 | 14 (10.7%) | 14 (10.7%) |
| **all** | 7,890 | 1,280 (16.2%) | 1,315 (16.7%) |

Strict multi-article by label (dev; cells are strict / claims):

| challenge tag | ALL | SUPPORTS | REFUTES | NOT ENOUGH INFO |
|---|---|---|---|---|
| Combining Tables and Text | 106 / 1,035 (10.2%) | 87 / 829 (10.5%) | 14 / 171 (8.2%) | 5 / 35 (14.3%) |
| Entity Disambiguation | 27 / 201 (13.4%) | 12 / 88 (13.6%) | 10 / 88 (11.4%) | 5 / 25 (20.0%) |
| Multi-hop Reasoning | 962 / 1,281 (75.1%) | 793 / 969 (81.8%) | 94 / 200 (47.0%) | 75 / 112 (67.0%) |
| Numerical Reasoning | 53 / 873 (6.1%) | 29 / 316 (9.2%) | 21 / 546 (3.8%) | 3 / 11 (27.3%) |
| Other | 118 / 4,369 (2.7%) | 52 / 1,662 (3.1%) | 42 / 2,447 (1.7%) | 24 / 260 (9.2%) |
| Search terms not in claim | 14 / 131 (10.7%) | 5 / 44 (11.4%) | 3 / 29 (10.3%) | 6 / 58 (10.3%) |
| **all** | 1,280 / 7,890 (16.2%) | 978 / 3,908 (25.0%) | 184 / 3,481 (5.3%) | 118 / 501 (23.6%) |

'Multi-hop Reasoning' tag as a predictor of multi-article evidence (dev):

| target | tag & multi (TP) | tag & not multi (FP) | no tag & multi (FN) | neither (TN) | precision (share of tagged) | recall (share of multi covered by tag) |
|---|---|---|---|---|---|---|
| strict | 962 | 319 | 318 | 6,291 | 75.1% | 75.2% |
| lenient | 973 | 308 | 342 | 6,267 | 76.0% | 74.0% |

### train+dev

| challenge tag | claims | strict multi-article | lenient multi-article |
|---|---|---|---|
| Combining Tables and Text | 11,118 | 1,377 (12.4%) | 1,423 (12.8%) |
| Entity Disambiguation | 1,554 | 265 (17.1%) | 273 (17.6%) |
| Multi-hop Reasoning | 12,905 | 10,107 (78.3%) | 10,247 (79.4%) |
| Numerical Reasoning | 8,087 | 562 (6.9%) | 585 (7.2%) |
| Other | 44,562 | 1,239 (2.8%) | 1,358 (3.0%) |
| Search terms not in claim | 955 | 150 (15.7%) | 155 (16.2%) |
| **all** | 79,181 | 13,700 (17.3%) | 14,041 (17.7%) |

Strict multi-article by label (train+dev; cells are strict / claims):

| challenge tag | ALL | SUPPORTS | REFUTES | NOT ENOUGH INFO |
|---|---|---|---|---|
| Combining Tables and Text | 1,377 / 11,118 (12.4%) | 1,207 / 9,580 (12.6%) | 143 / 1,320 (10.8%) | 27 / 218 (12.4%) |
| Entity Disambiguation | 265 / 1,554 (17.1%) | 159 / 851 (18.7%) | 58 / 559 (10.4%) | 48 / 144 (33.3%) |
| Multi-hop Reasoning | 10,107 / 12,905 (78.3%) | 8,981 / 10,947 (82.0%) | 726 / 1,398 (51.9%) | 400 / 560 (71.4%) |
| Numerical Reasoning | 562 / 8,087 (6.9%) | 382 / 3,377 (11.3%) | 165 / 4,624 (3.6%) | 15 / 86 (17.4%) |
| Other | 1,239 / 44,562 (2.8%) | 712 / 20,552 (3.5%) | 380 / 22,490 (1.7%) | 147 / 1,520 (9.7%) |
| Search terms not in claim | 150 / 955 (15.7%) | 88 / 436 (20.2%) | 22 / 305 (7.2%) | 40 / 214 (18.7%) |
| **all** | 13,700 / 79,181 (17.3%) | 11,529 / 45,743 (25.2%) | 1,494 / 30,696 (4.9%) | 677 / 2,742 (24.7%) |

'Multi-hop Reasoning' tag as a predictor of multi-article evidence (train+dev):

| target | tag & multi (TP) | tag & not multi (FP) | no tag & multi (FN) | neither (TN) | precision (share of tagged) | recall (share of multi covered by tag) |
|---|---|---|---|---|---|---|
| strict | 10,107 | 2,798 | 3,593 | 62,683 | 78.3% | 73.8% |
| lenient | 10,247 | 2,658 | 3,794 | 62,482 | 79.4% | 73.0% |

## Usable slices for a prose-only index

| split | label | claims | (a) >=1 sentence-only set | (b) >=1 sentence-only set with 2+ pages |
|---|---|---|---|---|
| train | ALL | 71,291 | 30,835 (43.3%) | 4,512 (6.3%) |
| train | SUPPORTS | 41,835 | 16,560 (39.6%) | 3,770 (9.0%) |
| train | REFUTES | 27,215 | 13,210 (48.5%) | 541 (2.0%) |
| train | NOT ENOUGH INFO | 2,241 | 1,065 (47.5%) | 201 (9.0%) |
| dev | ALL | 7,890 | 3,644 (46.2%) | 503 (6.4%) |
| dev | SUPPORTS | 3,908 | 1,591 (40.7%) | 361 (9.2%) |
| dev | REFUTES | 3,481 | 1,751 (50.3%) | 80 (2.3%) |
| dev | NOT ENOUGH INFO | 501 | 302 (60.3%) | 62 (12.4%) |
| train+dev | ALL | 79,181 | 34,479 (43.5%) | 5,015 (6.3%) |
| train+dev | SUPPORTS | 45,743 | 18,151 (39.7%) | 4,131 (9.0%) |
| train+dev | REFUTES | 30,696 | 14,961 (48.7%) | 621 (2.0%) |
| train+dev | NOT ENOUGH INFO | 2,742 | 1,367 (49.9%) | 263 (9.6%) |

| split | NEI claims | (c) NEI claims with non-empty evidence |
|---|---|---|
| train | 2,241 | 2,241 (100.0%) |
| dev | 501 | 501 (100.0%) |
| train+dev | 2,742 | 2,742 (100.0%) |

## Tier (strict multi-article, train+dev)

strict multi-article claims (train+dev): 13,700 -> tier 5  
lenient multi-article claims (train+dev): 14,041 -> tier 5 (same scale, for reference)  
excluding NOT ENOUGH INFO: strict 13,023 -> tier 5; lenient 13,358 -> tier 5

## Sensitivity and annotation-context proxies

Case-insensitive page names (page.casefold()) instead of exact strings:

| split | strict (exact) | strict (casefold) | lenient (exact) | lenient (casefold) |
|---|---|---|---|---|
| train | 12,420 (17.42%) | 12,420 (17.42%) | 12,726 (17.85%) | 12,726 (17.85%) |
| dev | 1,280 (16.22%) | 1,280 (16.22%) | 1,315 (16.67%) | 1,315 (16.67%) |
| train+dev | 13,700 (17.30%) | 13,700 (17.30%) | 14,041 (17.73%) | 14,041 (17.73%) |

Multi-unit evidence from annotation context: a unit is (page, set of section ids in the element's context). Strict = every set has 2+ units; lenient = some set has 2+ units. This approximates the tag's 'two or more sections or articles'. Not verified against the Wikipedia DB.

### Multi-unit (ctx): train

| challenge tag | claims | strict multi-unit (ctx) | lenient multi-unit (ctx) |
|---|---|---|---|
| Combining Tables and Text | 10,083 | 5,526 (54.8%) | 5,764 (57.2%) |
| Entity Disambiguation | 1,353 | 609 (45.0%) | 625 (46.2%) |
| Multi-hop Reasoning | 11,624 | 10,718 (92.2%) | 10,909 (93.8%) |
| Numerical Reasoning | 7,214 | 1,616 (22.4%) | 1,678 (23.3%) |
| Other | 40,193 | 4,924 (12.3%) | 5,282 (13.1%) |
| Search terms not in claim | 824 | 229 (27.8%) | 234 (28.4%) |
| **all** | 71,291 | 23,622 (33.1%) | 24,492 (34.4%) |

### Multi-unit (ctx): dev

| challenge tag | claims | strict multi-unit (ctx) | lenient multi-unit (ctx) |
|---|---|---|---|
| Combining Tables and Text | 1,035 | 529 (51.1%) | 565 (54.6%) |
| Entity Disambiguation | 201 | 86 (42.8%) | 94 (46.8%) |
| Multi-hop Reasoning | 1,281 | 1,138 (88.8%) | 1,169 (91.3%) |
| Numerical Reasoning | 873 | 180 (20.6%) | 192 (22.0%) |
| Other | 4,369 | 565 (12.9%) | 609 (13.9%) |
| Search terms not in claim | 131 | 24 (18.3%) | 25 (19.1%) |
| **all** | 7,890 | 2,522 (32.0%) | 2,654 (33.6%) |

### Multi-unit (ctx): train+dev

| challenge tag | claims | strict multi-unit (ctx) | lenient multi-unit (ctx) |
|---|---|---|---|
| Combining Tables and Text | 11,118 | 6,055 (54.5%) | 6,329 (56.9%) |
| Entity Disambiguation | 1,554 | 695 (44.7%) | 719 (46.3%) |
| Multi-hop Reasoning | 12,905 | 11,856 (91.9%) | 12,078 (93.6%) |
| Numerical Reasoning | 8,087 | 1,796 (22.2%) | 1,870 (23.1%) |
| Other | 44,562 | 5,489 (12.3%) | 5,891 (13.2%) |
| Search terms not in claim | 955 | 253 (26.5%) | 259 (27.1%) |
| **all** | 79,181 | 26,144 (33.0%) | 27,146 (34.3%) |

### Lead vs. later sections (ctx)

Lead vs. later sections from annotation context (an element is outside the lead when its context lists at least one section id). Gold sentences are counted once per claim (distinct element ids). Not verified against the Wikipedia DB.

| split | label | gold sentences | gold sentences outside lead (ctx) | claims with >=1 gold SENTENCE outside lead (ctx) | claims with >=1 gold element of ANY type outside lead (ctx) |
|---|---|---|---|---|---|
| train | ALL | 98,774 | 37,864 (38.3%) | 20,262 (28.4%) | 42,840 (60.1%) |
| train | SUPPORTS | 71,711 | 26,381 (36.8%) | 12,827 (30.7%) | 25,965 (62.1%) |
| train | REFUTES | 22,459 | 9,441 (42.0%) | 6,541 (24.0%) | 15,448 (56.8%) |
| train | NOT ENOUGH INFO | 4,604 | 2,042 (44.4%) | 894 (39.9%) | 1,427 (63.7%) |
| dev | ALL | 10,907 | 4,312 (39.5%) | 2,493 (31.6%) | 4,906 (62.2%) |
| dev | SUPPORTS | 6,934 | 2,591 (37.4%) | 1,323 (33.9%) | 2,522 (64.5%) |
| dev | REFUTES | 3,065 | 1,314 (42.9%) | 956 (27.5%) | 2,058 (59.1%) |
| dev | NOT ENOUGH INFO | 908 | 407 (44.8%) | 214 (42.7%) | 326 (65.1%) |
| train+dev | ALL | 109,681 | 42,176 (38.5%) | 22,755 (28.7%) | 47,746 (60.3%) |
| train+dev | SUPPORTS | 78,645 | 28,972 (36.8%) | 14,150 (30.9%) | 28,487 (62.3%) |
| train+dev | REFUTES | 25,524 | 10,755 (42.1%) | 7,497 (24.4%) | 17,506 (57.0%) |
| train+dev | NOT ENOUGH INFO | 5,512 | 2,449 (44.4%) | 1,108 (40.4%) | 1,753 (63.9%) |

