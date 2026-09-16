# FEVEROUS: how many claims need evidence from two or more Wikipedia pages?

Wikilense ground-truth assessment, using data downloaded on 2026-09-16.

**Where the numbers come from:**
- Dataset statistics and file-derived facts come from `feverous_analysis.py`. Its output is in `feverous_analysis_output.md`, with one row per claim in `feverous_claims.csv` and machine-readable counts in `feverous_stats.json`.
- File hashes and sizes, the Zenodo record, the remote database size, and the paper and thesis quotes come from separate checks, noted where they are used.

## 1. Conclusion

- **Strict multi-article claims, train+dev: 13,700 of 79,181 (17.30%) → tier 5.** A claim is strict when every alternative evidence set cites 2+ pages.
  Train 12,420 (17.42%), dev 1,280 (16.22%).
- **Lenient multi-article claims, train+dev: 14,041 (17.73%)**, also tier 5 on the same scale. A claim is lenient when at least one evidence set cites 2+ pages.
  Train 12,726, dev 1,315.
- **Without NOT ENOUGH INFO claims:** strict 13,023, lenient 13,358 (both still tier 5).
- **By label:** multi-article evidence is rare for REFUTES (4.87% strict), compared with SUPPORTS (25.20%) and NEI (24.69%) (train+dev).
- **Prose-only index:** 34,479 claims (43.5%) have at least one sentence-only evidence set, and 5,015 (6.3%) have a sentence-only set spanning 2+ pages.
- **Challenge tag:** the released "Multi-hop Reasoning" tag is an imperfect proxy for page counts. 78.3% of tagged claims are strict multi-article, and the tag covers 73.8% of strict multi-article claims (train+dev).
  - This tag was assigned by the *verifying* annotator.
  - The claim writer's tag (`expected_challenge`) is not in the released files, so comparing *that* tag with page counts is not determinable.
- **Paper counts:** the files match the paper's Table 2 and Table 5 exactly (claims, labels, all six verification-challenge tags, and five of six evidence-set counts). The exception is dev "Sentence+Cells": 2,468 in the paper against 2,198 in the files, and the paper is internally inconsistent there (section 5).
- **QA subset:** the paper's 8,474 duplicate-annotated QA samples (annotated two- or three-way) cannot be identified in the release, and the extra annotations were not released. The subset comparison is therefore not determinable (section 7).
- **Step 2 (Wikipedia DB) was not run.** The download is 10,353,775,701 bytes compressed and unpacks to a 53,486,538,752-byte SQLite file, over the 5 GB limit, so it waits for your approval.

## 2. Data

| file (fever.ai) | bytes | sha256 | lines |
|---|---|---|---|
| `data/feverous/feverous_train_challenges.jsonl` | 177,565,233 | `0c29ccba41e27c5b988ca5132085e8d67c7921f265707bea170bfbde12bceee7` | 71,292 (1 header + 71,291 claims) |
| `data/feverous/feverous_dev_challenges.jsonl` | 18,058,238 | `1ac8cfd964d4dcedc5de3375850fe3f93d39b89a73a475734bde864da1701f8f` | 7,891 (1 header + 7,890 claims) |

- **Source:** the "Training/Development Dataset" links on https://fever.ai/dataset/feverous.html. The unlabelled test set was skipped. Bytes, hashes and line counts were taken with `ls`, `sha256sum` and `wc -l`.
- **Licence:** per https://fever.ai/download/feverous/license.html, the annotations are "made available under the license terms described on the applicable Wikipedia article pages, or, where Wikipedia license terms are unavailable, under the Creative Commons Attribution-ShareAlike License (version 3.0)". The page metadata declares CC BY-SA 3.0.
- **Paper:** arXiv:2106.05707 v3 (NeurIPS 2021 Datasets and Benchmarks), used for all comparisons.
- **Reproduce (from the project root):** `python3 analysis/feverous/feverous_analysis.py --out-md analysis/feverous/feverous_analysis_output.md`.
  - It reads `data/feverous/` and also rewrites `feverous_claims.csv` and `feverous_stats.json` in `analysis/feverous/`.
  - It needs the standard library only and takes about 10 s here.
  - The CSV is UTF-8 with LF line endings and a header row. For MariaDB `LOAD DATA`, use `FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES`.

## 3. Definitions and parsing rule

- **Page of an element id:** `el.split("_")[0]` (the official rule). It was cross-checked against a strict regex `<page>_(sentence|cell|header_cell|table_caption|item)_<numbers>` on all 383,137 content ids, with 0 disagreements. No page id contains `_`. Pages are compared as exact strings. Casefolding changes no strict or lenient count. NFC normalisation doesn't change any count either, because no page id occurs in both forms (0 ids removed by NFC in the script output). It is still needed when joining to Wikipedia titles, which are stored in NFC.
- **Evidence:** only `content` counts. `context` is used only for the two proxies in Appendix A.3–A.4 and for format checks.
- **pages(set):** the number of distinct pages in the set's `content`. For each claim, min_pages and max_pages are taken over its evidence sets.
- **Strict multi-article:** min_pages ≥ 2. **Lenient:** max_pages ≥ 2.
- **Set modality:**
  - sentence-only: all ids are `sentence`
  - table-only: all ids are `cell`, `header_cell` or `table_caption`
  - list-only: all ids are `item`
  - mixed: anything else
- **Usable slices:**
  - (a) the claim has at least one sentence-only set
  - (b) the claim has at least one sentence-only set with 2+ pages
  - (c) an NEI claim with at least one non-empty evidence set
- **Percentages:** claim tables use claims as the denominator, set tables use evidence sets, and the lead table uses gold sentences (as labelled).
- **Tier (from the brief):** 5 = 10,000+; 4 = 5,000–9,999; 3 = 2,000–4,999; 2 = 500–1,999; 1 = 1–499; 0 = none.

## 4. Results

### 4.1 Counts against the paper

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

Rows starting with "sets:" count evidence sets, not claims. To match the paper's grouping, sets made of table and list elements with no sentence count as "cells only". In 4.4 they fall under "mixed".

### 4.2 Multi-article claims, per split and label

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

### 4.3 Pages per evidence set

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

### 4.4 Evidence modality per set

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

### 4.5 Challenge tag against strict multi-article (train+dev; train and dev in Appendix A.1)

The released `challenge` field is the **verification** challenge, chosen by the verifying annotator after verification (section 5, item 1).

| challenge tag | claims | strict multi-article | lenient multi-article |
|---|---|---|---|
| Combining Tables and Text | 11,118 | 1,377 (12.4%) | 1,423 (12.8%) |
| Entity Disambiguation | 1,554 | 265 (17.1%) | 273 (17.6%) |
| Multi-hop Reasoning | 12,905 | 10,107 (78.3%) | 10,247 (79.4%) |
| Numerical Reasoning | 8,087 | 562 (6.9%) | 585 (7.2%) |
| Other | 44,562 | 1,239 (2.8%) | 1,358 (3.0%) |
| Search terms not in claim | 955 | 150 (15.7%) | 155 (16.2%) |
| **all** | 79,181 | 13,700 (17.3%) | 14,041 (17.7%) |

| challenge tag | ALL | SUPPORTS | REFUTES | NOT ENOUGH INFO |
|---|---|---|---|---|
| Combining Tables and Text | 1,377 / 11,118 (12.4%) | 1,207 / 9,580 (12.6%) | 143 / 1,320 (10.8%) | 27 / 218 (12.4%) |
| Entity Disambiguation | 265 / 1,554 (17.1%) | 159 / 851 (18.7%) | 58 / 559 (10.4%) | 48 / 144 (33.3%) |
| Multi-hop Reasoning | 10,107 / 12,905 (78.3%) | 8,981 / 10,947 (82.0%) | 726 / 1,398 (51.9%) | 400 / 560 (71.4%) |
| Numerical Reasoning | 562 / 8,087 (6.9%) | 382 / 3,377 (11.3%) | 165 / 4,624 (3.6%) | 15 / 86 (17.4%) |
| Other | 1,239 / 44,562 (2.8%) | 712 / 20,552 (3.5%) | 380 / 22,490 (1.7%) | 147 / 1,520 (9.7%) |
| Search terms not in claim | 150 / 955 (15.7%) | 88 / 436 (20.2%) | 22 / 305 (7.2%) | 40 / 214 (18.7%) |
| **all** | 13,700 / 79,181 (17.3%) | 11,529 / 45,743 (25.2%) | 1,494 / 30,696 (4.9%) | 677 / 2,742 (24.7%) |

| target | tag & multi (TP) | tag & not multi (FP) | no tag & multi (FN) | neither (TN) | precision (share of tagged) | recall (share of multi covered by tag) |
|---|---|---|---|---|---|---|
| strict | 10,107 | 2,798 | 3,593 | 62,683 | 78.3% | 73.8% |
| lenient | 10,247 | 2,658 | 3,794 | 62,482 | 79.4% | 73.0% |

The tag's definition covers "two or more sections **or** articles", which may explain part of the gap. Counting (page, context-section-set) units instead of pages, 91.9% of tagged claims have 2+ units in every set (Appendix A.3, a proxy not checked against the Wikipedia DB).

### 4.6 Usable slices for a prose-only index

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

### 4.7 Tier

- Strict multi-article, train+dev: **13,700 → tier 5**
- Lenient multi-article, train+dev: **14,041 → tier 5**
- Excluding NEI: strict 13,023, lenient 13,358 (tier 5)

## 5. Mismatches and format notes

1. **Who assigns the challenge tag.** The brief says the claim writer assigns it. In the paper, the released `challenge` counts are the "Verification Challenges", chosen by the verifier.
   - Table 5 caption (v3): "Top: Expected verification challenges, selected during claim generation. Bottom: Verification challenges, selected by annotator after a claim was verified." All twelve train/dev tag counts in 4.1 match the bottom half.
   - The writer's `expected_challenge` (for example Multi-hop 17,248 train / 1,871 dev in Table 5) is listed in paper Sec 7.1 but is **not in the released files**: every record has exactly the six keys in the script's "Format checks". Comparing the writer's tag with page counts is therefore not determinable.
   - The "two or more sections or articles" definition is from Sec 3.1.1. The verification guidelines word it as "several pages/sections will be required for verification".
2. **Paper Table 2, dev "E_Sentence+Cells": the paper prints 2,468 and the files give 2,198 (−270).** The other five evidence-set cells match exactly under the grouping in 4.1. The paper itself is inconsistent:
   - Its Total column (25,395 = 20,865 + 2,468 + 2,062) uses 2,468.
   - Its dev percentages (43% / 32% / 25%) fit 2,198 (3,745 / 2,738 / 2,198 of 8,681 sets).
   - The TeX source has an author comment next to the table, `%77492 8681 (8467)`, which matches the train and dev evidence-set totals in the files (77,492 and 8,681).
3. **Other paper statistics:**
   - No count of pages per evidence set or per claim appears in v1–v3 (main text and appendix). The closest statement concerns claim writing, not evidence: "47300 annotations were prompted to use information from the same page, and 46428 from different pages" (Sec 7.5.1).
   - Table 1's "Evidence Sets by Type" (34,963 sentences, 28,760 tables, 24,667 combined) gives no split and does not equal Table 2's totals (38,941 / 30,574 / 25,395), so it was not compared.
4. **Element id format.**
   - The fever.ai page says the data "contains 5 fields" but lists 6, and the files have 6.
   - Its example shows `evidence` as a single object, but in the files it is always a list of 1–3 sets.
   - The pattern `[PAGE ID]_[EVIDENCE TYPE]_[NUMBER ID]`, which your brief also uses, implies one number. Cells and header cells carry 3 numbers, items 2, and sentences and captions 1 (0 exceptions). This does not affect the page rule (0 disagreements), so the analysis went ahead.
5. **Header line.** Line 1 of each file is a record with all six fields empty and must be skipped, as the official reader does.
6. **annotator_operations:**
   - `time` and `value` are strings.
   - 68 train / 77 dev claims have no `start` operation.
   - 694 train / 83 dev claims have operations after their first `finish`, and each of them has a timestamp above 10^6 after that point.
   - Time goes backwards in 2 train claims.
   - None of this is documented.
7. **Unicode and case in page ids.** 2,011 of the 3,130 non-ASCII page ids are not in NFC (they are decomposed). NFC normalisation merges no ids, but a join against an NFC-normalised Wikipedia table must normalise. One pair differs only by case ("Dog Boy" / "Dog boy"). Page `bupropion` (lowercase) is cited in train claims 81268 and 93433, whose context title is `Bupropion_title`.
8. **Duplicate evidence sets.** Train claims 31013 and 38669 each contain the same set twice, with ids in a different order. Both copies are counted as sets. This does not change strict, lenient or slice counts.
9. **The Zenodo DOI copy differs from the fever.ai files** (separate check via the Zenodo API). Record 10.5281/zenodo.4911508 (one version) has dev.jsonl at 17,827,949 bytes and train.jsonl at 175,493,294 bytes. Its dev records have no `challenge` field (dev checked in full; train by size only). Use the fever.ai `*_challenges.jsonl` files.
10. **Otherwise clean:**
    - no empty evidence sets, no ids repeated within a set, and context keys always equal content ids
    - evidence sets only have `content` and `context`
    - claim ids are unique and never shared between splits
    - all 2,742 NEI claims have non-empty evidence
    - no claim text repeats exactly. After normalising case, whitespace and punctuation there are 8 near-identical pairs (7 in train, 1 in dev), all with the same label on both sides (listed in the script output).

## 6. Not determined

- **Step 2 (Wikipedia database): not computed.** `feverous-wiki-pages-db.zip` is 10,353,775,701 bytes (HTTP Content-Length). Its central directory, read with a ranged HTTP request without downloading the file, lists one entry: `feverous_wikiv1.db`, 53,486,538,752 bytes uncompressed. That is over the 5 GB approval limit. Every Step 2 item is not computed, pending approval:
  - schema
  - page existence and misses
  - DB-based lead share
  - element types and paragraph markers
  - article lengths
  - metadata
- **Paragraph units.** Not determinable from the annotations: evidence ids contain only the five element types, but page structure lives in the DB.
- **Claim writer's challenge tag against page counts.** Not determinable, because `expected_challenge` is not in the released files (section 5, item 1).
- **Why the verifier's Multi-hop tag and page counts disagree.** Only the annotation-context proxy (Appendix A.3) is available.
- **Test split.** Not analysed; it is unlabelled and was skipped per the brief.

## 7. Quality-assurance subset (duplicate-annotated samples)

### 7.1 Answers

1. **Can the 8,474 QA samples be identified in the released files? No.** No field, flag, id list, file or documented rule identifies them.
2. **Are the second (or third) annotators' labels or evidence released? No.** The paper and the first author's thesis both say the duplicate annotations are not part of the dataset, and no official source hosts them. The label and evidence match counts are therefore not determinable.
3. **Does the subset look like the rest? Not determinable.** Without the subset, the comparison cannot be computed (label distribution and chi-square test, challenge tags, strict multi-article share, evidence modality). Following your instruction, no proxy was substituted.

### 7.2 What the sources say

- **Paper, arXiv v3, Sec 3.2 "Quality Control" (same text in v2 and the NeurIPS version):** "Around 10% of the claim verification annotations (8474 samples) were used for quality assurance. We measured two-way IAA using 66% of these samples, and three-way agreement with the remaining 33%. The samples were selected randomly proportionally to the number of annotations by each annotator. The κ over the verdict label was 0.65 both for two-way and three-way agreement. Duplicate annotations (and hence disagreements) due to measuring IAA are not considered for the dataset itself."
  - arXiv v1 lacks the sentences on random proportional sampling and on excluding duplicate annotations (checked in the v1 TeX source).
  - Two corrections to the request's wording: samples were annotated a second **or third** time (two-way and three-way), and the paper counts "samples" or "annotations", not claims.
- **R. Aly, PhD thesis "Neuro-symbolic fact verification" (Cambridge, Nov 2024; Apollo repository), Sec 3.2.3 "Quality control":** essentially the same passage, with minor wording changes. It adds "(5,648 samples)" for the two-way share and ends: "Duplicate annotations (and hence disagreements) due to measuring IAA are not included in the dataset itself."
- **Paper, Fig. 8 (annotation flowchart):** a box reads "Sample Claim from database, 10% duplicates as QA".
- **Paper, Sec 7.10.3 (describing Fig. 14, the QA interface):** "QA annotations with only partial agreement or complete disagreement are highlighted in the interface in red." Nothing says this data is released.

### 7.3 What was checked

| source | what was checked | result |
|---|---|---|
| Released train/dev files (script output, "Format checks") | Key sets of all records and evidence sets; annotator_operations structure and operation names; claim ids; exact and normalised repeated claim texts | Only `id, claim, label, evidence, annotator_operations, challenge` and `content, context`. One label per record, no annotator id, no QA flag, no second label or evidence. Ids are unique and never shared between splits. No exact text repeats. The 8 normalised near-identical pairs are separate records with their own ids, and nothing marks them as QA. |
| Operation logs (script output) | Operations after a claim's first `finish` | 777 claims (694 train, 83 dev). Undocumented, far from 8,474, and not used as a proxy. |
| Paper's QA-interface screenshot (Fig. 14) against the files (script lookup) | The id shown, "(31027)", and claims citing page "Days of War" | Released id 31027 is an unrelated claim ("The 1st Canadian Division …"), so interface ids are not released ids. The page lookup lists 3 released "Days of War" claims, and nothing links any of them to the screenshot. |
| fever.ai dataset page, licence page, dev dataset viewer, `/download/feverous/` | Full text, JSON-LD metadata, viewer fields, directory listing | No QA or agreement information. The viewer shows ID, verification challenge, evidence type, verdict, evidence and annotator operations. The directory listing is denied (403). |
| Zenodo 10.5281/zenodo.4911508 (concept 4911507) | REST API: versions and files | One version with 4 files (train, dev, two wiki zips). No QA file. |
| FEVER 2021 task and workshop pages; shared-task overview paper (ACL Anthology 2021.fever-1.1) | Full text, keyword search | Only the train/dev release and a blind test set. Nothing on QA data. |
| Paper arXiv v1–v3, NeurIPS proceedings and supplement | Full text, TeX sources, figures | QA described (7.2). The data-format section lists no QA field and no id list is mentioned. |
| First author's PhD thesis | Full text | As in 7.2. No id list or release. |
| Official repo github.com/Raldir/FEVEROUS | All files in the full git history, branches, tags, releases, every version of `download_data.sh`, all issues, PRs and comments | No data release beyond the fever.ai files, and no issue about QA. "QA Claims" appears only in internal annotation-platform code (`SELECT id FROM $claim_table WHERE evidence_annotators_num >= ?` with `$annotations_num = 2`). It runs on unreleased MySQL tables with no mapping to released ids. |
| github.com/Raldir/feverous-wip (early annotation-tool snapshot) | Full tree | Only calibration-phase internal row ids. |
| HuggingFace `fever/feverous` and author accounts | Dataset card, loader, author listings | The loader reads the fever.ai files; the annotation process is "[More Information Needed]". No QA data. |
| Web search (agreement / quality assurance / "8,474" / annotated twice) | — | Only mirrors of the paper and third-party material. |
| OpenReview reviews (h-flVCIlstW), Wayback Machine | Attempted | **Not reachable** (403 bot challenge; archive offline). A statement there cannot be ruled out. |

## Appendix A. Secondary tables (all from `feverous_analysis_output.md`)

### A.1 Challenge tag against multi-article, train

| challenge tag | claims | strict multi-article | lenient multi-article |
|---|---|---|---|
| Combining Tables and Text | 10,083 | 1,271 (12.6%) | 1,312 (13.0%) |
| Entity Disambiguation | 1,353 | 238 (17.6%) | 242 (17.9%) |
| Multi-hop Reasoning | 11,624 | 9,145 (78.7%) | 9,274 (79.8%) |
| Numerical Reasoning | 7,214 | 509 (7.1%) | 527 (7.3%) |
| Other | 40,193 | 1,121 (2.8%) | 1,230 (3.1%) |
| Search terms not in claim | 824 | 136 (16.5%) | 141 (17.1%) |
| **all** | 71,291 | 12,420 (17.4%) | 12,726 (17.9%) |

| challenge tag | ALL | SUPPORTS | REFUTES | NOT ENOUGH INFO |
|---|---|---|---|---|
| Combining Tables and Text | 1,271 / 10,083 (12.6%) | 1,120 / 8,751 (12.8%) | 129 / 1,149 (11.2%) | 22 / 183 (12.0%) |
| Entity Disambiguation | 238 / 1,353 (17.6%) | 147 / 763 (19.3%) | 48 / 471 (10.2%) | 43 / 119 (36.1%) |
| Multi-hop Reasoning | 9,145 / 11,624 (78.7%) | 8,188 / 9,978 (82.1%) | 632 / 1,198 (52.8%) | 325 / 448 (72.5%) |
| Numerical Reasoning | 509 / 7,214 (7.1%) | 353 / 3,061 (11.5%) | 144 / 4,078 (3.5%) | 12 / 75 (16.0%) |
| Other | 1,121 / 40,193 (2.8%) | 660 / 18,890 (3.5%) | 338 / 20,043 (1.7%) | 123 / 1,260 (9.8%) |
| Search terms not in claim | 136 / 824 (16.5%) | 83 / 392 (21.2%) | 19 / 276 (6.9%) | 34 / 156 (21.8%) |
| **all** | 12,420 / 71,291 (17.4%) | 10,551 / 41,835 (25.2%) | 1,310 / 27,215 (4.8%) | 559 / 2,241 (24.9%) |

| target | tag & multi (TP) | tag & not multi (FP) | no tag & multi (FN) | neither (TN) | precision (share of tagged) | recall (share of multi covered by tag) |
|---|---|---|---|---|---|---|
| strict | 9,145 | 2,479 | 3,275 | 56,392 | 78.7% | 73.6% |
| lenient | 9,274 | 2,350 | 3,452 | 56,215 | 79.8% | 72.9% |

### A.1 (cont.) Challenge tag against multi-article, dev

| challenge tag | claims | strict multi-article | lenient multi-article |
|---|---|---|---|
| Combining Tables and Text | 1,035 | 106 (10.2%) | 111 (10.7%) |
| Entity Disambiguation | 201 | 27 (13.4%) | 31 (15.4%) |
| Multi-hop Reasoning | 1,281 | 962 (75.1%) | 973 (76.0%) |
| Numerical Reasoning | 873 | 53 (6.1%) | 58 (6.6%) |
| Other | 4,369 | 118 (2.7%) | 128 (2.9%) |
| Search terms not in claim | 131 | 14 (10.7%) | 14 (10.7%) |
| **all** | 7,890 | 1,280 (16.2%) | 1,315 (16.7%) |

| challenge tag | ALL | SUPPORTS | REFUTES | NOT ENOUGH INFO |
|---|---|---|---|---|
| Combining Tables and Text | 106 / 1,035 (10.2%) | 87 / 829 (10.5%) | 14 / 171 (8.2%) | 5 / 35 (14.3%) |
| Entity Disambiguation | 27 / 201 (13.4%) | 12 / 88 (13.6%) | 10 / 88 (11.4%) | 5 / 25 (20.0%) |
| Multi-hop Reasoning | 962 / 1,281 (75.1%) | 793 / 969 (81.8%) | 94 / 200 (47.0%) | 75 / 112 (67.0%) |
| Numerical Reasoning | 53 / 873 (6.1%) | 29 / 316 (9.2%) | 21 / 546 (3.8%) | 3 / 11 (27.3%) |
| Other | 118 / 4,369 (2.7%) | 52 / 1,662 (3.1%) | 42 / 2,447 (1.7%) | 24 / 260 (9.2%) |
| Search terms not in claim | 14 / 131 (10.7%) | 5 / 44 (11.4%) | 3 / 29 (10.3%) | 6 / 58 (10.3%) |
| **all** | 1,280 / 7,890 (16.2%) | 978 / 3,908 (25.0%) | 184 / 3,481 (5.3%) | 118 / 501 (23.6%) |

| target | tag & multi (TP) | tag & not multi (FP) | no tag & multi (FN) | neither (TN) | precision (share of tagged) | recall (share of multi covered by tag) |
|---|---|---|---|---|---|---|
| strict | 962 | 319 | 318 | 6,291 | 75.1% | 75.2% |
| lenient | 973 | 308 | 342 | 6,267 | 76.0% | 74.0% |

### A.2 Composition of "mixed" evidence sets

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

### A.3 Multi-unit proxy by challenge tag, train+dev (annotation context, not checked against the DB)

A unit is (page, set of section ids in the element's context). Strict means every set has 2+ units, and lenient means some set does.

| challenge tag | claims | strict multi-unit (ctx) | lenient multi-unit (ctx) |
|---|---|---|---|
| Combining Tables and Text | 11,118 | 6,055 (54.5%) | 6,329 (56.9%) |
| Entity Disambiguation | 1,554 | 695 (44.7%) | 719 (46.3%) |
| Multi-hop Reasoning | 12,905 | 11,856 (91.9%) | 12,078 (93.6%) |
| Numerical Reasoning | 8,087 | 1,796 (22.2%) | 1,870 (23.1%) |
| Other | 44,562 | 5,489 (12.3%) | 5,891 (13.2%) |
| Search terms not in claim | 955 | 253 (26.5%) | 259 (27.1%) |
| **all** | 79,181 | 26,144 (33.0%) | 27,146 (34.3%) |

### A.4 Lead against later sections (annotation-context proxy, not a Step 2 result)

The official code gives each element a context listing the section it is in and that section's parent sections. An element with no section id in its context comes before the first section of its page, the lead. This is **not** the DB-based measurement requested in Step 2.3, which was not computed.

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
