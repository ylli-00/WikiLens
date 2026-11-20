# Selection of the 3,000 claims (scaled experiment)

Written by `scripts/scaled_build.py report` from `selection.json` and `corpora.json`;
every number below was measured by `scripts/scaled_build.py claims` and `corpora` on the
FEVEROUS train + dev files and `feverous_wikiv1.db`.

## Rule

- Kinds as `describe()` in `scripts/scaled_build.py`: **single** = every evidence set cites the
  same one page; **needs2** = every set cites two or more pages.
- Eligible: a single or needs2 claim with a text-only evidence set (sentences and list
  items only).
- Findable: a text-only set whose every element is in a chunk of `chunk_page(parse_page(page), 240, 1)`
  (the ingest's chunker; hatnotes and empty units are in no chunk by design). Unfindable
  claims are left out before sampling.
- `random.Random(20261111)`; per label `sample(single, 700)` then `sample(needs2, 300)`.
- The challenge tag is recorded, not balanced on.

## Pool per group, before and after the findable check

| Label | Kind | All claims of the kind | Eligible (text-only set) | Findable | Left out (unfindable) | Picked | Spare |
|---|---|---|---|---|---|---|---|
| SUPPORTS | single | 33,853 | 14,059 | 14,036 | 23 | 700 | 13,336 |
| SUPPORTS | needs2 | 11,529 | 4,215 | 4,178 | 37 | 300 | 3,878 |
| REFUTES | single | 28,852 | 14,319 | 14,298 | 21 | 700 | 13,598 |
| REFUTES | needs2 | 1,494 | 633 | 623 | 10 | 300 | 323 |
| NOT ENOUGH INFO | single | 2,050 | 1,158 | 1,147 | 11 | 700 | 447 |
| NOT ENOUGH INFO | needs2 | 677 | 324 | 314 | 10 | 300 | 14 |

Why the unfindable claims fail (all reasons over their text-only sets):

| Label | Kind | Reason | Claims |
|---|---|---|---|
| SUPPORTS | single | hatnote | 23 |
| SUPPORTS | needs2 | hatnote | 37 |
| REFUTES | single | hatnote | 21 |
| REFUTES | needs2 | hatnote | 10 |
| NOT ENOUGH INFO | single | hatnote | 11 |
| NOT ENOUGH INFO | needs2 | hatnote | 10 |

Elements of text-only sets in no chunk, by reason:

| Reason | Elements |
|---|---|
| hatnote | 124 |

Pages read for the findable check: 22,266 needed, 0 not in the DB, 0 whose parsed title differs; 51,286 of 51,410 needed elements are in a chunk.

## The 3,000 claims

| Label | single | needs2 | Total |
|---|---|---|---|
| SUPPORTS | 700 | 300 | 1,000 |
| REFUTES | 700 | 300 | 1,000 |
| NOT ENOUGH INFO | 700 | 300 | 1,000 |

Split per group:

| Group | train | dev |
|---|---|---|
| SUPPORTS / single | 625 | 75 |
| SUPPORTS / needs2 | 277 | 23 |
| REFUTES / single | 616 | 84 |
| REFUTES / needs2 | 261 | 39 |
| NOT ENOUGH INFO / single | 562 | 138 |
| NOT ENOUGH INFO / needs2 | 237 | 63 |
| all | 2,578 | 422 |

Challenge tags per group (recorded, not balanced on):

| Group | Other | Multi-hop Reasoning | Entity Disambiguation | Search terms not in claim | Numerical Reasoning | Combining Tables and Text |
|---|---|---|---|---|---|---|
| SUPPORTS / single | 574 | 53 | 31 | 16 | 6 | 20 |
| SUPPORTS / needs2 | 28 | 266 | 3 | 2 | 0 | 1 |
| REFUTES / single | 614 | 8 | 26 | 9 | 34 | 9 |
| REFUTES / needs2 | 95 | 161 | 20 | 6 | 17 | 1 |
| NOT ENOUGH INFO / single | 516 | 51 | 46 | 77 | 9 | 1 |
| NOT ENOUGH INFO / needs2 | 79 | 170 | 25 | 21 | 5 | 0 |
| all | 1,906 | 709 | 151 | 131 | 71 | 32 |

- needs2 claims by number of gold pages: {'2': 772, '3': 103, '4': 15, '5+': 10}.
- Evidence sets per claim: {'1': 2824, '2': 152, '3': 24}; findable sets per claim: {'1': 2975, '2': 21, '3': 4}.
- Claims with a set that holds a table cell, header cell or caption: 154 (their pages are gold pages too).
- Elements in the gold sets (`gold_evidence`): 7,739.
- Claim texts that are not NFC (kept as FEVEROUS has them): none.

## Gold pages

- Gold pages (every page cited by any evidence set of the 3,000 claims): **3,851**;
  3,845 of them are in a claim's gold set, 3,849 in some
  text-only set, 2 only in sets with table elements.
- Missing from the FEVEROUS DB: 0.
- Chunks per gold page: median 5, mean 11.2, p90 29 (`numpy.percentile`, linear), max 139; min 1; total 43,134.
- Gold pages without any chunk: 0.

The claims file is `claims_3000.jsonl`; its format is the contract with the LLM runner.
