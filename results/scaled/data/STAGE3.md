# Stage 3 of the scaled run: page metadata, 2026-11-15

Each S1 page's December 2020 revision, edit date and categories from the MediaWiki API, loaded
into `wikilense_s1` and `wikilense_s0` (S0's pages are a subset of S1's, so one file serves both).
Script: `scripts/scaled_stage3.py` (commit `a6da261`), which reuses `scripts/fetch_page_meta.py`'s
revision rule and its polite API client (one request at a time, at most 4 per second, 5 s pause
after any request over 1 s, `maxlag`, `Retry-After`, User-Agent with the repository URL).

## Fetch

- `fetch --pages scaled_data/S1/pages.jsonl --out scaled_data/stage3/page_meta_S1.jsonl`,
  02:47 to 07:43 (294 min), 20,795 API requests, **7,686 of 7,686 pages, 0 errors**
  (`fetch.log`; no `.errors.jsonl` file was written).

| method (same rule as the 100-page corpus) | pages |
|---|---|
| `last_before_end` (no edit in December 2020: the revision live at its end) | 2,921 |
| `best_text_match` (edited in December 2020: the revision whose text matches FEVEROUS best) | 4,620 |
| `unverified` (a revision exists, but its wikitext holds none of the page's sentences) | 112 |
| `missing` (no revision at the end of 2020) | 33 |

- Pages with a revision and an edit date: 7,541; without (unverified + missing): 145, stored as
  NULL, so an edit-date filter leaves them out.
- Pages under another title on Wikipedia (moved): 285. Pages without any category: 253.
- Pages whose chosen revision matched fewer than half of its sentence probes: 181 (counted only,
  not looked into); pages with no probes: 31.

## Load and verify

`load` then `verify` for each database (one transaction each; the counts below are read back
from the database and equal the counts computed from the file):

| database | pages | with revision and edit date | with a category | categories | page_category rows |
|---|---|---|---|---|---|
| `wikilense_s1` | 7,686 | 7,541 | 7,433 | 40,695 | 58,569 |
| `wikilense_s0` | 3,851 | 3,770 | 3,685 | 21,934 | 27,926 |

Spot check: `wikilense query ... --category "Living people" --edited-since 2020-06-01` on S1 and
`--edited-before 2015-01-01` on S0 both returned rows. The most common category is
"Living people" (910 S1 pages).

Not done: no timing or recall of the filtered searches on the scaled databases (not decided).
