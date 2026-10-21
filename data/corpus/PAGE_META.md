# Page metadata

Written by `scripts/fetch_page_meta.py` on 2026-10-21 from the MediaWiki API (https://en.wikipedia.org/w/api.php) with 309 requests. Each record names the Wikipedia revision whose text the corpus holds (the one live in December 2020, when FEVEROUS scraped the page), its timestamp and its categories; the script's docstring explains how the revision is chosen. The revisions lie in the past, so a re-run finds the same ones unless a page was deleted or moved since.

| Measure | Value |
|---|---|
| pages | 100 |
| revision found | 100 |
| method last_before_end (no edit in December 2020) | 7 |
| method best_text_match (edited in December 2020) | 93 |
| method missing (no revision at the end of 2020) | 0 |
| found under a title it was moved to after 2020 | 3 |
| pages with at least one category | 100 |
| page-category pairs | 1123 |
| distinct categories | 1005 |
| categories shared by two or more pages | 72 |
| FEVEROUS sentence probes found in the chosen revision | 27913 of 29544 (94.5%) |
| pages whose every probe was found | 5 of 100 |
| last_edited, earliest | 2019-12-23T23:29:30Z |
| last_edited, latest | 2020-12-31T17:18:20Z |

## Last edit before the snapshot, by year

| Year | Pages |
|---|---|
| 2019 | 1 |
| 2020 | 99 |

## Categories with the most pages

| Category | Pages |
|---|---|
| Member states of the United Nations | 10 |
| Countries in Europe | 6 |
| Member states of the Organisation of Islamic Cooperation | 6 |
| Countries in Asia | 5 |
| Member states of the Council of Europe | 5 |
| Republics | 5 |
| G-14 clubs | 4 |
| Landlocked countries | 4 |
| Member states of the Organisation internationale de la Francophonie | 4 |
| Member states of the Union for the Mediterranean | 4 |
| UEFA Cup Winners' Cup winning clubs | 4 |
| Coppa Italia winning clubs | 3 |
| Eastern European countries | 3 |
| Football clubs in Italy | 3 |
| Living people | 3 |
