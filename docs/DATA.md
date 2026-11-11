# The data: 100 Wikipedia pages, 75 claims, and where they come from

What is in `data/corpus/`, the rule that selected it, how the categories and edit dates were
fetched, and how to rebuild all of it. The [README](../README.md) has the summary.

`data/corpus/pages.jsonl` (8.0 MB) holds 100 Wikipedia pages in the FEVEROUS format: December 2020
articles with sentence, section, table and list elements. `data/corpus/claims.jsonl` (141 KB) has 75
FEVEROUS claims with their evidence sets. This file and `pages.jsonl` are committed, with SHA-256
checksums and the page list in [data/corpus/MANIFEST.md](../data/corpus/MANIFEST.md).

The selection rule takes all claims of FEVEROUS train and dev whose evidence pages are all inside
the first Wikipedia shard, `wiki_000.jsonl`: 75 claims (65 train, 10 dev; 7 SUPPORTS, 58 REFUTES, 10
NOT ENOUGH INFO), whose evidence cites 54 distinct pages. The other 46 pages are the pages those 54
link to most often, so the corpus contains realistic distractors and the link filters have something
to work on. 65 claims have an evidence set made only of sentences; 66 count for evidence recall,
because one more claim's set mixes sentences and list items, which are both text units. The 27
evidence ids that point at table cells cannot be matched, because tables are not indexed.

`data/corpus/page_meta.jsonl` gives each page's Wikipedia revision id, edit date and categories.
FEVEROUS keeps none of them; its pages stop before the category box and the "last edited" footer.
[scripts/fetch_page_meta.py](../scripts/fetch_page_meta.py) asks the MediaWiki API for the revision
whose text the corpus holds: the one live in December 2020, when FEVEROUS scraped Wikipedia (not
today's page). The revision's timestamp is the "last edited" date of the text we have, and its
`[[Category:...]]` links are the categories. For the 93 pages edited during December 2020, the
revision whose wikitext contains the most of the page's FEVEROUS sentences is chosen; on a tie, the
later one. Three renamed pages are followed through the move log.

All 100 pages were found, and 94.5% of the sentence probes occur in the chosen revisions
(`probes_matched` per page in the metadata file). The result is 1,005 categories and 1,123
page-category pairs, with edit dates from 2019-12-23 to 2020-12-31;
[data/corpus/PAGE_META.md](../data/corpus/PAGE_META.md) has the summary. The script follows the
Wikimedia Robot Policy: one request at a time, at most four per second, a User-Agent with this
repository's URL, `maxlag` and `Retry-After`. The 100 pages took 309 requests.

To rebuild the corpus (about 5 s, byte-identical):
1. Download the shard with `python3 scripts/fetch_wiki_shard.py wiki_000.jsonl`. It is 44 MB,
   fetched from inside the 9.9 GB archive with HTTP range requests and CRC-checked.
2. Put `feverous_train_challenges.jsonl` and `feverous_dev_challenges.jsonl` from
   https://fever.ai/dataset/feverous.html into `data/feverous/`.
3. Run `python3 scripts/build_corpus.py`.
4. Run `python3 scripts/fetch_page_meta.py` to refetch the metadata (a few minutes).

FEVEROUS: Aly et al., 2021 (https://arxiv.org/abs/2106.05707); Wikipedia text under CC BY-SA 3.0.
