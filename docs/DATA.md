# The data: the demo corpus, the 3,000 claims and the scaled corpora S0 and S1

There are two data sets, both from FEVEROUS and its December 2020 Wikipedia. The demo corpus in
`data/corpus/` (100 pages, 75 claims) is what `make setup` loads and what the README's first
screen and `make eval` use. The scaled run uses 3,000 FEVEROUS claims on two larger corpora, S0
and S1; its results are in [EVALUATION.md](EVALUATION.md).

## The demo corpus: 100 pages and 75 claims

`data/corpus/pages.jsonl` (8.0 MB) holds 100 Wikipedia pages in the FEVEROUS format: December 2020
articles with sentence, section, table and list elements. `data/corpus/claims.jsonl` (141 KB) has 75
FEVEROUS claims with their evidence sets. Both are committed, with SHA-256 checksums and the page
list in [data/corpus/MANIFEST.md](../data/corpus/MANIFEST.md).

The rule takes all claims of FEVEROUS train and dev whose evidence pages are all inside the first
Wikipedia shard, `wiki_000.jsonl`: 75 claims (65 train, 10 dev; 7 SUPPORTS, 58 REFUTES, 10 NOT
ENOUGH INFO), whose evidence cites 54 distinct pages. The other 46 pages are the pages those 54 link
to most often, so the corpus holds realistic distractors and the link filters have something to
work on. 65 claims have an evidence set made only of sentences; 66 count for evidence recall,
because one more claim's set mixes sentences and list items, which are both text units. The 27
evidence ids that point at table cells cannot be matched, because tables are not indexed.

**Categories and edit dates.** `data/corpus/page_meta.jsonl` gives each page's revision id, edit
date and categories, which FEVEROUS does not keep.
[scripts/fetch_page_meta.py](../scripts/fetch_page_meta.py) asks the MediaWiki API for the revision
whose text the corpus holds: the one live in December 2020, when FEVEROUS scraped Wikipedia, not
today's page. Its timestamp is the "last edited" date of our text, and its `[[Category:...]]` links
are the categories. For the 93 pages edited during December 2020, the revision whose wikitext
contains the most of the page's FEVEROUS sentences is chosen; on a tie, the later one. Three renamed
pages are followed through the move log. All 100 pages were found, and 94.5% of the sentence probes
occur in the chosen revisions (`probes_matched` per page): 1,005 categories, 1,123 page-category
pairs, edit dates from 2019-12-23 to 2020-12-31
([data/corpus/PAGE_META.md](../data/corpus/PAGE_META.md)). The script follows the Wikimedia Robot
Policy: one request at a time, at most four per second, a User-Agent with this repository's URL,
`maxlag` and `Retry-After`. The 100 pages took 309 requests.

**To rebuild the demo corpus** (about 5 s, byte-identical):
1. Download the shard with `python3 scripts/fetch_wiki_shard.py wiki_000.jsonl`. It is 44 MB,
   fetched from inside the 9.9 GB archive with HTTP range requests and CRC-checked.
2. Put `feverous_train_challenges.jsonl` and `feverous_dev_challenges.jsonl` from
   https://fever.ai/dataset/feverous.html into `data/feverous/`.
3. Run `python3 scripts/build_corpus.py`.
4. Run `python3 scripts/fetch_page_meta.py` to refetch the metadata (a few minutes).

## The scaled run: 3,000 claims, S0 and S1

The claims are committed as `data/scaled/claims_3000.jsonl`. S0 is their gold pages (database
`wikilense_s0`); S1 adds one distractor page per gold article with a usable link (database
`wikilense_s1`). The S0 and S1 folders are not in the repository
([Rebuilding the scaled data](#rebuilding-the-scaled-data)).

### Which claims

- **Train and dev together** (71,291 and 7,890 records): nothing is trained; dev alone is too small.
- **Six groups**, label × kind: *single* when every evidence set cites the same one page, *needs2*
  when every set cites two or more pages (two or more, not exactly two, so that enough NOT ENOUGH
  INFO claims qualify). Claims that fit neither kind are left out.
- **Text only**: an evidence set of sentences and list items only, as tables are not indexed.
  43,957 of the 79,181 train and dev claims have no such set, more than half of FEVEROUS.
- **Can be found**: every element of one text-only set lies in a chunk as the ingest makes it
  (240 words, overlap 1). Claims that fail are left out before sampling, so others take their place.
- **Seeded sample**: `random.Random(20261111)`; per label, 700 single claims, then 300 needs2.
  The FEVEROUS challenge tag is recorded and not balanced on, so recall can be given per tag.

| Label | Kind | All claims of the kind | With a text-only set | Can be found | Left out | Picked | Spare |
|---|---|---|---|---|---|---|---|
| SUPPORTS | single | 33,853 | 14,059 | 14,036 | 23 | 700 | 13,336 |
| SUPPORTS | needs2 | 11,529 | 4,215 | 4,178 | 37 | 300 | 3,878 |
| REFUTES | single | 28,852 | 14,319 | 14,298 | 21 | 700 | 13,598 |
| REFUTES | needs2 | 1,494 | 633 | 623 | 10 | 300 | 323 |
| NOT ENOUGH INFO | single | 2,050 | 1,158 | 1,147 | 11 | 700 | 447 |
| NOT ENOUGH INFO | needs2 | 677 | 324 | 314 | 10 | 300 | 14 |

All 112 left-out claims fail for the same reason, a hatnote: the note at the top of an article or
section, such as "This article is about the clothing brand." or "Main article: …". WikiLense keeps
hatnotes out of the chunks on purpose, so a set that cites one can never be fully retrieved.

The pick has 2,578 train and 422 dev claims. Challenge tags: Other 1,906, Multi-hop Reasoning 709,
Entity Disambiguation 151, Search terms not in claim 131, Numerical Reasoning 71, Combining Tables
and Text 32. The mix differs from FEVEROUS's: a third of the claims are NOT ENOUGH INFO, against
3.5% in FEVEROUS train and dev (45,743 SUPPORTS, 30,696 REFUTES, 2,742 NOT ENOUGH INFO), and 30% of
the REFUTES claims need two articles, against 4.9% there. Results are therefore given per group.
The gold pages are every page cited by any evidence set of the claims, table sets included:
3,851, none missing from the FEVEROUS database, with a median of 5 chunks each.

The claims file has 3,000 lines, one JSON object each, SHA-256
`905cf1281d01e310702e820121864ac96ab0c07d5e857c24ad2d1314f532ab8a`; its licence is in
[data/scaled/NOTICE.md](../data/scaled/NOTICE.md). Fields: `id`, `split` (`train` or `dev`),
`label`, `claim` and `challenge` as in FEVEROUS; `kind` (`single` or `needs2`); `gold_pages`, the
pages the evidence sets cite; `evidence_sets`, per FEVEROUS set its `elements` (element ids),
`text_only` and `findable`; `gold_set`, the index of the first findable set; and `gold_evidence`,
that set's elements with their `element`, `page` and December 2020 `text`. Titles are Unicode NFC.

### The distractor pages: rule D

Each gold article adds its favourite link: the page it links to most often (links counted with
duplicates, ties to the first mention). Gold pages, titles another article already took and titles
missing from the FEVEROUS database are skipped for the next favourite. The gold articles are walked
in `sorted(gold)` order shuffled with `random.Random(7)`. Every gold article with a usable link gets
one page, with no cap, so every claim is treated the same.

**The rules tried.** Four rules and four controls were compared on a test pick of 3,000 claims,
drawn with the same seed and groups but without the "can be found" check (43,264 chunks of gold
pages). Each rule added 3,000 pages; each control matches a rule's chunk count. The search was an
exact cosine top 20 per claim, computed offline with the ingest's chunks and embedding, not in
MariaDB. A claim counts for page-set recall when every page of one evidence set is among the
top-k chunks' pages, and for evidence recall when every element of one text-only set is inside
them. Claims lost at k = 10 against the gold pages alone (2,753 and 2,554 of 3,000 found):

| Added | How the pages are picked | Chunks | Page-set recall | Evidence recall |
|---|---|---|---|---|
| D, favourite link | the rule above, stopped at 3,000 pages | 72,063 | −58 | −67 |
| AD | A's top hub pages, up to D's chunk count | 72,075 | −45 | −49 |
| RD | random pages, up to D's chunk count | 72,066 | −26 | −24 |
| A, hubs | the non-gold pages linked from the most gold pages | 124,890 | −62 | −74 |
| RA | random pages, up to A's chunk count | 124,904 | −38 | −39 |
| B, spread | gold pages in a seeded order, each adds one random page it links to | 59,943 | −39 | −42 |
| RB | random pages, up to B's chunk count | 59,956 | −23 | −20 |
| C, random | 3,000 random pages of the database | 12,851 | −3 | −5 |

Why D:

- **Related pages distract more than the same amount of random text**: D loses 2.2 (page-set) to
  2.8 times (evidence) as many claims as RD, and 1.3 to 1.4 times as many as hub text of the same
  size (AD).
- **D distracts most per added chunk**: 9.3 evidence and 8.0 page-set claims lost per 10,000
  chunks (rough, as the corpora differ in size). It has about the hubs' total effect (−67 against
  −74 claims for evidence, −58 against −62 for page sets) with 58% of their chunks, and a smaller
  corpus is faster.
- **D's pages are on topic**: of the distractor chunks in a claim's top 10, 39% come from a page
  the claim's own gold article links to, against 26% for A, 29% for B and about 1% for random pages.

Random pages barely distract (row C), so none were added. A bigger corpus would add more linked
pages (each article's second favourite); none is built. Article recall hardly moves (99% at
k = 10 whatever is added). One seed ran per rule. The test's D stopped at 3,000 pages, so some
gold articles had no distractor of their own; on the 3,000 claims every gold article with a usable
link gets one (below).

On the 3,000 claims, all 3,851 gold articles were walked and 3,835 pages taken. 16 gold articles
have no usable link: each of their links was a gold page, taken or missing. Skips: gold page 551,
already taken 1,123, missing from the database 790. For 2,567 articles the most-linked title itself
was taken. Chunks per rule-D page: median 15, mean 23.27, max 196.

### S0 and S1

Each folder holds `pages.jsonl` (the pages from the FEVEROUS database, in Unicode NFC),
`claims.jsonl` (the 3,000 raw FEVEROUS records plus `"split"`, the same in both) and `MANIFEST.md`.
S0's pages are sorted by title; S1's `pages.jsonl` is S0's bytes followed by the rule-D pages.

| Count | S0 | S1 |
|---|---|---|
| Pages (gold + rule D) | 3,851 (3,851 + 0) | 7,686 (3,851 + 3,835) |
| Sections (lead included) | 35,571 | 103,823 |
| Text units (sentences and list items) | 284,302 | 906,342 |
| of them hatnotes / empty, in no chunk | 6,117 / 1,603 | 25,471 / 6,414 |
| Chunks | 43,134 | 132,359 |
| Links / to a page of the corpus | 357,532 / 11,413 | 1,159,328 / 154,762 |
| Evidence ids / with a page / with a text unit | 8,408 / 8,408 / 7,877 | 8,408 / 8,408 / 7,877 |
| Vector-index graph | 58,720,256 bytes | 163,577,856 bytes |
| Ingest: total / parse / load (inserts + index) | 54.6 / 9.3 / 43.6 s | 212.8 / 29.8 / 171.6 s |

**Loading.** Each corpus has its own database; the demo database `wikilense` is not touched. The
chunk texts were embedded once, for S1, as S0's are a subset: 132,359 texts in 499.5 s on the GPU
with BAAI/bge-small-en-v1.5 at its pinned revision, in parts of 10,000 (so not bit-identical to one
`wikilense ingest` call). The normal ingest then ran with the saved vectors, and the database
counts equal the build's. Both graphs stay below `mhnsw_max_cache_size` (512 MB).

### Categories and edit dates of S0 and S1

`scripts/scaled_stage3.py` uses the revision rule and the API client of `fetch_page_meta.py`, with
the same request rules, including a 5 s pause after any request over 1 s. S1's pages were fetched
once and the file is loaded into both databases, since S0's pages are a subset: 7,686 of 7,686
pages, 0 errors, 20,795 requests in 294 minutes; a stopped fetch resumes. 2,921 pages had
no edit in December 2020 and take the revision live at its end; 4,620 were edited then and take the
revision that matches their FEVEROUS sentences best. 145 pages have no revision id and no edit date
(112 whose revision holds none of their sentences, 33 with no revision at the end of 2020); they
are stored as NULL, so an edit-date filter leaves them out. 285 pages now have another title on
Wikipedia and are followed through the move log. For 181 pages the chosen revision holds fewer than
half of the sentence probes (counted, not looked into).

| Database | Pages | With revision and edit date | With a category | Categories | Page-category rows |
|---|---|---|---|---|---|
| `wikilense_s1` | 7,686 | 7,541 | 7,433 | 40,695 | 58,569 |
| `wikilense_s0` | 3,851 | 3,770 | 3,685 | 21,934 | 27,926 |

The most common category is "Living people" (910 S1 pages). With this metadata, all four filters
(category, length, edited since, linked from) work on S0 and S1 as on the demo corpus.

### Rebuilding the scaled data

This needs the FEVEROUS train and dev files in `data/feverous/`, the FEVEROUS Wikipedia database
(the 53.5 GB SQLite file `feverous_wikiv1.db`, read-only, its path in `WIKILENSE_FEVEROUS_DB`), a
GPU for the embedding, and hours, most of them for the metadata fetch. In order (the full commands
are in [SETUP.md](SETUP.md), "Reproduce the scaled run"):

1. `scripts/scaled_build.py` with `claims` (the pick, as `claims_3000.jsonl.tmp`), `distractors`
   (rule D), `corpora` (the S0 and S1 folders) and `report` (the selection and distractor reports).
2. `scripts/scaled_check.py`, an independent self-check that re-derives every rule with separate
   code. Then rename the `.tmp` file to `claims_3000.jsonl` by hand (the build never writes the
   final file) and compare its SHA-256 with the one above.
3. `scripts/scaled_stage2.py` with `create-dbs`, `embed` on S1, then `ingest` and `verify` per
   database; both refuse any database but `wikilense_s0` and `wikilense_s1`.
4. `scripts/scaled_stage3.py` with `fetch` on S1's pages, then `load` and `verify` per database.

FEVEROUS: Aly et al., 2021 (https://arxiv.org/abs/2106.05707); Wikipedia text under CC BY-SA 3.0,
with the licences in [data/corpus/NOTICE.md](../data/corpus/NOTICE.md) and
[data/scaled/NOTICE.md](../data/scaled/NOTICE.md).
