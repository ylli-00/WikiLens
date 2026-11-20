# A test load of S1 from the release file

On 2026-11-20 the release file `wikilense_s1_v1.tar.gz` (300,308,768 bytes, SHA-256
`86ad18f01ad98bfaf23b800181e9f76dc409bf9b0ce20ef606bcaea0e93e032c`) was loaded the way
[docs/SETUP.md](../../../docs/SETUP.md#get-the-scaled-database-s1) describes, into a scratch
database (`make load-s1 S1_CHECK=1` loads into `wikilense_s1_check`, so that the measured
`wikilense_s1` stayed as it was). The scratch database was dropped afterwards. As in the scaled
run, the CPU was capped at 50% and the GPU at 2,100 MHz.

## The load

- `make fetch-s1` from a local copy of the file: the size and SHA-256 matched, 34 files
  (607,719,671 bytes) unpacked, in 4.3 s.
- `make load-s1`: 221.1 s for the whole target. The ingest took 215.35 s, of which 174.5 s were
  the inserts and the index build ([ingest_S1_check.json](ingest_S1_check.json)); the first load
  of S1 took 212.84 s and 171.64 s ([ingest_S1.json](../data/ingest_S1.json)).
- Every count of `scaled_stage2.py verify` equals `corpora.json`: 7,686 pages, 132,359 chunks,
  19,149,694 chunk words, 3,000 claims, 8,408 evidence ids (8,408 with a page, 7,877 with a text
  unit). `scaled_stage3.py verify` read back 7,541 pages with a revision and an edit date, 7,433
  with a category, 40,695 categories and 58,569 page-category rows, the counts the file gives.
- On disk: the database's folder in the data volume held 1,375,712,270 bytes (the graph file
  `chunk#i#04.ibd` 163,577,856 of them); the free space of the partition fell by 1,418,092,544
  bytes during the load.

## Recall against the reported run

The same command as the reported run (`wikilense eval --ef-search 100`, k = 1, 3, 5, 10 and 20,
five timed passes): [s1_check.md](s1_check.md), [s1_check.json](s1_check.json), against
[s1_full_run1](../eval/s1_full_run1.md).

| k | article recall, test load | reported | evidence recall, test load | reported |
|---|---|---|---|---|
| 1 | 2814 | 2814 | 1304 | 1303 |
| 3 | 2911 | 2911 | 2042 | 2042 |
| 5 | 2934 | 2933 | 2256 | 2253 |
| 10 | 2948 | 2947 | 2478 | 2479 |
| 20 | 2957 | 2957 | 2607 | 2609 |

Counts of 3,000 claims. The vectors are the same, but the HNSW graph is built anew by each load,
so the two databases need not return the same hits. At k = 5 the test load finds a gold page for
2,934 claims; twenty builds of these vectors found 2,922 to 2,937
([build_variance.md](../run1/s1/build_variance.md)). Its top-5 hit lists (`hit_chunk_ids`, the
same chunk ids in both databases) equal the reported run's for 2,888 of the 3,000 claims; two of
the twenty builds agreed on 2,838 to 2,907. Both numbers are inside the measured build variance.
The two reported passes (`s1_full_run1`, `s1_full_run2`) ran on one graph and agree on all
3,000 lists.
