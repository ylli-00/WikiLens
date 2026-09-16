# Corpus manifest

Built by `scripts/build_corpus.py` from the FEVEROUS Wikipedia shard(s) and claim files (https://fever.ai/dataset/feverous.html). Every count below is recomputed from the two files next to this manifest; re-running the script reproduces them byte for byte.

## Rule

1. A claim is *in-shard* when every page cited in every `content` id of every evidence set is a shard title (Unicode NFC). Claims without any content id are not in-shard.
2. Evidence pages are the union of those pages.
3. Filler pages are link targets of the evidence pages that are shard titles and not evidence pages, ranked by the number of distinct evidence pages linking to them (descending), then title (ascending); up to 46. If fewer exist, the rest is taken in shard order from pages with at least 20 sentences.
4. `pages.jsonl` holds the raw shard lines in shard order; `claims.jsonl` holds the raw FEVEROUS records plus `"split"` (train records first, then dev, in file order).

Parameters: `--n-fill 46 --min-sentences 20`. Inputs: `wiki_000.jsonl` (9,996 pages), `feverous_train_challenges.jsonl` (71,291 claims), `feverous_dev_challenges.jsonl` (7,890 claims).

## Counts

| Count | Value |
|---|---|
| Pages (lines in pages.jsonl) | 100 |
| Evidence pages | 54 |
| Filler pages | 46 |
| Filler pages from links | 46 |
| Filler pages from shard order | 0 |
| Claims (lines in claims.jsonl) | 75 |
| Claims, train | 65 |
| Claims, dev | 10 |
| Claims, SUPPORTS | 7 |
| Claims, REFUTES | 58 |
| Claims, NOT ENOUGH INFO | 10 |
| Claims with a sentence-only evidence set | 65 |
| Evidence element ids | 114 |

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| pages.jsonl | 8007729 | edf0955892c37fe54bb840cda33a6df5a06c44394289e7176a922556d09621dc |
| claims.jsonl | 141397 | a9a36c5c14f9a3baea97df6b42361f866249a4c636e4504cd259d7e54725b8ac |

## Evidence pages (54)

Sorted by title; in brackets the number of claims citing the page.

- AFC Ajax (2)
- Acadia University (1)
- Actinopterygii (4)
- Adelaide (1)
- Agatha Christie (1)
- Alaric I (1)
- American Civil War (1)
- Amino acid (1)
- Andhra Pradesh (1)
- Andorra (1)
- Anime (1)
- Antlia (1)
- Arsenal F.C. (2)
- Asia (1)
- Association for Computing Machinery (1)
- Asteraceae (9)
- Asterales (3)
- Atlanta (2)
- Bangladesh (1)
- Biennial plant (1)
- Bird (1)
- British Columbia (1)
- Bulgaria (1)
- Canton of Aargau (1)
- Economy of Azerbaijan (1)
- Gyula Andrássy (1)
- J. K. Rowling (1)
- Jack Brabham (3)
- Jimmy Carter (1)
- Jules Verne (1)
- Juventus F.C. (3)
- KLM (1)
- Kabul (1)
- Kalmar Union (1)
- Kenesaw Mountain Landis (1)
- Kim Philby (1)
- King's Royal Rifle Corps (1)
- Lacrosse (1)
- Lamiales (1)
- Lebanese Armed Forces (1)
- Lincoln, England (1)
- List of township-level divisions of Hebei (1)
- London (1)
- London Underground (1)
- Longitude (1)
- Los Angeles (2)
- Louis Pasteur (1)
- Monreith House (1)
- Smithfield, Utah (1)
- The Apache Software Foundation (1)
- The Bronx (1)
- The Hunger Games (film) (1)
- Toshiki Kaifu (1)
- Zhang Quanyi (1)

## Filler pages (46)

Ranking reason: pages linked from the evidence pages are realistic distractors and make the "linked from" SQL filter meaningful. In rank order; in brackets the number of distinct evidence pages linking to the page (ties broken by title). Pages taken in shard order because too few linked pages existed are marked "shard order".

1. Latin (linked from 5)
2. Amsterdam (linked from 4)
3. Lebanon (linked from 4)
4. Afghanistan (linked from 3)
5. Agriculture (linked from 3)
6. Ancient Egypt (linked from 3)
7. Antisemitism (linked from 3)
8. Trade union (linked from 3)
9. Academy Awards (linked from 2)
10. African Americans (linked from 2)
11. Alabama (linked from 2)
12. Alaska (linked from 2)
13. Albania (linked from 2)
14. Alexander the Great (linked from 2)
15. American Revolutionary War (linked from 2)
16. Amine (linked from 2)
17. Amusement park (linked from 2)
18. Anglican Communion (linked from 2)
19. Ankara (linked from 2)
20. Anno Domini (linked from 2)
21. Annual plant (linked from 2)
22. Archery (linked from 2)
23. Art Deco (linked from 2)
24. Ashoka (linked from 2)
25. Athena (linked from 2)
26. Azerbaijan (linked from 2)
27. Belgium (linked from 2)
28. Black Sea (linked from 2)
29. Brazil (linked from 2)
30. Kazakhstan (linked from 2)
31. Korean language (linked from 2)
32. Leather (linked from 2)
33. Left-wing politics (linked from 2)
34. Limestone (linked from 2)
35. Literacy (linked from 2)
36. Liverpool (linked from 2)
37. Liverpool F.C. (linked from 2)
38. Los Angeles Dodgers (linked from 2)
39. Luftwaffe (linked from 2)
40. .bangla (linked from 1)
41. A.S. Roma (linked from 1)
42. ACF Fiorentina (linked from 1)
43. AZ Alkmaar (linked from 1)
44. Aarau (linked from 1)
45. Aare (linked from 1)
46. Abdur Rahman Khan (linked from 1)
