# Licence and attribution of the files in `data/corpus/`

The code of this repository is under the MIT licence ([LICENSE](../../LICENSE)). The data files in
this directory are not ours to license: they are Wikipedia text and FEVEROUS annotations, and they
are redistributed here under their own terms.

| File | What it is | Where it comes from | Licence |
|---|---|---|---|
| `pages.jsonl` | the text of 100 English Wikipedia articles as it stood in December 2020, in the FEVEROUS page format | Wikipedia contributors; the lines are copied unchanged from the FEVEROUS Wikipedia shard `wiki_000.jsonl` | Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) |
| `claims.jsonl` | 75 FEVEROUS claims with their evidence annotations, plus a `split` field | the FEVEROUS train and dev files, https://fever.ai/dataset/feverous.html | the FEVEROUS licence terms quoted below |
| `page_meta.jsonl`, `PAGE_META.md` | each page's Wikipedia revision id, its timestamp and its category names | the MediaWiki API of en.wikipedia.org, fetched by `scripts/fetch_page_meta.py` | the category names are part of the article text: CC BY-SA 3.0 |
| `MANIFEST.md` | counts, checksums and the page list | written by `scripts/build_corpus.py` | MIT, like the code |

## Wikipedia text

- Wikipedia's text is written by its editors. In December 2020 it was licensed under CC BY-SA 3.0
  Unported and, unless otherwise noted, the GNU Free Documentation License: the policy page of that
  date says so (https://en.wikipedia.org/w/index.php?title=Wikipedia:Copyrights&oldid=997413401).
  The licence text is at https://creativecommons.org/licenses/by-sa/3.0/. Wikipedia's policy today
  names version 4.0 for its current text (https://en.wikipedia.org/wiki/Wikipedia:Copyrights).
- Attribution: every page in `pages.jsonl` has its title in the `title` field, and
  `page_meta.jsonl` records the exact revision the text matches. The article, and through its
  history its authors, is at `https://en.wikipedia.org/w/index.php?oldid=<revision_id>`.
- Changes: the FEVEROUS data holds each article as sentences, tables and lists, with links kept
  inline as `[[target|text]]` and without the references and the category box. That processing is
  the FEVEROUS authors'. We changed nothing in the text.
- Share-alike: if you pass these files on, changed or not, keep them under CC BY-SA 3.0 and keep
  this notice.

## FEVEROUS annotations

The licence of the FEVEROUS data, in full, from https://fever.ai/download/feverous/license.html
(read on 2026-11-11):

> These data annotations incorporate material from Wikipedia, which is licensed pursuant to the
> Wikipedia Copyright Policy. These annotations are made available under the license terms
> described on the applicable Wikipedia article pages, or, where Wikipedia license terms are
> unavailable, under the Creative Commons Attribution-ShareAlike License (version 3.0), available
> at http://creativecommons.org/licenses/by-sa/3.0/ (collectively, the "License Terms"). You may
> not use these files except in compliance with the applicable License Terms.

FEVEROUS: Rami Aly, Zhijiang Guo, Michael Sejr Schlichtkrull, James Thorne, Andreas Vlachos,
Christos Christodoulopoulos, Oana Cocarascu and Arpit Mittal, "FEVEROUS: Fact Extraction and
VERification Over Unstructured and Structured information", NeurIPS 2021 Datasets and Benchmarks
(https://arxiv.org/abs/2106.05707; dataset DOI https://doi.org/10.5281/zenodo.4911508).
