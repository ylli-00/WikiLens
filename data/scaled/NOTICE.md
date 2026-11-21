# Licence and attribution of the files in `data/scaled/`

| File | What it is | Where it comes from | Licence |
|---|---|---|---|
| `claims_3000.jsonl` | the 3,000 FEVEROUS claims of the scaled run, with their evidence sets, gold pages and gold evidence text | the FEVEROUS train and dev files (https://fever.ai/dataset/feverous.html) and the FEVEROUS Wikipedia (`feverous_wikiv1.db`), picked by `scripts/scaled_build.py` | the FEVEROUS licence terms and CC BY-SA 3.0, as for `data/corpus/claims.jsonl` |
| `feverous_claims.csv` | one row per FEVEROUS train and dev claim: its id, split, label, challenge tag, and per evidence set the number of pages and the element types; no claim text ([docs/DATA.md](../../docs/DATA.md) lists the columns) | counted from the FEVEROUS train and dev files (https://fever.ai/dataset/feverous.html) | the FEVEROUS licence terms |

The gold evidence text in each record is Wikipedia text of December 2020. The licence of the
FEVEROUS annotations and of the Wikipedia text, with the attribution, is in
[data/corpus/NOTICE.md](../corpus/NOTICE.md); it applies to both files in the same way. If you pass
them on, changed or not, keep them under CC BY-SA 3.0 and keep both notices.

The release file `wikilense_s1_v1.tar.gz` (`make fetch-s1`,
[docs/SETUP.md](../../docs/SETUP.md#get-the-scaled-database-s1)) holds the pages, claims, vectors
and page metadata of S1 under the same terms; its own `NOTICE.md` gives the attribution and the
changes.
