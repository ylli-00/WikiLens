# The README's commands, run from a fresh clone

Run on 2026-11-11 on the author's laptop (i9-13900H, Ubuntu 26.04.1, Docker 29.1.3, Docker Compose
2.40.3, the system's Python 3.14.4; the GPU was not used). The repository was cloned from GitHub
into an empty directory, at commit 832d51e of the branch `fable-review-2026-11-11`, which `main`
was fast-forwarded to afterwards. The only later change is this record and the README lines copied
from it:

```
git clone --branch fable-review-2026-11-11 https://github.com/ylli-00/WikiLens.git WikiLens
```

The blocks below are copied from the logs of that run. `<clone>` stands for the clone's directory,
and pip's "Collecting", "Using cached" and build lines and the model loader's progress lines are
left out. The packages came from pip's local cache and the embedding model from
`~/.cache/huggingface`, so a first run on another machine also downloads them (about 134 MB for
the model).

## `make setup CPU=1`: 4 minutes 45 seconds

There was no `.env`. The run ended with exit code 0 after `real 4m44.810s`.

```
python3 scripts/make_env.py
Created .env from .env.example with two generated passwords (they are in that file, which git ignores).
python3 -m venv --clear .venv
docker compose up -d --wait
 Network wikilens_default  Creating
 Network wikilens_default  Created
 Volume wikilens_mariadb-data  Creating
 Volume wikilens_mariadb-data  Created
 Container wikilense-mariadb  Creating
 Container wikilense-mariadb  Created
 Container wikilense-mariadb  Starting
 Container wikilense-mariadb  Started
 Container wikilense-mariadb  Waiting
 Container wikilense-mariadb  Healthy
.venv/bin/pip install -r requirements-cpu.lock -e .
Ignoring colorama: markers 'sys_platform == "win32"' don't match your environment
Ignoring torch: markers 'sys_platform == "darwin"' don't match your environment
.venv/bin/wikilense ingest
schema: 11 statements applied (reset=True) in 0.10 s
metadata: 100 of 100 pages described, 1005 categories, 1123 page_category rows
pages: 100 pages, 3173 sections, 33637 sentences (372 empty, 1316 hatnotes), 40542 links (0 skipped), 4598 chunks; parse 0.75 s, load 0.94 s
embed: 4598 chunk texts with BAAI/bge-small-en-v1.5 (dim 384, prefix=True, batch_size 64) in 225.98 s
chunks: 4598 chunk rows and 33783 chunk_sentence rows written in 2.18 s
links: 373 of 40542 resolved to corpus pages in 0.00 s
analyze: statistics of chunk, page, section, link, category, page_category refreshed in 0.011 s
claims: 75 claims, 114 evidence ids (114 with a corpus page, 87 with a text unit)
meta: 19 ingest_meta rows written (model BAAI/bge-small-en-v1.5, corpus data/corpus)
ingest done: 100 pages, 4598 chunks, 75 claims in 230.0 s
ingest finished in 230.0 s (schema 0.1, parse 0.7, embed 226.0, load 3.1, resolve 0.0, analyze 0.0)
  pages 100, sections 3173, sentences 33637 (empty 372), chunks 4598
  links 40542 (resolved 373, skipped 0)
  claims 75, evidence ids 114 (page resolved 114, sentence resolved 87)
  page metadata 100 of 100 pages, categories 1005 (page_category 1123)
```

The four indented lines are the report the README prints. Embedding 4,598 chunks took 225.98 s on
the CPU; the same step takes about 17 s on the laptop's GPU (`results/SUMMARY.md`, section 8).

## `make test`

```
404 passed in 19.51s
```

Python 3.14.4, torch 2.14.0+cpu, PyMySQL 1.2.3, sentence-transformers 6.0.1, MariaDB 11.8.9-MariaDB-ubu2404 (from the versions block that `make eval` wrote in the clone).

## `make eval`

```
k   article_recall  evidence_recall  unit_coverage  sql_p50_ms  sql_p95_ms
1   0.840           0.591            0.596          0.39        0.88
3   0.987           0.788            0.808          0.39        0.81
5   0.987           0.833            0.852          0.39        0.85
10  0.987           0.909            0.917          0.44        0.93
20  0.987           0.985            0.985          0.56        1.15
claims 75 (evidence-eligible 66); query embedding p50 18.17 ms, p95 23.29 ms over 375 samples
```

Recall equals the published headline (`results/final_ef_100.md`): 0.987 at k=5, 0.909 at k=10 and
0.985 at k=20, on vectors embedded on the CPU.

## One question

```
$ .venv/bin/wikilense query "Which river rises and ends entirely within Switzerland?" --k 1
 1. 0.3091  Aare  [chunk_id 1, 127 words]
    The Aare (German: [ˈaːrə]) or Aar (German: [aːɐ̯] (listen)) is a tributary of the High Rhine and
    the longest river that both rises and ends entirely within Switzerland. Its total length from
    its source to its junction with the Rhine comprises about 295 kilometres (183 mi), during which
    distance it descends 1,565 m (5,135 ft), draining an area of 17,779 km (6,865 sq mi), almost
    entirely within Switzerland, and accounting for close to half the area of the country, including
    all of Central Switzerland. There are more than 40 hydroelectric plants along the course of the
    Aare. The river's name dates to at least the La Tène period, and it is attested as Nantaror
    "Aare valley" in the Berne zinc tablet. The name was Latinized as Arula/Arola/Araris.
1 hit; model load 3691 ms, embedding 15.2 ms, SQL 5.7 ms (BAAI/bge-small-en-v1.5 on cpu, strategy inline, ef_search 100)
```

The same command a second time ended with:

```
1 hit; model load 3782 ms, embedding 9.5 ms, SQL 6.0 ms (BAAI/bge-small-en-v1.5 on cpu, strategy inline, ef_search 100)
```

## `make examples`

Exit code 0, in `real 0m0.169s`: the ten examples of `sql/examples.sql` ran through the `mariadb`
client inside the container and printed 14 result tables. No Python is involved.

## `make serve`

The server answered 5 seconds after `make serve`. `GET /` returned 200 (11,831 bytes),
`GET /docs` returned 200, and
`GET /api/search?q=Which+river+rises+and+ends+entirely+within+Switzerland%3F&k=1` returned 200 with
the same chunk (Aare, chunk 1, distance 0.3091). The brief's four filters through the API
(`q=independence+from+the+Soviet+Union&k=3&category=Member+states+of+the+United+Nations&min_words=1000&edited_since=2020-12-01&linked_from=Asia`)
returned Azerbaijan (0.2522), Azerbaijan (0.3012), Kazakhstan (0.3087), and the first `EXPLAIN`
row was `chunk`, type `index`, key `embedding`. A request with a parameter the API does
not have (`explain=true`) was refused with 400 and the list of the parameters it accepts.

## A `.env` that does not fit the data volume

With the volume in place, `.env` was deleted and `make setup CPU=1` run again. It wrote a new
`.env` with new passwords, and the ingest stopped with one line:

```
wikilense: database error: Access denied for user 'wikilense'@'172.18.0.1' (using password: YES) (error 1045); the password in .env is not the one this MariaDB data volume was created with: put that password back in .env, or delete the volume with 'docker compose down -v' (this deletes the database) and run 'make setup' again
```

## Not run

Plain `make setup` (the version ranges of `requirements-dev.txt`), the command on the README's
first screen, was not run in this clone: on Linux it installs the CUDA build of PyTorch, several
GB. Its recipe differs from the run above only in the requirements file it gives to pip. The
ranges were checked by resolution only (`docs/SETUP.md`, "Platforms").
