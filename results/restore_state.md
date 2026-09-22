# Restored default state

Generated 2026-09-22T01:43:02+00:00. A fresh `run_ingest` with the defaults ('240 words, overlap 1, M=16, prefix on', what `wikilense ingest` produces), then `ingest_meta`, the index and the HNSW cache size checked.

| check | expected | found | ok |
|---|---|---|---|
| ingest_meta.chunk_max_words | 240 | 240 | yes |
| ingest_meta.chunk_overlap_units | 1 | 1 | yes |
| ingest_meta.embedding_prefix | true | true | yes |
| ingest_meta.index_m | 16 | 16 | yes |
| ingest_meta.index_distance | cosine | cosine | yes |
| ingest_meta.hatnote_pattern | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\b\|instance\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) | ^(?:(?:Main articles?\|See also\|Further information\|For other uses\|Not to be confused with)[:,]\|For (?!example\b\|instance\b)[^.]{0,80}?, see \|Not to be confused with \|This (?:article\|page) is about \|"[^"]{1,120}" redirects here) | yes |
| ingest_meta.embedding_model | BAAI/bge-small-en-v1.5 | BAAI/bge-small-en-v1.5 | yes |
| ingest_meta.embedding_dim | 384 | 384 | yes |
| vector index M (SHOW CREATE TABLE chunk) | 16 | 16 | yes |
| vector index DISTANCE | cosine | cosine | yes |
| @@GLOBAL.mhnsw_max_cache_size | 536870912 | 536870912 | yes |
| ingest_meta equals final_ef_100's (ingested_at, analyze_seconds excepted) | identical | identical | yes |

Ingest: 4598 chunks, 1316 hatnote units excluded, 21.0 s (embed 16.44 s); vector index tablespace 13631488 bytes.
