"""Retrieval for the ``retrieved`` condition with a fake embedder, search and connection: the
call it makes, the cache key and the cache's refusal of a different text. No database."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wikilense.llm.retrieval import Retriever, RetrievalError
from wikilense.search import Hit

SETTINGS = SimpleNamespace(ef_search=100, embedding_model="BAAI/bge-small-en-v1.5")


class FakeConn:
    def __init__(self):
        self.closed = False
        self.rolled_back = False

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def make(tmp_path, **kw):
    searches, embeds, conns = [], [], []

    def embed(texts):
        embeds.append(list(texts))
        return [[0.0] * 3 for _ in texts]

    def search(conn, qvec, **kwargs):
        searches.append(kwargs)
        return [Hit(chunk_id=10 + r, page_id=1, title="Paris", section_path="History",
                    chunk_ordinal=r, n_words=5, distance=0.1 * r, text=f"t{r}")
                for r in range(kwargs["k"])]

    def connect(settings, database):
        conn = FakeConn()
        conns.append((database, conn))
        return conn

    retriever = Retriever(SETTINGS, kw.pop("database", "wikilense"), tmp_path / "cache",
                          embed_queries=embed, search_fn=search, connect=connect, **kw)
    return retriever, searches, embeds, conns


def test_search_is_called_with_k_inline_ef_search_and_the_claim_text(tmp_path):
    retriever, searches, embeds, conns = make(tmp_path)
    hits = retriever.retrieve_many([(7, "Paris is old."), (8, "Rome is old.")])
    assert [h["chunk_id"] for h in hits[7]] == [10, 11, 12, 13, 14]
    assert hits[7][0]["rank"] == 1 and hits[7][0]["title"] == "Paris"
    assert searches[0] == {"k": 5, "strategy": "inline", "ef_search": 100,
                           "query_text": "Paris is old."}
    assert embeds == [["Paris is old.", "Rome is old."]]          # one batch
    assert conns[0][0] == "wikilense" and conns[0][1].rolled_back and conns[0][1].closed


def test_cache_is_keyed_by_database_k_ef_search_model_and_claim(tmp_path):
    retriever, searches, _, _ = make(tmp_path)
    retriever.retrieve_many([(7, "Paris is old.")])
    assert retriever.cache_path(7) == (tmp_path / "cache" / "wikilense"
                                       / "k5_inline_ef100_BAAI_bge-small-en-v1.5" / "7.json")
    retriever.retrieve_many([(7, "Paris is old.")])
    assert len(searches) == 1                                     # served from the cache
    other, other_searches, _, _ = make(tmp_path, database="wikilense_s1", k=3)
    other.retrieve_many([(7, "Paris is old.")])
    assert len(other_searches) == 1 and other.cache_dir != retriever.cache_dir


def test_cache_refuses_another_text_for_the_same_claim_id(tmp_path):
    retriever, _, _, _ = make(tmp_path)
    retriever.retrieve_many([(7, "Paris is old.")])
    with pytest.raises(RetrievalError, match="another text"):
        retriever.retrieve_many([(7, "Something else.")])


def test_ef_search_must_not_fall_back_to_the_server_value(tmp_path):
    with pytest.raises(RetrievalError):
        Retriever(SimpleNamespace(ef_search=0, embedding_model="m"), "wikilense", tmp_path)


def test_describe_records_the_cpu_device(tmp_path):
    retriever, _, _, _ = make(tmp_path)
    info = retriever.describe()
    assert info["device"] == "cpu" and info["k"] == 5 and info["ef_search"] == 100
    assert info["embedding_revision"]
