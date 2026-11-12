"""Retrieval for the ``retrieved`` condition: CPU query embedding, ``search.search``, a disk cache.

Per claim: ``Embedder(settings.embedding_model, device="cpu").embed_queries([claim])[0]`` (adds the
BGE query instruction and pins the model revision), then ``search.search(conn, qvec, k=k,
strategy=strategy, ef_search=..., query_text=claim)`` on ``db.connect(settings,
database=...)``. The device is fixed to the CPU because the Embedder takes the GPU by itself when
it sees one, and chat B needs the GPU. ``ef_search`` is always passed: without it the server's
value (20) applies instead of the repository's 100.

The connection only reads: nothing is committed, and it is rolled back and closed afterwards.

Cache: one JSON file per claim under
``<cache_dir>/<database>/k<k>_<strategy>_ef<ef_search>_<embedding model>/<claim_id>.json``, so the
key is (database, k, strategy, ef_search, embedding model, claim id). Each entry also stores the
SHA-256 of the claim text, and a cached entry for the same id but another text is refused
rather than reused.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from wikilense.llm.prompts import sha256_text

DEVICE = "cpu"
#: Torch threads for the CPU embedding: enough for a few claims, gentle on a laptop that once
#: shut down from heat (memory note laptop-heavy-jobs-crash).
TORCH_THREADS = 4


class RetrievalError(RuntimeError):
    """Retrieval cannot run as configured, or the cache holds a conflicting entry."""


def _safe(name: str) -> str:
    """Return ``name`` with every character outside ``[A-Za-z0-9._-]`` replaced by ``_``."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class Retriever:
    """Top-k chunks per claim from one database, cached on disk.

    ``embed_queries``, ``search_fn`` and ``connect`` exist for tests; by default they are the
    repository's ``Embedder(..., device="cpu").embed_queries``, ``search.search`` and
    ``db.connect``.
    """

    def __init__(
        self,
        settings: Any,
        database: str,
        cache_dir: Path,
        k: int = 5,
        strategy: str = "inline",
        ef_search: int | None = None,
        embed_queries: Callable[[Sequence[str]], Any] | None = None,
        search_fn: Callable[..., list[Any]] | None = None,
        connect: Callable[..., Any] | None = None,
    ) -> None:
        from wikilense import search as search_mod

        if k < 1:
            raise RetrievalError(f"k must be >= 1, got {k}")
        if strategy not in search_mod.STRATEGIES:
            raise RetrievalError(f"unknown strategy {strategy!r}")
        resolved = search_mod.resolve_ef_search(ef_search, settings.ef_search)
        if resolved is None:
            raise RetrievalError(
                "ef_search resolves to the server's value; pass an explicit ef_search"
            )
        self.settings = settings
        self.database = database
        self.k = k
        self.strategy = strategy
        self.ef_search = int(resolved)
        self.embedding_model = settings.embedding_model
        self.cache_root = Path(cache_dir)
        self._embed = embed_queries
        self._search = search_fn or search_mod.search
        self._connect = connect
        self._embedder: Any = None

    # ----------------------------------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        """Return the cache folder for this database, k, strategy, ef_search and model."""
        return (self.cache_root / _safe(self.database)
                / f"k{self.k}_{self.strategy}_ef{self.ef_search}_{_safe(self.embedding_model)}")

    def cache_path(self, claim_id: int) -> Path:
        """Return the cache file of one claim."""
        return self.cache_dir / f"{int(claim_id)}.json"

    def describe(self) -> dict[str, Any]:
        """Return what the manifest records about retrieval (no model load, no query)."""
        from wikilense.embedding import Embedder

        return {
            "database": self.database,
            "k": self.k,
            "strategy": self.strategy,
            "ef_search": self.ef_search,
            "embedding_model": self.embedding_model,
            "embedding_revision": Embedder(self.embedding_model, device=DEVICE).revision_label,
            "device": DEVICE,
            "cache_dir": str(self.cache_dir),
        }

    # ----------------------------------------------------------------------------------
    def _embed_texts(self, texts: list[str]) -> Any:
        """Embed claims on the CPU (the BGE instruction is added by ``embed_queries``)."""
        if self._embed is not None:
            return self._embed(texts)
        if self._embedder is None:
            import torch

            from wikilense.embedding import Embedder

            torch.set_num_threads(min(TORCH_THREADS, os.cpu_count() or 1))
            self._embedder = Embedder(self.embedding_model, device=DEVICE)
            if self._embedder.device != DEVICE:  # pragma: no cover - defensive
                raise RetrievalError(f"embedder on {self._embedder.device}, not {DEVICE}")
        return self._embedder.embed_queries(texts)

    def _read_cache(self, claim_id: int, claim_text: str) -> list[dict[str, Any]] | None:
        """Return the cached hits of one claim, None when absent; raise on a text mismatch."""
        path = self.cache_path(claim_id)
        if not path.is_file():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("claim_sha256") != sha256_text(claim_text):
            raise RetrievalError(
                f"retrieval cache {path} holds another text for claim {claim_id}; "
                "use another cache folder"
            )
        return list(entry["hits"])

    def _write_cache(self, claim_id: int, claim_text: str, hits: list[dict[str, Any]]) -> None:
        """Write one claim's hits atomically (temporary file, then rename)."""
        path = self.cache_path(claim_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "database": self.database, "k": self.k, "strategy": self.strategy,
            "ef_search": self.ef_search, "embedding_model": self.embedding_model,
            "device": DEVICE, "claim_id": int(claim_id),
            "claim_sha256": sha256_text(claim_text),
            "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hits": hits,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    def retrieve_many(self, claims: Sequence[tuple[int, str]]) -> dict[int, list[dict[str, Any]]]:
        """Return ``{claim_id: hits}`` for ``(claim_id, claim_text)`` pairs, filling the cache.

        Cached claims are read back; the others are embedded in one batch on the CPU and
        searched one by one over a single read-only connection. A hit is a dict with ``rank``
        (from 1), ``chunk_id``, ``page_id``, ``title``, ``section_path``, ``chunk_ordinal``,
        ``n_words``, ``distance`` and ``text``.
        """
        result: dict[int, list[dict[str, Any]]] = {}
        todo: list[tuple[int, str]] = []
        for claim_id, text in claims:
            cached = self._read_cache(claim_id, text)
            if cached is None:
                todo.append((int(claim_id), text))
            else:
                result[int(claim_id)] = cached
        if not todo:
            return result
        vectors = self._embed_texts([text for _, text in todo])
        conn = self._open()
        try:
            for (claim_id, text), qvec in zip(todo, vectors, strict=True):
                found = self._search(conn, qvec, k=self.k, strategy=self.strategy,
                                     ef_search=self.ef_search, query_text=text)
                hits = [_hit_dict(rank, hit) for rank, hit in enumerate(found, 1)]
                self._write_cache(claim_id, text, hits)
                result[claim_id] = hits
        finally:
            _close(conn)
        return result

    def count_chunks(self) -> int | None:
        """Return ``SELECT COUNT(*) FROM chunk`` of the database (recorded in the manifest)."""
        conn = self._open()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunk")
                row = cur.fetchone()
            return None if row is None else int(row[0])
        finally:
            _close(conn)

    def _open(self) -> Any:
        """Return a connection to :attr:`database`."""
        if self._connect is not None:
            return self._connect(self.settings, database=self.database)
        from wikilense import db

        return db.connect(self.settings, database=self.database)


def _hit_dict(rank: int, hit: Any) -> dict[str, Any]:
    """Return one ``search.Hit`` as a plain dict with its rank."""
    return {
        "rank": rank,
        "chunk_id": int(hit.chunk_id),
        "page_id": int(hit.page_id),
        "title": str(hit.title),
        "section_path": str(hit.section_path),
        "chunk_ordinal": int(hit.chunk_ordinal),
        "n_words": int(hit.n_words),
        "distance": float(hit.distance),
        "text": str(hit.text),
    }


def _close(conn: Any) -> None:
    """Roll back and close a read-only connection, ignoring errors on the way out."""
    for step in ("rollback", "close"):
        with suppress(Exception):       # closing must not hide the real error
            getattr(conn, step)()
