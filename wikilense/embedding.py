"""Embedding of chunk and query texts, and the vector <-> bytes / text helpers for MariaDB.

The model (chosen in phase 1, see docs/DESIGN.md, "Embedding") is ``BAAI/bge-small-en-v1.5``, 384
dimensions. Embeddings are L2-normalised so that the cosine distance MariaDB computes with
``VEC_DISTANCE_COSINE`` equals ``1 - dot``.

The helpers at the bottom of this module do not need the model. ``vector_to_bytes`` produces the
MariaDB ``VECTOR`` storage format (float32, little-endian, ``4 * dim`` bytes) and
``vector_to_text`` the ``[x,y,...]`` form accepted by ``VEC_FromText``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

#: Query instruction of the BGE v1.5 English models. It is prepended to queries only; passages
#: are embedded as-is (the BGE model card: "no instruction needed for passages").
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_VECTOR_DTYPE = np.dtype("<f4")  # float32, little-endian: the MariaDB VECTOR storage format


class Embedder:
    """Sentence-transformers wrapper that loads the model on first use.

    Args:
        model_name: Hugging Face model id or local path. The BGE query instruction is added
            in :meth:`embed_queries` only when the name starts with ``"BAAI/bge"``.
        device: ``"cuda"``, ``"cpu"``, ``"cuda:1"`` ...; ``None`` picks ``"cuda"`` when a GPU
            is available, else ``"cpu"``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        """Return the device string the model runs on (resolved without loading the model)."""
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def dim(self) -> int:
        """Return the embedding dimension (loads the model on first access)."""
        model = self.model
        # sentence-transformers 6 renamed the accessor; keep the old name for >= 3.0.
        getter = getattr(model, "get_embedding_dimension", None)
        if getter is None:
            getter = model.get_sentence_embedding_dimension
        dim = getter()
        if dim is None:  # pragma: no cover - only for models without a pooling module
            raise RuntimeError(f"model {self.model_name!r} does not report an embedding dimension")
        return int(dim)

    @property
    def query_instruction(self) -> str:
        """Return the text prepended to queries: the BGE instruction for BGE models, else ``""``."""
        return BGE_QUERY_INSTRUCTION if self.model_name.startswith("BAAI/bge") else ""

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` when the model has been loaded into memory."""
        return self._model is not None

    @property
    def model(self) -> SentenceTransformer:
        """Return the underlying ``SentenceTransformer``, loading it on the first call."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self) -> SentenceTransformer:
        """Load the model onto :attr:`device` and return it (downloads to ~/.cache if needed)."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, device=self.device)
        # Prompts are handled explicitly in embed_queries; never let a prompt shipped with the
        # model configuration be added implicitly.
        model.default_prompt_name = None
        model.eval()
        return model

    def embed_passages(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed passages and return a float32 array of shape ``(len(texts), dim)``.

        Rows are L2-normalised. No instruction is prepended. An empty input returns an array
        of shape ``(0, dim)``.
        """
        return self._encode(list(texts), batch_size=batch_size, show_progress=show_progress)

    def embed_queries(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed queries and return a float32 array of shape ``(len(texts), dim)``.

        Rows are L2-normalised. For BGE models the query instruction
        (:data:`BGE_QUERY_INSTRUCTION`) is prepended to every text; for other models the texts
        are embedded unchanged. An empty input returns an array of shape ``(0, dim)``.
        """
        prefix = self.query_instruction
        return self._encode(
            [prefix + text for text in texts], batch_size=batch_size, show_progress=show_progress
        )

    def _encode(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        """Run the model and return a C-contiguous float32 ``(n, dim)`` array with unit rows."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be strings")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        result = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        if result.ndim != 2 or result.shape[0] != len(texts):
            raise RuntimeError(f"unexpected embedding shape {result.shape} for {len(texts)} texts")
        return result


def vector_to_bytes(v: Sequence[float] | np.ndarray) -> bytes:
    """Return the vector as float32 little-endian bytes (exactly ``4 * dim`` bytes).

    This is the MariaDB ``VECTOR`` storage format. The input must be one-dimensional.
    """
    arr = np.asarray(v, dtype=_VECTOR_DTYPE)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {arr.shape}")
    return arr.tobytes()


def bytes_to_vector(b: bytes | bytearray | memoryview) -> np.ndarray:
    """Return a writable float32 array decoded from little-endian float32 bytes.

    Raises ``ValueError`` when the length is not a multiple of 4.
    """
    if len(b) % 4 != 0:
        raise ValueError(f"vector bytes must be a multiple of 4 bytes long, got {len(b)}")
    return np.frombuffer(b, dtype=_VECTOR_DTYPE).astype(np.float32, copy=True)


def vector_to_text(v: Sequence[float] | np.ndarray) -> str:
    """Return the vector as ``"[x,y,...]"`` for MariaDB ``VEC_FromText``.

    Each component is written as a Python float via ``json.dumps`` (shortest round-trip
    representation, no spaces). Non-finite values raise ``ValueError``.
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {arr.shape}")
    return json.dumps(arr.tolist(), separators=(",", ":"), allow_nan=False)


def text_to_vector(s: str) -> np.ndarray:
    """Return a float32 array parsed from ``"[x,y,...]"`` (the ``VEC_ToText`` format).

    Raises ``ValueError`` when the text is not a flat JSON array of numbers.
    """
    try:
        values = json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not a vector text: {s[:40]!r}") from exc
    if not isinstance(values, list) or not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in values
    ):
        raise ValueError("vector text must be a flat JSON array of numbers")
    return np.asarray(values, dtype=np.float32)
