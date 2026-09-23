"""Tests for wikilense.embedding: the model-free helpers, and the model itself (marked slow)."""

from __future__ import annotations

import numpy as np
import pytest

from wikilense import db as dbmod
from wikilense import embedding as embedding_module
from wikilense.embedding import (
    BGE_QUERY_INSTRUCTION,
    Embedder,
    text_to_vector,
    vector_to_text,
)

# --- helpers (no model) -------------------------------------------------------------------------


def test_vector_to_text_format() -> None:
    assert vector_to_text([0.1, 0.2]) == "[0.1,0.2]"
    assert vector_to_text(np.array([0.5, -0.25, 1.0], dtype=np.float32)) == "[0.5,-0.25,1.0]"
    assert vector_to_text([]) == "[]"
    text = vector_to_text(np.ones(384, dtype=np.float32))
    assert " " not in text
    assert text.startswith("[1.0,") and text.endswith(",1.0]")


def test_vector_to_text_rejects_non_finite_and_non_1d() -> None:
    with pytest.raises(ValueError):
        vector_to_text([1.0, float("nan")])
    with pytest.raises(ValueError):
        vector_to_text([1.0, float("inf")])
    with pytest.raises(ValueError):
        vector_to_text(np.zeros((2, 2)))


def test_text_round_trip_is_exact() -> None:
    rng = np.random.default_rng(1)
    v = rng.standard_normal(384).astype(np.float32)
    back = text_to_vector(vector_to_text(v))
    assert back.dtype == np.float32
    assert back.shape == (384,)
    assert np.array_equal(back, v)


def test_text_to_vector_parses_mariadb_style_text() -> None:
    v = text_to_vector("[0.1,0.2,-3e-05]")
    assert v.dtype == np.float32
    assert np.allclose(v, np.array([0.1, 0.2, -3e-05], dtype=np.float32))
    assert np.array_equal(text_to_vector("[1, 2]"), np.array([1.0, 2.0], dtype=np.float32))


def test_text_to_vector_rejects_bad_input() -> None:
    for bad in ("not json", "{}", "[[1,2]]", '["a"]', "[true]"):
        with pytest.raises(ValueError):
            text_to_vector(bad)


def test_bytes_and_text_encode_the_same_vector() -> None:
    rng = np.random.default_rng(2)
    v = rng.standard_normal(16).astype(np.float32)
    assert np.array_equal(dbmod.vec_from_bytes(dbmod.vec_param(v)), text_to_vector(vector_to_text(v)))


# --- Embedder, without loading the model --------------------------------------------------------


def test_embedder_is_lazy_and_reports_device() -> None:
    import torch

    embedder = Embedder()
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"
    assert not embedder.is_loaded
    assert embedder.device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert not embedder.is_loaded  # resolving the device must not load the model
    assert Embedder(device="cpu").device == "cpu"


def test_the_default_model_is_pinned_to_the_measured_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default model loads the Hub snapshot that results/ were measured with, whatever the
    Hub's main branch points at later; another model loads its main branch unless told."""
    from wikilense import config

    assert embedding_module.DEFAULT_MODEL_NAME == config.DEFAULT_EMBEDDING_MODEL
    assert Embedder().revision == config.DEFAULT_EMBEDDING_REVISION
    assert len(config.DEFAULT_EMBEDDING_REVISION) == 40  # a full commit hash, not a branch
    assert Embedder("sentence-transformers/all-MiniLM-L6-v2").revision is None
    assert Embedder("sentence-transformers/all-MiniLM-L6-v2", revision="abc").revision == "abc"

    loaded: list[dict] = []

    class RecordingModel:
        def __init__(self, name: str, **kwargs: object) -> None:
            loaded.append({"name": name, **kwargs})

        def eval(self) -> None:
            pass

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", RecordingModel)
    assert isinstance(Embedder(device="cpu").model, RecordingModel)
    assert loaded == [{"name": config.DEFAULT_EMBEDDING_MODEL, "device": "cpu",
                       "revision": config.DEFAULT_EMBEDDING_REVISION}]


def test_query_instruction_only_for_bge() -> None:
    assert Embedder("BAAI/bge-small-en-v1.5").query_instruction == BGE_QUERY_INSTRUCTION
    assert Embedder("BAAI/bge-base-en-v1.5").query_instruction == BGE_QUERY_INSTRUCTION
    assert Embedder("sentence-transformers/all-MiniLM-L6-v2").query_instruction == ""
    assert BGE_QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "


# --- Embedder with the real model (downloads it on first run) -----------------------------------

PASSAGES = ["The Aare is a river in Switzerland.", "Python is a programming language."]


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


@pytest.mark.slow
def test_model_shape_dim_and_device(embedder: Embedder) -> None:
    import torch

    out = embedder.embed_passages(PASSAGES)
    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    assert out.flags.c_contiguous
    assert embedder.dim == 384
    assert embedder.is_loaded
    assert embedder.device == ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.slow
def test_model_embeddings_are_normalised(embedder: Embedder) -> None:
    for out in (embedder.embed_passages(PASSAGES), embedder.embed_queries(PASSAGES)):
        norms = np.linalg.norm(out, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-4)


@pytest.mark.slow
def test_model_is_deterministic(embedder: Embedder) -> None:
    a = embedder.embed_passages(PASSAGES)
    b = embedder.embed_passages(PASSAGES)
    assert np.allclose(a, b, atol=1e-6)
    qa = embedder.embed_queries(PASSAGES)
    qb = embedder.embed_queries(PASSAGES)
    assert np.allclose(qa, qb, atol=1e-6)


@pytest.mark.slow
def test_model_query_and_passage_embeddings_differ(embedder: Embedder) -> None:
    passages = embedder.embed_passages(PASSAGES)
    queries = embedder.embed_queries(PASSAGES)
    assert queries.shape == passages.shape
    assert not np.allclose(passages, queries, atol=1e-3)


@pytest.mark.slow
def test_model_empty_input(embedder: Embedder) -> None:
    assert embedder.embed_passages([]).shape == (0, 384)
    assert embedder.embed_queries([]).shape == (0, 384)


@pytest.mark.slow
def test_model_semantic_sanity(embedder: Embedder) -> None:
    passages = embedder.embed_passages(PASSAGES)
    query = embedder.embed_queries(["Aare river in Switzerland"])[0]
    # Normalised vectors: cosine distance = 1 - dot, as VEC_DISTANCE_COSINE computes it.
    d_aare = 1.0 - float(passages[0] @ query)
    d_python = 1.0 - float(passages[1] @ query)
    assert d_aare < d_python
    assert d_aare < 0.3
