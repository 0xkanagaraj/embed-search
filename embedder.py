"""
embedder.py

Multilingual sentence-transformer embedder.

Model: intfloat/multilingual-e5-small
  - 100+ languages
  - 384-dimensional output (same DIM as the previous MiniLM model, but the
    vector spaces are NOT compatible — a full re-index of existing
    documents is required after switching. Delete/rebuild data/index/*)
  - ~470 MB download on first run; cached to ~/.cache/huggingface after that
  - E5 models were trained with an instruction-style prefix and need it at
    inference time too: "query: " for search queries and "passage: " for
    indexed text. Skipping the prefix measurably hurts retrieval quality,
    so embed_query()/embed_passages() add it automatically — always use
    those instead of calling the model directly.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # silence fork warning

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384

# Module-level singleton — loaded once, reused for every request.
# sentence_transformers is imported lazily so that importing this module
# (e.g. for DIM / MODEL_NAME in store.py and the tests) stays instant.
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIM), dtype="float32")
    return _get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype("float32")


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed document chunks for indexing. Adds the required 'passage: ' prefix."""
    return _encode([f"passage: {t}" for t in texts])


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query. Adds the required 'query: ' prefix."""
    return _encode([f"query: {text}"])[0]

