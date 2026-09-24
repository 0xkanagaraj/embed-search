"""
embedder.py

Multilingual sentence-transformer embedder.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - 50+ languages out of the box
  - 384-dimensional output (same DIM as before — no re-indexing needed
    IF you switch from all-MiniLM-L6-v2; the vector spaces differ so a
    full re-index of existing documents is still required after switching)
  - ~470 MB download on first run; cached to ~/.cache/huggingface after that
  - ~100 ms/query on CPU — negligible vs. LLM latency

IMPORTANT FIX: removed `import streamlit as st` and `@st.cache_resource`
that were left over from the Streamlit version. Those would crash the
server on startup since Streamlit is no longer installed.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # silence fork warning

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384

# Module-level singleton — loaded once, reused for every request.
# Previously used @st.cache_resource which crashes without Streamlit.
_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIM), dtype="float32")
    return _get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype("float32")


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
