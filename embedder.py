import os
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384


@st.cache_resource(show_spinner=False)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIM), dtype="float32")
    return _model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]