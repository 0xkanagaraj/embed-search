import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        # Lazy import keeps `import embedder` instant (store.py and tests only need DIM/MODEL_NAME).
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


# E5 models need these prefixes at inference time; without them retrieval quality drops.
def embed_passages(texts: list[str]) -> np.ndarray:
    return _encode([f"passage: {t}" for t in texts])


def embed_query(text: str) -> np.ndarray:
    return _encode([f"query: {text}"])[0]
