"""
reranker.py — optional cross-encoder re-ranking pass.

Bi-encoder embeddings (what embedder.py produces) are fast and scale to
huge corpora, but a cross-encoder that looks at the (query, passage)
pair jointly is meaningfully more accurate at judging relevance. Used
here as a second-stage re-rank: retrieve a wider candidate set cheaply
with vectors + BM25, then re-score just those top candidates with the
cross-encoder before returning the final top-k.

This is optional and fails soft: if the model can't be loaded (e.g. no
network on first run, or intentionally disabled), rerank() returns the
input order unchanged rather than breaking search.
"""

import math
import os

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingual
RERANK_ENABLED = os.environ.get("DISABLE_RERANK", "").lower() not in ("1", "true", "yes")

_MODEL = None
_load_failed = False


def _get_model():
    global _MODEL, _load_failed
    if _MODEL is None and not _load_failed:
        try:
            from sentence_transformers import CrossEncoder
            _MODEL = CrossEncoder(RERANK_MODEL)
        except Exception as e:
            print(f"⚠️  Reranker unavailable ({e}); continuing without re-ranking.")
            _load_failed = True
    return _MODEL


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """
    Re-score `hits` (each a dict with a 'text' key) against `query` with
    the cross-encoder and return the best `top_k`, each with an added
    'rerank_score'. Falls back to hits[:top_k] unchanged if disabled or
    unavailable.
    """
    if not RERANK_ENABLED or not hits:
        return hits[:top_k]

    model = _get_model()
    if model is None:
        return hits[:top_k]

    pairs = [(query, h["text"]) for h in hits]
    scores = model.predict(pairs)
    for h, s in zip(hits, scores):
        s_val = float(s)
        # Sigmoid maps unbounded logits to clean [0, 1] probability
        prob = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, s_val))))
        h["raw_rerank_score"] = s_val
        h["rerank_score"] = prob
        h["score"] = prob  # Set primary score so UI displays the metric used to rank
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]

