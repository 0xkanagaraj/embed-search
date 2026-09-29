import math
import os

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
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


# Fails soft: if the model can't load, hybrid order is returned unchanged.
def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    if not RERANK_ENABLED or not hits:
        return hits[:top_k]

    model = _get_model()
    if model is None:
        return hits[:top_k]

    pairs = [(query, h["text"]) for h in hits]
    scores = model.predict(pairs)
    for h, s in zip(hits, scores):
        s_val = float(s)
        prob = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, s_val))))
        h["raw_rerank_score"] = s_val
        h["rerank_score"] = prob
        h["score"] = prob
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]
