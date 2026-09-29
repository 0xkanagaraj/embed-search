"""
store.py — vector + BM25 store with in-memory cache.

Layout on disk:
    data/index/
        chunks.jsonl      one JSON object per line
        vectors.npy       (N, DIM) float32, L2-normalised
        meta.json         {model, dim, count, updated_at}

Retrieval is hybrid: dense cosine similarity (via the embedding model)
combined with BM25 lexical scoring, so exact keywords/IDs/names that a
small embedding model might blur are still found. The BM25 index is
built in memory from chunks.jsonl on load — it's cheap to rebuild and
keeping it out of the on-disk format avoids a second file to keep in
sync.

Performance:
    The first query after startup (or after an index write) reads from
    disk and rebuilds the BM25 index. Every subsequent query is served
    entirely from RAM. Cache is invalidated automatically via mtime
    comparison so it stays correct even if files are written by another
    process.
"""

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from embedder import DIM, MODEL_NAME

# Hybrid scoring weights — vector similarity carries more weight, BM25
# is there mainly to rescue exact keyword/ID matches a small embedding
# model can miss.
VECTOR_WEIGHT = 0.65
BM25_WEIGHT   = 0.35
# How many hybrid candidates to hand up to the (optional) reranker.
# The reranker scores (k × CANDIDATE_MULTIPLIER) candidates, then trims to k.
# Higher multiplier = better recall at the cost of cross-encoder latency.
CANDIDATE_MULTIPLIER = 4
CANDIDATE_MIN = 10   # minimum pool size even when k is very small (k=1,2)

# \w+ is unicode-aware; the embedding model is multilingual, so BM25 must be too.
_TOKEN_RE = re.compile(r"\w+")


class IndexModelMismatch(Exception):
    """Raised when the on-disk index was built with a different embedding
    model than the one currently configured. The vector spaces of two
    different models are not comparable, so results would be meaningless
    — the caller must re-index (delete data/index/ and re-upload,
    or run a migration script) before querying again."""
    def __init__(self, indexed_model: str, current_model: str):
        self.indexed_model = indexed_model
        self.current_model = current_model
        super().__init__(
            f"Index was built with model '{indexed_model}' but the server "
            f"is now running '{current_model}'. Delete data/index/ and "
            f"re-upload your files to rebuild the index."
        )


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ── Paths ─────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
_INDEX_DIR = _HERE / "data" / "index"

def _chunks_path() -> Path:  return _INDEX_DIR / "chunks.jsonl"
def _vectors_path() -> Path: return _INDEX_DIR / "vectors.npy"
def _meta_path()   -> Path:  return _INDEX_DIR / "meta.json"

_INDEX_DIR.mkdir(parents=True, exist_ok=True)


# Serialises read-modify-write cycles (append / remove) so two concurrent
# uploads can't overwrite each other's chunks.
_WRITE_LOCK = threading.RLock()


# ── Cache: (chunks, vectors, bm25_index, mtime) ────────────────────
_CACHE: tuple[list[dict], np.ndarray, object, float] | None = None


def _disk_mtime() -> float:
    cp, vp = _chunks_path(), _vectors_path()
    if not cp.exists() or not vp.exists():
        return 0.0
    return max(cp.stat().st_mtime, vp.stat().st_mtime)


def _invalidate() -> None:
    """Drop the in-memory cache (call after every write)."""
    global _CACHE
    _CACHE = None


def _build_bm25(chunks: list[dict]):
    """Build (or skip, if rank_bm25 isn't installed) a BM25 index."""
    if not chunks:
        return None
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None
    tokenized = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def _check_model() -> None:
    """Raise IndexModelMismatch if the on-disk index used a different model."""
    mp = _meta_path()
    if not mp.exists():
        return
    try:
        with open(mp) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    indexed_model = meta.get("model")
    if indexed_model and indexed_model != MODEL_NAME and meta.get("count", 0) > 0:
        raise IndexModelMismatch(indexed_model, MODEL_NAME)


# ── Core I/O ──────────────────────────────────────────────────────
def load(check_model: bool = True) -> tuple[list[dict], np.ndarray, object]:
    global _CACHE
    if check_model:
        _check_model()

    cp, vp = _chunks_path(), _vectors_path()
    if not cp.exists() or not vp.exists():
        return [], np.zeros((0, DIM), dtype="float32"), None

    mtime = _disk_mtime()
    if _CACHE is not None and _CACHE[3] >= mtime:
        return _CACHE[0], _CACHE[1], _CACHE[2]

    # Cache miss: read from disk and populate cache
    chunks: list[dict] = []
    with open(cp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    vectors = np.load(vp)
    bm25 = _build_bm25(chunks)
    _CACHE = (chunks, vectors, bm25, mtime)
    return chunks, vectors, bm25


def save(chunks: list[dict], vectors: np.ndarray) -> None:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(_chunks_path(), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(_vectors_path(), vectors)
    with open(_meta_path(), "w") as f:
        json.dump({
            "model":      MODEL_NAME,
            "dim":        DIM,
            "count":      len(chunks),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    _invalidate()


# ── Public API ────────────────────────────────────────────────────
def append(new_chunks: list[dict], new_vectors: np.ndarray) -> None:
    with _WRITE_LOCK:
        chunks, vectors, _ = load()
        chunks  = chunks + new_chunks
        vectors = np.vstack([vectors, new_vectors]) if len(vectors) else new_vectors
        save(chunks, vectors)


def remove_file(file_id: int, filename: Optional[str] = None) -> None:
    """Drop every chunk belonging to `file_id` (or, for stale data, `filename`)."""
    with _WRITE_LOCK:
        # Deleting a file doesn't depend on vector-space compatibility.
        chunks, vectors, _ = load(check_model=False)
        if not chunks:
            return
        keep = np.array(
            [
                not (
                    c.get("file_id") == file_id
                    or (filename is not None and c.get("filename") == filename)
                )
                for c in chunks
            ],
            dtype=bool,
        )
        if keep.all():
            return                       # nothing to remove — skip the rewrite
        if len(vectors) != len(chunks):
            # chunks.jsonl and vectors.npy are out of sync (corrupt index).
            # Never write chunks without matching vectors; reset both.
            save([], np.zeros((0, DIM), dtype="float32"))
            return
        save([c for c, k in zip(chunks, keep) if k], vectors[keep])


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def search(
    query_vector: np.ndarray,
    query_text:   str = "",
    file_ids:     Optional[list[int]] = None,
    k:            int = 5,
) -> list[dict]:
    """
    Hybrid retrieval: combine cosine similarity with BM25 lexical score
    (when rank_bm25 is installed; falls back to pure vector search
    otherwise) and return the top `k * CANDIDATE_MULTIPLIER` candidates
    for the caller to optionally re-rank down to `k`.
    """
    chunks, vectors, bm25 = load()   # served from RAM after first call
    if not chunks:
        return []

    if file_ids is not None:
        if not file_ids:
            return []
        wanted = set(file_ids)
        mask = np.array([c.get("file_id") in wanted for c in chunks])
        if not mask.any():
            return []
        idx       = np.where(mask)[0]
        chunks_f  = [chunks[i] for i in idx]
        vectors_f = vectors[idx]
    else:
        idx = np.arange(len(chunks))
        chunks_f, vectors_f = chunks, vectors

    vec_scores = vectors_f @ query_vector          # cosine (vectors are normalised)
    vec_norm   = _minmax(vec_scores)

    if bm25 is not None and query_text:
        # rank_bm25 scores the whole corpus; slice down to our filtered subset.
        full_bm25   = np.asarray(bm25.get_scores(_tokenize(query_text)))
        bm25_scores = full_bm25[idx]
        bm25_norm   = _minmax(bm25_scores)
        combined    = VECTOR_WEIGHT * vec_norm + BM25_WEIGHT * bm25_norm
    else:
        combined = vec_scores

    n_candidates = max(k * CANDIDATE_MULTIPLIER, CANDIDATE_MIN)
    top = np.argsort(combined)[::-1][:n_candidates]
    return [
        {
            **chunks_f[i],
            "score":        float(combined[i]),
            "vec_score":    float(vec_scores[i]),
            "hybrid_score": float(combined[i]),
        }
        for i in top
    ]
