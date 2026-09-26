"""
store.py — per-user vector + BM25 store with in-memory cache.

Layout on disk:
    data/index/<user>/
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

Thread safety:
    Python's GIL protects dict reads/writes. Safe for single-process
    uvicorn. For multi-worker deployments, use Redis or a shared vector
    DB instead (see README "Scaling beyond one process").
"""

import json
import re
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
CANDIDATE_MULTIPLIER = 4
CANDIDATE_MIN = 20

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class IndexModelMismatch(Exception):
    """Raised when the on-disk index was built with a different embedding
    model than the one currently configured. The vector spaces of two
    different models are not comparable, so results would be meaningless
    — the caller must re-index (delete data/index/<user>/ and re-upload,
    or run a migration script) before querying again."""
    def __init__(self, indexed_model: str, current_model: str):
        self.indexed_model = indexed_model
        self.current_model = current_model
        super().__init__(
            f"Index was built with model '{indexed_model}' but the server "
            f"is now running '{current_model}'. Re-index this user's files "
            f"before querying (the two models' vectors aren't comparable)."
        )


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ── Cache: username → (chunks, vectors, bm25_index, mtime) ─────────
_CACHE: dict[str, tuple[list[dict], np.ndarray, object, float]] = {}


# ── Paths ─────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

def store_dir(username: str) -> Path:
    d = _HERE / "data" / "index" / username
    d.mkdir(parents=True, exist_ok=True)
    return d

def _chunks_path(u: str) -> Path: return store_dir(u) / "chunks.jsonl"
def _vectors_path(u: str) -> Path: return store_dir(u) / "vectors.npy"
def _meta_path(u: str)   -> Path: return store_dir(u) / "meta.json"


def _disk_mtime(username: str) -> float:
    cp, vp = _chunks_path(username), _vectors_path(username)
    if not cp.exists() or not vp.exists():
        return 0.0
    return max(cp.stat().st_mtime, vp.stat().st_mtime)


def _invalidate(username: str) -> None:
    """Drop cached data for this user (call after every write)."""
    _CACHE.pop(username, None)


def _build_bm25(chunks: list[dict]):
    """Build (or skip, if the dependency isn't installed) a BM25 index."""
    if not chunks:
        return None
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None
    tokenized = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def _check_model(username: str) -> None:
    """Raise IndexModelMismatch if the on-disk index used a different model."""
    mp = _meta_path(username)
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
def load(username: str, check_model: bool = True) -> tuple[list[dict], np.ndarray, object]:
    if check_model:
        _check_model(username)

    cp, vp = _chunks_path(username), _vectors_path(username)
    if not cp.exists() or not vp.exists():
        return [], np.zeros((0, DIM), dtype="float32"), None

    mtime = _disk_mtime(username)
    cached = _CACHE.get(username)

    # Cache hit: same or newer than disk
    if cached is not None and cached[3] >= mtime:
        return cached[0], cached[1], cached[2]

    # Cache miss: read from disk and populate cache
    chunks: list[dict] = []
    with open(cp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    vectors = np.load(vp)
    bm25 = _build_bm25(chunks)
    _CACHE[username] = (chunks, vectors, bm25, mtime)
    return chunks, vectors, bm25


def save(username: str, chunks: list[dict], vectors: np.ndarray) -> None:
    with open(_chunks_path(username), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(_vectors_path(username), vectors)
    with open(_meta_path(username), "w") as f:
        json.dump({
            "model":      MODEL_NAME,
            "dim":        DIM,
            "count":      len(chunks),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    _invalidate(username)   # keep cache consistent after every write


# ── Public API ────────────────────────────────────────────────────
def append(username: str, new_chunks: list[dict], new_vectors: np.ndarray) -> None:
    chunks, vectors, _ = load(username)
    chunks  = chunks + new_chunks
    vectors = np.vstack([vectors, new_vectors]) if len(vectors) else new_vectors
    save(username, chunks, vectors)


def remove_file(username: str, file_id: int, filename: Optional[str] = None) -> None:
    # Deleting a file doesn't depend on vector-space compatibility, so it
    # should work even while an index is mid-migration to a new model.
    chunks, vectors, _ = load(username, check_model=False)
    if not chunks:
        return
    fid_int = int(file_id) if file_id is not None else None
    keep = []
    for c in chunks:
        c_fid = int(c.get("file_id", -1))
        c_fn = c.get("filename")
        if fid_int is not None and c_fid == fid_int:
            keep.append(False)
        elif filename is not None and c_fn == filename:
            keep.append(False)
        else:
            keep.append(True)
    keep_arr = np.array(keep, dtype=bool)
    if len(vectors) == len(chunks):
        new_vectors = vectors[keep_arr] if keep_arr.any() else np.zeros((0, DIM), dtype="float32")
    else:
        new_vectors = np.zeros((0, DIM), dtype="float32")
    save(username,
         [c for c, k in zip(chunks, keep_arr) if k],
         new_vectors)


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def search(
    username:     str,
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
    chunks, vectors, bm25 = load(username)   # served from RAM after first call
    if not chunks:
        return []

    if file_ids is not None:
        if not file_ids:
            return []
        mask = np.array([c.get("file_id") in file_ids for c in chunks])
        if not mask.any():
            return []
        idx = np.where(mask)[0]
        chunks_f  = [chunks[i] for i in idx]
        vectors_f = vectors[idx]
    else:
        idx = np.arange(len(chunks))
        chunks_f, vectors_f = chunks, vectors

    vec_scores = vectors_f @ query_vector          # cosine (vectors are normalised)
    vec_norm   = _minmax(vec_scores)

    if bm25 is not None and query_text:
        # rank_bm25 scores the whole corpus; slice down to our filtered subset.
        full_bm25 = np.asarray(bm25.get_scores(_tokenize(query_text)))
        bm25_scores = full_bm25[idx]
        bm25_norm = _minmax(bm25_scores)
        combined = VECTOR_WEIGHT * vec_norm + BM25_WEIGHT * bm25_norm
    else:
        combined = vec_scores

    n_candidates = max(k * CANDIDATE_MULTIPLIER, CANDIDATE_MIN)
    top = np.argsort(combined)[::-1][:n_candidates]
    return [
        {
            **chunks_f[i],
            "score": float(combined[i]),
            "vec_score": float(vec_scores[i]),
            "hybrid_score": float(combined[i]),
        }
        for i in top
    ]


def stats(username: str) -> dict:
    p = _meta_path(username)
    if not p.exists():
        return {"count": 0}
    with open(p) as f:
        return json.load(f)
