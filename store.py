"""
store.py — per-user vector store with in-memory cache.

Layout on disk:
    data/index/<user>/
        chunks.jsonl      one JSON object per line
        vectors.npy       (N, DIM) float32, L2-normalised
        meta.json         {model, dim, count, updated_at}

Performance:
    The first query after startup (or after an index write) reads from disk.
    Every subsequent query is served entirely from RAM — no file I/O, no
    deserialization. Cache is invalidated automatically via mtime comparison
    so it stays correct even if files are written by another process.

Thread safety:
    Python's GIL protects dict reads/writes. Safe for single-process uvicorn.
    For multi-worker deployments, use Redis or a shared vector DB instead.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from embedder import DIM, MODEL_NAME

# ── Cache: username → (chunks, vectors, mtime) ───────────────────
_CACHE: dict[str, tuple[list[dict], np.ndarray, float]] = {}


# ── Paths ─────────────────────────────────────────────────────────
def store_dir(username: str) -> Path:
    d = Path("data/index") / username
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


# ── Core I/O ──────────────────────────────────────────────────────
def load(username: str) -> tuple[list[dict], np.ndarray]:
    cp, vp = _chunks_path(username), _vectors_path(username)
    if not cp.exists() or not vp.exists():
        return [], np.zeros((0, DIM), dtype="float32")

    mtime = _disk_mtime(username)
    cached = _CACHE.get(username)

    # Cache hit: same or newer than disk
    if cached is not None and cached[2] >= mtime:
        return cached[0], cached[1]

    # Cache miss: read from disk and populate cache
    chunks: list[dict] = []
    with open(cp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    vectors = np.load(vp)
    _CACHE[username] = (chunks, vectors, mtime)
    return chunks, vectors


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
    chunks, vectors = load(username)
    chunks  = chunks + new_chunks
    vectors = np.vstack([vectors, new_vectors]) if len(vectors) else new_vectors
    save(username, chunks, vectors)


def remove_file(username: str, file_id: int) -> None:
    chunks, vectors = load(username)
    if not chunks:
        return
    keep   = np.array([c["file_id"] != file_id for c in chunks])
    save(username,
         [c for c, k in zip(chunks, keep) if k],
         vectors[keep])


def search(
    username:     str,
    query_vector: np.ndarray,
    file_ids:     Optional[list[int]] = None,
    k:            int = 5,
) -> list[dict]:
    chunks, vectors = load(username)   # served from RAM after first call
    if not chunks:
        return []

    if file_ids is not None:
        mask = np.array([c["file_id"] in file_ids for c in chunks])
        if not mask.any():
            return []
        chunks  = [c for c, m in zip(chunks, mask) if m]
        vectors = vectors[mask]

    scores = vectors @ query_vector           # pure numpy, fast
    top    = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


def stats(username: str) -> dict:
    p = _meta_path(username)
    if not p.exists():
        return {"count": 0}
    with open(p) as f:
        return json.load(f)
