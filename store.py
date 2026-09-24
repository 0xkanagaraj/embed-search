"""
Per-user vector store.

Layout:
    data/index/<user>/
        chunks.jsonl     one JSON per chunk {file_id, filename, text}
        vectors.npy      (N, 384) float32, L2-normalized
        meta.json        {model, dim, count, updated_at}

The store is a derived artifact — it can be rebuilt from raw files.
"""
from pathlib import Path
import json
import pickle
from datetime import datetime

import numpy as np

from embedder import DIM


def store_dir(username: str) -> Path:
    d = Path("data/index") / username
    d.mkdir(parents=True, exist_ok=True)
    return d


def _chunks_path(username: str) -> Path:
    return store_dir(username) / "chunks.jsonl"


def _vectors_path(username: str) -> Path:
    return store_dir(username) / "vectors.npy"


def _meta_path(username: str) -> Path:
    return store_dir(username) / "meta.json"


def load(username: str) -> tuple[list[dict], np.ndarray]:
    cp, vp = _chunks_path(username), _vectors_path(username)
    if not cp.exists() or not vp.exists():
        return [], np.zeros((0, DIM), dtype="float32")

    chunks = []
    with open(cp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    vectors = np.load(vp)
    return chunks, vectors


def save(username: str, chunks: list[dict], vectors: np.ndarray):
    with open(_chunks_path(username), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(_vectors_path(username), vectors)
    with open(_meta_path(username), "w") as f:
        json.dump({
            "count": len(chunks),
            "dim": DIM,
            "model": "all-MiniLM-L6-v2",
            "updated_at": datetime.utcnow().isoformat(),
        }, f, indent=2)


def append(username: str, new_chunks: list[dict], new_vectors: np.ndarray):
    chunks, vectors = load(username)
    chunks = chunks + new_chunks
    vectors = np.vstack([vectors, new_vectors]) if len(vectors) else new_vectors
    save(username, chunks, vectors)


def remove_file(username: str, file_id: int):
    chunks, vectors = load(username)
    if not chunks:
        return
    keep = np.array([c["file_id"] != file_id for c in chunks])
    save(username,
         [c for c, k in zip(chunks, keep) if k],
         vectors[keep])


def search(username: str,
           query_vector: np.ndarray,
           file_ids: list[int] | None = None,
           k: int = 5) -> list[dict]:
    chunks, vectors = load(username)
    if not chunks:
        return []

    if file_ids is not None:
        mask = np.array([c["file_id"] in file_ids for c in chunks])
        if not mask.any():
            return []
        chunks = [c for c, m in zip(chunks, mask) if m]
        vectors = vectors[mask]

    scores = vectors @ query_vector
    top = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


def stats(username: str) -> dict:
    p = _meta_path(username)
    if not p.exists():
        return {"count": 0}
    with open(p) as f:
        return json.load(f)