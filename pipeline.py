"""
Orchestrates ingestion and retrieval. The only module that talks
to both ingest+embedder (ingestion side) and store+embedder (retrieval side).
"""
from ingest import ingest_file
from embedder import embed_passages, embed_query
from reranker import rerank
import store


def index_file(username: str, file_id: int, filename: str, path: str) -> int:
    """Chunk + embed + append to the user's store. Returns chunk count."""
    pieces = ingest_file(path)   # [{"text": ..., "location": ...}, ...]
    if not pieces:
        return 0

    texts   = [p["text"] for p in pieces]
    vectors = embed_passages(texts)
    chunks = [
        {
            "file_id":  file_id,
            "filename": filename,
            "text":     p["text"],
            "location": p["location"],
        }
        for p in pieces
    ]
    store.append(username, chunks, vectors)
    return len(pieces)


def remove_file(username: str, file_id: int):
    store.remove_file(username, file_id)


def retrieve(username: str,
             query: str,
             file_ids: list[int] | None = None,
             k: int = 5) -> list[dict]:
    """
    Embed the query, pull a wider hybrid (vector + BM25) candidate set
    from the user's store, then re-rank down to the final top-k with the
    cross-encoder (when available; otherwise the hybrid order stands).
    """
    qv = embed_query(query)
    candidates = store.search(username, qv, query_text=query, file_ids=file_ids, k=k)
    return rerank(query, candidates, k)


def _cite(h: dict) -> str:
    return f"{h['filename']}, {h['location']}" if h.get("location") else h["filename"]


def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM, citing
    filename + page/slide/sheet where available."""
    return "\n\n---\n\n".join(
        f"[{_cite(h)}]\n{h['text']}" for h in hits
    )
