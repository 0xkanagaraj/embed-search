from ingest import ingest_file
from embedder import embed_passages, embed_query
from reranker import rerank
import store


def index_file(file_id: int, filename: str, path: str) -> int:
    pieces = ingest_file(path)
    if not pieces:
        return 0

    texts   = [p["text"] for p in pieces]
    vectors = embed_passages(texts)
    chunks  = [
        {
            "file_id":  file_id,
            "filename": filename,
            "text":     p["text"],
            "location": p["location"],
        }
        for p in pieces
    ]
    store.append(chunks, vectors)
    return len(pieces)


def remove_file(file_id: int, filename: str | None = None) -> None:
    store.remove_file(file_id, filename)


def retrieve(
    query:    str,
    file_ids: list[int] | None = None,
    k:        int = 5,
) -> list[dict]:
    if file_ids is not None and len(file_ids) == 0:
        return []
    qv         = embed_query(query)
    candidates = store.search(qv, query_text=query, file_ids=file_ids, k=k)
    return rerank(query, candidates, k)


def _cite(h: dict) -> str:
    return f"{h['filename']}, {h['location']}" if h.get("location") else h["filename"]


def build_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{_cite(h)}]\n{h['text']}" for h in hits
    )
