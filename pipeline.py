"""
Orchestrates ingestion and retrieval. The only module that talks
to both ingest+embedder (ingestion side) and store+embedder (retrieval side).
"""
from ingest import ingest_file
from embedder import embed, embed_one
import store


def index_file(username: str, file_id: int, filename: str, path: str) -> int:
    """Chunk + embed + append to the user's store. Returns chunk count."""
    texts = ingest_file(path)
    if not texts:
        return 0

    vectors = embed(texts)
    chunks = [
        {"file_id": file_id, "filename": filename, "text": t}
        for t in texts
    ]
    store.append(username, chunks, vectors)
    return len(texts)


def remove_file(username: str, file_id: int):
    store.remove_file(username, file_id)


def retrieve(username: str,
             query: str,
             file_ids: list[int] | None = None,
             k: int = 5) -> list[dict]:
    """Embed the query and return top-k chunks from the user's store."""
    qv = embed_one(query)
    return store.search(username, qv, file_ids=file_ids, k=k)


def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    return "\n\n---\n\n".join(
        f"[{h['filename']}]\n{h['text']}" for h in hits
    )