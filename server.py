"""
server.py — SemanticSearch FastAPI backend (no-auth, single-user mode)

Run:
    python server.py
    uvicorn server:app --host 0.0.0.0 --port 8502 --reload

Speed notes:
  - Embedding model is pre-loaded at startup (lifespan hook) so the first
    query doesn't pay the cold-start penalty (~3-5 s model load).
  - The vector store is cached in RAM after the first load; subsequent
    queries hit memory only.
  - The query endpoint streams SSE events so the user sees sources within
    ~400-700 ms and then watches the answer appear token-by-token.
"""

import asyncio
import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, field_validator

import db
import validation
from ingest import SUPPORTED_EXTENSIONS, IngestLimitExceeded
from llm import stream_generate
from pipeline import build_context, index_file, remove_file, retrieve
from store import IndexModelMismatch
import reranker as _reranker_mod

_HERE      = Path(__file__).parent
FILES_ROOT = _HERE / "data" / "files"
_UI        = _HERE / "ui.html"

_ACCEPT_EXTS = SUPPORTED_EXTENSIONS   # e.g. {'.pdf', '.docx', ...}


# ── Warmup ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load and warm the embedding and reranker models before accepting requests."""
    print("⏳  [1/2] Loading embedding model (Bi-Encoder) …")
    from embedder import embed_query
    embed_query("warmup")
    print("⏳  [2/2] Loading re-ranking model (Cross-Encoder) …")
    from reranker import _get_model
    _get_model()
    print("✅  All models loaded — server is live at http://localhost:8502")
    yield


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="SemanticSearch",
    docs_url=None, redoc_url=None,
    lifespan=lifespan,
)


def _files_dir() -> Path:
    FILES_ROOT.mkdir(parents=True, exist_ok=True)
    return FILES_ROOT


# ── UI ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_UI.read_text(encoding="utf-8"))


# ── Files ─────────────────────────────────────────────────────────
@app.get("/api/files")
async def get_files():
    return db.list_files()


@app.post("/api/files")
async def upload_files(files: list[UploadFile] = File(...)):
    root = _files_dir()
    results = []
    for uf in files:
        ext = Path(uf.filename).suffix.lower()
        if ext not in _ACCEPT_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"'{ext}' is not supported. "
                       f"Accepted: {', '.join(sorted(_ACCEPT_EXTS))}",
            )

        content = await uf.read()
        try:
            validation.validate_upload(uf.filename, content)
        except validation.ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        dest = root / uf.filename
        dest.write_bytes(content)
        file_id = db.add_file(uf.filename, str(dest), 0)
        remove_file(file_id, uf.filename)   # clear stale chunks for this file

        try:
            n = index_file(file_id, uf.filename, str(dest))
        except IngestLimitExceeded as e:
            db.delete_file(file_id)
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(e))
        except IndexModelMismatch as e:
            db.delete_file(file_id)
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail=f"{e} Delete data/index/ and re-upload all files to rebuild.",
            )

        db.update_file_chunks(file_id, n)
        results.append({"filename": uf.filename, "n_chunks": n, "file_id": file_id})
    return results


@app.delete("/api/files/{file_id}")
async def delete_file_route(file_id: int):
    files = db.list_files()
    f = next((x for x in files if x["id"] == file_id), None)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    db.delete_file(file_id)
    f_path = Path(f["path"])
    if not f_path.is_absolute():
        f_path = _HERE / f_path
    f_path.unlink(missing_ok=True)
    remove_file(file_id, f.get("filename"))
    return {"ok": True}


@app.get("/api/files/{file_id}/chunks")
async def preview_file_chunks(file_id: int):
    """Preview indexed chunks for one file — lets you verify extraction."""
    files = db.list_files()
    if not any(f["id"] == file_id for f in files):
        raise HTTPException(status_code=404, detail="File not found")

    import store
    chunks, _, _ = store.load(check_model=False)
    file_chunks = [c for c in chunks if c["file_id"] == file_id]
    return {
        "file_id":  file_id,
        "n_chunks": len(file_chunks),
        "chunks":   [{"text": c["text"], "location": c.get("location")} for c in file_chunks],
    }


# ── Debug ─────────────────────────────────────────────────────────
@app.get("/api/debug/search")
async def debug_search(q: str, k: int = 5):
    """
    Diagnostic endpoint — runs the full retrieval+rerank pipeline and
    returns a detailed score breakdown so you can verify accuracy.

    Usage:
        GET /api/debug/search?q=your+query
        GET /api/debug/search?q=your+query&k=10

    stage1_candidates: wide hybrid (vector+BM25) candidate pool fed to the cross-encoder.
    stage2_top_k_shown_in_ui: final top-k in the order shown in the UI.
    """
    if not q:
        raise HTTPException(status_code=400, detail="?q= is required")

    from embedder import embed_query
    from store import search as vector_search, CANDIDATE_MULTIPLIER, CANDIDATE_MIN
    import math

    qv         = embed_query(q)
    candidates = vector_search(qv, query_text=q, k=k)

    stage1 = [
        {
            "rank":         i + 1,
            "filename":     c.get("filename"),
            "location":     c.get("location"),
            "hybrid_score": round(c.get("hybrid_score", 0), 4),
            "vec_score":    round(c.get("vec_score", c.get("score", 0)), 4),
            "text_preview": (c.get("text") or "")[:120] + ("…" if len(c.get("text", "")) > 120 else ""),
        }
        for i, c in enumerate(candidates)
    ]

    reranker_active = _reranker_mod.RERANK_ENABLED and not _reranker_mod._load_failed
    model = _reranker_mod._get_model()
    stage2 = []
    if model is not None and candidates:
        pairs      = [(q, c["text"]) for c in candidates]
        raw_scores = model.predict(pairs)
        annotated  = [
            {
                "stage1_rank":  i + 1,
                "filename":     candidates[i].get("filename"),
                "location":     candidates[i].get("location"),
                "vec_score":    round(float(candidates[i].get("vec_score", candidates[i].get("score", 0))), 4),
                "hybrid_score": round(float(candidates[i].get("hybrid_score", 0)), 4),
                "rerank_logit": round(float(raw_scores[i]), 4),
                "rerank_prob":  round(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(raw_scores[i]))))), 4),
                "text_preview": (candidates[i].get("text") or "")[:120] + "…",
            }
            for i in range(len(candidates))
        ]
        annotated.sort(key=lambda x: x["rerank_prob"], reverse=True)
        for j, a in enumerate(annotated):
            a["final_rank"] = j + 1
        stage2 = annotated[:k]

    return {
        "query":            q,
        "k_requested":      k,
        "reranker_model":   _reranker_mod.RERANK_MODEL,
        "reranker_active":  reranker_active,
        "pipeline_summary": (
            f"Stage 1 fetched {len(candidates)} hybrid candidates "
            f"(k={k} × CANDIDATE_MULTIPLIER={CANDIDATE_MULTIPLIER}, "
            f"min={CANDIDATE_MIN}). "
            f"Stage 2 cross-encoder scored all {len(candidates)} and "
            f"returned top {len(stage2)} to the UI."
        ),
        "stage1_candidates":        stage1,
        "stage2_top_k_shown_in_ui": stage2,
    }


# ── Observability ─────────────────────────────────────────────────
@app.get("/api/stats")
async def stats():
    return db.stats()


# ── Query (SSE streaming) ─────────────────────────────────────────
class _HistoryTurn(BaseModel):
    role:    str   # "user" | "assistant"
    content: str


class _QueryBody(BaseModel):
    question: str
    file_ids: list[int]
    mode:     str                          # "search" | "llm" | "rag"
    history:  list[_HistoryTurn] = []      # prior turns, for follow-up questions
    k:        int = 5                      # number of chunks to retrieve & display

    @field_validator("k")
    @classmethod
    def _clamp_k(cls, v: int) -> int:
        return max(1, min(v, 50))


@app.post("/api/query")
async def query(body: _QueryBody):
    """
    Server-Sent Events stream.

    Events (each ending with \\n\\n):
      {"type": "sources",  "sources": [...hits...]}
      {"type": "token",    "text": "..."}          (0-N times)
      {"type": "done"}
    """
    use_search = body.mode in ("search", "rag")
    use_llm    = body.mode in ("llm",    "rag")
    loop       = asyncio.get_running_loop()
    t_start    = time.perf_counter()
    history    = [h.model_dump() for h in body.history]

    async def event_stream():
        # ── Phase 1: Retrieval ──────────────────────────────────
        hits: list[dict] = []
        t_retrieval_ms = 0.0
        if use_search:
            if body.file_ids is not None and len(body.file_ids) == 0:
                hits = []
            else:
                t0 = time.perf_counter()
                try:
                    hits = await loop.run_in_executor(
                        None,
                        lambda: retrieve(body.question, body.file_ids, body.k),
                    )
                except IndexModelMismatch as e:
                    yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
                t_retrieval_ms = (time.perf_counter() - t0) * 1000
        yield f"data: {json.dumps({'type': 'sources', 'sources': hits})}\n\n"

        # ── Phase 2: LLM streaming ──────────────────────────────
        if use_llm:
            if use_search and hits:
                context = build_context(hits)
            elif use_search:
                context = "(no relevant context found in the selected files)"
            else:
                context = "(no document context — answering from model knowledge)"

            tok_q: queue.SimpleQueue = queue.SimpleQueue()

            def _run_llm():
                try:
                    for tok in stream_generate(body.question, context, history):
                        tok_q.put(("tok", tok))
                except Exception as exc:
                    tok_q.put(("err", str(exc)))
                finally:
                    tok_q.put(("end", None))

            threading.Thread(target=_run_llm, daemon=True).start()

            while True:
                kind, val = await loop.run_in_executor(None, tok_q.get)
                if kind == "end":
                    break
                etype = "token" if kind == "tok" else "error"
                yield f"data: {json.dumps({'type': etype, 'text': val})}\n\n"

        total_ms = (time.perf_counter() - t_start) * 1000
        db.log_query(body.mode, len(body.question), len(hits), t_retrieval_ms, total_ms)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8502, reload=False)
