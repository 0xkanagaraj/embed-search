"""
server.py — SemanticSearch FastAPI backend (single-user, no auth).

Run:
    python server.py
    uvicorn server:app --host 0.0.0.0 --port 8502 --reload

Notes:
  - Embedding + reranker models are pre-loaded at startup (lifespan hook) so
    the first query doesn't pay the cold-start penalty.
  - Heavy work (embedding, OCR, retrieval) never runs on the event loop:
    blocking routes are plain `def` (FastAPI runs them in a thread pool) and
    the query route pushes retrieval into an executor. Otherwise one upload
    would freeze every other request, including live SSE streams.
  - /api/query streams SSE events: sources first, then LLM tokens.
"""

import asyncio
import json
import math
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, field_validator

import db
import reranker as _reranker_mod
import store
from embedder import embed_query
from ingest import SUPPORTED_EXTENSIONS, IngestLimitExceeded
from llm import stream_generate
from pipeline import build_context, index_file, remove_file, retrieve
from store import CANDIDATE_MIN, CANDIDATE_MULTIPLIER, IndexModelMismatch

_HERE      = Path(__file__).parent
FILES_ROOT = _HERE / "data" / "files"
_UI        = _HERE / "ui.html"

# If the LLM produces nothing for this long, give up instead of hanging the stream.
LLM_IDLE_TIMEOUT_S = 60


# ── Warmup ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load and warm the embedding and reranker models before accepting requests."""
    print("⏳  [1/2] Loading embedding model (Bi-Encoder) …")
    embed_query("warmup")
    print("⏳  [2/2] Loading re-ranking model (Cross-Encoder) …")
    _reranker_mod._get_model()
    print("✅  All models loaded — server is live at http://localhost:8502")
    yield


app = FastAPI(title="SemanticSearch", docs_url=None, redoc_url=None, lifespan=lifespan)


# ── UI ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(_UI.read_text(encoding="utf-8"))


# ── Files ─────────────────────────────────────────────────────────
@app.get("/api/files")
def get_files():
    return db.list_files()


def _discard(file_id: int, filename: str, dest: Path) -> None:
    """Undo a failed upload: DB row, saved file and any half-written chunks."""
    db.delete_file(file_id)
    dest.unlink(missing_ok=True)
    try:
        remove_file(file_id, filename)
    except Exception:
        pass


@app.post("/api/files")
def upload_files(files: list[UploadFile] = File(...)):
    FILES_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for uf in files:
        # .name strips any directory part, so "../../x.txt" can't escape FILES_ROOT.
        filename = Path(uf.filename or "").name
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"'{ext or filename}' is not supported. "
                       f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        content = uf.file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"'{filename}' is empty.")

        dest = FILES_ROOT / filename
        dest.write_bytes(content)
        file_id = db.add_file(filename, str(dest), 0)
        remove_file(file_id, filename)   # clear stale chunks if this filename was uploaded before

        try:
            n = index_file(file_id, filename, str(dest))
        except IngestLimitExceeded as e:
            _discard(file_id, filename, dest)
            raise HTTPException(status_code=400, detail=str(e))
        except IndexModelMismatch as e:
            _discard(file_id, filename, dest)
            raise HTTPException(
                status_code=409,
                detail=f"{e} Delete data/index/ and re-upload all files to rebuild.",
            )
        except Exception as e:           # corrupt/unreadable file, missing extractor lib, ...
            _discard(file_id, filename, dest)
            raise HTTPException(status_code=400, detail=f"Could not read '{filename}': {e}")

        if n == 0:
            _discard(file_id, filename, dest)
            raise HTTPException(
                status_code=400,
                detail=f"No text could be extracted from '{filename}'.",
            )

        db.update_file_chunks(file_id, n)
        results.append({"filename": filename, "n_chunks": n, "file_id": file_id})
    return results


@app.delete("/api/files/{file_id}")
def delete_file_route(file_id: int):
    f = next((x for x in db.list_files() if x["id"] == file_id), None)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    db.delete_file(file_id)
    f_path = Path(f["path"])
    if not f_path.is_absolute():
        f_path = _HERE / f_path
    f_path.unlink(missing_ok=True)
    remove_file(file_id, f["filename"])
    return {"ok": True}


@app.get("/api/files/{file_id}/chunks")
def preview_file_chunks(file_id: int):
    """Preview indexed chunks for one file — lets you verify extraction."""
    if not any(f["id"] == file_id for f in db.list_files()):
        raise HTTPException(status_code=404, detail="File not found")

    chunks, _, _ = store.load(check_model=False)
    file_chunks = [c for c in chunks if c["file_id"] == file_id]
    return {
        "file_id":  file_id,
        "n_chunks": len(file_chunks),
        "chunks":   [{"text": c["text"], "location": c.get("location")} for c in file_chunks],
    }


# ── Debug ─────────────────────────────────────────────────────────
@app.get("/api/debug/search")
def debug_search(q: str, k: int = 5):
    """
    Diagnostic endpoint — runs retrieval + rerank and returns the score
    breakdown so you can sanity-check ranking quality.

        GET /api/debug/search?q=your+query&k=10
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="?q= is required")
    k = max(1, min(k, 50))

    try:
        candidates = store.search(embed_query(q), query_text=q, k=k)
    except IndexModelMismatch as e:
        raise HTTPException(status_code=409, detail=str(e))

    def preview(c: dict) -> str:
        t = c.get("text") or ""
        return t[:120] + ("…" if len(t) > 120 else "")

    stage1 = [
        {
            "rank":         i + 1,
            "filename":     c.get("filename"),
            "location":     c.get("location"),
            "hybrid_score": round(c["hybrid_score"], 4),
            "vec_score":    round(c["vec_score"], 4),
            "text_preview": preview(c),
        }
        for i, c in enumerate(candidates)
    ]

    model = _reranker_mod._get_model() if _reranker_mod.RERANK_ENABLED else None
    stage2 = []
    if model is not None and candidates:
        raw = model.predict([(q, c["text"]) for c in candidates])
        annotated = [
            {
                "stage1_rank":  i + 1,
                "filename":     c.get("filename"),
                "location":     c.get("location"),
                "vec_score":    round(c["vec_score"], 4),
                "hybrid_score": round(c["hybrid_score"], 4),
                "rerank_logit": round(float(raw[i]), 4),
                "rerank_prob":  round(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(raw[i]))))), 4),
                "text_preview": preview(c),
            }
            for i, c in enumerate(candidates)
        ]
        annotated.sort(key=lambda x: x["rerank_prob"], reverse=True)
        for j, a in enumerate(annotated):
            a["final_rank"] = j + 1
        stage2 = annotated[:k]

    return {
        "query":           q,
        "k_requested":     k,
        "reranker_model":  _reranker_mod.RERANK_MODEL,
        "reranker_active": model is not None,
        "pipeline_summary": (
            f"Stage 1 fetched {len(candidates)} hybrid candidates "
            f"(k={k} × CANDIDATE_MULTIPLIER={CANDIDATE_MULTIPLIER}, min={CANDIDATE_MIN}). "
            f"Stage 2 cross-encoder scored them and returned the top {len(stage2)}."
        ),
        "stage1_candidates":        stage1,
        "stage2_top_k_shown_in_ui": stage2,
    }


# ── Stats ─────────────────────────────────────────────────────────
@app.get("/api/stats")
def stats():
    return db.stats()


# ── Query (SSE streaming) ─────────────────────────────────────────
class _HistoryTurn(BaseModel):
    role:    str   # "user" | "assistant"
    content: str


class _QueryBody(BaseModel):
    question: str
    file_ids: list[int]
    mode:     Literal["search", "llm", "rag"]
    history:  list[_HistoryTurn] = []      # prior turns, for follow-up questions
    k:        int = 5                      # number of chunks to retrieve & display

    @field_validator("question")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v

    @field_validator("k")
    @classmethod
    def _clamp_k(cls, v: int) -> int:
        return max(1, min(v, 50))


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/query")
async def query(body: _QueryBody):
    """
    Server-Sent Events stream. Each event is `data: {json}\\n\\n`:
      {"type": "sources", "sources": [...hits...]}
      {"type": "token",   "text": "..."}          (0-N times)
      {"type": "error",   "text": "..."}          (optional)
      {"type": "done"}                            (always last)
    """
    use_search = body.mode in ("search", "rag")
    use_llm    = body.mode in ("llm",    "rag")
    loop       = asyncio.get_running_loop()
    t_start    = time.perf_counter()
    history    = [h.model_dump() for h in body.history]

    async def event_stream():
        stop = threading.Event()     # set when the client goes away → stops the LLM thread
        try:
            # ── Phase 1: Retrieval ──────────────────────────────
            hits: list[dict] = []
            t_retrieval_ms = 0.0
            if use_search and body.file_ids:
                t0 = time.perf_counter()
                try:
                    hits = await loop.run_in_executor(
                        None, lambda: retrieve(body.question, body.file_ids, body.k)
                    )
                except Exception as e:      # incl. IndexModelMismatch
                    yield _sse({"type": "error", "text": str(e)})
                    yield _sse({"type": "done"})
                    return
                t_retrieval_ms = (time.perf_counter() - t0) * 1000
            yield _sse({"type": "sources", "sources": hits})

            # ── Phase 2: LLM streaming ──────────────────────────
            if use_llm and use_search and not hits:
                # Nothing to ground an answer on — don't spend an LLM call.
                yield _sse({"type": "token",
                            "text": "I couldn't find anything relevant in the selected files."})
            elif use_llm:
                context = (build_context(hits) if use_search
                           else "(no document context — answering from model knowledge)")

                aq: asyncio.Queue = asyncio.Queue()

                def _push(item):
                    try:
                        loop.call_soon_threadsafe(aq.put_nowait, item)
                    except RuntimeError:    # event loop already closed
                        pass

                def _run_llm():
                    try:
                        for tok in stream_generate(body.question, context, history):
                            if stop.is_set():
                                break
                            _push(("tok", tok))
                    except Exception as exc:
                        _push(("err", str(exc)))
                    finally:
                        _push(("end", None))

                threading.Thread(target=_run_llm, daemon=True).start()

                while True:
                    try:
                        kind, val = await asyncio.wait_for(aq.get(), LLM_IDLE_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        yield _sse({"type": "error",
                                    "text": f"The model didn't respond within {LLM_IDLE_TIMEOUT_S}s."})
                        break
                    if kind == "end":
                        break
                    yield _sse({"type": "token" if kind == "tok" else "error", "text": val})

            total_ms = (time.perf_counter() - t_start) * 1000
            db.log_query(body.mode, len(body.question), len(hits), t_retrieval_ms, total_ms)
            yield _sse({"type": "done"})
        finally:
            stop.set()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8502, reload=False)
