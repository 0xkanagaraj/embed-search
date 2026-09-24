"""
server.py — SemanticSearch FastAPI backend

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
    This gives sub-second perceived latency even for long answers.
"""

import asyncio
import json
import queue
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from auth import login as auth_login
from auth import signup as auth_signup
from db import add_file, delete_file, list_files
from ingest import SUPPORTED_EXTENSIONS
from llm import stream_generate
from pipeline import build_context, index_file, remove_file, retrieve

FILES_ROOT = Path("data/users")
_UI        = Path(__file__).parent / "ui.html"

# ── Accepted MIME / extension list for upload validation ──────────
_ACCEPT_EXTS = SUPPORTED_EXTENSIONS   # e.g. {'.pdf', '.docx', ...}


# ── Warmup ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load and warm the embedding model before accepting requests."""
    print("⏳  Loading embedding model …")
    from embedder import embed
    embed(["warmup query"])          # JIT-compile + download weights if needed
    print("✅  Model ready — server is live at http://localhost:8502")
    yield
    # nothing to clean up on shutdown


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="SemanticSearch",
    docs_url=None, redoc_url=None,
    lifespan=lifespan,
)

# ── Session store (in-memory) ─────────────────────────────────────
_SESSIONS: dict[str, str] = {}   # token → username


def _new_session(username: str) -> str:
    token = secrets.token_hex(32)
    _SESSIONS[token] = username
    return token

def _get_user(request: Request) -> str | None:
    token = request.cookies.get("session")
    return _SESSIONS.get(token) if token else None

def _require_user(request: Request) -> str:
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

def _files_dir(username: str) -> Path:
    d = FILES_ROOT / username / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── UI ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_UI.read_text(encoding="utf-8"))


# ── Auth ──────────────────────────────────────────────────────────
class _AuthBody(BaseModel):
    username: str
    password: str

@app.get("/api/auth/me")
async def me(request: Request):
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": user}

@app.post("/api/auth/login")
async def login(body: _AuthBody, response: Response):
    ok, msg = auth_login(body.username, body.password)
    if not ok:
        raise HTTPException(status_code=401, detail=msg)
    token = _new_session(msg)
    response.set_cookie("session", token, httponly=True,
                        samesite="lax", max_age=86_400 * 7)
    return {"user": msg}

@app.post("/api/auth/signup")
async def signup(body: _AuthBody):
    ok, msg = auth_signup(body.username, body.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    if token:
        _SESSIONS.pop(token, None)
    response.delete_cookie("session")
    return {"ok": True}


# ── Files ─────────────────────────────────────────────────────────
@app.get("/api/files")
async def get_files(request: Request):
    return list_files(_require_user(request))

@app.post("/api/files")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    user = _require_user(request)
    root = _files_dir(user)
    results = []
    for uf in files:
        ext = Path(uf.filename).suffix.lower()
        if ext not in _ACCEPT_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"'{ext}' is not supported. "
                       f"Accepted: {', '.join(sorted(_ACCEPT_EXTS))}",
            )
        dest = root / uf.filename
        dest.write_bytes(await uf.read())
        file_id = add_file(user, uf.filename, str(dest), 0)
        remove_file(user, file_id)                   # clear stale vectors
        n = index_file(user, file_id, uf.filename, str(dest))
        add_file(user, uf.filename, str(dest), n)
        results.append({"filename": uf.filename, "n_chunks": n, "file_id": file_id})
    return results

@app.delete("/api/files/{file_id}")
async def delete_file_route(file_id: int, request: Request):
    user  = _require_user(request)
    files = list_files(user)
    f     = next((x for x in files if x["id"] == file_id), None)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    delete_file(user, file_id)
    Path(f["path"]).unlink(missing_ok=True)
    remove_file(user, file_id)
    return {"ok": True}


# ── Query (SSE streaming) ─────────────────────────────────────────
class _QueryBody(BaseModel):
    question: str
    file_ids: list[int]
    mode:     str   # "search" | "llm" | "rag"


@app.post("/api/query")
async def query(body: _QueryBody, request: Request):
    """
    Server-Sent Events stream.

    Events (each ending with \\n\\n):
      {"type": "sources",  "sources": [...hits...]}
      {"type": "token",    "text": "..."}          (0-N times)
      {"type": "done"}

    The client receives sources within ~400-700 ms and the answer streams
    in token-by-token — giving perceived latency well under 1 s.
    """
    user       = _require_user(request)
    use_search = body.mode in ("search", "rag")
    use_llm    = body.mode in ("llm",    "rag")
    loop       = asyncio.get_event_loop()

    async def event_stream():
        # ── Phase 1: Retrieval ──────────────────────────────────
        hits: list[dict] = []
        if use_search:
            # retrieve() is sync (numpy) — run in thread pool
            hits = await loop.run_in_executor(
                None,
                lambda: retrieve(user, body.question,
                                 body.file_ids or None, 5),
            )
        yield f"data: {json.dumps({'type': 'sources', 'sources': hits})}\n\n"

        # ── Phase 2: LLM streaming ──────────────────────────────
        if use_llm:
            if use_search and hits:
                context = build_context(hits)
            elif use_search:
                context = "(no relevant context found in the selected files)"
            else:
                context = "(no document context — answering from model knowledge)"

            # Bridge: sync Gemini generator → async SSE yield via a queue
            tok_q: queue.SimpleQueue = queue.SimpleQueue()

            def _run_llm():
                try:
                    for tok in stream_generate(body.question, context):
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

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",     # prevent nginx from buffering SSE
        },
    )


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8502, reload=False)
