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

Security notes:
  - Sessions are persisted in SQLite (db.py) rather than an in-process
    dict, so they survive a restart.
  - Auth endpoints are rate-limited per client IP (ratelimit.py).
  - State-changing requests (upload/delete/query) require a CSRF token
    issued at login, sent back via the X-CSRF-Token header — see ui.html.
  - Uploads are size-capped and content-sniffed against their extension
    (validation.py) before anything is written to disk.
"""

import asyncio
import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import db
import ratelimit
import validation
from auth import login as auth_login
from auth import signup as auth_signup
from db import add_file, delete_file, list_files, get_user, create_user
from ingest import SUPPORTED_EXTENSIONS, IngestLimitExceeded
from llm import stream_generate
from pipeline import build_context, index_file, remove_file, retrieve
from store import IndexModelMismatch

FILES_ROOT = Path("data/users")
_UI        = Path(__file__).parent / "ui.html"

# ── Accepted MIME / extension list for upload validation ──────────
_ACCEPT_EXTS = SUPPORTED_EXTENSIONS   # e.g. {'.pdf', '.docx', ...}

# ── Rate limit tunables ─────────────────────────────────────────────
LOGIN_MAX_ATTEMPTS  = 8
LOGIN_WINDOW_SECS   = 60
SIGNUP_MAX_ATTEMPTS = 5
SIGNUP_WINDOW_SECS  = 300


# ── Warmup ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load and warm the embedding model before accepting requests."""
    print("⏳  Loading embedding model …")
    from embedder import embed_query
    embed_query("warmup")            # JIT-compile + download weights if needed
    db.purge_expired_sessions()
    print("✅  Model ready — server is live at http://localhost:8502")
    yield
    # nothing to clean up on shutdown


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="SemanticSearch",
    docs_url=None, redoc_url=None,
    lifespan=lifespan,
)


def _client_ip(request: Request) -> str:
    # Trust X-Forwarded-For's first hop only if you're behind a known
    # reverse proxy that sets it; otherwise this is spoofable. Fine for
    # single-instance deployments behind e.g. nginx/Caddy.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_user(request: Request) -> str | None:
    token = request.cookies.get("session")
    if not token:
        return None
    sess = db.get_session(token)
    return sess["username"] if sess else None


def _require_user(request: Request) -> str:
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _require_csrf(request: Request) -> None:
    """Double-submit CSRF check for state-changing endpoints. The token is
    issued (non-httponly) at login and must be echoed back in a header —
    a cross-site request can't read the cookie to do that, but our own
    same-origin JS can."""
    token = request.cookies.get("session")
    sess = db.get_session(token) if token else None
    if not sess:
        raise HTTPException(status_code=401, detail="Not authenticated")
    header_csrf = request.headers.get("x-csrf-token")
    if not header_csrf or header_csrf != sess["csrf_token"]:
        raise HTTPException(status_code=403, detail="Missing or invalid CSRF token")


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
async def login(body: _AuthBody, request: Request, response: Response):
    key = f"login:{_client_ip(request)}:{body.username.strip().lower()}"
    try:
        ratelimit.check(key, LOGIN_MAX_ATTEMPTS, LOGIN_WINDOW_SECS)
    except ratelimit.RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Try again in {e.retry_after:.0f}s.",
        )

    ok, msg = auth_login(body.username, body.password)
    if not ok:
        raise HTTPException(status_code=401, detail=msg)

    ratelimit.reset(key)
    token, csrf_token = db.create_session(msg)
    response.set_cookie("session", token, httponly=True,
                        samesite="lax", max_age=db.SESSION_TTL_SECONDS)
    # Deliberately NOT httponly: our own JS needs to read it to echo it
    # back as the CSRF header. It's not the auth credential — "session"
    # (httponly) is — so exposing it to JS doesn't weaken the session cookie.
    response.set_cookie("csrf", csrf_token, httponly=False,
                        samesite="lax", max_age=db.SESSION_TTL_SECONDS)
    return {"user": msg}

@app.post("/api/auth/signup")
async def signup(body: _AuthBody, request: Request):
    key = f"signup:{_client_ip(request)}"
    try:
        ratelimit.check(key, SIGNUP_MAX_ATTEMPTS, SIGNUP_WINDOW_SECS)
    except ratelimit.RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail=f"Too many signup attempts. Try again in {e.retry_after:.0f}s.",
        )

    ok, msg = auth_signup(body.username, body.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    if token:
        db.delete_session(token)
    response.delete_cookie("session")
    response.delete_cookie("csrf")
    return {"ok": True}


@app.post("/api/auth/guest")
async def guest_login(response: Response):
    """Auto-create and log in a shared 'guest' account.
    Called silently by the UI on load so no login screen is needed."""
    _GUEST_USER = "guest"
    _GUEST_PASS = "guestpass"
    # Create the guest account if it doesn't exist yet
    if not get_user(_GUEST_USER):
        from auth import hash_password
        create_user(_GUEST_USER, hash_password(_GUEST_PASS))
    token, csrf_token = db.create_session(_GUEST_USER)
    response.set_cookie("session", token, httponly=True,
                        samesite="lax", max_age=db.SESSION_TTL_SECONDS)
    response.set_cookie("csrf", csrf_token, httponly=False,
                        samesite="lax", max_age=db.SESSION_TTL_SECONDS)
    return {"user": _GUEST_USER}


# ── Files ─────────────────────────────────────────────────────────
@app.get("/api/files")
async def get_files(request: Request):
    return list_files(_require_user(request))

@app.post("/api/files")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    user = _require_user(request)
    _require_csrf(request)
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

        content = await uf.read()
        try:
            validation.validate_upload(uf.filename, content)
        except validation.ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        dest = root / uf.filename
        dest.write_bytes(content)
        file_id = add_file(user, uf.filename, str(dest), 0)
        remove_file(user, file_id)                   # clear stale vectors

        try:
            n = index_file(user, file_id, uf.filename, str(dest))
        except IngestLimitExceeded as e:
            delete_file(user, file_id)
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(e))
        except IndexModelMismatch as e:
            delete_file(user, file_id)
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail=f"{e} Delete this account's existing index "
                       f"(data/index/{user}/) or re-upload all files to rebuild it.",
            )

        add_file(user, uf.filename, str(dest), n)
        results.append({"filename": uf.filename, "n_chunks": n, "file_id": file_id})
    return results

@app.delete("/api/files/{file_id}")
async def delete_file_route(file_id: int, request: Request):
    user  = _require_user(request)
    _require_csrf(request)
    files = list_files(user)
    f     = next((x for x in files if x["id"] == file_id), None)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    delete_file(user, file_id)
    Path(f["path"]).unlink(missing_ok=True)
    remove_file(user, file_id)
    return {"ok": True}

@app.get("/api/files/{file_id}/chunks")
async def preview_file_chunks(file_id: int, request: Request):
    """Preview the indexed chunks for one file — lets a user sanity-check
    that extraction/chunking actually captured what they expect."""
    user  = _require_user(request)
    files = list_files(user)
    if not any(f["id"] == file_id for f in files):
        raise HTTPException(status_code=404, detail="File not found")

    import store
    chunks, _, _ = store.load(user, check_model=False)
    file_chunks = [c for c in chunks if c["file_id"] == file_id]
    return {
        "file_id": file_id,
        "n_chunks": len(file_chunks),
        "chunks": [
            {"text": c["text"], "location": c.get("location")}
            for c in file_chunks
        ],
    }


# ── Observability ─────────────────────────────────────────────────
@app.get("/api/stats")
async def stats(request: Request):
    user = _require_user(request)
    return db.user_stats(user)


# ── Query (SSE streaming) ─────────────────────────────────────────
class _HistoryTurn(BaseModel):
    role:    str   # "user" | "assistant"
    content: str

class _QueryBody(BaseModel):
    question: str
    file_ids: list[int]
    mode:     str                          # "search" | "llm" | "rag"
    history:  list[_HistoryTurn] = []      # prior turns, for follow-up questions


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
    _require_csrf(request)
    use_search = body.mode in ("search", "rag")
    use_llm    = body.mode in ("llm",    "rag")
    loop       = asyncio.get_event_loop()
    t_start    = time.perf_counter()
    history    = [h.model_dump() for h in body.history]

    async def event_stream():
        # ── Phase 1: Retrieval ──────────────────────────────────
        hits: list[dict] = []
        t_retrieval_ms = 0.0
        if use_search:
            t0 = time.perf_counter()
            try:
                # retrieve() is sync (numpy) — run in thread pool
                hits = await loop.run_in_executor(
                    None,
                    lambda: retrieve(user, body.question,
                                     body.file_ids or None, 5),
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

            # Bridge: sync Gemini generator → async SSE yield via a queue
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
        db.log_query(user, body.mode, len(body.question), len(hits),
                     t_retrieval_ms, total_ms)
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
