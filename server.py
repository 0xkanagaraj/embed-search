"""
server.py — SemanticSearch (FastAPI backend)

Replaces app.py / Streamlit entirely.

Run:
    python server.py
  or
    uvicorn server:app --host 0.0.0.0 --port 8502 --reload
"""

import secrets
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from auth import login as auth_login
from auth import signup as auth_signup
from db import add_file, delete_file, list_files
from llm import generate
from pipeline import build_context, index_file, remove_file, retrieve

# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="SemanticSearch", docs_url=None, redoc_url=None)

FILES_ROOT = Path("data/users")
_UI        = Path(__file__).parent / "ui.html"

# ─────────────────────────────────────────────────────────────────
# Session store (in-memory, sufficient for local/dev use)
# ─────────────────────────────────────────────────────────────────
_SESSIONS: dict[str, str] = {}  # token → username


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


# ─────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_UI.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────
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
    token = _new_session(msg)  # msg is the normalised username on success
    response.set_cookie(
        "session", token,
        httponly=True, samesite="lax", max_age=86_400 * 7,
    )
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


# ─────────────────────────────────────────────────────────────────
# Files
# ─────────────────────────────────────────────────────────────────
@app.get("/api/files")
async def get_files(request: Request):
    user = _require_user(request)
    return list_files(user)


@app.post("/api/files")
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
):
    user = _require_user(request)
    root = _files_dir(user)
    results = []
    for uf in files:
        dest = root / uf.filename
        dest.write_bytes(await uf.read())
        # Bug fix: clear stale vectors before re-indexing the same file
        file_id = add_file(user, uf.filename, str(dest), 0)
        remove_file(user, file_id)
        n = index_file(user, file_id, uf.filename, str(dest))
        add_file(user, uf.filename, str(dest), n)
        results.append({
            "filename": uf.filename,
            "n_chunks": n,
            "file_id":  file_id,
        })
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


# ─────────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────────
class _QueryBody(BaseModel):
    question: str
    file_ids: list[int]
    mode:     str   # "search" | "llm" | "rag"


@app.post("/api/query")
async def query(body: _QueryBody, request: Request):
    user       = _require_user(request)
    use_search = body.mode in ("search", "rag")
    use_llm    = body.mode in ("llm",    "rag")

    hits:   list[dict] = []
    answer: str        = ""

    if use_search:
        hits = retrieve(user, body.question,
                        file_ids=body.file_ids or None, k=5)

    if use_llm:
        if use_search and hits:
            context = build_context(hits)
        elif use_search:
            context = "(no relevant context found in selected files)"
        else:
            context = "(no context — answering from model knowledge)"
        answer = generate(body.question, context)
    elif use_search:
        answer = ""   # sources are enough; no stub text needed

    return {"answer": answer, "sources": hits, "mode": body.mode}


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8502, reload=True)
