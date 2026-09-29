"""End-to-end API tests with the heavy models faked out (no downloads, no network)."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from embedder import DIM


def _vec(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=DIM).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture
def client(isolated_db, isolated_store, tmp_path, monkeypatch):
    import pipeline
    import reranker
    import server

    monkeypatch.setattr(server, "FILES_ROOT", tmp_path / "files")
    monkeypatch.setattr(pipeline, "embed_passages", lambda texts: np.stack([_vec(t) for t in texts]))
    monkeypatch.setattr(pipeline, "embed_query", _vec)
    monkeypatch.setattr(reranker, "RERANK_ENABLED", False)
    monkeypatch.setattr(server, "stream_generate", lambda q, ctx, hist: iter(["Hello ", "world"]))
    return TestClient(server.app)   # not used as a context manager → lifespan (model loading) is skipped


def _events(resp) -> list[dict]:
    return [json.loads(b[6:]) for b in resp.text.split("\n\n") if b.startswith("data: ")]


def _upload(client, name="notes.txt", body=b"The quick brown fox jumps over the lazy dog. " * 20):
    return client.post("/api/files", files=[("files", (name, body))])


def test_upload_list_delete(client):
    r = _upload(client)
    assert r.status_code == 200
    fid = r.json()[0]["file_id"]
    assert r.json()[0]["n_chunks"] > 0

    assert [f["id"] for f in client.get("/api/files").json()] == [fid]
    assert client.get(f"/api/files/{fid}/chunks").json()["n_chunks"] > 0

    assert client.delete(f"/api/files/{fid}").status_code == 200
    assert client.get("/api/files").json() == []
    assert client.delete(f"/api/files/{fid}").status_code == 404


def test_reupload_replaces_instead_of_duplicating(client):
    _upload(client)
    fid = _upload(client).json()[0]["file_id"]
    assert len(client.get("/api/files").json()) == 1
    n = client.get(f"/api/files/{fid}/chunks").json()["n_chunks"]
    assert n == client.get("/api/files").json()[0]["n_chunks"]


def test_upload_rejects_bad_files_and_leaves_no_orphans(client, tmp_path):
    assert _upload(client, "x.exe", b"MZ").status_code == 400
    assert _upload(client, "x.doc", b"junk").status_code == 400          # legacy format unsupported
    assert _upload(client, "empty.txt", b"").status_code == 400
    # a corrupt docx passes the extension check but fails extraction
    r = _upload(client, "broken.docx", b"not really a docx")
    assert r.status_code == 400 and "broken.docx" in r.json()["detail"]
    # a file with no extractable text
    assert _upload(client, "blank.txt", b"   \n  ").status_code == 400

    assert client.get("/api/files").json() == []
    assert not list((tmp_path / "files").glob("*"))


def test_upload_filename_cannot_escape_files_dir(client, tmp_path):
    r = _upload(client, "../../evil.txt")
    assert r.status_code == 200
    assert r.json()[0]["filename"] == "evil.txt"
    assert (tmp_path / "files" / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_query_rag_streams_sources_tokens_done(client):
    fid = _upload(client).json()[0]["file_id"]
    r = client.post("/api/query", json={"question": "fox?", "file_ids": [fid], "mode": "rag", "k": 3})
    ev = _events(r)
    assert [e["type"] for e in ev][0] == "sources"
    assert ev[0]["sources"]
    assert "".join(e["text"] for e in ev if e["type"] == "token") == "Hello world"
    assert ev[-1]["type"] == "done"
    assert client.get("/api/stats").json()["n_queries"] == 1


def test_query_search_only_has_no_tokens(client):
    fid = _upload(client).json()[0]["file_id"]
    ev = _events(client.post("/api/query", json={"question": "fox", "file_ids": [fid], "mode": "search"}))
    assert not any(e["type"] == "token" for e in ev)
    assert ev[-1]["type"] == "done"


def test_query_rag_with_no_files_skips_llm(client, monkeypatch):
    import server
    def boom(*a, **k):
        raise AssertionError("LLM should not be called with nothing to ground on")
    monkeypatch.setattr(server, "stream_generate", boom)
    ev = _events(client.post("/api/query", json={"question": "hi", "file_ids": [], "mode": "rag"}))
    assert ev[0] == {"type": "sources", "sources": []}
    assert any("couldn't find" in e.get("text", "") for e in ev)
    assert ev[-1]["type"] == "done"


def test_query_validation(client):
    assert client.post("/api/query", json={"question": "  ", "file_ids": [], "mode": "rag"}).status_code == 422
    assert client.post("/api/query", json={"question": "x", "file_ids": [], "mode": "bogus"}).status_code == 422


def test_llm_stall_times_out_instead_of_hanging(client, monkeypatch):
    import threading
    import server
    release = threading.Event()
    def stalled(q, ctx, hist):
        release.wait(5)
        return iter([])
    monkeypatch.setattr(server, "stream_generate", stalled)
    monkeypatch.setattr(server, "LLM_IDLE_TIMEOUT_S", 0.3)
    try:
        ev = _events(client.post("/api/query", json={"question": "x", "file_ids": [], "mode": "llm"}))
    finally:
        release.set()
    assert any(e["type"] == "error" for e in ev)
    assert ev[-1]["type"] == "done"


def test_retrieval_error_is_reported_not_crashed(client, monkeypatch):
    import server
    def boom(*a, **k):
        raise RuntimeError("index exploded")
    monkeypatch.setattr(server, "retrieve", boom)
    ev = _events(client.post("/api/query", json={"question": "x", "file_ids": [1], "mode": "search"}))
    assert ev[0]["type"] == "error" and "exploded" in ev[0]["text"]
    assert ev[-1]["type"] == "done"
