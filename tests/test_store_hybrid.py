import json

import numpy as np
import pytest

st = pytest.importorskip("sentence_transformers", reason="requires sentence-transformers (model download)")

import store
from embedder import DIM, MODEL_NAME


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store._CACHE.clear()
    yield store
    store._CACHE.clear()


def _fake_vec(seed: int) -> np.ndarray:
    """A deterministic, L2-normalised fake embedding (store doesn't care
    that it wasn't produced by the real model — it just needs vectors of
    the right shape to exercise the ranking math)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype("float32")
    return v / np.linalg.norm(v)


def test_minmax_normalizes_to_unit_range():
    scores = np.array([0.1, 0.5, 0.9, 0.3])
    n = store._minmax(scores)
    assert n.min() == 0.0
    assert n.max() == 1.0


def test_minmax_handles_degenerate_equal_scores():
    scores = np.array([0.4, 0.4, 0.4])
    n = store._minmax(scores)
    assert (n == 0).all()


def test_append_and_search_roundtrip(isolated_store):
    chunks = [
        {"file_id": 1, "filename": "a.txt", "text": "apples and oranges", "location": None},
        {"file_id": 1, "filename": "a.txt", "text": "bananas and grapes", "location": None},
    ]
    vectors = np.stack([_fake_vec(1), _fake_vec(2)])
    isolated_store.append("u1", chunks, vectors)

    results = isolated_store.search("u1", _fake_vec(1), query_text="apples", k=2)
    assert len(results) == 2
    assert {r["text"] for r in results} == {"apples and oranges", "bananas and grapes"}
    assert all("hybrid_score" in r for r in results)


def test_search_filters_by_file_ids(isolated_store):
    chunks = [
        {"file_id": 1, "filename": "a.txt", "text": "content one", "location": None},
        {"file_id": 2, "filename": "b.txt", "text": "content two", "location": None},
    ]
    vectors = np.stack([_fake_vec(10), _fake_vec(20)])
    isolated_store.append("u2", chunks, vectors)

    results = isolated_store.search("u2", _fake_vec(10), query_text="content", file_ids=[1], k=5)
    assert all(r["file_id"] == 1 for r in results)


def test_remove_file_drops_only_its_chunks(isolated_store):
    chunks = [
        {"file_id": 1, "filename": "a.txt", "text": "keep me", "location": None},
        {"file_id": 2, "filename": "b.txt", "text": "remove me", "location": None},
    ]
    vectors = np.stack([_fake_vec(30), _fake_vec(40)])
    isolated_store.append("u3", chunks, vectors)

    isolated_store.remove_file("u3", 2)
    remaining, _, _ = isolated_store.load("u3", check_model=False)
    assert len(remaining) == 1
    assert remaining[0]["file_id"] == 1


def test_model_mismatch_raises_on_search(isolated_store):
    chunks = [{"file_id": 1, "filename": "a.txt", "text": "hi", "location": None}]
    vectors = np.stack([_fake_vec(1)])
    isolated_store.append("u4", chunks, vectors)

    # Simulate having previously indexed with a different embedding model.
    meta_path = isolated_store._meta_path("u4")
    meta = json.loads(meta_path.read_text())
    meta["model"] = "some-other-model"
    meta_path.write_text(json.dumps(meta))
    isolated_store._CACHE.clear()

    with pytest.raises(isolated_store.IndexModelMismatch):
        isolated_store.search("u4", _fake_vec(1), query_text="hi", k=1)


def test_model_mismatch_does_not_block_remove_file(isolated_store):
    chunks = [{"file_id": 1, "filename": "a.txt", "text": "hi", "location": None}]
    vectors = np.stack([_fake_vec(1)])
    isolated_store.append("u5", chunks, vectors)

    meta_path = isolated_store._meta_path("u5")
    meta = json.loads(meta_path.read_text())
    meta["model"] = "some-other-model"
    meta_path.write_text(json.dumps(meta))
    isolated_store._CACHE.clear()

    # Deleting a file shouldn't require vector-space compatibility.
    isolated_store.remove_file("u5", 1)
    remaining, _, _ = isolated_store.load("u5", check_model=False)
    assert remaining == []


def test_model_mismatch_blocks_append(isolated_store):
    chunks = [{"file_id": 1, "filename": "a.txt", "text": "hi", "location": None}]
    vectors = np.stack([_fake_vec(1)])
    isolated_store.append("u6", chunks, vectors)

    meta_path = isolated_store._meta_path("u6")
    meta = json.loads(meta_path.read_text())
    meta["model"] = "some-other-model"
    meta_path.write_text(json.dumps(meta))
    isolated_store._CACHE.clear()

    new_chunks = [{"file_id": 2, "filename": "b.txt", "text": "new", "location": None}]
    new_vectors = np.stack([_fake_vec(2)])
    with pytest.raises(isolated_store.IndexModelMismatch):
        isolated_store.append("u6", new_chunks, new_vectors)
