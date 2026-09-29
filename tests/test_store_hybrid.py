import json

import numpy as np
import pytest

from embedder import DIM


def _fake_vec(seed: int) -> np.ndarray:
    """Deterministic L2-normalised fake embedding — store only needs the right shape."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype("float32")
    return v / np.linalg.norm(v)


def _corrupt_model_name(store):
    meta = json.loads(store._meta_path().read_text())
    meta["model"] = "some-other-model"
    store._meta_path().write_text(json.dumps(meta))
    store._invalidate()


def test_minmax_normalizes_to_unit_range(isolated_store):
    n = isolated_store._minmax(np.array([0.1, 0.5, 0.9, 0.3]))
    assert n.min() == 0.0 and n.max() == 1.0


def test_minmax_handles_degenerate_equal_scores(isolated_store):
    assert (isolated_store._minmax(np.array([0.4, 0.4, 0.4])) == 0).all()


def test_tokenizer_is_unicode_aware(isolated_store):
    toks = isolated_store._tokenize("Café 你好 தமிழ்")
    assert "café" in toks and "你好" in toks


def test_append_and_search_roundtrip(isolated_store):
    chunks = [
        {"file_id": 1, "filename": "a.txt", "text": "apples and oranges", "location": None},
        {"file_id": 1, "filename": "a.txt", "text": "bananas and grapes", "location": None},
    ]
    isolated_store.append(chunks, np.stack([_fake_vec(1), _fake_vec(2)]))

    results = isolated_store.search(_fake_vec(1), query_text="apples", k=2)
    assert len(results) == 2
    assert results[0]["text"] == "apples and oranges"
    assert all("hybrid_score" in r for r in results)


def test_search_filters_by_file_ids(isolated_store):
    chunks = [
        {"file_id": 1, "filename": "a.txt", "text": "content one", "location": None},
        {"file_id": 2, "filename": "b.txt", "text": "content two", "location": None},
    ]
    isolated_store.append(chunks, np.stack([_fake_vec(10), _fake_vec(20)]))

    results = isolated_store.search(_fake_vec(10), query_text="content", file_ids=[1], k=5)
    assert results and all(r["file_id"] == 1 for r in results)
    assert isolated_store.search(_fake_vec(10), query_text="content", file_ids=[], k=5) == []


def test_remove_file_drops_only_its_chunks(isolated_store):
    chunks = [
        {"file_id": 1, "filename": "a.txt", "text": "keep me", "location": None},
        {"file_id": 2, "filename": "b.txt", "text": "remove me", "location": None},
    ]
    isolated_store.append(chunks, np.stack([_fake_vec(30), _fake_vec(40)]))

    isolated_store.remove_file(2)
    remaining, vectors, _ = isolated_store.load(check_model=False)
    assert [c["file_id"] for c in remaining] == [1]
    assert len(vectors) == 1


def test_remove_file_by_filename_clears_stale_chunks(isolated_store):
    chunks = [{"file_id": 99, "filename": "a.txt", "text": "stale", "location": None}]
    isolated_store.append(chunks, np.stack([_fake_vec(5)]))
    isolated_store.remove_file(1, "a.txt")   # new id, same filename
    assert isolated_store.load(check_model=False)[0] == []


def test_remove_file_survives_out_of_sync_index(isolated_store):
    chunks = [{"file_id": 1, "filename": "a.txt", "text": "x", "location": None},
              {"file_id": 2, "filename": "b.txt", "text": "y", "location": None}]
    isolated_store.append(chunks, np.stack([_fake_vec(1), _fake_vec(2)]))
    np.save(isolated_store._vectors_path(), np.stack([_fake_vec(1)]))   # corrupt: 1 vec, 2 chunks
    isolated_store._invalidate()

    isolated_store.remove_file(2)
    chunks_after, vectors_after, _ = isolated_store.load(check_model=False)
    assert len(chunks_after) == len(vectors_after)   # never left inconsistent


def test_model_mismatch_raises_on_search(isolated_store):
    isolated_store.append([{"file_id": 1, "filename": "a.txt", "text": "hi", "location": None}],
                          np.stack([_fake_vec(1)]))
    _corrupt_model_name(isolated_store)
    with pytest.raises(isolated_store.IndexModelMismatch):
        isolated_store.search(_fake_vec(1), query_text="hi", k=1)


def test_model_mismatch_does_not_block_remove_file(isolated_store):
    isolated_store.append([{"file_id": 1, "filename": "a.txt", "text": "hi", "location": None}],
                          np.stack([_fake_vec(1)]))
    _corrupt_model_name(isolated_store)
    isolated_store.remove_file(1)
    assert isolated_store.load(check_model=False)[0] == []


def test_model_mismatch_blocks_append(isolated_store):
    isolated_store.append([{"file_id": 1, "filename": "a.txt", "text": "hi", "location": None}],
                          np.stack([_fake_vec(1)]))
    _corrupt_model_name(isolated_store)
    with pytest.raises(isolated_store.IndexModelMismatch):
        isolated_store.append([{"file_id": 2, "filename": "b.txt", "text": "new", "location": None}],
                              np.stack([_fake_vec(2)]))
