import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.db")
    db.init_db()
    return db


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    import store
    monkeypatch.setattr(store, "_INDEX_DIR", tmp_path / "index")
    store._invalidate()
    yield store
    store._invalidate()
