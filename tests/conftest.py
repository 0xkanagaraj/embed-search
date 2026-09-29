import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point db.py at a throwaway SQLite file (DB_PATH is fixed at import time,
    so chdir alone would NOT protect the real data/app.db)."""
    import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "app.db")
    db.init_db()
    return db


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point store.py at a throwaway index dir and clear its RAM cache."""
    import store
    monkeypatch.setattr(store, "_INDEX_DIR", tmp_path / "index")
    store._invalidate()
    yield store
    store._invalidate()
