import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """
    Run a test against a throwaway SQLite DB + data/ dir instead of the
    real one. db.py resolves its path relative to the CWD at import time,
    so we chdir *before* importing/reloading it.
    """
    monkeypatch.chdir(tmp_path)
    import db
    importlib.reload(db)
    yield db
