import sqlite3
from contextlib import contextmanager
from pathlib import Path

_HERE   = Path(__file__).parent
DB_PATH = _HERE / "data" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# sqlite3's own `with conn:` commits but never closes, hence the wrapper.
@contextmanager
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        with c:
            yield c
    finally:
        c.close()


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL UNIQUE,
                path        TEXT NOT NULL,
                n_chunks    INTEGER DEFAULT 0,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                mode         TEXT NOT NULL,
                question_len INTEGER NOT NULL,
                n_hits       INTEGER NOT NULL,
                retrieval_ms REAL NOT NULL,
                total_ms     REAL NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def add_file(filename: str, path: str, n_chunks: int = 0) -> int:
    with _conn() as c:
        c.execute(
            "INSERT INTO files (filename, path, n_chunks) VALUES (?, ?, ?) "
            "ON CONFLICT(filename) DO UPDATE SET path=excluded.path, n_chunks=excluded.n_chunks",
            (filename, path, n_chunks),
        )
        # cursor.lastrowid is unreliable after an upsert-update, so look the id up.
        row = c.execute("SELECT id FROM files WHERE filename=?", (filename,)).fetchone()
        return row["id"]


def list_files() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, filename, path, n_chunks, uploaded_at FROM files ORDER BY uploaded_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_file(file_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM files WHERE id = ?", (file_id,))


def update_file_chunks(file_id: int, n_chunks: int) -> None:
    with _conn() as c:
        c.execute("UPDATE files SET n_chunks = ? WHERE id = ?", (n_chunks, file_id))


def log_query(mode: str, question_len: int, n_hits: int,
              retrieval_ms: float, total_ms: float) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO query_log (mode, question_len, n_hits, retrieval_ms, total_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (mode, question_len, n_hits, retrieval_ms, total_ms),
        )


def stats() -> dict:
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n_queries, "
            "       AVG(total_ms) AS avg_total_ms, "
            "       AVG(retrieval_ms) AS avg_retrieval_ms, "
            "       AVG(n_hits) AS avg_hits "
            "FROM query_log"
        ).fetchone()
        by_mode = c.execute(
            "SELECT mode, COUNT(*) AS n FROM query_log GROUP BY mode"
        ).fetchall()
        n_files = c.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(n_chunks),0) AS chunks FROM files"
        ).fetchone()
    return {
        "n_queries":        row["n_queries"] or 0,
        "avg_total_ms":     round(row["avg_total_ms"] or 0, 1),
        "avg_retrieval_ms": round(row["avg_retrieval_ms"] or 0, 1),
        "avg_hits":         round(row["avg_hits"] or 0, 2),
        "by_mode":          {r["mode"]: r["n"] for r in by_mode},
        "n_files":          n_files["n"],
        "n_chunks":         n_files["chunks"],
    }


init_db()
