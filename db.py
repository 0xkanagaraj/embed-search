import sqlite3
from pathlib import Path

DB_PATH = Path("data/users.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username   TEXT PRIMARY KEY,
                pwd_hash   TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT NOT NULL,
                filename    TEXT NOT NULL,
                path        TEXT NOT NULL,
                n_chunks    INTEGER DEFAULT 0,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(username, filename)
            )
        """)


def create_user(username: str, pwd_hash: str) -> bool:
    try:
        with _conn() as c:
            c.execute("INSERT INTO users (username, pwd_hash) VALUES (?, ?)",
                      (username, pwd_hash))
        return True
    except sqlite3.IntegrityError:
        return False


def get_user(username: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?",
                        (username,)).fetchone()
    return dict(row) if row else None


def add_file(username: str, filename: str, path: str, n_chunks: int = 0) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO files (username, filename, path, n_chunks) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(username, filename) DO UPDATE SET "
            "path=excluded.path, n_chunks=excluded.n_chunks",
            (username, filename, path, n_chunks),
        )
        if cur.lastrowid:
            return cur.lastrowid
        row = c.execute(
            "SELECT id FROM files WHERE username=? AND filename=?",
            (username, filename),
        ).fetchone()
        return row["id"]


def list_files(username: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, filename, path, n_chunks, uploaded_at FROM files "
            "WHERE username = ? ORDER BY uploaded_at DESC",
            (username,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_file(username: str, file_id: int):
    with _conn() as c:
        c.execute("DELETE FROM files WHERE username = ? AND id = ?",
                  (username, file_id))


init_db()