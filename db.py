import secrets
import sqlite3
import time
from pathlib import Path

# Anchor to the directory containing this file so the DB is found regardless
# of what working directory the server process was launched from.
_HERE    = Path(__file__).parent
DB_PATH  = _HERE / "data" / "users.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SESSION_TTL_SECONDS = 86_400 * 7   # 7 days, matches the old cookie max_age


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
        # Sessions persisted in SQLite instead of an in-process dict, so a
        # server restart (or a second worker sharing the same data/ volume)
        # doesn't silently log everyone out / desync auth state.
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                username   TEXT NOT NULL,
                csrf_token TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
        """)
        # Lightweight observability: one row per /api/query call.
        c.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT NOT NULL,
                mode         TEXT NOT NULL,
                question_len INTEGER NOT NULL,
                n_hits       INTEGER NOT NULL,
                retrieval_ms REAL NOT NULL,
                total_ms     REAL NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def update_file_chunks(file_id: int, n_chunks: int) -> None:
    with _conn() as c:
        c.execute("UPDATE files SET n_chunks = ? WHERE id = ?", (n_chunks, file_id))



# ── Sessions ──────────────────────────────────────────────────────
def create_session(username: str) -> tuple[str, str]:
    """Create a session. Returns (session_token, csrf_token)."""
    token      = secrets.token_hex(32)
    csrf_token = secrets.token_hex(16)
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (token, username, csrf_token, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (token, username, csrf_token, now, now + SESSION_TTL_SECONDS),
        )
    return token, csrf_token


def get_session(token: str) -> dict | None:
    """Return {'username', 'csrf_token'} for a valid, unexpired session."""
    with _conn() as c:
        row = c.execute(
            "SELECT username, csrf_token, expires_at FROM sessions WHERE token = ?",
            (token,),
        ).fetchone()
    if not row or row["expires_at"] < time.time():
        return None
    return {"username": row["username"], "csrf_token": row["csrf_token"]}


def delete_session(token: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


def purge_expired_sessions() -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))


# ── Query log / observability ────────────────────────────────────
def log_query(username: str, mode: str, question_len: int, n_hits: int,
              retrieval_ms: float, total_ms: float) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO query_log "
            "(username, mode, question_len, n_hits, retrieval_ms, total_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username, mode, question_len, n_hits, retrieval_ms, total_ms),
        )


def user_stats(username: str) -> dict:
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n_queries, "
            "       AVG(total_ms) AS avg_total_ms, "
            "       AVG(retrieval_ms) AS avg_retrieval_ms, "
            "       AVG(n_hits) AS avg_hits "
            "FROM query_log WHERE username = ?",
            (username,),
        ).fetchone()
        by_mode = c.execute(
            "SELECT mode, COUNT(*) AS n FROM query_log "
            "WHERE username = ? GROUP BY mode",
            (username,),
        ).fetchall()
        n_files = c.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(n_chunks),0) AS chunks "
            "FROM files WHERE username = ?",
            (username,),
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