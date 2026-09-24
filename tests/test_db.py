def test_session_create_get_delete(isolated_db):
    db = isolated_db
    db.create_user("alice", "hashed-pw")

    token, csrf = db.create_session("alice")
    sess = db.get_session(token)
    assert sess is not None
    assert sess["username"] == "alice"
    assert sess["csrf_token"] == csrf

    db.delete_session(token)
    assert db.get_session(token) is None


def test_session_expiry_and_purge(isolated_db):
    db = isolated_db
    db.create_user("bob", "hashed-pw")
    token, _ = db.create_session("bob")

    # Force it into the past.
    with db._conn() as c:
        c.execute("UPDATE sessions SET expires_at = 0 WHERE token = ?", (token,))

    # An expired session is treated as absent even before an explicit purge.
    assert db.get_session(token) is None

    db.purge_expired_sessions()
    with db._conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM sessions WHERE token = ?", (token,)
        ).fetchone()
    assert row["n"] == 0


def test_unknown_session_token_returns_none(isolated_db):
    db = isolated_db
    assert db.get_session("does-not-exist") is None


def test_query_log_and_user_stats(isolated_db):
    db = isolated_db
    db.create_user("carol", "hashed-pw")

    db.log_query("carol", "rag", 20, 5, 120.5, 800.2)
    db.log_query("carol", "search", 10, 3, 90.0, 95.0)
    db.log_query("carol", "rag", 15, 4, 100.0, 700.0)

    stats = db.user_stats("carol")
    assert stats["n_queries"] == 3
    assert stats["by_mode"] == {"rag": 2, "search": 1}
    assert stats["avg_hits"] == round((5 + 3 + 4) / 3, 2)


def test_user_stats_empty_user(isolated_db):
    db = isolated_db
    stats = db.user_stats("nobody")
    assert stats["n_queries"] == 0
    assert stats["by_mode"] == {}
    assert stats["n_files"] == 0
