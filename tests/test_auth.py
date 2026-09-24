import importlib

import pytest

pytest.importorskip("bcrypt", reason="requires bcrypt")


@pytest.fixture
def isolated_auth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import db
    import auth
    importlib.reload(db)
    importlib.reload(auth)
    yield auth


def test_signup_rejects_short_username(isolated_auth):
    ok, msg = isolated_auth.signup("ab", "longenough")
    assert not ok
    assert "3 characters" in msg


def test_signup_rejects_short_password(isolated_auth):
    ok, msg = isolated_auth.signup("someuser", "short")
    assert not ok
    assert "6 characters" in msg


def test_signup_then_login_roundtrip(isolated_auth):
    ok, msg = isolated_auth.signup("dave", "password123")
    assert ok, msg

    ok, msg = isolated_auth.login("dave", "password123")
    assert ok
    assert msg == "dave"


def test_login_rejects_wrong_password(isolated_auth):
    isolated_auth.signup("erin", "correcthorse")
    ok, msg = isolated_auth.login("erin", "wrongpassword")
    assert not ok
    assert msg == "Invalid credentials."


def test_signup_rejects_duplicate_username(isolated_auth):
    isolated_auth.signup("frank", "password123")
    ok, msg = isolated_auth.signup("frank", "password456")
    assert not ok
    assert "already exists" in msg


def test_username_is_case_and_whitespace_normalized(isolated_auth):
    isolated_auth.signup("  Grace  ", "password123")
    ok, msg = isolated_auth.login("grace", "password123")
    assert ok
    assert msg == "grace"
