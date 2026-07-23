"""Multi-user login + identity resolution."""

import asyncio
from contextlib import asynccontextmanager

import pytest

import auth_utils
import deps
import persistence


@pytest.fixture(autouse=True)
def _clean_users():
    with persistence._db_lock:
        persistence._conn.execute("DELETE FROM users")
        persistence._conn.commit()
    deps._refresh_key_map()
    yield
    with persistence._db_lock:
        persistence._conn.execute("DELETE FROM users")
        persistence._conn.commit()
    deps._refresh_key_map()


def _seed(name, pw, key):
    s, h = auth_utils.hash_password(pw)
    persistence.save_user(name, s, h, key, "2026-07-23T00:00:00+00:00")
    deps._refresh_key_map()


# -- password hashing ------------------------------------------------
def test_hash_roundtrip_and_reject():
    s, h = auth_utils.hash_password("hunter2")
    assert auth_utils.verify_password("hunter2", s, h)
    assert not auth_utils.verify_password("nope", s, h)


def test_hash_is_salted():
    s1, h1 = auth_utils.hash_password("same")
    s2, h2 = auth_utils.hash_password("same")
    assert s1 != s2 and h1 != h2  # per-user salt -> different hashes


# -- get_current_user ------------------------------------------------
def test_current_user_default_when_auth_disabled():
    # no users seeded, no env keys -> auth off -> everyone is the default user
    assert asyncio.run(deps.get_current_user(None)) == "default"


def test_current_user_maps_key_to_username():
    _seed("alice", "pw", "k_alice")
    assert asyncio.run(deps.get_current_user("k_alice")) == "alice"


def test_current_user_rejects_unknown_key():
    _seed("alice", "pw", "k_alice")
    with pytest.raises(Exception):  # HTTPException 401
        asyncio.run(deps.get_current_user("wrong"))


# -- POST /login -----------------------------------------------------
@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture(scope="module")
def client():
    import api_server
    from starlette.testclient import TestClient
    original = api_server.app.router.lifespan_context
    api_server.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(api_server.app, raise_server_exceptions=True) as c:
            yield c
    finally:
        api_server.app.router.lifespan_context = original


def test_login_returns_key_on_correct_password(client):
    _seed("alice", "s3cret", "k_alice")
    r = client.post("/login", json={"username": "alice", "password": "s3cret"})
    assert r.status_code == 200
    assert r.json() == {"username": "alice", "api_key": "k_alice"}


def test_login_wrong_password_is_401(client):
    _seed("alice", "s3cret", "k_alice")
    r = client.post("/login", json={"username": "alice", "password": "wrong"})
    assert r.status_code == 401
    assert "k_alice" not in r.text  # never leak the key on failure


def test_login_unknown_user_is_401(client):
    r = client.post("/login", json={"username": "ghost", "password": "x"})
    assert r.status_code == 401
