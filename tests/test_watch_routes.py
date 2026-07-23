"""Endpoint tests for /watch CRUD (Phase 6 registration)."""

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture(scope="module")
def client():
    import api_server
    from starlette.testclient import TestClient

    original_lifespan = api_server.app.router.lifespan_context
    api_server.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(api_server.app, raise_server_exceptions=True) as c:
            yield c
    finally:
        api_server.app.router.lifespan_context = original_lifespan


@pytest.fixture(autouse=True)
def _clear_watches():
    import persistence
    with persistence._db_lock:
        persistence._conn.execute("DELETE FROM watches")
        persistence._conn.commit()
    yield


def test_watch_disabled_returns_404(client):
    with patch("config.WATCH_ENABLE", False):
        resp = client.post("/watch", json={"user_id": "u1", "topic": "graphene"})
        assert resp.status_code == 404


def test_create_watch_returns_registered_watch(client):
    with patch("config.WATCH_ENABLE", True):
        resp = client.post("/watch", json={
            "user_id": "u1", "topic": "terahertz antennas",
            "language": "hi", "cadence": "daily",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"]
        assert body["topic"] == "terahertz antennas"
        assert body["language"] == "hi"
        assert body["cadence"] == "daily"
        assert body["seen_count"] == 0
        assert body["has_digest"] is False
        assert body["next_run"] and body["last_run"] is None


def test_create_watch_defaults_cadence_and_language(client):
    with patch("config.WATCH_ENABLE", True), patch("config.WATCH_DEFAULT_CADENCE", "weekly"):
        resp = client.post("/watch", json={"user_id": "u1", "topic": "spintronics"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["language"] == "en"
        assert body["cadence"] == "weekly"


def test_create_watch_rejects_bad_cadence(client):
    with patch("config.WATCH_ENABLE", True):
        resp = client.post("/watch", json={
            "user_id": "u1", "topic": "x", "cadence": "hourly",
        })
        assert resp.status_code == 422


def test_create_watch_rejects_empty_topic(client):
    with patch("config.WATCH_ENABLE", True):
        resp = client.post("/watch", json={"user_id": "u1", "topic": ""})
        assert resp.status_code == 422


def test_list_watches_scoped_to_authenticated_user(client):
    """Each user sees only their own watches; owner comes from the API key,
    not a client-supplied field (multi-user support)."""
    import persistence
    import deps
    import auth_utils
    for name, key in (("alice", "k_alice"), ("bob", "k_bob")):
        s, h = auth_utils.hash_password("pw")
        persistence.save_user(name, s, h, key, "2026-07-23T00:00:00+00:00")
    deps._refresh_key_map()
    try:
        with patch("config.WATCH_ENABLE", True):
            client.post("/watch", json={"topic": "a"}, headers={"X-API-Key": "k_alice"})
            client.post("/watch", json={"topic": "b"}, headers={"X-API-Key": "k_alice"})
            client.post("/watch", json={"topic": "c"}, headers={"X-API-Key": "k_bob"})

            alice = client.get("/watch", headers={"X-API-Key": "k_alice"}).json()
            bob = client.get("/watch", headers={"X-API-Key": "k_bob"}).json()
            assert len(alice) == 2 and all(w["user_id"] == "alice" for w in alice)
            assert len(bob) == 1 and bob[0]["user_id"] == "bob"

            # bob cannot see or delete alice's watch — 404, not 403 (no existence leak)
            a_id = alice[0]["id"]
            assert client.get(f"/watch/{a_id}", headers={"X-API-Key": "k_bob"}).status_code == 404
            assert client.delete(f"/watch/{a_id}", headers={"X-API-Key": "k_bob"}).status_code == 404
            assert client.get(f"/watch/{a_id}", headers={"X-API-Key": "k_alice"}).status_code == 200
    finally:
        for name in ("alice", "bob"):
            persistence.delete_user(name)
        deps._refresh_key_map()


def test_get_watch_roundtrip_and_404(client):
    with patch("config.WATCH_ENABLE", True):
        wid = client.post("/watch", json={"user_id": "u1", "topic": "a"}).json()["id"]
        assert client.get(f"/watch/{wid}").json()["id"] == wid
        assert client.get("/watch/nope").status_code == 404


def test_get_digest_returns_stored_and_404(client):
    import persistence
    with patch("config.WATCH_ENABLE", True):
        wid = client.post("/watch", json={"user_id": "u1", "topic": "a"}).json()["id"]
        # no run yet → digest is null
        assert client.get(f"/watch/{wid}/digest").json()["digest"] is None
        # persist a digest, then it comes back
        w = persistence.get_watch(wid)
        w["latest_digest"] = "digest text [2401.00001]"
        persistence.save_watch(w)
        assert client.get(f"/watch/{wid}/digest").json()["digest"] == "digest text [2401.00001]"
        assert client.get("/watch/nope/digest").status_code == 404


def test_delete_watch_and_404(client):
    with patch("config.WATCH_ENABLE", True):
        wid = client.post("/watch", json={"user_id": "u1", "topic": "a"}).json()["id"]
        assert client.delete(f"/watch/{wid}").json()["status"] == "deleted"
        assert client.get(f"/watch/{wid}").status_code == 404
        assert client.delete(f"/watch/{wid}").status_code == 404
