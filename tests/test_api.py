"""
FastAPI endpoint tests — no real server, no model loading.

Strategy: swap the app's lifespan with a no-op before opening the TestClient so
that BGE-M3 / ChromaDB are never initialised during tests.
"""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch


@asynccontextmanager
async def _noop_lifespan(app):
    """Replaces the real lifespan so model-loading is skipped."""
    yield


@pytest.fixture(scope="module")
def client():
    import api_server
    from starlette.testclient import TestClient

    # ponytail: bypass model loading; restore after module tests
    original_lifespan = api_server.app.router.lifespan_context
    api_server.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(api_server.app, raise_server_exceptions=True) as c:
            yield c
    finally:
        api_server.app.router.lifespan_context = original_lifespan


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_healthy(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"


# ---------------------------------------------------------------------------
# DELETE /papers/{paper_id}
# ---------------------------------------------------------------------------

def test_delete_paper_not_found(client):
    """DELETE a nonexistent paper returns 404."""
    with patch("vector_store.delete_by_paper_id", return_value=0):
        resp = client.delete("/papers/nonexistent-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /papers/{paper_id}
# ---------------------------------------------------------------------------

def test_patch_paper_invalid_field(client):
    """
    PATCH with only unknown fields: Pydantic v2 ignores extras (no extra='forbid'),
    so updates dict is empty → endpoint raises 400 "no valid fields to update".
    """
    resp = client.patch("/papers/someid", json={"invalid_field": "x"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Rate limiting wired up
# ---------------------------------------------------------------------------

def test_rate_limit_wired_up():
    """The slowapi limiter is attached to app.state so endpoints can use it."""
    import api_server
    assert hasattr(api_server.app.state, "limiter")
    assert api_server.app.state.limiter is not None
