"""
FastAPI endpoint tests — no real server, no model loading.

Strategy: swap the app's lifespan with a no-op before opening the TestClient so
that BGE-M3 / ChromaDB are never initialised during tests.
"""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import patch


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
# POST /chat — paper_ids + tags filter combination
# ---------------------------------------------------------------------------

def test_chat_combines_paper_and_tags_filter(client):
    """routes/chat.py must reach rag.answer_with_history with the same
    combine_filters(paper_filter, tags_filter) wiring routes/query.py uses."""
    import rag

    with patch("rag.answer_with_history") as mock_chat:
        mock_chat.return_value = {
            "answer": "Test answer", "language": "en", "language_name": "English",
            "chunks_used": 1, "citations": [],
        }
        resp = client.post("/chat", json={
            "message": "What is IndicRAG?",
            "paper_ids": ["p1"],
            "tags": "transformer, efficiency",
        })

    assert resp.status_code == 200
    _, kwargs = mock_chat.call_args
    assert kwargs["filter_dict"] == {
        "$and": [
            {"paper_id": {"$in": ["p1"]}},
            {rag._TAGS_SENTINEL: ["transformer", "efficiency"]},
        ]
    }


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


def test_query_response_includes_confidence_and_evidence(client):
    claim = {"claim": "x", "support": 0.82, "grounded": True,
              "supporting_chunk": "chunk text", "supporting_chunk_index": 0}
    with patch("rag.answer_question") as mock_query:
        mock_query.return_value = {
            "answer": "Test answer",
            "language": "en",
            "language_name": "English",
            "chunks_used": 1,
            "citations": [],
            "processing_time": 0.1,
            "timestamp": "2026-06-30T00:00:00Z",
            "answer_confidence": 0.82,
            "faithfulness": [claim],
        }
        resp = client.post("/query", json={"question": "What is IndicRAG?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == 0.82
    assert body["evidence"] == [claim]


def test_query_response_defaults_confidence_and_evidence_when_absent(client):
    with patch("rag.answer_question") as mock_query:
        mock_query.return_value = {
            "answer": "Test answer",
            "language": "en",
            "language_name": "English",
            "chunks_used": 1,
            "citations": [],
            "processing_time": 0.1,
            "timestamp": "2026-06-30T00:00:00Z",
        }
        resp = client.post("/query", json={"question": "What is IndicRAG?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == 0.0
    assert body["evidence"] == []


# ---------------------------------------------------------------------------
# Rate limiting wired up
# ---------------------------------------------------------------------------

def test_rate_limit_headers_present(client):
    """Rate-limited endpoints enforce rate limits and return 429 when exceeded."""
    with patch("rag.answer_question") as mock_query:
        # Mock the query handler to return a dummy response
        mock_query.return_value = {
            "answer": "Test answer",
            "language": "en",
            "language_name": "English",
            "chunks_used": 1,
            "citations": [],
            "processing_time": 0.1,
            "timestamp": "2026-06-30T00:00:00Z"
        }

        # Make 31 POST requests to /query (rate limit is 30/minute)
        # The 31st request should be rate limited
        rate_limit_exceeded = False
        for _ in range(31):
            resp = client.post("/query", json={"question": "What is IndicRAG?"})
            if resp.status_code == 429:
                rate_limit_exceeded = True
                break

        # Verify that we did hit the rate limit (at least on the 31st request)
        assert rate_limit_exceeded, "Expected to hit rate limit after 31 requests"
