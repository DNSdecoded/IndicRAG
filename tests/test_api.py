"""
FastAPI endpoint tests — no real server, no model loading.

Strategy: swap the app's lifespan with a no-op before opening the TestClient so
that BGE-M3 / ChromaDB are never initialised during tests.
"""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import patch, MagicMock


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
# GET /papers
# ---------------------------------------------------------------------------

def test_list_papers_includes_paper_id(client, tmp_path, monkeypatch):
    """paper_id (the value /compare, /ingest/reindex, etc. actually need) must
    be in the response — filename alone doesn't tell the UI what to send."""
    (tmp_path / "smith_2020_transformer.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("config.PAPERS_DIR", tmp_path)

    resp = client.get("/papers")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["paper_id"] == "smith_2020_transformer"
    assert body[0]["filename"] == "smith_2020_transformer.pdf"


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------

def test_compare_rejects_single_paper(client):
    resp = client.post("/compare", json={"paper_ids": ["p1"], "dimensions": ["methodology"]})
    assert resp.status_code == 422


def test_compare_runs_job_and_status_returns_matrix(client):
    fake_matrix = {"dimensions": ["methodology"], "matrix": {"p1": {"methodology": "x"}, "p2": {"methodology": "y"}}}
    with patch("rag.compare_papers", return_value=fake_matrix) as mock_compare:
        resp = client.post("/compare", json={"paper_ids": ["p1", "p2"], "dimensions": ["methodology"]})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        status_resp = client.get(f"/compare/status/{job_id}")

    mock_compare.assert_called_once_with(["p1", "p2"], ["methodology"], None)
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["status"] == "success"
    assert body["matrix"] == fake_matrix["matrix"]


def test_compare_status_404_for_unknown_job(client):
    resp = client.get("/compare/status/does-not-exist")
    assert resp.status_code == 404


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
# GET /ingest/health
# ---------------------------------------------------------------------------

def test_ingest_health_reports_indexed_and_failed_papers(client, tmp_path, monkeypatch):
    """A PDF with 0 indexed chunks (extraction failed, corrupted figure, etc.)
    must show up as failed rather than silently vanishing from the count."""
    (tmp_path / "good_paper.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "failed_paper.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("config.PAPERS_DIR", tmp_path)

    with patch("vector_store.get_paper_chunk_counts", return_value={"good_paper": 5}):
        resp = client.get("/ingest/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["paper_count"] == 2
    assert body["chunk_count"] == 5
    assert body["failed_count"] == 1
    by_id = {p["paper_id"]: p for p in body["papers"]}
    assert by_id["good_paper"]["status"] == "indexed"
    assert by_id["good_paper"]["chunks"] == 5
    assert by_id["failed_paper"]["status"] == "failed"
    assert by_id["failed_paper"]["chunks"] == 0


def test_ingest_health_empty_papers_dir(client, tmp_path, monkeypatch):
    monkeypatch.setattr("config.PAPERS_DIR", tmp_path)

    with patch("vector_store.get_paper_chunk_counts", return_value={}):
        resp = client.get("/ingest/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"paper_count": 0, "chunk_count": 0, "failed_count": 0, "papers": []}


# ---------------------------------------------------------------------------
# POST /ingest/from-url
# ---------------------------------------------------------------------------

def test_ingest_from_url_rejects_unresolvable_input(client):
    resp = client.post("/ingest/from-url", json={})
    assert resp.status_code == 400


def test_ingest_from_url_accepts_direct_url(client):
    resp = client.post("/ingest/from-url", json={"url": "http://example.com/paper.pdf"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    assert body["job_id"]


# ---------------------------------------------------------------------------
# DELETE /papers/{paper_id}
# ---------------------------------------------------------------------------

def test_delete_paper_not_found(client):
    """DELETE a nonexistent paper returns 404."""
    with patch("vector_store.delete_by_paper_id", return_value=0):
        resp = client.delete("/papers/nonexistent-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /quality
# ---------------------------------------------------------------------------

def test_quality_returns_report_when_present(client, tmp_path, monkeypatch):
    report = {"overall": 0.94, "num_queries": 3}
    report_path = tmp_path / "eval_report.json"
    report_path.write_text(__import__("json").dumps(report), encoding="utf-8")
    monkeypatch.setattr("routes.management._EVAL_REPORT_PATH", report_path)

    resp = client.get("/quality")

    assert resp.status_code == 200
    assert resp.json() == report


def test_quality_returns_error_when_absent(client, tmp_path, monkeypatch):
    monkeypatch.setattr("routes.management._EVAL_REPORT_PATH", tmp_path / "does_not_exist.json")

    resp = client.get("/quality")

    assert resp.status_code == 200
    assert resp.json() == {"error": "No eval report available"}


# ---------------------------------------------------------------------------
# GET /corpus/map
# ---------------------------------------------------------------------------

def test_kmeans_separates_two_clear_clusters():
    import numpy as np
    from routes.management import _kmeans

    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],
    ], dtype=np.float32)
    labels = _kmeans(X, k=2)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_kmeans_k_greater_than_n_does_not_crash():
    import numpy as np
    from routes.management import _kmeans

    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    labels = _kmeans(X, k=10)
    assert len(labels) == 3
    assert len(set(labels.tolist())) <= 3


def test_corpus_map_returns_clusters_and_timeline(client):
    metadatas = [
        {"title": "Paper A", "year": "2020"}, {"title": "Paper A", "year": "2020"},
        {"title": "Paper B", "year": "2021"},
        {"title": "Paper C", "year": "2021"}, {"title": "Paper C", "year": "2021"},
    ]
    embeddings = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0]]
    fake_collection = MagicMock()
    with patch("vector_store.get_or_create_collection", return_value=fake_collection), \
         patch("vector_store._chroma_call", return_value={"embeddings": embeddings, "metadatas": metadatas}):
        resp = client.get("/corpus/map")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["clusters"]) >= 1
    total_chunks = sum(c["chunk_count"] for c in body["clusters"])
    assert total_chunks == 5
    # 5 chunks but only 3 distinct papers -> at least one cluster must dedupe
    # a repeated title down to a smaller paper_count than its raw chunk_count.
    assert any(c["chunk_count"] > c["paper_count"] for c in body["clusters"])
    years = {t["year"]: t["count"] for t in body["timeline"]}
    assert years == {"2020": 2, "2021": 3}


def test_corpus_map_empty_collection(client):
    with patch("vector_store.get_or_create_collection", return_value=MagicMock()), \
         patch("vector_store._chroma_call", return_value={"embeddings": [], "metadatas": []}):
        resp = client.get("/corpus/map")

    assert resp.status_code == 200
    assert resp.json() == {"clusters": [], "timeline": []}


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


# ---------------------------------------------------------------------------
# GET /graph  (Task 3.3 — knowledge/citation graph)
# ---------------------------------------------------------------------------

def test_graph_returns_all_edges(client):
    fake = [{"source": "A", "target": "B", "type": "co_citation", "score": 2.0}]
    with patch("persistence.get_all_edges", return_value=fake):
        resp = client.get("/graph")
    assert resp.status_code == 200
    assert resp.json()["edges"] == fake


def test_graph_filters_by_paper_id(client):
    fake = [{"source": "A", "target": "B", "type": "co_citation", "score": 2.0, "metadata": {}}]
    with patch("persistence.get_paper_edges", return_value=fake) as m:
        resp = client.get("/graph", params={"paper_id": "A"})
    assert resp.status_code == 200
    m.assert_called_once_with("A")
    assert resp.json()["edges"] == fake
