"""Endpoint tests for /feedback, /prefs/{user_id}, /search/export (Phase 3 C-2/C-3/C-4)."""

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


def test_submit_feedback_records_and_returns_id(client):
    resp = client.post("/feedback", json={"query_id": "q-1", "rating": "up"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "recorded"
    assert body["feedback_id"]


def test_submit_feedback_rejects_invalid_rating(client):
    resp = client.post("/feedback", json={"query_id": "q-1", "rating": "sideways"})
    assert resp.status_code == 422


def test_get_prefs_404_when_disabled(client):
    with patch("config.ENABLE_USER_PREFS", False):
        resp = client.get("/prefs")
    assert resp.status_code == 404


def test_put_and_get_prefs_when_enabled(client):
    # Owner comes from auth; with auth disabled the caller is the default user.
    with patch("config.ENABLE_USER_PREFS", True):
        put_resp = client.put("/prefs", json={"prefs": {"language": "hi"}})
        assert put_resp.status_code == 200
        assert put_resp.json()["prefs"] == {"language": "hi"}

        get_resp = client.get("/prefs")
        assert get_resp.status_code == 200
        assert get_resp.json()["prefs"] == {"language": "hi"}


def test_search_export_rejects_unknown_format(client):
    resp = client.get("/search/export", params={"query": "diabetes", "format": "ris"})
    assert resp.status_code == 400


def test_list_feedback_returns_query_context(client):
    import persistence
    persistence.log_query(
        query_id="q-ctx-1", question="What is BERT?", answer="A transformer model.",
        mode="standard_A", model="default", language="en", confidence=0.8, coverage=0.5,
        created_at="2026-07-20T00:00:00+00:00",
    )
    client.post("/feedback", json={"query_id": "q-ctx-1", "rating": "up"})

    resp = client.get("/feedback")
    assert resp.status_code == 200
    entry = next(e for e in resp.json() if e["query_id"] == "q-ctx-1")
    assert entry["question"] == "What is BERT?"
    assert entry["answer"] == "A transformer model."
    assert entry["rating"] == "up"


def test_feedback_stats_aggregates_by_language(client):
    import persistence
    persistence.log_query(
        query_id="q-stats-en", question="q", answer="a", mode="standard_A",
        model="default", language="en", confidence=0.9, coverage=0.5,
        created_at="2026-07-20T00:00:00+00:00",
    )
    persistence.log_query(
        query_id="q-stats-hi", question="q", answer="a", mode="standard_A",
        model="default", language="hi", confidence=0.9, coverage=0.5,
        created_at="2026-07-20T00:00:00+00:00",
    )
    client.post("/feedback", json={"query_id": "q-stats-en", "rating": "up"})
    client.post("/feedback", json={"query_id": "q-stats-hi", "rating": "down"})

    resp = client.get("/feedback/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["by_language"]["en"]["approval_rate"] == 1.0
    assert body["by_language"]["hi"]["approval_rate"] == 0.0


def test_export_bibtex_from_answer_sources(client):
    resp = client.post("/export/bibtex", json={
        "sources": [
            {"title": "A {Great} Paper", "authors": "John Smith, Jane Doe", "year": "2020"},
            {"title": "Another Paper", "authors": "", "year": ""},
        ]
    })
    assert resp.status_code == 200
    assert "@article{Smith2020," in resp.text
    assert "title = {A Great Paper}" in resp.text
    assert "author = {John Smith, Jane Doe}" in resp.text
    assert "year = {2020}" in resp.text
    # No authors/year → falls back to a ref-style key, no dangling empty fields
    assert "@article{ref2," in resp.text
    assert "author = {}" not in resp.text
    assert "year = {}" not in resp.text


def test_export_bibtex_leading_comma_authors_no_crash(client):
    """Regression: authors=', John Smith' must not IndexError inside _cite_key."""
    resp = client.post("/export/bibtex", json={
        "sources": [{"title": "Solo Title", "authors": ", John Smith", "year": "2021"}]
    })
    assert resp.status_code == 200
    assert "@article{ref1," in resp.text


def test_export_bibtex_no_authors_or_year_at_index_zero(client):
    resp = client.post("/export/bibtex", json={"sources": [{"title": "Only A Title"}]})
    assert resp.status_code == 200
    assert "@article{ref1," in resp.text


def test_export_bibtex_dedupes_colliding_keys(client):
    resp = client.post("/export/bibtex", json={
        "sources": [
            {"title": "Paper One", "authors": "John Smith", "year": "2020"},
            {"title": "Paper Two", "authors": "John Smith", "year": "2020"},
        ]
    })
    assert resp.status_code == 200
    assert "@article{Smith2020," in resp.text
    assert "@article{Smith2020a," in resp.text


def test_export_bibtex_rejects_empty_sources(client):
    resp = client.post("/export/bibtex", json={"sources": []})
    assert resp.status_code == 422


def test_search_export_returns_bibtex(client):
    fake_context = {
        "chunks": ["chunk text"],
        "metadatas": [{"paper_id": "smith2020", "title": "A {Great} Paper", "authors": "J. Smith", "year": "2020"}],
        "distances": [0.1],
        "formatted_context": "",
        "chunks_used": 1,
    }
    with patch("rag.retrieve_context", return_value=fake_context):
        resp = client.get("/search/export", params={"query": "diabetes"})
    assert resp.status_code == 200
    assert "@article{smith2020," in resp.text
    assert "title = {A Great Paper}" in resp.text
    assert "author = {J. Smith}" in resp.text
    assert "year = {2020}" in resp.text
