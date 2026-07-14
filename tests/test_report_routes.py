"""Endpoint tests for /report (Phase 7 literature-review workflow)."""

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

    original = api_server.app.router.lifespan_context
    api_server.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(api_server.app, raise_server_exceptions=True) as c:
            yield c
    finally:
        api_server.app.router.lifespan_context = original


_FAKE = {
    "topic": "graphene sensors", "language": "en",
    "sections": ["Background", "Findings"],
    "markdown": "# Literature Review: graphene sensors\n\n## Background\n\ntext [1]\n",
    "citation_count": 3,
}


def test_report_disabled_returns_404(client):
    with patch("config.REPORT_ENABLE", False):
        assert client.post("/report", json={"topic": "x"}).status_code == 404


def test_report_end_to_end(client):
    # BackgroundTasks run synchronously in TestClient after the response, so by the
    # time we poll status the job has completed.
    with patch("config.REPORT_ENABLE", True), \
         patch("report_runner.run_report", return_value=_FAKE):
        r = client.post("/report", json={"topic": "graphene sensors"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        st = client.get(f"/report/status/{job_id}")
        assert st.status_code == 200
        body = st.json()
        assert body["status"] == "success"
        assert body["sections"] == ["Background", "Findings"]
        assert body["citation_count"] == 3
        assert "markdown" not in body  # markdown is download-only, not in status

        dl = client.get(f"/report/{job_id}/download")
        assert dl.status_code == 200
        assert dl.headers["content-type"].startswith("text/markdown")
        assert "attachment" in dl.headers["content-disposition"]
        assert "# Literature Review: graphene sensors" in dl.text


def test_report_rejects_empty_topic(client):
    with patch("config.REPORT_ENABLE", True):
        assert client.post("/report", json={"topic": ""}).status_code == 422


def test_report_status_404(client):
    with patch("config.REPORT_ENABLE", True):
        assert client.get("/report/status/nope").status_code == 404
        assert client.get("/report/nope/download").status_code == 404


def test_download_before_ready_409(client):
    with patch("config.REPORT_ENABLE", True), \
         patch("report_runner.run_report", side_effect=RuntimeError("boom")):
        job_id = client.post("/report", json={"topic": "x"}).json()["job_id"]
        assert client.get(f"/report/status/{job_id}").json()["status"] == "failed"
        assert client.get(f"/report/{job_id}/download").status_code == 409
