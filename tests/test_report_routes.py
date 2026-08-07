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


def test_plan_sections_prompt_names_language_natively():
    import config
    import rag
    import report_runner

    seen = {}

    def _capture(prompt, **kwargs):
        seen["prompt"] = prompt
        return '["x"]'

    with patch.object(rag, "llm_generate", _capture):
        report_runner.plan_sections("some topic", "hi")

    assert config.LANGUAGE_NAMES["hi"] in seen["prompt"]
    assert "'hi'" not in seen["prompt"]
    assert "Do not write them in English." in seen["prompt"]

    with patch.object(rag, "llm_generate", _capture):
        report_runner.plan_sections("some topic", "en")

    # English must not get the self-contradictory "in English, not in English"
    assert "Do not write them in English." not in seen["prompt"]
    assert "MUST be written in English." in seen["prompt"]


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


# ---------------------------------------------------------------------------
# Report persistence (survives restart, unlike the job store's 24h prune)
# ---------------------------------------------------------------------------

def test_save_get_list_report_roundtrip():
    import persistence

    persistence.save_report(
        report_id="r1", watch_id="w1", topic="graphene sensors", language="en",
        markdown="# Review\n\ntext [1]\n", citation_count=3,
        created_at="2026-07-20T00:00:00+00:00",
    )

    got = persistence.get_report("r1")
    assert got["topic"] == "graphene sensors"
    assert got["markdown"] == "# Review\n\ntext [1]\n"
    assert got["citation_count"] == 3

    listed = persistence.list_reports(watch_id="w1")
    assert any(r["id"] == "r1" for r in listed)
    assert "markdown" not in listed[0]  # list is summary-only, not the full body


def test_save_report_upsert_overwrites_same_id():
    import persistence

    persistence.save_report(
        report_id="r2", watch_id="", topic="t", language="en",
        markdown="v1", citation_count=1, created_at="2026-07-20T00:00:00+00:00",
    )
    persistence.save_report(
        report_id="r2", watch_id="", topic="t", language="en",
        markdown="v2", citation_count=2, created_at="2026-07-21T00:00:00+00:00",
    )
    assert persistence.get_report("r2")["markdown"] == "v2"


def test_get_report_missing_returns_none():
    import persistence
    assert persistence.get_report("does-not-exist") is None


# ---------------------------------------------------------------------------
# GET /reports, GET /reports/{report_id}
# ---------------------------------------------------------------------------

def test_list_reports_endpoint(client):
    import persistence
    persistence.save_report(
        report_id="r3", watch_id="", topic="t3", language="en",
        markdown="body", citation_count=1, created_at="2026-07-20T00:00:00+00:00",
    )
    resp = client.get("/reports")
    assert resp.status_code == 200
    assert any(r["id"] == "r3" for r in resp.json())


def test_get_report_endpoint_404(client):
    assert client.get("/reports/does-not-exist").status_code == 404


def test_get_report_endpoint_returns_full_report(client):
    import persistence
    persistence.save_report(
        report_id="r4", watch_id="", topic="t4", language="en",
        markdown="full body [1]", citation_count=1, created_at="2026-07-20T00:00:00+00:00",
    )
    resp = client.get("/reports/r4")
    assert resp.status_code == 200
    assert resp.json()["markdown"] == "full body [1]"
