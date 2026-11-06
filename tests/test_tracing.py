"""Research traces (v3 OBS-1): spans, served model, redaction, persistence, completeness,
and the cache miss->hit contract for each cache."""

import contextvars
import threading
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

import config
import metrics
import persistence
import tracing


@pytest.fixture(autouse=True)
def _tracing_on(monkeypatch):
    monkeypatch.setattr(config, "TRACE_ENABLED", True)
    monkeypatch.setattr(config, "TRACE_STORE_TEXT", False)
    # Each test starts with no trace bound, whatever an earlier test left in this context.
    token = tracing._current.set(None)
    yield
    tracing._current.reset(token)


# ── recording ───────────────────────────────────────────────────────────────

def test_stages_and_served_model_are_recorded():
    t = tracing.start("query", "t-1", "q", "owner-a")
    with metrics.stage("retrieval_dense"):
        pass
    tracing.note_llm("gemini", "g-main", attempt=1)
    tracing.note_llm("nvidia", "n-model", {"prompt": 3, "completion": 2}, attempt=3)
    rec = tracing.build_record(t, "q", "a")
    assert [s["stage"] for s in rec["spans"]] == ["retrieval_dense"]
    assert rec["served_by"]["model"] == "n-model" and rec["served_by"]["attempt"] == 3


def test_nothing_is_recorded_without_a_trace():
    with metrics.stage("rerank_cross_encoder"):
        pass
    tracing.note(retrieval={"chunks": []})
    assert tracing.current() is None


def test_disabled_tracing_starts_nothing(monkeypatch):
    monkeypatch.setattr(config, "TRACE_ENABLED", False)
    assert tracing.start("query") is None
    tracing.finish(None, "a")   # a no-op, not an error


def test_question_and_answer_text_are_redacted_by_default(monkeypatch):
    t = tracing.start("query", "t-2", "my private question", None)
    rec = tracing.build_record(t, t.question, "the answer")
    assert "my private question" not in str(rec) and "the answer" not in str(rec)
    assert rec["question_len"] == len("my private question") and len(rec["question_sha"]) == 16
    monkeypatch.setattr(config, "TRACE_STORE_TEXT", True)
    assert tracing.build_record(t, t.question, "the answer")["question"] == "my private question"


def test_repro_stamp_tracks_models_prompts_and_knobs(monkeypatch):
    a = tracing.repro_stamp()
    assert set(a) == {"models", "prompt_version", "config_hash", "chunker_version"} and a["models"]
    monkeypatch.setattr(config, "RERANK_POOL_MULT", 99)
    assert tracing.repro_stamp()["config_hash"] != a["config_hash"]


def test_copied_context_follows_work_into_threads():
    t = tracing.start("agent", "t-3")

    def work():
        with metrics.stage("tool_in_thread"):
            pass
    th = threading.Thread(target=contextvars.copy_context().run, args=(work,))
    th.start()
    th.join()
    plain = threading.Thread(target=work)      # without the copy, the span is lost
    plain.start()
    plain.join()
    assert [s["stage"] for s in t.spans] == ["tool_in_thread"]


# ── persistence ─────────────────────────────────────────────────────────────

def test_finish_persists_and_scopes_by_owner():
    t = tracing.start("query", "t-persist", "q", "owner-a")
    with metrics.stage("llm_generate"):
        pass
    tracing.finish(t, "answer", mode="standard_A")
    got = persistence.get_trace("t-persist", owner="owner-a")
    assert got["kind"] == "query" and got["mode"] == "standard_A" and got["spans"][0]["stage"] == "llm_generate"
    assert persistence.get_trace("t-persist", owner="owner-b") is None


def test_a_failed_write_never_reaches_the_caller(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(persistence, "save_trace", boom)
    tracing.finish(tracing.start("query", "t-boom"), "answer")   # must not raise


def test_retention_prunes_old_traces():
    persistence.save_trace("t-old", "query", None, "2000-01-01T00:00:00+00:00", {})
    assert persistence.prune_traces("2001-01-01T00:00:00+00:00") >= 1
    assert persistence.get_trace("t-old") is None


def test_migration_is_recorded():
    names = {r[0] for r in persistence._conn.execute("SELECT name FROM schema_migrations")}
    assert "0004_traces" in names


# ── completeness: every answer route saves exactly one trace ────────────────

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
        with TestClient(api_server.app) as c:
            yield c
    finally:
        api_server.app.router.lifespan_context = original


def _pipeline(answer="Traced answer [1]."):
    """A stand-in pipeline that records what the real one would: a stage and an LLM call."""
    def run(*a, **k):
        with metrics.stage("retrieval_dense"):
            pass
        tracing.note_llm("gemini", "g-main", {"prompt": 10, "completion": 5})
        return {"answer": answer, "language": "en", "language_name": "English", "chunks_used": 1,
                "citations": [], "processing_time": 0.1, "timestamp": "2026-11-06T00:00:00Z"}
    return run


def test_query_route_saves_its_trace(client):
    with patch("rag.answer_question", side_effect=_pipeline()):
        resp = client.post("/query", json={"question": "What is IndicRAG?"})
    assert resp.status_code == 200
    qid = resp.json()["query_id"]
    got = persistence.get_trace(qid)
    assert got is not None and got["kind"] == "query"
    assert got["served_by"]["model"] == "g-main" and any(s["stage"] == "retrieval_dense" for s in got["spans"])
    assert "What is IndicRAG?" not in str(got)


def test_chat_route_saves_its_trace(client):
    with patch("rag.answer_with_history", side_effect=_pipeline()):
        resp = client.post("/chat", json={"message": "What is IndicRAG?"})
    assert resp.status_code == 200
    got = persistence.get_trace(resp.json()["query_id"])
    assert got is not None and got["kind"] == "chat" and got["served_by"]["model"] == "g-main"


def test_stream_finishes_its_trace_with_the_final_answer():
    """sse_stream finishes the request's trace before the done event (both stream routes)."""
    import asyncio
    import sse_utils

    async def run():
        t = tracing.start("query_stream", "t-stream", "q", None)
        # sse_utils calls llm_client.llm_generate_stream directly: patch it there, or this
        # test would make a live LLM call.
        with patch("llm_client.llm_generate_stream", side_effect=lambda *a, **k: iter(["streamed ", "answer"])):
            async for _ in sse_utils.sse_stream("prompt", [], "en", query_id="t-stream"):
                pass
        return t
    asyncio.run(run())
    got = persistence.get_trace("t-stream")
    assert got is not None and got["answer_len"] == len("streamed answer")


# ── cache contract: a repeat is a hit, and the hit is counted ───────────────

def test_llm_cache_miss_then_hit():
    import rag
    from cache import llm_cache
    from providers.base import LLMResponse
    llm_cache.invalidate()
    calls = []
    with patch("llm_client.generate_with_failover",
               side_effect=lambda *a, **k: calls.append(1) or LLMResponse(text="cached answer")):
        assert rag.llm_generate("contract prompt") == "cached answer"
        assert rag.llm_generate("contract prompt") == "cached answer"
    assert len(calls) == 1


def test_tool_cache_miss_then_hit():
    import dataclasses
    from agent import tool_registry as reg
    from agent.nodes import tool_executor_node as node
    from cache import tool_cache
    tool_cache.invalidate()
    calls = []
    fake = dataclasses.replace(reg.REGISTRY["web_search"], run=lambda a: calls.append(1) or {"passages": [{"text": "x"}]})
    with patch.dict(reg.REGISTRY, {"web_search": fake}):
        for _ in range(2):
            node._run_tool("web_search", {"query": "contract query"})
    assert len(calls) == 1


def test_cache_hits_and_misses_are_exported():
    from cache import TTLCache
    seen = []
    with patch.object(metrics, "record_cache", side_effect=lambda name, hit: seen.append((name, hit))):
        c = TTLCache(max_size=4, ttl_seconds=60, name="contract")
        c.get("k")
        c.put("k", 1)
        c.get("k")
    assert seen == [("contract", False), ("contract", True)]
