"""Tests for the two degraded-mode fixes.

1. A ChromaDB that keeps timing out used to cost every request the full 5s
   timeout and park an uncancellable worker in the pool each time — the pool
   drained in seconds and took down requests that never touched Chroma. It now
   trips a circuit breaker and fails fast.
2. An embedding failure (OOM, corrupt weights, driver fault) used to fail the
   whole query, including ones the in-process BM25 index could answer. It now
   degrades to sparse-only and marks the result.
"""

import pytest

import config
import rag
import vector_store


@pytest.fixture(autouse=True)
def _reset_breaker():
    """The breaker is module state; leaking a tripped one poisons later tests."""
    vector_store._circuit_until = 0.0
    vector_store._circuit_failures = 0
    yield
    vector_store._circuit_until = 0.0
    vector_store._circuit_failures = 0


def _hang():
    import time
    time.sleep(5)


# --------------------------------------------------------------------------
# ChromaDB circuit breaker
# --------------------------------------------------------------------------
def test_breaker_starts_closed():
    assert not vector_store.chroma_circuit_open()


def test_healthy_calls_pass_through_untouched():
    assert vector_store._chroma_call(lambda: "ok") == "ok"
    assert not vector_store.chroma_circuit_open()


def test_breaker_opens_only_after_repeated_timeouts():
    """One slow call is not an outage — the breaker must not trip on a blip."""
    for i in range(vector_store._CIRCUIT_TRIP_AFTER - 1):
        with pytest.raises(TimeoutError):
            vector_store._chroma_call(_hang, timeout=0.01)
        assert not vector_store.chroma_circuit_open(), f"tripped early after {i + 1}"

    with pytest.raises(TimeoutError):
        vector_store._chroma_call(_hang, timeout=0.01)
    assert vector_store.chroma_circuit_open()


def test_open_breaker_fails_fast_instead_of_waiting():
    """The whole point: no new call is attempted, so no worker is parked."""
    vector_store._circuit_until = float("inf")
    called = []

    with pytest.raises(vector_store.ChromaUnavailable):
        vector_store._chroma_call(lambda: called.append(1))
    assert called == []


def test_a_success_clears_the_failure_streak():
    """Consecutive means consecutive: an intervening success resets the count."""
    with pytest.raises(TimeoutError):
        vector_store._chroma_call(_hang, timeout=0.01)
    assert vector_store._circuit_failures == 1
    vector_store._chroma_call(lambda: "ok")
    assert vector_store._circuit_failures == 0


# --------------------------------------------------------------------------
# Sparse-only fallback
# --------------------------------------------------------------------------
class _FakeCollection:
    name = "fake"

    def __init__(self, docs):
        self._docs = docs

    def count(self):
        return len(self._docs)

    def get(self, ids=None, include=None, **kwargs):
        # Deliberately returns a DIFFERENT order than requested — Chroma makes no
        # ordering promise, and BM25 rank is the only ranking left in this path.
        items = list(self._docs.items()) if ids is None else [
            (i, self._docs[i]) for i in reversed(ids) if i in self._docs
        ]
        return {
            "ids": [i for i, _ in items],
            "documents": [d for _, d in items],
            "metadatas": [{"paper_id": i, "title": i, "section": "abstract"} for i, _ in items],
        }


def _install_index(monkeypatch, docs):
    import bm25_search
    # Every fake collection is called "fake", so a persisted cache from one test
    # would be loaded by the next one.
    monkeypatch.setattr(config, "BM25_PERSIST", False)
    bm25_search.invalidate()
    monkeypatch.setattr(config, "USE_HYBRID_SEARCH", True)
    return _FakeCollection(docs)


def test_sparse_fallback_returns_results_and_flags_degradation(monkeypatch):
    coll = _install_index(monkeypatch, {
        "p1": "antenna optimization using machine learning",
        "p2": "protein folding with transformers",
    })
    out = rag._sparse_only_retrieval("antenna optimization", top_k=2, collection=coll)
    assert out is not None
    # Never silently pass a degraded answer off as a normal one.
    assert out["degraded"] == "sparse_only"
    assert out["chunks_used"] >= 1


def test_sparse_fallback_preserves_bm25_rank_order(monkeypatch):
    """collection.get() reorders; the fallback must restore BM25 ranking."""
    coll = _install_index(monkeypatch, {
        "p1": "antenna optimization using machine learning",
        "p2": "unrelated text about baking bread",
    })
    out = rag._sparse_only_retrieval("antenna optimization", top_k=2, collection=coll)
    assert out["metadatas"][0]["paper_id"] == "p1"


def test_no_fallback_when_hybrid_search_is_disabled(monkeypatch):
    coll = _install_index(monkeypatch, {"p1": "antenna"})
    monkeypatch.setattr(config, "USE_HYBRID_SEARCH", False)
    assert rag._sparse_only_retrieval("antenna", 2, coll) is None


def test_no_fallback_when_corpus_is_empty(monkeypatch):
    """Nothing to degrade to => None, so the caller re-raises the real error
    instead of serving a confidently empty answer."""
    coll = _install_index(monkeypatch, {})
    assert rag._sparse_only_retrieval("antenna", 2, coll) is None


def test_no_fallback_when_chroma_is_also_down(monkeypatch):
    """BM25 stores ids, not text — if Chroma can't return documents there is no
    degraded answer to give, and pretending otherwise is worse than failing."""
    coll = _install_index(monkeypatch, {"p1": "antenna optimization"})

    def _boom(*a, **k):
        raise TimeoutError("chroma down")

    monkeypatch.setattr(vector_store, "_chroma_call", _boom)
    assert rag._sparse_only_retrieval("antenna optimization", 2, coll) is None
