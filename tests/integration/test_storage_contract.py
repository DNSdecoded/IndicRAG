"""Storage contract suite (v3 STO-1): the behaviour every storage backend must match.

Parametrized over backends; today only `local` (vector_store + persistence, as
they are). STO-2 adds the Postgres backend to BACKENDS and must pass unchanged.
Real ChromaDB in a temp dir; the embedder is replaced by deterministic vectors at
its external boundary, so everything on our side of it is exercised.
"""

import types
import uuid
import zlib
from datetime import datetime, timezone

import numpy as np
import pytest

import bm25_search
import cache
import config
import persistence
import vector_store
from storage.base import DocumentStore, StateStore

pytestmark = pytest.mark.integration

BACKENDS = ["local"]

PAPERS = {
    "p_antenna": ("Antenna Design With Learning", ["patch antenna gain optimisation", "pixel antenna surrogate model",
                                                    "antenna bandwidth results"]),
    "p_qubit": ("Qubit Error Correction", ["surface code qubit lattice", "syndrome extraction circuits"]),
}


def _vec(texts, dim=32):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, zlib.crc32(tok.encode()) % dim] += 1.0
        out[i] /= np.linalg.norm(out[i]) or 1.0
    return out


@pytest.fixture(params=BACKENDS)
def store(request, tmp_path, monkeypatch):
    """(documents, state, collection) loaded with PAPERS."""
    if request.param != "local":
        pytest.skip(f"backend {request.param} not available")
    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "BM25_CACHE_DIR", tmp_path / "bm25")
    monkeypatch.setattr(vector_store, "_chroma_client", None)
    bm25_search.invalidate()
    cache.retrieval_cache.invalidate()
    name = f"contract_{uuid.uuid4().hex[:8]}"
    col = vector_store.get_or_create_collection(name, reset=True)
    texts, metas, ids = [], [], []
    for pid, (title, chunks) in PAPERS.items():
        for i, text in enumerate(chunks):
            texts.append(text)
            metas.append({"paper_id": pid, "title": title, "section": "body", "chunk_index": i})
            ids.append(f"{pid}_{i}")
    vector_store.add_documents(texts, _vec(texts), metas, ids, collection=col)
    yield vector_store, persistence, col
    bm25_search.invalidate()
    vector_store._chroma_client = None


# ── conformance ─────────────────────────────────────────────────────────────

def test_local_modules_satisfy_the_protocols():
    assert isinstance(vector_store, DocumentStore)
    assert isinstance(persistence, StateStore)


def test_protocol_check_can_fail():
    incomplete = types.SimpleNamespace(**{k: getattr(vector_store, k) for k in dir(vector_store)
                                          if not k.startswith("_") and k != "get_chunks"})
    assert not isinstance(incomplete, DocumentStore)


# ── DocumentStore behaviour ─────────────────────────────────────────────────

def test_count_after_add_and_upsert_is_idempotent(store):
    docs, _, col = store
    assert docs.count(col) == 5
    docs.add_documents(["patch antenna gain optimisation"], _vec(["patch antenna gain optimisation"]),
                       [{"paper_id": "p_antenna", "title": "Antenna Design With Learning",
                         "section": "body", "chunk_index": 0}], ["p_antenna_0"], collection=col)
    assert docs.count(col) == 5


def test_get_chunks_by_ids_where_limit_and_include(store):
    docs, _, col = store
    got = docs.get_chunks(col, ids=["p_qubit_1", "p_antenna_2"])
    assert set(got["ids"]) == {"p_qubit_1", "p_antenna_2"}
    assert {m["paper_id"] for m in got["metadatas"]} == {"p_qubit", "p_antenna"} and len(got["documents"]) == 2
    assert sorted(docs.get_chunks(col, where={"paper_id": "p_qubit"})["ids"]) == ["p_qubit_0", "p_qubit_1"]
    assert len(docs.get_chunks(col, where={"paper_id": "p_antenna"}, limit=1)["ids"]) == 1
    ids_only = docs.get_chunks(col, include=[])
    assert len(ids_only["ids"]) == 5 and not ids_only.get("documents")


def test_direct_and_bounded_reads_agree(store):
    docs, _, col = store
    assert docs.count(col, timeout=None) == docs.count(col)
    a = docs.get_chunks(col, where={"paper_id": "p_qubit"}, timeout=None)
    b = docs.get_chunks(col, where={"paper_id": "p_qubit"})
    assert sorted(a["ids"]) == sorted(b["ids"])


def test_search_returns_nearest_first(store):
    docs, _, col = store
    res = docs.search(_vec(["syndrome extraction circuits"])[0], top_k=3, collection=col)
    assert set(res) >= {"ids", "documents", "metadatas", "distances"}
    assert res["ids"][0] == "p_qubit_1"
    assert res["distances"] == sorted(res["distances"]) and len(res["ids"]) == 3


def test_search_filter(store):
    docs, _, col = store
    res = docs.search(_vec(["antenna"])[0], top_k=5, filter_dict={"paper_id": "p_qubit"}, collection=col)
    assert {m["paper_id"] for m in res["metadatas"]} == {"p_qubit"}


def test_paper_counts_and_metadata_update(store):
    docs, _, col = store
    assert docs.get_paper_chunk_counts(col) == {"p_antenna": 3, "p_qubit": 2}
    assert docs.update_paper_metadata("p_qubit", {"year": "2026"}, collection=col) == 2
    assert {m["year"] for m in docs.get_chunks(col, where={"paper_id": "p_qubit"})["metadatas"]} == {"2026"}


def test_delete_stale_keeps_listed_ids(store):
    docs, _, col = store
    assert docs.delete_stale_chunks("p_antenna", ["p_antenna_0"], collection=col) == 2
    assert docs.get_chunks(col, where={"paper_id": "p_antenna"}, include=[])["ids"] == ["p_antenna_0"]


def test_delete_by_paper_removes_only_that_paper(store):
    docs, _, col = store
    assert docs.delete_by_paper_id("p_antenna", collection=col) == 3
    assert docs.count(col) == 2 and docs.get_paper_chunk_counts(col) == {"p_qubit": 2}


def test_find_similar_paper(store):
    docs, _, col = store
    assert docs.find_similar_paper("Antenna design with learning", collection=col) == "p_antenna"
    assert docs.find_similar_paper("Protein folding with transformers", collection=col) is None


# ── StateStore behaviour ────────────────────────────────────────────────────

def test_ingest_log_round_trip(store):
    _, state, _ = store
    pid = f"p_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc).isoformat()
    # ingest.py passes event_id=paper_id: the log keeps one row per paper, keyed on it.
    state.record_ingest(pid, pid, "hash1", "T", "/x.pdf", ["c0", "c1"], [{"i": 0}, {"i": 1}],
                        [f"{pid}_0", f"{pid}_1"], "BAAI/bge-m3", 1, now, embed_backend="onnx-int8")
    events = state.get_ingest_events(pid)
    assert len(events) == 1 and events[0]["paper_id"] == pid
    state.record_ingest(pid, pid, "hash2", "T", "/x.pdf", ["c0"], [{"i": 0}], [f"{pid}_0"],
                        "BAAI/bge-m3", 1, now)
    events = state.get_ingest_events(pid)
    assert len(events) == 1 and events[0]["content_hash"] == "hash2"   # re-ingest replaces, never appends
    assert state.delete_ingest_events(pid) == 1 and state.get_ingest_events(pid) == []


def test_session_round_trip(store):
    _, state, _ = store
    sid = f"s_{uuid.uuid4().hex[:6]}"
    now = datetime.now(timezone.utc).isoformat()
    state.save_session(sid, {"created_at": now, "updated_at": now, "messages": [{"role": "user", "content": "hi"}]})
    assert state.load_sessions()[sid]["messages"][0]["content"] == "hi"
    state.delete_session(sid)
    assert sid not in state.load_sessions()
