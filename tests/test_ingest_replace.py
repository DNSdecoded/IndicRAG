"""Re-ingesting a paper must never pass through a state where it is gone.

The old order was delete-then-add: delete_by_paper_id removed the paper from
Chroma, BM25 AND the ingest log, then the upsert ran. A failed upsert left the
paper absent everywhere, with no log row to replay it from. The new order writes
the new chunks first and only then drops the old ids the new version no longer has.
"""
import concurrent.futures

import numpy as np
import pytest

import bm25_search
import config
import embeddings
import ingest
import persistence
import vector_store


class MemCollection:
    """In-memory stand-in for a ChromaDB collection: get / upsert / delete / count."""
    name = "test_replace"

    def __init__(self, rows=None):
        self.rows = dict(rows or {})  # id -> (text, metadata)
        self.fail_upsert = False

    def _match(self, where):
        return [i for i, (_, m) in self.rows.items()
                if all(m.get(k) == v for k, v in (where or {}).items())]

    def get(self, ids=None, where=None, limit=None, include=None, **_):
        found = [i for i in ids if i in self.rows] if ids is not None else self._match(where)
        found = found[:limit] if limit is not None else found
        return {"ids": found, "metadatas": [self.rows[i][1] for i in found],
                "documents": [self.rows[i][0] for i in found]}

    def upsert(self, documents, embeddings, metadatas, ids):
        if self.fail_upsert:
            raise RuntimeError("chroma write timed out")
        for i, d, m in zip(ids, documents, metadatas):
            self.rows[i] = (d, m)

    def delete(self, ids=None, where=None):
        for i in (ids if ids is not None else self._match(where)):
            self.rows.pop(i, None)

    def count(self):
        return len(self.rows)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(vector_store, "_chroma_call", lambda fn, *a, timeout=None, **kw: fn(*a, **kw))
    monkeypatch.setattr(embeddings, "embed_passages", lambda texts: np.zeros((len(texts), 4)))
    log = {}
    monkeypatch.setattr(persistence, "record_ingest", lambda **kw: log.__setitem__(kw["paper_id"], kw["ids"]))
    monkeypatch.setattr(persistence, "delete_ingest_events", lambda pid: log.pop(pid, None))
    monkeypatch.setattr(config, "DEDUP_PAPERS", False)
    monkeypatch.setattr(config, "BM25_PERSIST", False)
    bm25_search.invalidate()
    yield log
    bm25_search.invalidate()


def _old_paper():
    meta = {"paper_id": "p", "title": "T", "file_hash": "old"}
    return MemCollection({f"p_introduction_{n}": (f"old text {n}", meta) for n in range(3)})


def _sections(word):
    return [("introduction", " ".join([word] * 60) + ".")]


def test_failed_write_on_reingest_keeps_the_old_version(env):
    coll = _old_paper()
    env["p"] = sorted(coll.rows)
    coll.fail_upsert = True

    with pytest.raises(RuntimeError):
        ingest.ingest_paper("p", "T", _sections("new"), {"file_hash": "new"}, coll)

    assert sorted(coll.rows) == ["p_introduction_0", "p_introduction_1", "p_introduction_2"]
    assert "p" in env  # the ingest log still describes the paper


def test_reingest_drops_old_chunks_the_new_version_no_longer_has(env):
    coll = _old_paper()
    n = ingest.ingest_paper("p", "T", _sections("new"), {"file_hash": "new"}, coll)

    assert n == 1
    assert sorted(coll.rows) == ["p_introduction_0"]
    assert coll.rows["p_introduction_0"][0].startswith("new")
    assert env["p"] == ["p_introduction_0"]


def test_bulk_reingest_drops_stale_chunks_after_writing(env, monkeypatch, tmp_path):
    coll = _old_paper()
    (tmp_path / "p.pdf").write_bytes(b"%PDF-1.4 changed")
    monkeypatch.setattr(ingest, "_extract_worker", lambda path, metadata=None: (
        path, "p", {"title": "T", "text": "x", "sections": _sections("new")}, {"file_hash": "new"}))

    class _InlineExecutor:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def submit(self, fn, *a):
            f = concurrent.futures.Future()
            f.set_result(fn(*a))
            return f

    monkeypatch.setattr(ingest.concurrent.futures, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(vector_store, "get_collection_stats", lambda c: {"name": c.name, "count": c.count()})
    stats = ingest.ingest_directory(str(tmp_path), collection=coll)

    assert stats["successful"] == 1
    assert sorted(coll.rows) == ["p_introduction_0"]
    assert coll.rows["p_introduction_0"][0].startswith("new")


def test_force_reingests_an_unchanged_file(env):
    coll = _old_paper()

    assert ingest.ingest_paper("p", "T", _sections("same"), {"file_hash": "old"}, coll) == 0
    assert ingest.ingest_paper("p", "T", _sections("same"), {"file_hash": "old"}, coll, force=True) == 1
    assert sorted(coll.rows) == ["p_introduction_0"]


def test_add_documents_folds_new_chunks_into_a_live_bm25_index(env):
    coll = MemCollection({"a": ("antenna gain measured", {"paper_id": "x"})})
    assert bm25_search.get_or_build_index(coll) is not None

    vector_store.add_documents(["quantum error correction"], np.zeros((1, 4)),
                               [{"paper_id": "y"}], ["b"], coll)

    assert "b" in bm25_search.get_or_build_index(coll).doc_ids
