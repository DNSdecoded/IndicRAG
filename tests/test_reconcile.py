"""Tests for the log/index reconciler and the paper_index mirror.

The ingest log is the system of record and the search indexes are derived from
it, but nothing ever verified they still agree — and every divergence class here
is silent from the outside: a ghost citation, a paper that quietly stops being
retrievable, a replay that resurrects something deleted.
"""

import uuid

import check_db
import persistence

TS = "2026-01-01T00:00:00+00:00"


class FakeCollection:
    """Minimal stand-in: only .get(include=[...]) is exercised by the reconciler."""

    name = "test_collection"

    def __init__(self, ids_to_paper: dict):
        self._ids_to_paper = dict(ids_to_paper)

    def get(self, **kwargs):
        ids = list(self._ids_to_paper)
        return {
            "ids": ids,
            "metadatas": [{"paper_id": self._ids_to_paper[i]} for i in ids],
        }


def _record(paper_id, n_chunks, year=None):
    metas = [{"paper_id": paper_id, "title": f"Title {paper_id}", "chunk_index": i}
             for i in range(n_chunks)]
    if year:
        for m in metas:
            m["year"] = year
    persistence.record_ingest(
        event_id=paper_id, paper_id=paper_id, content_hash="h" + paper_id,
        title=f"Title {paper_id}", source_path=f"papers/{paper_id}.pdf",
        chunks=[f"chunk {i}" for i in range(n_chunks)],
        metadatas=metas,
        ids=[f"{paper_id}_{i}" for i in range(n_chunks)],
        embed_model="BAAI/bge-m3", chunker_version=1, created_at=TS,
    )
    return [f"{paper_id}_{i}" for i in range(n_chunks)]


def _reconcile(collection):
    # BM25 is compared only when there is a live index; these tests are about the
    # log/Chroma axis, and _bm25_live_ids returns None with no index built.
    return check_db.reconcile(collection=collection)


def test_agreeing_log_and_collection_are_consistent():
    pid = str(uuid.uuid4())
    ids = _record(pid, 3)
    try:
        report = _reconcile(FakeCollection({i: pid for i in ids}))
        assert report["consistent"], report["divergences"]
        assert report["chunks_in_log"] == 3
        assert report["chunks_in_chroma"] == 3
    finally:
        persistence.delete_ingest_events(pid)


def test_chunks_the_log_claims_but_chroma_lost_are_reported():
    """A rolled-back upsert or a half-applied delete: retrieval silently lost
    content and only a replay restores it."""
    pid = str(uuid.uuid4())
    ids = _record(pid, 3)
    try:
        report = _reconcile(FakeCollection({ids[0]: pid}))
        kinds = {d["kind"] for d in report["divergences"] if d["paper_id"] == pid}
        assert "missing_from_chroma" in kinds
        assert not report["consistent"]
        missing = next(d for d in report["divergences"] if d["kind"] == "missing_from_chroma")
        assert missing["count"] == 2
    finally:
        persistence.delete_ingest_events(pid)


def test_paper_chroma_holds_but_the_log_never_recorded_is_reported():
    """The failed-rollback case: a rebuild would delete these chunks outright."""
    ghost = str(uuid.uuid4())
    report = _reconcile(FakeCollection({f"{ghost}_0": ghost}))
    assert any(d["kind"] == "not_in_log" and d["paper_id"] == ghost
               for d in report["divergences"])


def test_extra_chunks_for_a_logged_paper_are_reported():
    pid = str(uuid.uuid4())
    ids = _record(pid, 1)
    try:
        report = _reconcile(FakeCollection({ids[0]: pid, f"{pid}_stray": pid}))
        assert any(d["kind"] == "extra_in_chroma" and d["paper_id"] == pid
                   for d in report["divergences"])
    finally:
        persistence.delete_ingest_events(pid)


def test_last_result_is_cached_for_quality_endpoint():
    """/quality must never trigger a full scan of its own."""
    pid = str(uuid.uuid4())
    ids = _record(pid, 1)
    try:
        fresh = _reconcile(FakeCollection({i: pid for i in ids}))
        assert check_db.last_result()["checked_at"] == fresh["checked_at"]
    finally:
        persistence.delete_ingest_events(pid)


# ── paper_index mirror (D2) ─────────────────────────────────────────────────

def test_record_ingest_maintains_the_paper_mirror():
    pid = str(uuid.uuid4())
    _record(pid, 4, year="2024")
    try:
        row = next(r for r in persistence.list_papers() if r["paper_id"] == pid)
        assert row["title"] == f"Title {pid}"
        assert row["year"] == "2024"
        assert row["chunk_count"] == 4
    finally:
        persistence.delete_ingest_events(pid)


def test_re_ingest_replaces_the_mirror_row_rather_than_duplicating_it():
    pid = str(uuid.uuid4())
    _record(pid, 4)
    _record(pid, 2)
    try:
        rows = [r for r in persistence.list_papers() if r["paper_id"] == pid]
        assert len(rows) == 1
        assert rows[0]["chunk_count"] == 2
    finally:
        persistence.delete_ingest_events(pid)


def test_deleting_a_paper_drops_its_mirror_row():
    """A stale mirror row would make dedup skip a paper that is gone."""
    pid = str(uuid.uuid4())
    _record(pid, 2)
    persistence.delete_ingest_events(pid)
    assert not [r for r in persistence.list_papers() if r["paper_id"] == pid]


def test_dedup_matches_from_the_mirror_without_touching_the_collection():
    """D2: the dedup check used to pull every chunk's metadata out of ChromaDB
    on every ingest."""
    import vector_store

    pid = str(uuid.uuid4())
    persistence.record_ingest(
        event_id=pid, paper_id=pid, content_hash="h", title="Attention Is All You Need",
        source_path=f"papers/{pid}.pdf", chunks=["c"],
        metadatas=[{"paper_id": pid, "title": "Attention Is All You Need"}],
        ids=[f"{pid}_0"], embed_model="BAAI/bge-m3", chunker_version=1, created_at=TS,
    )

    import config

    class ExplodingCollection:
        # The live collection: the mirror describes exactly this one, so dedup
        # must answer from it without touching ChromaDB.
        name = config.COLLECTION_NAME

        def get(self, **kwargs):
            raise AssertionError("dedup must read the mirror, not the collection")

    try:
        found = vector_store.find_similar_paper(
            "Attention Is All You Need", collection=ExplodingCollection())
        assert found == pid
    finally:
        persistence.delete_ingest_events(pid)


def test_dedup_scans_a_foreign_collection_instead_of_the_mirror():
    """The mirror is derived from the ingest log, which describes the LIVE
    collection only. Answering a staging rebuild from it would report duplicates
    that collection does not hold, and miss the ones it does."""
    import vector_store

    pid = str(uuid.uuid4())
    persistence.record_ingest(
        event_id=pid, paper_id=pid, content_hash="h", title="Attention Is All You Need",
        source_path=f"papers/{pid}.pdf", chunks=["c"],
        metadatas=[{"paper_id": pid, "title": "Attention Is All You Need"}],
        ids=[f"{pid}_0"], embed_model="BAAI/bge-m3", chunker_version=1, created_at=TS,
    )

    scanned = []

    class StagingCollection:
        name = "staging_rebuild"

        def get(self, **kwargs):
            scanned.append(kwargs)
            return {"ids": [], "metadatas": []}

    try:
        found = vector_store.find_similar_paper(
            "Attention Is All You Need", collection=StagingCollection())
        assert scanned, "a non-live collection must be scanned directly"
        assert found is None, "the live corpus must not answer for a staging collection"
    finally:
        persistence.delete_ingest_events(pid)
