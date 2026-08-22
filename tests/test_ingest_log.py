"""Tests for the ingest log — the system of record the indexes derive from.

Rebuilding the vector store used to mean re-parsing every PDF and re-calling the
VLM captioner: hours of work, not reproducible, and impossible once a source
file had moved. That made an embedding-model or chunking change expensive enough
that it never happened. The log records the chunks each ingestion produced, so a
rebuild becomes a replay.
"""

import uuid

import persistence
import reindex

TS = "2026-01-01T00:00:00+00:00"


def _record(paper_id, chunks, embed_model="BAAI/bge-m3", chunker_version=1):
    persistence.record_ingest(
        event_id=paper_id, paper_id=paper_id, content_hash="h" + paper_id,
        title=f"Title {paper_id}", source_path=f"papers/{paper_id}.pdf",
        chunks=chunks,
        metadatas=[{"paper_id": paper_id, "title": f"Title {paper_id}",
                    "section": "abstract", "chunk_index": i}
                   for i in range(len(chunks))],
        ids=[f"{paper_id}_{i}" for i in range(len(chunks))],
        embed_model=embed_model, chunker_version=chunker_version, created_at=TS,
    )


def test_recorded_chunks_round_trip_exactly():
    """A replay re-embeds these strings; anything lossy here silently changes
    the corpus on the next rebuild."""
    pid = str(uuid.uuid4())
    chunks = ["first chunk about antennas", "second chunk about qubits"]
    _record(pid, chunks)
    try:
        events = persistence.get_ingest_events(pid)
        assert len(events) == 1
        e = events[0]
        assert e["chunks"] == chunks
        assert e["ids"] == [f"{pid}_0", f"{pid}_1"]
        assert e["metadatas"][1]["chunk_index"] == 1
        assert e["embed_model"] == "BAAI/bge-m3"
        assert e["source_path"] == f"papers/{pid}.pdf"
    finally:
        persistence.delete_ingest_events(pid)


def test_reingesting_replaces_rather_than_appends():
    """The log describes the CURRENT index contents, not history. A second row
    for the same paper would make a replay reinstate superseded chunks."""
    pid = str(uuid.uuid4())
    _record(pid, ["v1 chunk"])
    _record(pid, ["v2 chunk a", "v2 chunk b"])
    try:
        events = persistence.get_ingest_events(pid)
        assert len(events) == 1
        assert events[0]["chunks"] == ["v2 chunk a", "v2 chunk b"]
    finally:
        persistence.delete_ingest_events(pid)


def test_deleting_a_paper_removes_it_from_the_log():
    """Otherwise a rebuild resurrects a deleted paper — the exact inconsistency
    a system of record is supposed to prevent."""
    pid = str(uuid.uuid4())
    _record(pid, ["chunk"])
    assert persistence.delete_ingest_events(pid) == 1
    assert persistence.get_ingest_events(pid) == []


def test_events_replay_in_ingest_order():
    """Chunk ids and citation numbering follow insertion order, so a replay has
    to reproduce it."""
    a, b = "aaa-" + str(uuid.uuid4()), "bbb-" + str(uuid.uuid4())
    persistence.record_ingest(a, a, "h", "A", "", ["x"], [{}], ["a_0"], "m", 1,
                              "2026-01-01T00:00:00+00:00")
    persistence.record_ingest(b, b, "h", "B", "", ["y"], [{}], ["b_0"], "m", 1,
                              "2026-01-02T00:00:00+00:00")
    try:
        ordered = [e["paper_id"] for e in persistence.get_ingest_events()]
        assert ordered.index(a) < ordered.index(b)
    finally:
        persistence.delete_ingest_events(a)
        persistence.delete_ingest_events(b)


# --------------------------------------------------------------------------
# Drift detection
# --------------------------------------------------------------------------
def test_no_drift_when_the_log_matches_the_config():
    import config
    import vector_store
    events = [{"embed_model": config.EMBEDDING_MODEL_NAME,
               "chunker_version": vector_store.CHUNKER_VERSION}]
    assert reindex._drift_report(events) == []


def test_changed_embedding_model_is_reported_as_replayable():
    import vector_store
    events = [{"embed_model": "some/old-model",
               "chunker_version": vector_store.CHUNKER_VERSION}]
    problems = reindex._drift_report(events)
    assert any("differs" in p and "re-embed" in p for p in problems)


def test_mixed_embedding_models_are_reported():
    """Two embedding spaces in one collection: cosine distance between them is
    meaningless, and nothing else in the system would notice."""
    import config
    import vector_store
    events = [
        {"embed_model": config.EMBEDDING_MODEL_NAME,
         "chunker_version": vector_store.CHUNKER_VERSION},
        {"embed_model": "some/other-model",
         "chunker_version": vector_store.CHUNKER_VERSION},
    ]
    assert any("MIXED embedding models" in p for p in reindex._drift_report(events))


def test_changed_chunker_is_reported_as_NOT_replayable():
    """The important half of the honesty: replay reuses recorded chunk
    boundaries, so it cannot implement a chunker change. Claiming otherwise
    would leave the user believing they had migrated when they had not."""
    import config
    import vector_store
    events = [{"embed_model": config.EMBEDDING_MODEL_NAME,
               "chunker_version": vector_store.CHUNKER_VERSION + 1}]
    problems = reindex._drift_report(events)
    assert any("CANNOT" in p and "Re-ingest" in p for p in problems)
