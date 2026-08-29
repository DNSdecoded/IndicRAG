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


# ── snapshots (backup.py) ───────────────────────────────────────────────────

def test_snapshot_and_restore_round_trip(tmp_path):
    """A snapshot is the whole recovery story: the indexes are derived, so the
    log is the only thing whose loss is unrecoverable."""
    import backup

    kept = str(uuid.uuid4())
    _record(kept, ["kept chunk"])
    snapshot = backup.create(out_dir=tmp_path)

    after = str(uuid.uuid4())
    _record(after, ["written after the snapshot"])
    assert persistence.get_ingest_events(after)

    result = persistence.restore_from(snapshot)
    try:
        assert result["papers"] >= 1
        assert persistence.get_ingest_events(kept), "restore lost a recorded paper"
        assert not persistence.get_ingest_events(after), (
            "restore kept a paper the snapshot predates")
    finally:
        persistence.delete_ingest_events(kept)


def test_snapshot_manifest_describes_what_it_holds(tmp_path):
    import json as _json
    import backup

    pid = str(uuid.uuid4())
    _record(pid, ["a", "b"])
    try:
        snapshot = backup.create(out_dir=tmp_path)
        manifest = _json.loads(snapshot.with_suffix(".json").read_text(encoding="utf-8"))
        assert manifest["papers"] >= 1
        assert manifest["chunks"] >= 2
        assert "BAAI/bge-m3" in manifest["embed_models"]
    finally:
        persistence.delete_ingest_events(pid)


def test_restore_refuses_a_database_that_is_not_indicrag(tmp_path):
    """Restoring an arbitrary SQLite file would silently wipe the system of record."""
    import sqlite3

    import pytest

    stranger = tmp_path / "not-indicrag.db"
    conn = sqlite3.connect(stranger)
    conn.execute("CREATE TABLE unrelated (x TEXT)")
    conn.commit()
    conn.close()

    pid = str(uuid.uuid4())
    _record(pid, ["still here"])
    try:
        with pytest.raises(ValueError):
            persistence.restore_from(stranger)
        assert persistence.get_ingest_events(pid), "a refused restore must change nothing"
    finally:
        persistence.delete_ingest_events(pid)


def test_batch_writes_commits_once_and_rolls_back_as_a_unit():
    """A partial log is the exact divergence the log exists to prevent, so a bulk
    write must land whole or not at all."""
    import pytest

    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    try:
        with pytest.raises(RuntimeError):
            with persistence.batch_writes():
                _record(a, ["one"])
                _record(b, ["two"])
                raise RuntimeError("failure mid-batch")
        assert not persistence.get_ingest_events(a)
        assert not persistence.get_ingest_events(b)

        with persistence.batch_writes():
            _record(a, ["one"])
            _record(b, ["two"])
        assert persistence.get_ingest_events(a)
        assert persistence.get_ingest_events(b)
    finally:
        persistence.delete_ingest_events(a)
        persistence.delete_ingest_events(b)
