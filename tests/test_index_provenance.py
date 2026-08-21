"""Tests for embedding/chunker provenance stamps.

Nothing recorded which model produced a vector, so swapping the embedding model
— or changing the per-section chunk sizes — silently mixed incompatible vectors
into one collection. Cosine distance between two different embedding spaces is
meaningless but never raises, so the only symptom was retrieval quality quietly
getting worse, which is nearly undebuggable from outside.
"""

import numpy as np

import config
import vector_store


class _Collection:
    name = "prov"

    def __init__(self, metas=None):
        self.upserted = None
        self._metas = metas

    def upsert(self, **kwargs):
        self.upserted = kwargs

    def count(self):
        return len(self.upserted["ids"]) if self.upserted else 0

    def peek(self, limit=1):
        if self._metas is None:
            return {"metadatas": []}
        return {"metadatas": self._metas[:limit]}


def test_every_chunk_is_stamped_on_write():
    coll = _Collection()
    vector_store.add_documents(
        texts=["some text"],
        embeddings=np.zeros((1, 4)),
        metadatas=[{"paper_id": "p1", "title": "T"}],
        ids=["c1"],
        collection=coll,
    )
    meta = coll.upserted["metadatas"][0]
    assert meta["embed_model"] == config.EMBEDDING_MODEL_NAME
    assert meta["embed_dim"] == config.EMBEDDING_DIMENSION
    assert meta["chunker_version"] == vector_store.CHUNKER_VERSION
    assert meta["schema_version"] == vector_store.SCHEMA_VERSION
    assert meta["paper_id"] == "p1"  # original fields survive


def test_stamping_does_not_mutate_the_callers_metadata():
    """add_documents receives the ingest pipeline's own dicts; stamping in place
    would leak index bookkeeping back into the caller."""
    coll = _Collection()
    original = {"paper_id": "p1"}
    vector_store.add_documents(["t"], np.zeros((1, 4)), [original], ["c1"], collection=coll)
    assert original == {"paper_id": "p1"}


def test_empty_collection_has_nothing_to_be_incompatible_with():
    assert vector_store.check_index_compatibility(_Collection(metas=None)) is None


def test_matching_provenance_reports_no_problem():
    coll = _Collection(metas=[{
        "embed_model": config.EMBEDDING_MODEL_NAME,
        "embed_dim": config.EMBEDDING_DIMENSION,
        "chunker_version": vector_store.CHUNKER_VERSION,
        "schema_version": vector_store.SCHEMA_VERSION,
    }])
    assert vector_store.check_index_compatibility(coll) is None


def test_a_changed_embedding_model_is_reported():
    coll = _Collection(metas=[{
        "embed_model": "some/other-model",
        "embed_dim": config.EMBEDDING_DIMENSION,
        "chunker_version": vector_store.CHUNKER_VERSION,
        "schema_version": vector_store.SCHEMA_VERSION,
    }])
    problem = vector_store.check_index_compatibility(coll)
    assert problem and "some/other-model" in problem


def test_a_changed_dimension_is_reported():
    coll = _Collection(metas=[{
        "embed_model": config.EMBEDDING_MODEL_NAME,
        "embed_dim": 384,
        "chunker_version": vector_store.CHUNKER_VERSION,
        "schema_version": vector_store.SCHEMA_VERSION,
    }])
    problem = vector_store.check_index_compatibility(coll)
    assert problem and "384" in problem


def test_a_changed_chunker_is_reported():
    coll = _Collection(metas=[{
        "embed_model": config.EMBEDDING_MODEL_NAME,
        "embed_dim": config.EMBEDDING_DIMENSION,
        "chunker_version": vector_store.CHUNKER_VERSION + 1,
        "schema_version": vector_store.SCHEMA_VERSION,
    }])
    assert "chunker version" in vector_store.check_index_compatibility(coll)


def test_unstamped_legacy_chunks_are_reported_not_assumed_compatible():
    """Chunks written before stamping existed cannot be compared against
    anything, so say so rather than silently calling them fine."""
    coll = _Collection(metas=[{"paper_id": "old"}])
    problem = vector_store.check_index_compatibility(coll)
    assert problem and "provenance" in problem


def test_fingerprint_reads_back_what_was_stamped():
    coll = _Collection(metas=[{
        "embed_model": "BAAI/bge-m3", "embed_dim": 1024,
        "chunker_version": 1, "schema_version": 1,
    }])
    assert vector_store.index_fingerprint(coll) == {
        "embed_model": "BAAI/bge-m3", "embed_dim": 1024,
        "chunker_version": 1, "schema_version": 1,
    }
