"""Integration tests across the real retrieval pipeline.

Every one of the unit-test files mocks the LLM, the vector store and the
embeddings, so the seams BETWEEN stages were never exercised — prompt assembly,
context formatting, citation compaction, chunk truncation, cache population, the
argument shape one stage hands the next. That is exactly where this codebase's
shipped bugs came from. From the v2.4 patch notes alone:

  * verify.check_claims was handed (index, text) tuples instead of chunk text,
    so every answer reported confidence 0.0;
  * the retrieval cache never populated, because cacheability was checked after
    the collection was materialized;
  * the answer generator used the configured default instead of the requested
    model.

None of those is catchable by a mock that returns whatever the test author
assumed the real thing returns.

Real here: ChromaDB (a genuine on-disk collection), the BM25 index, RRF fusion,
format_context, citation extraction and compaction, and the TTL caches.

Stubbed, and why: the embedding model and the LLM. Those are true external
boundaries — a 2.5 GB model download would make this suite unrunnable in CI, and
it is not our code under test. The stub embedder returns real vectors of a real
dimension through the real Chroma code path, so everything on our side of that
boundary is genuinely exercised.

Marked `integration`: slower than the unit suite and excluded by CI's default
selector.
"""

import zlib

import numpy as np
import pytest

import bm25_search
import cache
import config
import embeddings
import rag
import vector_store

pytestmark = pytest.mark.integration


CORPUS = [
    # (paper_id, title, section, text)
    ("antenna_ml", "Machine Learning for Antenna Design", "abstract",
     "We apply machine learning to antenna design, optimizing patch antenna "
     "geometry with neural networks for wideband performance."),
    ("antenna_ml", "Machine Learning for Antenna Design", "methods",
     "The tandem neural network uses a smooth thresholding function to force "
     "discrete pixel values in the antenna geometry during optimization."),
    ("antenna_ml", "Machine Learning for Antenna Design", "results",
     "The optimized antennas are 50 percent more compact while maintaining "
     "gain and radiation efficiency across the band."),
    ("qubit_ec", "Surface Codes for Qubit Error Correction", "abstract",
     "Surface codes provide fault tolerant quantum error correction for "
     "superconducting qubit arrays with high threshold."),
    ("qubit_ec", "Surface Codes for Qubit Error Correction", "methods",
     "Syndrome extraction circuits measure stabilizers on the qubit lattice "
     "without collapsing the encoded logical state."),
    ("protein_fold", "Transformers for Protein Folding", "abstract",
     "Transformer architectures predict protein tertiary structure from amino "
     "acid sequence using attention over residue pairs."),
]


def _fake_embed(texts):
    """Deterministic bag-of-words vectors.

    Not a random stub: texts sharing vocabulary land near each other, so cosine
    distance is meaningful and the dense leg actually ranks. That is what lets
    these tests assert on retrieval ORDER rather than merely on plumbing.
    """
    dim = 64
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in str(t).lower().split():
            # crc32, not hash(): str hashing is salted per process, so bucket
            # collisions — and with them the ranking these tests assert on —
            # would change from run to run.
            out[i, zlib.crc32(tok.encode("utf-8")) % dim] += 1.0
        norm = np.linalg.norm(out[i])
        if norm:
            out[i] /= norm
    return out


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """A real ChromaDB collection loaded with the corpus above."""
    monkeypatch.setattr(config, "CHROMA_DB_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "BM25_CACHE_DIR", tmp_path / "bm25")
    monkeypatch.setattr(config, "COLLECTION_NAME", "seams")
    monkeypatch.setattr(config, "USE_RERANKER", False)      # avoids a model download
    monkeypatch.setattr(config, "USE_HYBRID_SEARCH", True)
    monkeypatch.setattr(vector_store, "_chroma_client", None)

    monkeypatch.setattr(embeddings, "embed_texts", lambda texts, **k: _fake_embed(texts))
    monkeypatch.setattr(embeddings, "embed_query", lambda q: _fake_embed([q])[0])
    monkeypatch.setattr(embeddings, "embed_passages", lambda p, **k: _fake_embed(p))
    monkeypatch.setattr(rag.embeddings, "embed_query", lambda q: _fake_embed([q])[0])

    bm25_search.invalidate()
    cache.retrieval_cache.invalidate()

    collection = vector_store.get_or_create_collection("seams", reset=True)
    texts = [c[3] for c in CORPUS]
    metas = [{"paper_id": p, "title": t, "section": s, "chunk_index": i}
             for i, (p, t, s, _) in enumerate(CORPUS)]
    ids = [f"{p}_{i}" for i, (p, _, _, _) in enumerate(CORPUS)]
    vector_store.add_documents(texts, _fake_embed(texts), metas, ids, collection=collection)

    yield collection

    bm25_search.invalidate()
    cache.retrieval_cache.invalidate()
    vector_store._chroma_client = None


# --------------------------------------------------------------------------
# Retrieval through the real store
# --------------------------------------------------------------------------
def test_retrieval_returns_the_relevant_paper_first(pipeline):
    out = rag.retrieve_context("antenna design neural network", top_k=3, collection=pipeline)
    assert out["chunks_used"] > 0
    assert out["metadatas"][0]["paper_id"] == "antenna_ml"


def test_retrieval_is_stamped_with_provenance_end_to_end(pipeline):
    """The stamp must survive the round trip through Chroma, not merely be
    present on the dict handed to upsert."""
    out = rag.retrieve_context("antenna", top_k=1, collection=pipeline)
    assert out["metadatas"][0]["embed_model"] == config.EMBEDDING_MODEL_NAME
    assert vector_store.check_index_compatibility(pipeline) is None


def test_hybrid_retrieval_finds_a_lexical_match(pipeline):
    """Exercises dense + BM25 + RRF together. 'syndrome' is a rare term lexical
    search nails; assert BM25 saw it too, so a broken RRF can't pass on the
    dense leg alone."""
    idx = bm25_search.get_or_build_index(pipeline)
    sparse_ids, _ = idx.search("syndrome stabilizers", top_k=3)
    assert any("qubit_ec" in i for i in sparse_ids)

    out = rag.retrieve_context("syndrome extraction stabilizers", top_k=3, collection=pipeline)
    assert "qubit_ec" in {m["paper_id"] for m in out["metadatas"]}


# --------------------------------------------------------------------------
# The seams the mocks never covered
# --------------------------------------------------------------------------
def test_format_context_and_citation_numbering_agree(pipeline):
    """citation_number_map, format_context and extract_citations each number
    papers independently. If they disagree, an answer's [N] resolves to the
    wrong source — a silent correctness bug with no exception to catch it."""
    out = rag.retrieve_context("antenna qubit protein", top_k=6, collection=pipeline)
    metas = out["metadatas"]
    context = out["formatted_context"]

    num_to_meta = rag.citation_number_map(metas)
    for num, meta in num_to_meta.items():
        # Every number the map hands out must appear in the text the model sees.
        assert f"[{num}]" in context or f"Cite:{num}" in context, \
            f"citation {num} ({meta['title']}) missing from formatted context"

    answer = "Antennas shrink by half [1]."
    cites = rag.extract_citations(answer, metas, visible_chunks=out["chunks_used"])
    assert cites and cites[0]["title"] == num_to_meta[1]["title"]


def test_compact_citations_renumbers_and_rewrites_together(pipeline):
    """The answer text and the citation panel must be renumbered as one unit.
    Renumbering only the panel is how [1]...[4] ends up beside a two-entry list."""
    out = rag.retrieve_context("antenna qubit protein", top_k=6, collection=pipeline)
    metas = out["metadatas"]
    n = len(rag.citation_number_map(metas))
    assert n >= 3, "need >=3 distinct papers to test sparse citation renumbering"

    answer = f"First claim [1]. Third claim [{n}]."
    rewritten, cites = rag.compact_citations(answer, metas, out.get("chunks"),
                                             visible_chunks=out["chunks_used"])
    assert [c["number"] for c in cites] == ["1", "2"]   # dense 1..M
    assert "[1]" in rewritten and "[2]" in rewritten
    assert f"[{n}]" not in rewritten                    # old numbering is gone


def test_dangling_citation_is_dropped_not_left_in_the_answer(pipeline):
    """A number the model invented past the context must not survive — it would
    read as a real source that was never shown."""
    out = rag.retrieve_context("antenna", top_k=2, collection=pipeline)
    rewritten, cites = rag.compact_citations("Claim with a bogus source [99].",
                                             out["metadatas"], out.get("chunks"),
                                             visible_chunks=out["chunks_used"])
    assert "[99]" not in rewritten
    assert all(c["number"] != "99" for c in cites)


def test_faithfulness_receives_chunk_text_not_indices(pipeline, monkeypatch):
    """Regression for the v2.4 bug: _run_faithfulness passed (index, text) tuples
    into the NLI model, so every call raised and every answer reported
    confidence 0.0. Assert the seam carries real strings."""
    seen = {}

    def _capture(answer, chunks, metadatas=None):
        seen["chunks"] = chunks
        return [{"claim": "c", "support": 0.9, "grounded": True}]

    import verify
    monkeypatch.setattr(verify, "check_claims", _capture)

    out = rag.retrieve_context("antenna design", top_k=2, collection=pipeline)
    rag._run_faithfulness("Antennas shrink [1].", out["chunks"], out["metadatas"])

    # check_claims itself is patched, so the seam is exercised whatever the
    # faithfulness config says — a conditional assert here would let the
    # regression back in silently.
    assert "chunks" in seen, "faithfulness never reached verify.check_claims"
    assert seen["chunks"], "faithfulness ran with no chunks"
    assert all(isinstance(c, str) for c in seen["chunks"]), \
        f"expected chunk text, got {[type(c).__name__ for c in seen['chunks']]}"


def test_an_explicit_collection_is_deliberately_not_cached(pipeline):
    """`cacheable = collection is None and filter_dict is None`. The cache key
    does not include the collection, so caching a caller-supplied one would let
    a result from one collection be served for another. Pinned so the condition
    is not "simplified" away later."""
    cache.retrieval_cache.invalidate()
    rag.retrieve_context("antenna design neural network", top_k=3, collection=pipeline)
    assert cache.retrieval_cache.stats["size"] == 0


def test_retrieval_cache_actually_populates(pipeline):
    """Regression for the v2.4 bug where the cacheability check ran after the
    collection was materialized, so nothing was ever stored.

    Goes through the default-collection path, which is the one that caches — the
    fixture has already pointed COLLECTION_NAME and CHROMA_DB_DIR at the
    temporary corpus.
    """
    cache.retrieval_cache.invalidate()
    before = cache.retrieval_cache.stats["size"]
    out = rag.retrieve_context("antenna design neural network", top_k=3)
    assert out["chunks_used"] > 0, "default-collection path retrieved nothing"
    assert cache.retrieval_cache.stats["size"] > before, "retrieval cache did not store anything"


def test_cached_entry_cannot_be_mutated_by_a_caller(pipeline):
    """Cached results are handed out by reference unless copied — one caller
    trimming its list would corrupt every later hit."""
    cache.retrieval_cache.invalidate()
    q = "antenna design neural network"
    first = rag.retrieve_context(q, top_k=3)
    assert first["metadatas"]
    first["metadatas"].clear()
    second = rag.retrieve_context(q, top_k=3)
    assert second["metadatas"], "a caller mutating its result corrupted the cache"


# --------------------------------------------------------------------------
# Ingest / delete consistency across both indexes
# --------------------------------------------------------------------------
def test_deleting_a_paper_removes_it_from_both_indexes(pipeline):
    """Chroma and BM25 are two indexes over one corpus. A delete reaching only
    one keeps citing the deleted paper from the other."""
    bm25_search.get_or_build_index(pipeline)          # build before deleting
    removed = vector_store.delete_by_paper_id("qubit_ec", collection=pipeline)
    assert removed > 0

    idx = bm25_search._indices.get("seams")
    assert idx is not None
    sparse_ids, _ = idx.search("syndrome stabilizers", top_k=5)
    assert not any("qubit_ec" in i for i in sparse_ids), "BM25 still serves deleted chunks"

    out = rag.retrieve_context("syndrome extraction", top_k=5, collection=pipeline)
    assert "qubit_ec" not in {m["paper_id"] for m in out["metadatas"]}


def test_reindex_replays_the_log_into_a_working_collection(pipeline, monkeypatch):
    """The whole justification for the ingest log: rebuild the vector store from
    recorded chunks, with no PDFs and no re-captioning, and get a collection that
    retrieves the same way.

    This is what makes an embedding-model change a routine operation instead of
    an hours-long non-reproducible one.
    """
    import uuid
    import persistence
    import reindex

    pid = "replay_" + uuid.uuid4().hex[:8]
    chunks = [c[3] for c in CORPUS if c[0] == "antenna_ml"]
    persistence.record_ingest(
        event_id=pid, paper_id=pid, content_hash="h", title="Replayed Antennas",
        source_path="", chunks=chunks,
        metadatas=[{"paper_id": pid, "title": "Replayed Antennas",
                    "section": "abstract", "chunk_index": i} for i in range(len(chunks))],
        ids=[f"{pid}_{i}" for i in range(len(chunks))],
        embed_model=config.EMBEDDING_MODEL_NAME, chunker_version=vector_store.CHUNKER_VERSION,
        created_at="2026-01-01T00:00:00+00:00")
    try:
        rc = reindex.reindex("replayed", dry_run=False, batch_size=8)
        assert rc == 0

        rebuilt = vector_store.get_or_create_collection("replayed")
        # Only this paper's rows: the ingest log is process-wide, so a total count
        # would depend on whatever else earlier tests recorded.
        assert len(rebuilt.get(where={"paper_id": pid}).get("ids", [])) == len(chunks)

        # The rebuilt collection must actually retrieve, not merely hold rows.
        out = rag.retrieve_context("antenna design neural network", top_k=3,
                                   collection=rebuilt)
        assert out["chunks_used"] > 0
        assert out["metadatas"][0]["paper_id"] == pid
        # And it is stamped with the model that just re-embedded it.
        assert vector_store.check_index_compatibility(rebuilt) is None
    finally:
        persistence.delete_ingest_events(pid)


def test_newly_ingested_chunks_are_searchable_without_a_rebuild(pipeline):
    """The incremental BM25 path: an ingest must not require dropping the index."""
    bm25_search.get_or_build_index(pipeline)
    text = "Photonic crystal waveguides confine light in periodic dielectric lattices."
    vector_store.add_documents(
        [text], _fake_embed([text]),
        [{"paper_id": "photonics", "title": "Photonic Crystals", "section": "abstract",
          "chunk_index": 0}],
        ["photonics_0"], collection=pipeline)
    assert bm25_search.add_to_index(["photonics_0"], [text]) is True

    idx = bm25_search._indices["seams"]
    ids, _ = idx.search("photonic crystal waveguides", top_k=3)
    assert "photonics_0" in ids
