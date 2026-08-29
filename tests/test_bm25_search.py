"""Coverage for the BM25 lexical index.

This module had no dedicated test — it was only reached incidentally through
test_agent.py — which matters because it is the half of hybrid retrieval that is
next in line to be rewritten (inverted index, incremental updates, disk
persistence). These tests pin the behavior that must survive that rewrite:
ranking order, Unicode/Indic tokenization, RRF fusion, and index invalidation.
"""

import pytest

import bm25_search


def _index(docs: dict) -> bm25_search.BM25Index:
    idx = bm25_search.BM25Index()
    idx.build(list(docs.keys()), list(docs.values()))
    return idx


def test_ranks_the_document_that_actually_contains_the_term():
    idx = _index({
        "a": "quantum entanglement in superconducting qubits",
        "b": "antenna design for millimetre wave radar",
        "c": "a survey of antenna optimization using machine learning",
    })
    ids, scores = idx.search("antenna optimization", top_k=3)
    assert ids[0] == "c"          # matches both query terms
    assert scores[0] > scores[1]
    assert "a" not in ids[:2]     # matches neither


def test_rarer_term_outweighs_common_one():
    """IDF must actually discriminate: a term in every doc carries no signal."""
    idx = _index({
        "a": "machine learning applied to antenna design",
        "b": "machine learning applied to protein folding",
        "c": "machine learning applied to weather models",
    })
    ids, _ = idx.search("machine antenna", top_k=3)
    assert ids[0] == "a"  # 'antenna' is rare, 'machine' is in all three


def test_empty_index_returns_empty_not_error():
    idx = bm25_search.BM25Index()
    idx.build([], [])
    assert idx.search("anything", top_k=5) == ([], [])


def test_no_query_term_matches_returns_zero_scores():
    idx = _index({"a": "antenna design"})
    ids, scores = idx.search("thermodynamics", top_k=5)
    assert scores == [0.0] * len(ids)


def test_tokenizer_handles_devanagari_and_tamil():
    """The Unicode-aware tokenizer is why Indic lexical search works at all —
    an ASCII/whitespace tokenizer would drop these scripts entirely."""
    tok = bm25_search.BM25Index._tokenize
    assert tok("यंत्र अधिगम") == ["यंत्र", "अधिगम"]
    assert tok("இயந்திர கற்றல்") == ["இயந்திர", "கற்றல்"]
    # Combining marks stay attached to their base character, not split off.
    assert tok("हिन्दी") == ["हिन्दी"]


def test_tokenizer_strips_punctuation_and_casefolds():
    assert bm25_search.BM25Index._tokenize("Antenna-Design, v2.0!") == [
        "antenna", "design", "v2", "0"
    ]


def test_indic_query_retrieves_indic_document():
    idx = _index({
        "hi": "यंत्र अधिगम का उपयोग करके एंटीना अनुकूलन",
        "en": "antenna optimization using machine learning",
    })
    ids, _ = idx.search("एंटीना अनुकूलन", top_k=2)
    assert ids[0] == "hi"


def test_rrf_rewards_agreement_between_both_rankers():
    """A doc both rankers like must beat one that only tops a single list."""
    dense = ["x", "a", "b"]
    sparse = ["y", "a", "c"]
    fused = bm25_search.rrf(dense, sparse, k=60)
    assert fused[0] == "a"
    assert set(fused) == {"a", "b", "c", "x", "y"}


def test_rrf_handles_one_empty_list():
    assert bm25_search.rrf([], ["a", "b"], k=60) == ["a", "b"]


def test_invalidate_drops_cached_indices():
    bm25_search._indices["probe"] = _index({"a": "text"})
    bm25_search.invalidate()
    assert bm25_search._indices == {}


def test_get_or_build_index_returns_none_for_empty_collection():
    """An empty corpus must yield None, not a zero-doc index — callers branch on
    None to skip the sparse leg entirely."""
    class _EmptyCollection:
        name = "empty"

        def count(self):
            return 0

    bm25_search.invalidate()
    assert bm25_search.get_or_build_index(_EmptyCollection()) is None


def test_add_documents_makes_new_text_searchable_without_a_rebuild():
    idx = _index({"a": "antenna design"})
    idx.add_documents(["b"], ["quantum error correction"])
    ids, _ = idx.search("quantum", top_k=5)
    assert ids == ["b"]
    assert idx.n_docs == 2


def test_add_documents_updates_idf_not_just_the_postings():
    """Corpus statistics have to move with the corpus, or scores drift from a
    freshly built index the longer the process runs."""
    idx = _index({"a": "antenna"})
    idx.add_documents(["b", "c"], ["antenna", "unrelated"])
    fresh = _index({"a": "antenna", "b": "antenna", "c": "unrelated"})
    assert idx.n_docs == fresh.n_docs
    assert idx.avg_dl == pytest.approx(fresh.avg_dl)
    assert idx.search("antenna", 5)[1][0] == pytest.approx(fresh.search("antenna", 5)[1][0])


def test_re_adding_an_id_replaces_it_rather_than_double_counting():
    idx = _index({"a": "antenna design"})
    idx.add_documents(["a"], ["antenna design revised"])
    ids, _ = idx.search("antenna", top_k=5)
    assert ids == ["a"]        # one hit, not two
    assert idx.n_docs == 1


def test_removed_documents_stop_being_retrieved():
    """A deleted paper must not keep getting cited."""
    idx = _index({"a": "antenna design", "b": "antenna optimization"})
    assert idx.remove_documents(["a"]) == 1
    ids, _ = idx.search("antenna", top_k=5)
    assert ids == ["b"]
    assert idx.n_docs == 1


def test_removing_everything_leaves_a_valid_empty_index():
    idx = _index({"a": "antenna", "b": "qubit"})
    idx.remove_documents(["a", "b"])
    assert idx.n_docs == 0
    assert idx.search("antenna", top_k=5) == ([], [])
    assert idx.avg_dl == 1.0  # no ZeroDivisionError


def test_remove_is_idempotent():
    idx = _index({"a": "antenna"})
    assert idx.remove_documents(["a"]) == 1
    assert idx.remove_documents(["a"]) == 0
    assert idx.remove_documents(["never-existed"]) == 0


def test_incremental_result_matches_a_full_rebuild():
    """The whole justification for incremental updates: identical ranking."""
    docs = {
        "a": "antenna optimization using machine learning",
        "b": "protein folding with transformers",
        "c": "millimetre wave radar antenna arrays",
    }
    incremental = _index({"a": docs["a"]})
    incremental.add_documents(["b"], [docs["b"]])
    incremental.add_documents(["c"], [docs["c"]])
    assert incremental.search("antenna arrays", 3)[0] == _index(docs).search("antenna arrays", 3)[0]


def test_add_to_index_reports_when_there_is_no_index_yet():
    """False tells the caller to leave it alone and let the next query build it,
    rather than forcing a rebuild it doesn't need."""
    bm25_search.invalidate()
    assert bm25_search.add_to_index(["a"], ["text"]) is False


def test_add_to_index_updates_a_live_index():
    bm25_search.invalidate()
    bm25_search._indices["live"] = _index({"a": "antenna"})
    try:
        assert bm25_search.add_to_index(["b"], ["qubit entanglement"]) is True
        assert bm25_search._indices["live"].search("qubit", 5)[0] == ["b"]
    finally:
        bm25_search.invalidate()


def test_remove_from_index_updates_a_live_index():
    bm25_search.invalidate()
    bm25_search._indices["live"] = _index({"a": "antenna", "b": "qubit"})
    try:
        assert bm25_search.remove_from_index(["a"]) is True
        assert bm25_search._indices["live"].search("antenna", 5)[0] == []
    finally:
        bm25_search.invalidate()


class _Coll:
    """Minimal stand-in for a ChromaDB collection."""

    def __init__(self, docs, name="persist"):
        self.name = name
        self._docs = docs
        self.get_calls = 0
        self.doc_fetches = 0

    def count(self):
        return len(self._docs)

    def get(self, include=None):
        self.get_calls += 1
        # The staleness check fetches ids only (include=[]); a rebuild fetches
        # documents. Counted apart, because "did not re-read the corpus" is about
        # the text payload, not about touching the collection at all.
        if include:
            self.doc_fetches += 1
        return {"ids": list(self._docs), "documents": list(self._docs.values())}


def _simulate_restart():
    """Drop in-memory indices only. invalidate() also deletes the saved cache
    (it means "the corpus moved"), which is the opposite of a restart."""
    bm25_search._indices = {}


@pytest.fixture
def _cache_dir(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "BM25_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "BM25_PERSIST", True)
    bm25_search.invalidate()
    yield tmp_path
    bm25_search.invalidate()


def test_index_is_persisted_and_reloaded_without_rereading_the_corpus(_cache_dir):
    """The point of persisting: a restart must not re-read every document."""
    coll = _Coll({"a": "antenna optimization", "b": "protein folding"})
    bm25_search.get_or_build_index(coll)
    assert coll.doc_fetches == 1

    _simulate_restart()               # the process restarts; the file stays
    idx = bm25_search.get_or_build_index(coll)

    # Served from disk: the id set is re-read to prove the cache still matches
    # the collection, but the documents — the expensive part — are not.
    assert coll.doc_fetches == 1
    assert idx.search("antenna", 5)[0] == ["a"]


def test_reloaded_index_ranks_identically_to_a_freshly_built_one(_cache_dir):
    docs = {
        "a": "antenna optimization using machine learning",
        "b": "protein folding with transformers",
        "c": "millimetre wave radar antenna arrays",
    }
    coll = _Coll(docs)
    fresh = bm25_search.get_or_build_index(coll).search("antenna arrays", 3)
    _simulate_restart()
    reloaded = bm25_search.get_or_build_index(coll).search("antenna arrays", 3)
    assert reloaded[0] == fresh[0]
    assert reloaded[1] == pytest.approx(fresh[1])


def test_stale_cache_is_ignored_when_the_corpus_changed(_cache_dir):
    """A saved index that no longer matches the corpus must be rebuilt, not
    trusted — serving a stale index is worse than paying for a rebuild."""
    coll = _Coll({"a": "antenna"})
    bm25_search.get_or_build_index(coll)
    _simulate_restart()

    coll._docs["b"] = "newly ingested qubit paper"
    idx = bm25_search.get_or_build_index(coll)

    assert coll.doc_fetches == 2             # rebuilt from the corpus
    assert idx.search("qubit", 5)[0] == ["b"]


def test_corrupt_cache_falls_back_to_rebuild(_cache_dir):
    coll = _Coll({"a": "antenna"})
    bm25_search.get_or_build_index(coll)
    _simulate_restart()

    bm25_search._cache_path(coll.name).write_bytes(b"not a gzip file at all")
    idx = bm25_search.get_or_build_index(coll)

    assert idx is not None                   # degraded to a rebuild, did not raise
    assert idx.search("antenna", 5)[0] == ["a"]


def test_cache_from_an_older_format_is_ignored(_cache_dir):
    import gzip
    import json
    coll = _Coll({"a": "antenna"})
    bm25_search.get_or_build_index(coll)
    _simulate_restart()

    path = bm25_search._cache_path(coll.name)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    payload["format"] = bm25_search._CACHE_FORMAT + 1
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    assert bm25_search.get_or_build_index(coll) is not None
    assert coll.doc_fetches == 2             # ignored the future-format file


def test_persisted_cache_is_not_pickle(_cache_dir):
    """Regression: this file is read back and trusted at startup, so it must not
    be a format that can execute code on load."""
    import gzip
    coll = _Coll({"a": "antenna"})
    bm25_search.get_or_build_index(coll)
    raw = bm25_search._cache_path(coll.name).read_bytes()
    assert raw[:2] == b"\x1f\x8b"            # gzip magic
    assert gzip.decompress(raw).lstrip().startswith(b"{")   # JSON object


def test_persistence_can_be_turned_off(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "BM25_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "BM25_PERSIST", False)
    bm25_search.invalidate()
    try:
        coll = _Coll({"a": "antenna"}, name="nopersist")
        bm25_search.get_or_build_index(coll)
        assert list(tmp_path.iterdir()) == []
    finally:
        bm25_search.invalidate()


def test_a_cache_that_no_longer_matches_the_collection_is_rejected(_cache_dir):
    """The saved index holds one live doc while Chroma still holds both, so the
    file must be thrown away and rebuilt rather than served."""
    coll = _Coll({"a": "antenna design", "b": "antenna optimization"})
    idx = bm25_search.get_or_build_index(coll)
    idx.remove_documents(["a"])
    bm25_search.save_index(coll.name)
    _simulate_restart()

    # count() still reports 2 rows in Chroma but the index holds 1 live doc, so
    # the staleness check correctly rejects it and rebuilds.
    reloaded = bm25_search.get_or_build_index(coll)
    assert reloaded is not None
    # Rebuilt from the collection, so "a" is back — the point is that what gets
    # served matches Chroma, not that the stale tombstone survives.
    assert reloaded.search("antenna design", 5)[0][:1] == ["a"]


def test_get_or_build_index_caches_per_collection():
    class _Collection:
        name = "cached"

        def count(self):
            return 2

        def get(self, include=None):
            return {"ids": ["a", "b"], "documents": ["antenna", "qubit"]}

    bm25_search.invalidate()
    coll = _Collection()
    first = bm25_search.get_or_build_index(coll)
    second = bm25_search.get_or_build_index(coll)
    assert first is second  # rebuilt once, then served from cache
    bm25_search.invalidate()


def test_cache_is_rejected_when_content_changed_but_count_did_not(_cache_dir):
    """The count-only check could not see this: delete one paper and ingest
    another of the same size and the count is identical while every chunk
    differs. Loading that cache serves deleted text and misses its replacement."""
    coll = _Coll({"a1": "antenna design", "a2": "antenna results"})
    bm25_search.get_or_build_index(coll)
    _simulate_restart()

    coll._docs = {"b1": "qubit surface codes", "b2": "qubit syndrome"}   # same count
    idx = bm25_search.get_or_build_index(coll)

    assert idx.search("antenna", 5)[0] == [], "stale cache served deleted content"
    assert set(idx.search("qubit", 5)[0]) == {"b1", "b2"}


def test_removing_documents_persists_the_index(_cache_dir):
    """A delete mutates the in-memory index; leaving the disk copy untouched is
    how it diverges from the collection across a restart."""
    _simulate_restart()
    bm25_search._indices["persist"] = _index({"a": "antenna", "b": "qubit"})
    try:
        bm25_search.remove_from_index(["a"], "persist")
        assert bm25_search._cache_path("persist").exists()
    finally:
        _simulate_restart()


def test_invalidate_drops_the_saved_cache_too(_cache_dir):
    """Chunk ids are positional (paper_id_section_N), so re-ingesting a changed
    paper reuses them and the id fingerprint alone cannot spot the new text.
    invalidate() therefore has to take the file with it."""
    coll = _Coll({"a": "antenna"})
    bm25_search.get_or_build_index(coll)
    assert bm25_search._cache_path(coll.name).exists()

    bm25_search.invalidate()
    assert not bm25_search._cache_path(coll.name).exists()


# ── df counters + compaction (B1) ───────────────────────────────────────────

def _small_index():
    idx = bm25_search.BM25Index()
    idx.build(
        ["a", "b", "c"],
        ["antenna optimization deep learning",
         "antenna design microstrip",
         "quantum error correction"],
    )
    return idx


def test_df_counters_track_tombstones():
    """df used to be recomputed per query term by walking that term's whole
    posting list against the tombstone set."""
    idx = _small_index()
    assert idx.df["antenna"] == 2
    assert idx.df["quantum"] == 1

    idx.remove_documents(["b"])
    assert idx.df["antenna"] == 1, "a tombstoned doc must stop counting toward df"
    assert idx.df["quantum"] == 1


def test_readding_an_id_does_not_double_count_df():
    idx = _small_index()
    idx.add_documents(["a"], ["antenna antenna antenna"])
    assert idx.df["antenna"] == 2, "the replaced copy must be tombstoned, not counted twice"


def test_compaction_reclaims_slots_and_preserves_search():
    idx = _small_index()
    before_ids, _ = idx.search("antenna")
    assert set(before_ids) == {"a", "b"}

    idx.remove_documents(["b"])
    assert idx.deleted_ratio > 0
    freed = idx.compact()

    assert freed == 1
    assert idx._deleted == set()
    assert idx.doc_ids == ["a", "c"]
    assert idx.n_docs == 2
    after_ids, _ = idx.search("antenna")
    assert after_ids == ["a"], "compaction must not change what the index returns"
    assert idx.search("quantum")[0] == ["c"], "surviving docs must keep their postings"


def test_compaction_of_a_clean_index_is_a_no_op():
    idx = _small_index()
    assert idx.compact() == 0
    assert idx.doc_ids == ["a", "b", "c"]


def test_rebuild_derived_recovers_df_after_a_cache_load():
    """df and doc_terms are not persisted — the loader reconstructs them."""
    idx = _small_index()
    idx.remove_documents(["c"])
    idx.df = {}
    idx.doc_terms = []
    idx._rebuild_derived()

    assert idx.df["antenna"] == 2
    assert idx.df["quantum"] == 0, "a tombstoned doc must not be counted on reload"
