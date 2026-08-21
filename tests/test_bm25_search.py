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
