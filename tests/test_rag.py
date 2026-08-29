"""Unit tests for rag.py — citation parsing, prompt building, context formatting."""

from unittest.mock import patch


def test_run_faithfulness_returns_mean_confidence():
    from rag import _run_faithfulness

    claims = [
        {"claim": "a", "support": 0.9, "grounded": True},
        {"claim": "b", "support": 0.3, "grounded": False},
    ]
    with patch("verify.check_claims", return_value=claims):
        result = _run_faithfulness("answer text", ["chunk"], [{"title": "P"}])

    assert result["claims"] == claims
    assert result["confidence"] == 0.6


def test_run_faithfulness_zero_confidence_when_no_claims():
    from rag import _run_faithfulness

    with patch("verify.check_claims", return_value=[]):
        result = _run_faithfulness("answer text", ["chunk"], [{"title": "P"}])

    assert result == {"claims": [], "confidence": 0.0}


def test_compare_papers_builds_matrix():
    """compare_papers must scope retrieval per paper and ask one extraction
    per dimension, keyed matrix[paper_id][dimension]."""
    from rag import compare_papers

    scoped_by_paper = {
        "p1": {"chunks": ["p1 text"], "metadatas": [{"paper_id": "p1"}],
               "formatted_context": "[1] Paper One - body:\np1 text\n", "chunks_used": 1},
        "p2": {"chunks": ["p2 text"], "metadatas": [{"paper_id": "p2"}],
               "formatted_context": "[1] Paper Two - body:\np2 text\n", "chunks_used": 1},
    }

    def fake_retrieve_scoped(filter_dict, collection):
        pid = filter_dict["paper_id"]["$in"][0]
        return scoped_by_paper[pid]

    def fake_llm_generate(prompt, max_tokens=None, model=None):
        # Echo back which paper/dimension the prompt was built for, so the
        # test can assert the matrix cell actually came from the right call.
        return f"answer for {prompt.splitlines()[-1]}"

    with patch("rag._retrieve_scoped", side_effect=fake_retrieve_scoped), \
         patch("rag.llm_generate", side_effect=fake_llm_generate), \
         patch("vector_store.get_or_create_collection", return_value=object()):
        result = compare_papers(["p1", "p2"], ["methodology", "dataset"])

    assert result["dimensions"] == ["methodology", "dataset"]
    assert set(result["matrix"].keys()) == {"p1", "p2"}
    assert "p1 text" in result["matrix"]["p1"]["methodology"]
    assert "p2 text" in result["matrix"]["p2"]["dataset"]


def test_compare_papers_missing_paper_marks_na_without_llm_call():
    """A paper_id with no chunks in the corpus must not burn an LLM call per
    dimension — every cell for it is N/A."""
    from rag import compare_papers

    with patch("rag._retrieve_scoped", return_value={"chunks": [], "metadatas": [],
                                                       "formatted_context": "", "chunks_used": 0}), \
         patch("rag.llm_generate") as mock_llm, \
         patch("vector_store.get_or_create_collection", return_value=object()):
        result = compare_papers(["missing_paper"], ["methodology"])

    assert result["matrix"]["missing_paper"]["methodology"] == "N/A — paper not found in corpus"
    mock_llm.assert_not_called()


def test_extract_citations_cite_format():
    from rag import extract_citations

    metadatas = [
        {"title": "Paper A", "section": "intro"},
        {"title": "Paper B", "section": "body"},
    ]
    result = extract_citations("text [1] more [2]", metadatas)
    assert len(result) == 2
    assert result[0]["number"] == "1"
    assert result[1]["number"] == "2"
    assert result[0]["title"] == "Paper A"
    assert result[1]["title"] == "Paper B"


def test_compact_citations_closes_numbering_gaps():
    """Citing papers 1 and 4 of 4 must render as [1],[2] against a 2-entry panel."""
    from rag import compact_citations

    metadatas = [
        {"title": "Paper A", "section": "intro"},
        {"title": "Paper B", "section": "body"},
        {"title": "Paper C", "section": "body"},
        {"title": "Paper D", "section": "results"},
    ]
    answer, cites = compact_citations("reward [1] and bandwidth [4], both [1, 4]", metadatas)

    assert answer == "reward [1] and bandwidth [2], both [1, 2]"
    assert [c["number"] for c in cites] == ["1", "2"]
    assert [c["title"] for c in cites] == ["Paper A", "Paper D"]


def test_dangling_marker_on_its_own_line_leaves_no_blank_line():
    """A marker alone on a line must take its newline with it, not leave a blank
    line that markdown renders as a paragraph break — and must not splice the
    surrounding lines together either."""
    from rag import compact_citations

    metadatas = [{"title": "Paper A", "section": "intro"}]
    answer, _ = compact_citations("first line\n[99]\nsecond line", metadatas)

    assert answer == "first line\nsecond line"


def test_citations_ignore_papers_the_prompt_never_showed():
    """A marker past the prompt's truncation point must not resolve to a real paper.

    format_context truncates by chunk count and by length, but callers hold the
    FULL retrieved metadata. Without visible_chunks, an invented [3] resolves to
    Paper C — a paper the model was never shown — producing a citation that looks
    legitimate. Numbering only the visible slice makes it dangle, so it is dropped.
    """
    from rag import compact_citations, extract_citations

    metadatas = [
        {"title": "Paper A", "section": "intro"},
        {"title": "Paper B", "section": "body"},
        {"title": "Paper C", "section": "results"},  # truncated out of the prompt
    ]
    answer = "grounded [1] and invented [3]"

    # Only the first two chunks reached the prompt.
    compacted, cites = compact_citations(answer, metadatas, visible_chunks=2)
    assert [c["title"] for c in cites] == ["Paper A"]
    assert compacted == "grounded [1] and invented"

    # Same call without the slice is what produced the phantom citation.
    assert [c["title"] for c in extract_citations(answer, metadatas)] == ["Paper A", "Paper C"]


def test_compact_citations_drops_dangling_marker():
    """A number the model invented resolves to no paper — drop it, don't renumber around it."""
    from rag import compact_citations

    metadatas = [{"title": "Paper A", "section": "intro"}]
    answer, cites = compact_citations("grounded [1] invented [7] and mid [7] sentence", metadatas)

    # dropping a marker must not leave a trailing or doubled space behind
    assert answer == "grounded [1] invented and mid sentence"
    assert [c["number"] for c in cites] == ["1"]


def test_extract_citations_no_false_match():
    from rag import extract_citations

    # Range notation like "[10-15] mg" must NOT produce citations
    result = extract_citations("[10-15] mg dose was administered", [])
    assert result == []


def test_build_prompt_returns_string():
    from rag import build_prompt

    result = build_prompt("What is BERT?", "some retrieved context", "en")
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_paper_filter_scopes_to_ids():
    from routes.query import build_paper_filter

    assert build_paper_filter(["p1", "p2"]) == {"paper_id": {"$in": ["p1", "p2"]}}
    # Whitespace trimmed, blanks dropped
    assert build_paper_filter([" p1 ", "", "  "]) == {"paper_id": {"$in": ["p1"]}}


def test_build_paper_filter_none_when_empty():
    from routes.query import build_paper_filter

    # No scope → None means whole-corpus retrieval (unchanged default behaviour)
    assert build_paper_filter(None) is None
    assert build_paper_filter([]) is None
    assert build_paper_filter(["", "   "]) is None


def test_build_tags_filter_parses_comma_separated():
    from routes.query import build_tags_filter
    import rag

    assert build_tags_filter("transformer, efficiency") == {
        rag._TAGS_SENTINEL: ["transformer", "efficiency"]
    }


def test_build_tags_filter_none_when_blank():
    from routes.query import build_tags_filter

    assert build_tags_filter(None) is None
    assert build_tags_filter("") is None
    assert build_tags_filter("  ,  ") is None


def test_combine_filters_ands_multiple_present():
    from routes.query import combine_filters

    assert combine_filters({"paper_id": {"$in": ["p1"]}}, {"tags": {"$in": ["t1"]}}) == {
        "$and": [{"paper_id": {"$in": ["p1"]}}, {"tags": {"$in": ["t1"]}}]
    }
    assert combine_filters({"paper_id": {"$in": ["p1"]}}, None) == {"paper_id": {"$in": ["p1"]}}
    assert combine_filters(None, None) is None


def test_extract_tags_post_filter_sentinel_only():
    from rag import _extract_tags_post_filter, _TAGS_SENTINEL

    chroma_safe, tags = _extract_tags_post_filter({_TAGS_SENTINEL: ["a", "b"]})
    assert chroma_safe is None
    assert tags == ["a", "b"]


def test_extract_tags_post_filter_combined_with_and():
    """paper_id must resurface as a top-level key after extraction, so the
    paper-scoped retrieval branch (`'paper_id' in filter_dict`) still fires."""
    from rag import _extract_tags_post_filter, _TAGS_SENTINEL

    combined = {"$and": [{"paper_id": {"$in": ["p1"]}}, {_TAGS_SENTINEL: ["a"]}]}
    chroma_safe, tags = _extract_tags_post_filter(combined)
    assert chroma_safe == {"paper_id": {"$in": ["p1"]}}
    assert tags == ["a"]


def test_extract_tags_post_filter_no_sentinel_passthrough():
    from rag import _extract_tags_post_filter

    original = {"year": {"$gte": "2020"}}
    chroma_safe, tags = _extract_tags_post_filter(original)
    assert chroma_safe == original
    assert tags is None


def test_apply_tags_post_filter_matches_multi_tag_comma_joined_paper():
    """The actual regression this card exists to fix: PATCH /papers stores tags
    as one unsplit string (e.g. 'transformer,efficiency'), so a paper tagged with
    2+ tags must still match a filter for just one of them."""
    from rag import _apply_tags_post_filter

    results = {
        "chunks": ["c1", "c2"],
        "metadatas": [{"tags": "transformer,efficiency"}, {"tags": "unrelated"}],
        "distances": [0.1, 0.2],
    }
    filtered = _apply_tags_post_filter(results, ["transformer"])
    assert filtered["chunks"] == ["c1"]
    assert filtered["metadatas"] == [{"tags": "transformer,efficiency"}]
    assert filtered["distances"] == [0.1]


def test_apply_tags_post_filter_no_match_returns_empty():
    from rag import _apply_tags_post_filter

    results = {"chunks": ["c1"], "metadatas": [{"tags": "unrelated"}], "distances": [0.1]}
    filtered = _apply_tags_post_filter(results, ["transformer"])
    assert filtered["chunks"] == []


class _ScopedFakeCollection:
    """Returns all stored chunks for a where-filter; ignores include details."""
    def __init__(self, rows):
        self._rows = rows  # list of (id, document, metadata)

    def get(self, where=None, include=None, **kw):
        return {
            "ids": [r[0] for r in self._rows],
            "documents": [r[1] for r in self._rows],
            "metadatas": [r[2] for r in self._rows],
        }


def test_retrieve_scoped_returns_all_in_document_order():
    from rag import _retrieve_scoped

    # Chunks arrive out of order (2, 0, 1) — scoped retrieval must reorder by chunk_index
    rows = [
        ("p1_2", "third",  {"paper_id": "p1", "chunk_index": 2, "title": "T", "section": "results"}),
        ("p1_0", "first",  {"paper_id": "p1", "chunk_index": 0, "title": "T", "section": "intro"}),
        ("p1_1", "second", {"paper_id": "p1", "chunk_index": 1, "title": "T", "section": "methods"}),
    ]
    out = _retrieve_scoped({"paper_id": {"$in": ["p1"]}}, _ScopedFakeCollection(rows))

    assert out["chunks_used"] == 3  # exhaustive: every chunk kept, not top-k
    assert out["chunks"] == ["first", "second", "third"]  # document order


def test_retrieve_scoped_empty():
    from rag import _retrieve_scoped

    out = _retrieve_scoped({"paper_id": {"$in": ["missing"]}}, _ScopedFakeCollection([]))
    assert out["chunks_used"] == 0
    assert out["chunks"] == []


def test_format_context_uses_cite_labels():
    from rag import format_context

    chunks = ["chunk text here"]
    metadatas = [{"title": "My Paper", "section": "introduction"}]
    context, count = format_context(chunks, metadatas)
    assert "[1]" in context
    assert count == 1
    assert "My Paper" in context


def test_format_context_numbers_by_paper_not_chunk():
    """Chunks from the same paper must share one citation number."""
    from rag import format_context

    chunks = ["a1", "a2", "b1"]
    metadatas = [
        {"title": "Paper A", "section": "intro"},
        {"title": "Paper A", "section": "methods"},
        {"title": "Paper B", "section": "body"},
    ]
    context, count = format_context(chunks, metadatas)
    assert count == 3
    # Both Paper A chunks are [1], Paper B is [2] — not [1,2,3]
    assert context.count("[1]") == 2
    assert "[2]" in context
    assert "[3]" not in context


def test_extract_citations_maps_by_paper():
    """[N] must resolve to the Nth unique paper, matching format_context numbering."""
    from rag import extract_citations

    metadatas = [
        {"title": "Paper A", "section": "intro"},
        {"title": "Paper A", "section": "methods"},
        {"title": "Paper B", "section": "body"},
    ]
    result = extract_citations("see [2] for details", metadatas)
    assert len(result) == 1
    assert result[0]["number"] == "2"
    assert result[0]["title"] == "Paper B"


def test_extract_citations_comma_separated():
    """Comma-separated markers like [1, 2] must produce one citation per number."""
    from rag import extract_citations

    metadatas = [
        {"title": "Paper A", "section": "intro"},
        {"title": "Paper B", "section": "methods"},
    ]
    result = extract_citations("see [1, 2] for details", metadatas)
    assert len(result) == 2
    assert result[0]["number"] == "1"
    assert result[0]["title"] == "Paper A"
    assert result[1]["number"] == "2"
    assert result[1]["title"] == "Paper B"


def _fake_search_results(n_untagged: int, n_tagged: int, tag: str = "thz"):
    """Dense results where the tagged chunks rank last — the starvation shape."""
    docs = [f"untagged {i}" for i in range(n_untagged)] + [f"tagged {i}" for i in range(n_tagged)]
    metas = ([{"title": f"U{i}", "section": "body"} for i in range(n_untagged)]
             + [{"title": f"T{i}", "section": "body", "tags": f"{tag},review"} for i in range(n_tagged)])
    return {
        "ids": [f"id{i}" for i in range(len(docs))],
        "documents": docs,
        "metadatas": metas,
        "distances": [0.1 * (i + 1) for i in range(len(docs))],
    }


def test_tags_filter_overfetches_so_low_ranked_tagged_chunks_survive():
    """Regression: the tags post-filter runs after retrieval, so fetching exactly
    top_k returned zero results whenever tagged papers ranked below the cut."""
    import config
    import rag

    captured = {}

    def fake_search(query_embedding, top_k, filter_dict, collection):
        captured["top_k"] = top_k
        return _fake_search_results(n_untagged=12, n_tagged=1)

    with patch("vector_store.search", side_effect=fake_search), \
         patch("vector_store.get_or_create_collection", return_value=object()), \
         patch("embeddings.embed_query", return_value=[0.0]), \
         patch.object(config, "USE_HYBRID_SEARCH", False), \
         patch.object(config, "USE_RERANKER", False), \
         patch.object(config, "USE_COLBERT_RERANK", False):
        result = rag.retrieve_context(
            "q", top_k=5, filter_dict={rag._TAGS_SENTINEL: ["thz"]}
        )

    assert captured["top_k"] > 5, "must widen the fetch when a tags post-filter is active"
    assert result["chunks_used"] == 1
    assert result["chunks"] == ["tagged 0"]


def test_tags_filter_result_never_exceeds_requested_top_k():
    import config
    import rag

    with patch("vector_store.search", return_value=_fake_search_results(0, 9)), \
         patch("vector_store.get_or_create_collection", return_value=object()), \
         patch("embeddings.embed_query", return_value=[0.0]), \
         patch.object(config, "USE_HYBRID_SEARCH", False), \
         patch.object(config, "USE_RERANKER", False), \
         patch.object(config, "USE_COLBERT_RERANK", False):
        result = rag.retrieve_context("q", top_k=3, filter_dict={rag._TAGS_SENTINEL: ["thz"]})

    assert len(result["chunks"]) == 3
    assert len(result["metadatas"]) == 3
    assert len(result["distances"]) == 3


def test_untagged_query_fetch_width_unchanged():
    """No tags → the fetch width must stay exactly top_k (no extra Chroma work)."""
    import config
    import rag

    captured = {}

    def fake_search(query_embedding, top_k, filter_dict, collection):
        captured["top_k"] = top_k
        return _fake_search_results(n_untagged=4, n_tagged=0)

    with patch("vector_store.search", side_effect=fake_search), \
         patch("vector_store.get_or_create_collection", return_value=object()), \
         patch("embeddings.embed_query", return_value=[0.0]), \
         patch.object(config, "USE_HYBRID_SEARCH", False), \
         patch.object(config, "USE_RERANKER", False), \
         patch.object(config, "USE_COLBERT_RERANK", False):
        rag.retrieve_context("q", top_k=5, filter_dict={"year": {"$gte": "2020"}})

    assert captured["top_k"] == 5


def test_retrieval_cache_actually_stores_and_hits():
    """Regression: the store step re-tested `collection is None` after the local
    had already been assigned a collection, so nothing was ever cached and every
    repeat query paid a full embed + search + rerank."""
    import config
    import rag
    from cache import retrieval_cache

    retrieval_cache.invalidate()
    searches = []

    def fake_search(**kwargs):
        searches.append(1)
        return _fake_search_results(3, 0)

    with patch("vector_store.search", side_effect=fake_search), \
         patch("vector_store.get_or_create_collection", return_value=object()), \
         patch("embeddings.embed_query", return_value=[0.0]), \
         patch.object(config, "USE_HYBRID_SEARCH", False), \
         patch.object(config, "USE_RERANKER", False), \
         patch.object(config, "USE_COLBERT_RERANK", False):
        rag.retrieve_context("repeat me", top_k=3)
        rag.retrieve_context("repeat me", top_k=3)

    assert len(searches) == 1, "second identical query must be served from the cache"
    assert retrieval_cache.stats["size"] == 1
    retrieval_cache.invalidate()


def test_retrieval_cache_not_used_for_explicit_collection():
    """A caller-supplied collection stays uncached: the entry would key off
    caller state that a post-ingest invalidation cannot reason about."""
    import config
    import rag
    from cache import retrieval_cache

    retrieval_cache.invalidate()
    with patch("vector_store.search", side_effect=lambda **kw: _fake_search_results(3, 0)), \
         patch("vector_store.get_or_create_collection", return_value=object()), \
         patch("embeddings.embed_query", return_value=[0.0]), \
         patch.object(config, "USE_HYBRID_SEARCH", False), \
         patch.object(config, "USE_RERANKER", False), \
         patch.object(config, "USE_COLBERT_RERANK", False):
        rag.retrieve_context("q", top_k=3, collection=object())

    assert retrieval_cache.stats["size"] == 0
    retrieval_cache.invalidate()


def test_filtered_query_is_cached_and_keyed_by_its_filter():
    """C4: a filtered repeat must hit the cache, and must not collide with the
    same question asked unfiltered — the key already hashes the filter."""
    import config
    import rag
    from cache import retrieval_cache

    retrieval_cache.invalidate()
    searches = []

    def fake_search(**kw):
        searches.append(kw)
        return _fake_search_results(3, 0)

    with patch("vector_store.search", side_effect=fake_search), \
         patch("vector_store.get_or_create_collection", return_value=object()), \
         patch("embeddings.embed_query", return_value=[0.0]), \
         patch.object(config, "USE_HYBRID_SEARCH", False), \
         patch.object(config, "USE_RERANKER", False), \
         patch.object(config, "USE_COLBERT_RERANK", False):
        rag.retrieve_context("q", top_k=3, filter_dict={"year": {"$gte": "2020"}})
        rag.retrieve_context("q", top_k=3, filter_dict={"year": {"$gte": "2020"}})
        assert len(searches) == 1, "filtered repeat must be served from the cache"

        rag.retrieve_context("q", top_k=3)
        assert len(searches) == 2, "unfiltered query must not read the filtered entry"

    assert retrieval_cache.stats["size"] == 2
    retrieval_cache.invalidate()


def test_paper_scoped_query_is_cached():
    """C4: paper-scoped retrieval is the exhaustive path — the expensive one —
    and used to bypass the cache entirely by returning early."""
    import rag
    from cache import retrieval_cache

    retrieval_cache.invalidate()
    calls = []

    def fake_scoped(chroma_filter_dict, collection):
        calls.append(chroma_filter_dict)
        return {"chunks": ["c"], "metadatas": [{"title": "P"}], "distances": [0.1],
                "formatted_context": "ctx", "chunks_used": 1}

    with patch("vector_store.get_or_create_collection", return_value=object()), \
         patch.object(rag, "_retrieve_scoped", side_effect=fake_scoped):
        first = rag.retrieve_context("q", top_k=3, filter_dict={"paper_id": "p1"})
        second = rag.retrieve_context("q", top_k=3, filter_dict={"paper_id": "p1"})

    assert len(calls) == 1, "scoped repeat must be served from the cache"
    assert second["chunks"] == first["chunks"]
    assert second["chunks"] is not first["chunks"], "cache must hand out copies"
    retrieval_cache.invalidate()


def test_retrieval_cache_hit_returns_independent_lists():
    """A caller trimming the returned lists must not corrupt the cached entry."""
    import config
    import rag
    from cache import retrieval_cache

    retrieval_cache.invalidate()
    # side_effect (not return_value): a shared mock dict would let the first
    # caller's mutation come back through the mock itself, hiding the real
    # question — whether the *cache* handed out an aliased list.
    with patch("vector_store.search", side_effect=lambda **kw: _fake_search_results(4, 0)), \
         patch("vector_store.get_or_create_collection", return_value=object()), \
         patch("embeddings.embed_query", return_value=[0.0]), \
         patch.object(config, "USE_HYBRID_SEARCH", False), \
         patch.object(config, "USE_RERANKER", False), \
         patch.object(config, "USE_COLBERT_RERANK", False):
        first = rag.retrieve_context("cache probe", top_k=4)
        del first["chunks"][:]                      # hostile caller
        second = rag.retrieve_context("cache probe", top_k=4)

    assert len(second["chunks"]) == 4, "cache entry was mutated by a previous caller"
    retrieval_cache.invalidate()


def test_extract_citations_comma_no_spaces():
    """No-space variant [1,2] must also work."""
    from rag import extract_citations

    metadatas = [
        {"title": "Paper A", "section": "intro"},
        {"title": "Paper B", "section": "methods"},
    ]
    result = extract_citations("results in [1,2] above", metadatas)
    assert len(result) == 2
    assert result[0]["number"] == "1"
    assert result[0]["title"] == "Paper A"
    assert result[1]["number"] == "2"
    assert result[1]["title"] == "Paper B"
