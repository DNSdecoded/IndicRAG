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
