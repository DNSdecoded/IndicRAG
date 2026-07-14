"""Unit tests for rag.py — citation parsing, prompt building, context formatting."""


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
