"""Unit tests for rag.py — citation parsing, prompt building, context formatting."""
from unittest.mock import patch


def test_extract_citations_cite_format():
    from rag import extract_citations

    metadatas = [
        {"title": "Paper A", "section": "intro"},
        {"title": "Paper B", "section": "body"},
    ]
    result = extract_citations("text [Cite:1] more [Cite:2]", metadatas)
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


def test_format_context_uses_cite_labels():
    from rag import format_context

    chunks = ["chunk text here"]
    metadatas = [{"title": "My Paper", "section": "introduction"}]
    context, count = format_context(chunks, metadatas)
    assert "[Cite:1]" in context
    assert count == 1
    assert "My Paper" in context
