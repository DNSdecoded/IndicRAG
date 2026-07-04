"""Unit tests for metadata_enrich.py (arXiv lookup) and vector_store.find_similar_paper (dedup)."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import metadata_enrich
import vector_store


def _fake_arxiv_result(title, authors, published_year, doi=""):
    r = MagicMock()
    r.title = title
    r.authors = [MagicMock(name=a) for a in authors]
    for mock_author, name in zip(r.authors, authors):
        mock_author.name = name
    r.published = datetime(published_year, 1, 1)
    r.doi = doi
    return r


def test_enrich_from_arxiv_returns_none_for_empty_title():
    assert metadata_enrich.enrich_from_arxiv("") is None
    assert metadata_enrich.enrich_from_arxiv("   ") is None


def test_enrich_from_arxiv_matches_and_returns_metadata():
    result = _fake_arxiv_result("Attention Is All You Need", ["A. Vaswani", "N. Shazeer"], 2017, doi="10.1234/x")
    fake_client = MagicMock()
    fake_client.results.return_value = iter([result])

    with patch("arxiv.Client", return_value=fake_client), patch("arxiv.Search"):
        enriched = metadata_enrich.enrich_from_arxiv("Attention Is All You Need")

    assert enriched == {"authors": "A. Vaswani, N. Shazeer", "year": "2017", "doi": "10.1234/x"}


def test_enrich_from_arxiv_rejects_weak_title_match():
    result = _fake_arxiv_result("A Completely Unrelated Paper About Birds", ["X"], 2020)
    fake_client = MagicMock()
    fake_client.results.return_value = iter([result])

    with patch("arxiv.Client", return_value=fake_client), patch("arxiv.Search"):
        enriched = metadata_enrich.enrich_from_arxiv("Attention Is All You Need")

    assert enriched is None


def test_enrich_from_arxiv_returns_none_on_no_results():
    fake_client = MagicMock()
    fake_client.results.return_value = iter([])

    with patch("arxiv.Client", return_value=fake_client), patch("arxiv.Search"):
        assert metadata_enrich.enrich_from_arxiv("Some Paper Title") is None


def test_enrich_from_arxiv_never_raises_on_network_error():
    with patch("arxiv.Client", side_effect=RuntimeError("network down")):
        assert metadata_enrich.enrich_from_arxiv("Some Paper Title") is None


def test_find_similar_paper_detects_near_duplicate_title():
    fake_collection = MagicMock()
    fake_collection.get.return_value = {
        "metadatas": [
            {"paper_id": "paper_a", "title": "Attention Is All You Need", "year": "2017"},
            {"paper_id": "paper_b", "title": "Deep Residual Learning", "year": "2015"},
        ]
    }

    dup = vector_store.find_similar_paper(
        "Attention is all you need", year="2017", threshold=0.9, collection=fake_collection
    )
    assert dup == "paper_a"


def test_find_similar_paper_returns_none_when_no_match():
    fake_collection = MagicMock()
    fake_collection.get.return_value = {
        "metadatas": [{"paper_id": "paper_a", "title": "Deep Residual Learning", "year": "2015"}]
    }

    dup = vector_store.find_similar_paper("Attention Is All You Need", collection=fake_collection)
    assert dup is None


def test_find_similar_paper_respects_year_filter():
    fake_collection = MagicMock()
    fake_collection.get.return_value = {
        "metadatas": [{"paper_id": "paper_a", "title": "Attention Is All You Need", "year": "2019"}]
    }

    dup = vector_store.find_similar_paper(
        "Attention Is All You Need", year="2017", threshold=0.9, collection=fake_collection
    )
    assert dup is None
