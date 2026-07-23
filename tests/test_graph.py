"""Task 3.3 — knowledge/citation graph (Phase 1): edge persistence + co-citation."""

import pytest

import persistence


@pytest.fixture(autouse=True)
def _clear_edges():
    with persistence._db_lock:
        persistence._conn.execute("DELETE FROM graph_edges")
        persistence._conn.commit()
    yield


def test_save_and_get_all_edges():
    persistence.save_graph_edge("Paper A", "Paper B", "co_citation", 1.0)
    edges = persistence.get_all_edges()
    assert len(edges) == 1
    e = edges[0]
    assert {e["source"], e["target"]} == {"Paper A", "Paper B"}
    assert e["type"] == "co_citation"
    assert e["score"] == 1.0


def test_repeated_edge_accumulates_score_not_duplicates():
    """Same query cited twice must not create two rows — score accumulates."""
    for _ in range(3):
        persistence.save_graph_edge("Paper A", "Paper B", "co_citation", 1.0)
    edges = persistence.get_all_edges()
    assert len(edges) == 1
    assert edges[0]["score"] == 3.0


def test_edges_are_undirected():
    """(A,B) and (B,A) are the same edge."""
    persistence.save_graph_edge("Paper B", "Paper A", "co_citation", 1.0)
    persistence.save_graph_edge("Paper A", "Paper B", "co_citation", 1.0)
    assert len(persistence.get_all_edges()) == 1


def test_different_edge_types_are_distinct():
    persistence.save_graph_edge("Paper A", "Paper B", "co_citation", 1.0)
    persistence.save_graph_edge("Paper A", "Paper B", "contradiction", 0.9)
    assert len(persistence.get_all_edges()) == 2


def test_get_paper_edges_filters_by_paper():
    persistence.save_graph_edge("Paper A", "Paper B", "co_citation", 1.0)
    persistence.save_graph_edge("Paper C", "Paper D", "co_citation", 1.0)
    a_edges = persistence.get_paper_edges("Paper A")
    assert len(a_edges) == 1
    assert {a_edges[0]["source"], a_edges[0]["target"]} == {"Paper A", "Paper B"}
    assert a_edges[0]["metadata"] == {}


def test_save_graph_edges_batch():
    persistence.save_graph_edges([
        ("A", "B", "co_citation", 1.0, None),
        ("B", "C", "co_citation", 1.0, {"note": "x"}),
    ])
    assert len(persistence.get_all_edges()) == 2


def test_extract_co_citations_pairs_cited_papers():
    import rag
    citations = [{"title": "Paper A"}, {"title": "Paper B"}, {"title": "Paper C"}]
    pairs = rag.extract_co_citations(citations)
    assert set(map(frozenset, pairs)) == {
        frozenset({"Paper A", "Paper B"}),
        frozenset({"Paper A", "Paper C"}),
        frozenset({"Paper B", "Paper C"}),
    }


def test_extract_co_citations_single_paper_has_no_pairs():
    import rag
    assert rag.extract_co_citations([{"title": "Only One"}]) == []


def test_extract_co_citations_dedups_repeat_titles():
    import rag
    pairs = rag.extract_co_citations([{"title": "A"}, {"title": "A"}, {"title": "B"}])
    assert set(map(frozenset, pairs)) == {frozenset({"A", "B"})}
