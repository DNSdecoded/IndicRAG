from unittest.mock import patch, MagicMock
import pytest


def test_hard_stop_at_max_iterations():
    from agent.nodes.reflexion_evaluator import reflexion_evaluator_node, MAX_REFLEXION

    state = {
        "reflexion_count": MAX_REFLEXION,
        "draft_answer": "partial answer",
        "reflexion_history": [],
        "original_query": "test",
        "retrieved_contexts": [],
    }
    result = reflexion_evaluator_node(state)
    assert "final_answer" in result, "Must finalise — never loop past MAX_REFLEXION"


def test_tool_executor_dispatches_correctly():
    from agent.tool_executor import execute_calculate

    result = execute_calculate("2 + 2")
    assert "4" in result["text"]


def test_answer_generator_reuses_rag_functions():
    with patch("rag.format_context", return_value=("mock context", 3)) as mock_fc, \
         patch("rag.build_prompt", return_value="mock prompt") as mock_bp, \
         patch("rag.llm_generate", return_value="mock answer") as mock_gen:
        from agent.nodes.answer_generator import answer_generator_node

        state = {
            "retrieved_contexts": [{"text": "t", "title": "T", "section": "body"}],
            "original_query": "q",
            "detected_language": "en",
            "strategy": "A",
        }
        result = answer_generator_node(state)
        mock_fc.assert_called_once()
        mock_bp.assert_called_once()
        mock_gen.assert_called_once()
        assert result["draft_answer"] == "mock answer"


def test_state_schema_fields():
    from agent.state import AgentState

    state = AgentState(
        original_query="test",
        detected_language="en",
        query_plan=[],
        tool_calls_requested=[],
        retrieved_contexts=[],
        draft_answer=None,
        final_answer=None,
        reflexion_count=0,
        reflexion_history=[],
        tool_calls_log=[],
        session_id="test-001",
        strategy="A",
    )
    assert state["original_query"] == "test"
    assert state["strategy"] == "A"


def test_finalizer_uses_final_answer():
    from agent.nodes.finalizer import finalizer_node

    state = {"final_answer": "  the answer  ", "draft_answer": "draft"}
    result = finalizer_node(state)
    assert result["final_answer"] == "the answer"


def test_finalizer_falls_back_to_draft():
    from agent.nodes.finalizer import finalizer_node

    state = {"final_answer": None, "draft_answer": "draft answer"}
    result = finalizer_node(state)
    assert result["final_answer"] == "draft answer"


def test_route_reflexion_accept():
    from agent.graph import _route_reflexion

    state = {
        "reflexion_count": 1,
        "final_answer": None,
        "reflexion_history": [{"action": "accept"}],
    }
    assert _route_reflexion(state) == "finalizer"


def test_route_reflexion_retrieve_more():
    from agent.graph import _route_reflexion

    state = {
        "reflexion_count": 1,
        "final_answer": None,
        "reflexion_history": [{"action": "retrieve_more"}],
    }
    assert _route_reflexion(state) == "tool_selector"


def test_route_reflexion_max_count():
    from agent.graph import _route_reflexion

    state = {
        "reflexion_count": 3,
        "final_answer": None,
        "reflexion_history": [{"action": "retrieve_more"}],
    }
    assert _route_reflexion(state) == "finalizer"


@pytest.mark.network
def test_arxiv_search_returns_passages():
    from agent.tool_executor import execute_arxiv_search

    result = execute_arxiv_search("attention is all you need", max_results=2)
    assert "passages" in result
    assert len(result["passages"]) > 0
    paper = result["passages"][0]
    assert "title" in paper
    assert "source" in paper
    assert "pdf_url" in paper
    assert "arxiv" in paper["source"]


@pytest.mark.network
def test_open_access_search_returns_passages():
    from agent.tool_executor import execute_open_access_search

    result = execute_open_access_search("transformer neural network", max_results=2)
    assert "passages" in result
    assert len(result["passages"]) > 0
    paper = result["passages"][0]
    assert "title" in paper
    assert "source" in paper


@pytest.mark.network
def test_open_access_search_year_filter():
    from agent.tool_executor import execute_open_access_search

    result = execute_open_access_search(
        "large language models", max_results=3, year_range="2024-2025"
    )
    assert "passages" in result
    for paper in result["passages"]:
        assert "2024" in paper["text"] or "2025" in paper["text"] or "N/A" in paper["text"]


def test_tool_dispatch_has_new_tools():
    from agent.tool_executor import TOOL_DISPATCH

    assert "arxiv_search" in TOOL_DISPATCH
    assert "open_access_search" in TOOL_DISPATCH


def test_ttl_cache_hit_and_miss():
    from cache import TTLCache

    c = TTLCache(max_size=4, ttl_seconds=60)
    c.put("k1", "v1")
    assert c.get("k1") == "v1"
    assert c.get("k2") is None
    assert c.stats["hits"] == 1
    assert c.stats["misses"] == 1


def test_ttl_cache_expiration():
    import time as _time
    from cache import TTLCache

    c = TTLCache(max_size=4, ttl_seconds=0.1)
    c.put("k1", "v1")
    assert c.get("k1") == "v1"
    _time.sleep(0.15)
    assert c.get("k1") is None


def test_ttl_cache_lru_eviction():
    from cache import TTLCache

    c = TTLCache(max_size=3, ttl_seconds=60)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.put("d", 4)
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("d") == 4


def test_ttl_cache_invalidate():
    from cache import TTLCache

    c = TTLCache(max_size=4, ttl_seconds=60)
    c.put("k1", "v1")
    c.put("k2", "v2")
    c.invalidate("k1")
    assert c.get("k1") is None
    assert c.get("k2") == "v2"
    c.invalidate()
    assert c.get("k2") is None


def test_make_key_deterministic():
    from cache import make_key

    k1 = make_key("indicrag_retrieval", {"query": "test", "expand_query": False})
    k2 = make_key("indicrag_retrieval", {"query": "test", "expand_query": False})
    k3 = make_key("indicrag_retrieval", {"query": "different"})
    assert k1 == k2
    assert k1 != k3


@pytest.mark.integration
def test_full_graph_terminates():
    from agent.graph import agent_graph
    from agent.state import AgentState
    from agent.nodes.reflexion_evaluator import MAX_REFLEXION

    state = AgentState(
        original_query="What is the main finding of the paper?",
        detected_language="",
        query_plan=[],
        tool_calls_requested=[],
        retrieved_contexts=[],
        draft_answer=None,
        final_answer=None,
        reflexion_count=0,
        reflexion_history=[],
        tool_calls_log=[],
        session_id="test-001",
        strategy="A",
    )
    result = agent_graph.invoke(state)
    assert result["final_answer"] is not None
    assert result["reflexion_count"] <= MAX_REFLEXION
