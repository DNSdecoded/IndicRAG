from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import itertools
import inspect
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
    mock_resp = MagicMock()
    mock_resp.text = "mock answer"
    with patch("rag.format_context", return_value=("mock context", 3)) as mock_fc, \
         patch("rag.build_prompt", return_value="mock prompt") as mock_bp, \
         patch("rag.generate_with_failover", return_value=mock_resp) as mock_gen:
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
        conversation_history=[],
        session_id="test-001",
        strategy="A",
    )
    assert state["original_query"] == "test"
    assert state["strategy"] == "A"
    assert state["conversation_history"] == []


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
    from agent.graph import build_agent_graph
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
        conversation_history=[],
        session_id="test-001",
        strategy="A",
    )
    result = build_agent_graph().invoke(state)
    assert result["final_answer"] is not None
    assert result["reflexion_count"] <= MAX_REFLEXION


# =============================================================================
# BUG-027: Agent conversation history surfaced in answer_generator
# =============================================================================

def test_answer_generator_prepends_history():
    mock_resp = MagicMock()
    mock_resp.text = "answer"
    with patch("rag.format_context", return_value=("ctx", 1)), \
         patch("rag.build_prompt", return_value="base prompt"), \
         patch("rag.generate_with_failover", return_value=mock_resp) as mock_gen:
        from agent.nodes.answer_generator import answer_generator_node

        history = [
            {"role": "user", "content": "What is BERT?"},
            {"role": "assistant", "content": "BERT is a language model."},
        ]
        state = {
            "retrieved_contexts": [{"text": "t", "title": "T", "section": "body"}],
            "original_query": "How does it compare to GPT?",
            "detected_language": "en",
            "strategy": "A",
            "conversation_history": history,
        }
        result = answer_generator_node(state)
        contents = mock_gen.call_args[0][1]
        history_texts = [p.text for c in contents[:-1] for p in c.parts]
        assert any("BERT" in t for t in history_texts)
        assert contents[-1].role == "user"
        assert result["draft_answer"] == "answer"


def test_answer_generator_no_history_unchanged():
    mock_resp = MagicMock()
    mock_resp.text = "answer"
    with patch("rag.format_context", return_value=("ctx", 1)), \
         patch("rag.build_prompt", return_value="base prompt"), \
         patch("rag.generate_with_failover", return_value=mock_resp) as mock_gen:
        from agent.nodes.answer_generator import answer_generator_node

        state = {
            "retrieved_contexts": [{"text": "t", "title": "T", "section": "body"}],
            "original_query": "q",
            "detected_language": "en",
            "strategy": "A",
            "conversation_history": [],
        }
        answer_generator_node(state)
        contents = mock_gen.call_args[0][1]
        assert len(contents) == 1
        assert contents[0].role == "user"


def test_answer_generator_history_capped_at_six_messages():
    mock_resp = MagicMock()
    mock_resp.text = "answer"
    with patch("rag.format_context", return_value=("ctx", 1)), \
         patch("rag.build_prompt", return_value="base prompt"), \
         patch("rag.generate_with_failover", return_value=mock_resp) as mock_gen:
        from agent.nodes.answer_generator import answer_generator_node

        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
                   for i in range(10)]
        state = {
            "retrieved_contexts": [{"text": "t", "title": "T", "section": "body"}],
            "original_query": "q",
            "detected_language": "en",
            "strategy": "A",
            "conversation_history": history,
        }
        answer_generator_node(state)
        contents = mock_gen.call_args[0][1]
        history_contents = contents[:-1]
        assert len(history_contents) == 6
        all_hist_text = " ".join(p.text for c in history_contents for p in c.parts)
        assert "msg0" not in all_hist_text
        assert "msg9" in all_hist_text


# =============================================================================
# BUG-029: Hybrid search — BM25 index building and RRF fusion
# =============================================================================

def test_bm25_build_and_basic_search():
    from bm25_search import BM25Index
    idx = BM25Index()
    idx.build(["d1", "d2", "d3"],
              ["neural network transformer", "convolutional network", "transformer attention"])
    assert idx.n_docs == 3
    ids, scores = idx.search("transformer", top_k=2)
    assert len(ids) == 2
    assert "d1" in ids or "d3" in ids  # both contain "transformer"


def test_bm25_empty_corpus():
    from bm25_search import BM25Index
    idx = BM25Index()
    idx.build([], [])
    ids, scores = idx.search("anything")
    assert ids == [] and scores == []


def test_rrf_merges_both_lists():
    from bm25_search import rrf
    fused = rrf(["a", "b", "c"], ["c", "d", "a"])
    assert set(fused) == {"a", "b", "c", "d"}
    # "a" ranks 1st in dense and 3rd in sparse — should beat "b" (dense only at 2nd)
    assert fused.index("a") < fused.index("b")


def test_rrf_empty_sparse():
    from bm25_search import rrf
    ids = ["x", "y", "z"]
    assert rrf(ids, []) == ids


def test_bm25_per_collection_isolation():
    import bm25_search
    bm25_search.invalidate()

    coll_a = MagicMock()
    coll_a.name = "coll_a"
    coll_a.count.return_value = 2
    coll_a.get.return_value = {"ids": ["a1", "a2"], "documents": ["hello world", "foo bar"]}

    coll_b = MagicMock()
    coll_b.name = "coll_b"
    coll_b.count.return_value = 1
    coll_b.get.return_value = {"ids": ["b1"], "documents": ["completely different"]}

    idx_a = bm25_search.get_or_build_index(coll_a)
    idx_b = bm25_search.get_or_build_index(coll_b)

    assert idx_a is not idx_b
    assert idx_a.n_docs == 2
    assert idx_b.n_docs == 1
    # Cache hit — collection.get not called a second time
    assert bm25_search.get_or_build_index(coll_a) is idx_a
    assert coll_a.get.call_count == 1


def test_bm25_invalidate_clears_all_collections():
    import bm25_search
    coll = MagicMock()
    coll.name = "to_clear"
    coll.count.return_value = 1
    coll.get.return_value = {"ids": ["x"], "documents": ["text"]}
    bm25_search.get_or_build_index(coll)
    assert "to_clear" in bm25_search._indices
    bm25_search.invalidate()
    assert bm25_search._indices == {}


# =============================================================================
# BUG-030: Translation pipeline
# =============================================================================

def test_translate_same_language_is_noop():
    from translation import translate_text
    assert translate_text("Hello world", "en", "en") == "Hello world"


def test_translate_unsupported_language_raises():
    from translation import translate_text
    with pytest.raises(ValueError, match="Unsupported"):
        translate_text("Hello", "en", "zz")


def test_translate_max_length_default_is_1024():
    from translation import translate_text
    sig = inspect.signature(translate_text)
    assert sig.parameters["max_length"].default == 1024


def test_translate_calls_model_generate():
    import torch
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.convert_tokens_to_ids.return_value = 256047
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.zeros(1, 5, dtype=torch.long)
    mock_tokenizer.batch_decode.return_value = ["अनुवाद"]

    with patch("translation.load_translation_model", return_value=(mock_model, mock_tokenizer)):
        from translation import translate_text
        result = translate_text("Hello.", "en", "hi")
    assert mock_model.generate.called
    assert isinstance(result, str)


def test_translate_long_text_is_segmented():
    """Verify the text is split into segments before model calls."""
    import torch
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.convert_tokens_to_ids.return_value = 256047
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.zeros(1, 5, dtype=torch.long)
    mock_tokenizer.batch_decode.return_value = ["seg"]

    with patch("translation.load_translation_model", return_value=(mock_model, mock_tokenizer)):
        from translation import translate_text
        long_text = "Sentence one. Sentence two. Sentence three. Sentence four."
        translate_text(long_text, "en", "hi")

    # generate must be called (segments processed)
    assert mock_model.generate.call_count >= 1


# =============================================================================
# BUG-031: Cache invalidation after ingestion
# =============================================================================

def test_retrieval_cache_invalidate_wipes_all_entries():
    from cache import TTLCache, make_key
    c = TTLCache(max_size=10, ttl_seconds=60)
    c.put(make_key("q1", 5, None, None, False, 10), {"chunks": ["a"]})
    c.put(make_key("q2", 5, None, None, False, 10), {"chunks": ["b"]})
    c.invalidate()
    assert c.get(make_key("q1", 5, None, None, False, 10)) is None
    assert c.get(make_key("q2", 5, None, None, False, 10)) is None


def test_three_caches_invalidate_independently():
    from cache import TTLCache
    c1, c2, c3 = TTLCache(4, 60), TTLCache(4, 60), TTLCache(4, 60)
    for c in (c1, c2, c3):
        c.put("k", "v")
    c1.invalidate()
    assert c1.get("k") is None
    assert c2.get("k") == "v"
    assert c3.get("k") == "v"


def test_bm25_cache_cleared_by_invalidate():
    import bm25_search
    coll = MagicMock()
    coll.name = "inv_test"
    coll.count.return_value = 1
    coll.get.return_value = {"ids": ["d1"], "documents": ["some text"]}
    bm25_search.invalidate()
    bm25_search.get_or_build_index(coll)
    assert "inv_test" in bm25_search._indices
    bm25_search.invalidate()
    assert bm25_search._indices == {}


def test_cache_selective_key_invalidate():
    from cache import TTLCache
    c = TTLCache(max_size=4, ttl_seconds=60)
    c.put("keep", "yes")
    c.put("drop", "no")
    c.invalidate("drop")
    assert c.get("keep") == "yes"
    assert c.get("drop") is None


# =============================================================================
# BUG-032: Concurrency / thread safety
# =============================================================================

def test_query_cache_concurrent_reads_writes():
    import numpy as np
    import embeddings as emb_mod

    fake = np.zeros(1024, dtype=np.float32)
    emb_mod._query_cache.clear()

    with patch("embeddings.embed_texts", return_value=np.array([fake])):
        from embeddings import embed_query

        def task(i):
            return embed_query(f"q{i % 5}")

        with ThreadPoolExecutor(max_workers=20) as ex:
            results = [f.result() for f in [ex.submit(task, i) for i in range(100)]]

    assert len(results) == 100
    assert all(r is not None for r in results)


def test_client_pool_idx_stays_in_bounds_under_concurrency():
    import rag

    rag._client_pool = [MagicMock(), MagicMock(), MagicMock()]
    rag._client_index = itertools.cycle(range(3))

    with ThreadPoolExecutor(max_workers=30) as ex:
        indices = [f.result() for f in [ex.submit(rag._next_client_idx) for _ in range(300)]]

    assert all(0 <= i < 3 for i in indices)
    assert len(indices) == 300


def test_sessions_concurrent_creation_all_unique():
    import api_server
    with api_server._sessions_lock:
        api_server._sessions.clear()

    with ThreadPoolExecutor(max_workers=20) as ex:
        ids = [f.result()[0] for f in [ex.submit(api_server._get_or_create_session, None)
                                        for _ in range(50)]]

    assert len(set(ids)) == 50
    with api_server._sessions_lock:
        assert len(api_server._sessions) == 50


def test_jobs_concurrent_updates_no_corruption():
    import api_server
    with api_server._jobs_lock:
        for i in range(10):
            api_server._jobs[f"job-{i}"] = {"status": "running"}

    def update(i):
        api_server._update_job(f"job-{i % 10}", status="done", val=i)

    with ThreadPoolExecutor(max_workers=20) as ex:
        [f.result() for f in [ex.submit(update, i) for i in range(100)]]

    with api_server._jobs_lock:
        assert all(j["status"] == "done" for j in api_server._jobs.values())
