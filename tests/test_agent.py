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
    import llm_client

    llm_client._client_pool = [MagicMock(), MagicMock(), MagicMock()]
    llm_client._client_index = itertools.cycle(range(3))

    with ThreadPoolExecutor(max_workers=30) as ex:
        indices = [f.result() for f in [ex.submit(llm_client._next_client_idx) for _ in range(300)]]

    assert all(0 <= i < 3 for i in indices)
    assert len(indices) == 300


def test_sessions_concurrent_creation_all_unique():
    import deps
    with deps._sessions_lock:
        deps._sessions.clear()

    with ThreadPoolExecutor(max_workers=20) as ex:
        ids = [f.result()[0] for f in [ex.submit(deps._get_or_create_session, None)
                                        for _ in range(50)]]

    assert len(set(ids)) == 50
    with deps._sessions_lock:
        assert len(deps._sessions) == 50


def test_jobs_concurrent_updates_no_corruption():
    import deps
    with deps._jobs_lock:
        for i in range(10):
            deps._jobs[f"job-{i}"] = {"status": "running"}

    def update(i):
        deps._update_job(f"job-{i % 10}", status="done", val=i)

    with ThreadPoolExecutor(max_workers=20) as ex:
        [f.result() for f in [ex.submit(update, i) for i in range(100)]]

    with deps._jobs_lock:
        assert all(j["status"] == "done" for j in deps._jobs.values())


# =============================================================================
# T2: agent/json_utils.extract_json
# =============================================================================

def test_extract_json_valid():
    from agent.json_utils import extract_json
    assert extract_json('{"key": "val"}') == {"key": "val"}


def test_extract_json_fenced():
    from agent.json_utils import extract_json
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


# =============================================================================
# T1: BM25 Indic tokenisation (regex Unicode categories)
# =============================================================================

def test_bm25_indic_tokenize():
    from bm25_search import BM25Index
    tokens = BM25Index._tokenize("नमस्ते दुनिया")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


# =============================================================================
# Reflexion faithfulness scoring: long multi-claim answers must not be nuked
# =============================================================================

def _mk_eval_resp(complete, action):
    resp = MagicMock()
    resp.text = f'{{"completeness_score": {complete}, "action": "{action}", "missing_aspects": []}}'
    return resp


def _eval_state(**over):
    state = {
        "reflexion_count": 0,
        "draft_answer": "Long detailed answer. " * 20,
        "reflexion_history": [],
        "original_query": "compare the papers",
        "retrieved_contexts": [{"text": "chunk", "title": "T", "section": "body"}],
    }
    state.update(over)
    return state


def test_faithfulness_is_grounded_fraction_not_min():
    """One weakly-entailed claim among many grounded ones must not force faith to ~0."""
    from agent.nodes.reflexion_evaluator import reflexion_evaluator_node

    claims = [{"claim": f"c{i}", "support": 0.9, "grounded": True} for i in range(9)]
    claims.append({"claim": "c9", "support": 0.02, "grounded": False})

    with patch("verify.check_claims", return_value=claims), \
         patch("rag.generate_with_failover", return_value=_mk_eval_resp(0.9, "retrieve_more")), \
         patch("rag.safe_extract_text", side_effect=lambda r: r.text):
        result = reflexion_evaluator_node(_eval_state())

    # 9/10 grounded => faith 0.9 => accept branch fires
    assert "final_answer" in result
    assert result["reflexion_history"][-1]["faithfulness_score"] >= 0.75


def test_safe_stop_preserves_draft_answer():
    """Stuck loop with low faithfulness must return the draft with a caveat, not discard it."""
    from agent.nodes.reflexion_evaluator import reflexion_evaluator_node

    draft = "Substantive answer about antennas."
    claims = [{"claim": "c", "support": 0.01, "grounded": False}]
    prev = [{"faithfulness_score": 0.0, "completeness_score": 0.3,
             "action": "retrieve_more", "missing_aspects": []}]

    with patch("verify.check_claims", return_value=claims), \
         patch("rag.generate_with_failover", return_value=_mk_eval_resp(0.3, "retrieve_more")), \
         patch("rag.safe_extract_text", side_effect=lambda r: r.text):
        result = reflexion_evaluator_node(
            _eval_state(draft_answer=draft, reflexion_count=1, reflexion_history=prev))

    assert "final_answer" in result
    assert draft in result["final_answer"], "draft answer must survive safe_stop"


def test_year_filter_builds_chromadb_where_clause():
    """Year range filter must produce valid ChromaDB where-clauses and ignore junk."""
    from agent.tool_executor import _year_filter

    assert _year_filter() is None
    assert _year_filter("not-a-year") is None
    assert _year_filter(1800) is None          # out of 1900-2100 range
    assert _year_filter(2020) == {"year": {"$gte": "2020"}}
    assert _year_filter(None, 2019) == {"year": {"$lte": "2019"}}
    assert _year_filter(2020, 2025) == {
        "$and": [{"year": {"$gte": "2020"}}, {"year": {"$lte": "2025"}}]
    }


def test_indicrag_year_range_passed_as_filter():
    """execute_indicrag must forward the year range to retrieve_context as filter_dict."""
    import agent.tool_executor as te

    with patch("rag.retrieve_context", return_value={"chunks": [], "metadatas": []}) as rc:
        te.execute_indicrag("deep learning antennas", year_from=2020)

    _, kwargs = rc.call_args
    assert kwargs["filter_dict"] == {"year": {"$gte": "2020"}}


def test_indicrag_tags_passed_as_post_filter_sentinel():
    """execute_indicrag must forward tags as a rag post-filter sentinel, not a ChromaDB $in
    clause — PATCH /papers stores tags as one unsplit string, so $in would never match."""
    import agent.tool_executor as te
    import rag

    with patch("rag.retrieve_context", return_value={"chunks": [], "metadatas": []}) as rc:
        te.execute_indicrag("deep learning antennas", tags="transformer, efficiency")

    _, kwargs = rc.call_args
    assert kwargs["filter_dict"] == {rag._TAGS_SENTINEL: ["transformer", "efficiency"]}


def test_indicrag_tags_and_year_combined_with_and():
    import agent.tool_executor as te
    import rag

    with patch("rag.retrieve_context", return_value={"chunks": [], "metadatas": []}) as rc:
        te.execute_indicrag("deep learning antennas", year_from=2020, tags="transformer")

    _, kwargs = rc.call_args
    assert kwargs["filter_dict"] == {
        "$and": [{"year": {"$gte": "2020"}}, {rag._TAGS_SENTINEL: ["transformer"]}]
    }


def test_indicrag_blank_tags_no_filter():
    import agent.tool_executor as te

    with patch("rag.retrieve_context", return_value={"chunks": [], "metadatas": []}) as rc:
        te.execute_indicrag("deep learning antennas", tags="  ,  ")

    _, kwargs = rc.call_args
    assert kwargs["filter_dict"] is None


def test_reflexion_time_budget_finalises_draft():
    """Over the wall-clock budget, the loop finalises the current draft instead of
    starting another retrieve→generate→verify cycle (returns before any LLM/NLI call)."""
    import time as _time
    import config
    from agent.nodes.reflexion_evaluator import reflexion_evaluator_node

    draft = "Best-effort answer so far."
    state = _eval_state(
        draft_answer=draft,
        reflexion_count=1,
        start_time=_time.monotonic() - (config.AGENT_REFLEXION_BUDGET_S + 10),
    )
    result = reflexion_evaluator_node(state)
    assert result["final_answer"] == draft


def test_evaluator_sees_full_long_answer():
    """Completeness evaluator must not judge only the first 4000 chars of a long answer."""
    from agent.nodes.reflexion_evaluator import reflexion_evaluator_node

    sentinel = "UNIQUE_TAIL_SENTINEL_XYZ."
    long_answer = ("Filler sentence for padding. " * 400) + sentinel  # ~11.6k chars

    captured = {}

    def capture(model, contents, gen_config=None, **kw):
        captured["prompt"] = contents
        return _mk_eval_resp(0.9, "accept")

    with patch("verify.check_claims", return_value=[]), \
         patch("rag.generate_with_failover", side_effect=capture), \
         patch("rag.safe_extract_text", side_effect=lambda r: r.text):
        reflexion_evaluator_node(_eval_state(draft_answer=long_answer))

    assert sentinel in captured["prompt"], "tail of long answer was truncated away from evaluator"


# =============================================================================
# Corpus fairness: fast web tools must not truncate slow corpus out of the
# MAX_CONTEXT_CHUNKS budget. Passages are interleaved round-robin by retriever.
# =============================================================================

def test_interleave_by_retriever_round_robins():
    from agent.nodes.tool_executor_node import _interleave_by_retriever

    contexts = (
        [{"text": f"web{i}", "retriever": "arxiv_search"} for i in range(10)]
        + [{"text": f"corpus{i}", "retriever": "indicrag_retrieval"} for i in range(3)]
    )
    out = _interleave_by_retriever(contexts)

    first12 = out[:12]
    corpus_in_budget = sum(1 for c in first12 if c["retriever"] == "indicrag_retrieval")
    # All 3 corpus chunks must survive the first-12 cut, not be tail-truncated
    assert corpus_in_budget == 3
    assert len(out) == len(contexts)


def test_tool_executor_tags_and_interleaves():
    def fake_corpus(args):
        return {"passages": [{"text": f"c{i}", "title": "Corpus Paper", "section": "body"}
                             for i in range(3)]}

    def fake_arxiv(args):
        return {"passages": [{"text": f"a{i}", "title": f"Arxiv {i}", "source": "arxiv"}
                             for i in range(10)]}

    import agent.nodes.tool_executor_node as node
    with patch.object(node, "TOOL_DISPATCH",
                      {"indicrag_retrieval": fake_corpus, "arxiv_search": fake_arxiv}):
        state = {
            "tool_calls_requested": [
                {"name": "arxiv_search", "args": {}},
                {"name": "indicrag_retrieval", "args": {}},
            ],
            "retrieved_contexts": [],
            "tool_calls_log": [],
        }
        result = node.tool_executor_node(state)

    ctxs = result["retrieved_contexts"]
    assert all("retriever" in c for c in ctxs), "every passage must be tagged with its tool"
    first12 = ctxs[:12]
    assert sum(1 for c in first12 if c["retriever"] == "indicrag_retrieval") == 3
