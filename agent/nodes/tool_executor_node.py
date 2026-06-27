import hashlib
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.state import AgentState
from agent.tool_executor import TOOL_DISPATCH
from cache import tool_cache, make_key

logger = logging.getLogger(__name__)

_CACHEABLE_TOOLS = {"indicrag_retrieval", "arxiv_search", "open_access_search", "web_search"}
_MAX_PARALLEL_TOOLS = 4


def _run_tool(name: str, args: dict) -> tuple[str, dict, dict, float]:
    start = time.monotonic()
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        logger.warning(f"[ToolExecutor] Unknown tool: {name}")
        return name, args, {"passages": [], "error": f"Unknown tool: {name}"}, 0.0

    cache_key = make_key(name, args) if name in _CACHEABLE_TOOLS else None
    result = tool_cache.get(cache_key) if cache_key else None

    if result is not None:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(f"[ToolExecutor] {name} cache hit ({latency_ms:.0f}ms)")
    else:
        try:
            result = fn(args)
        except Exception as e:
            logger.error(f"[ToolExecutor] {name} failed: {e}", exc_info=True)
            result = {"passages": [], "error": str(e)}
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        if cache_key and result and "error" not in result:
            tool_cache.put(cache_key, result)
        logger.info(f"[ToolExecutor] {name} completed in {latency_ms:.0f}ms")

    return name, args, result, latency_ms


def _passage_key(p: dict) -> str:
    return hashlib.sha256(p.get("text", "").encode()).hexdigest()


def _collect_result(name, args, result, latency_ms, contexts, seen_hashes, log):
    if "error" not in result and "passages" in result:
        for p in result["passages"]:
            h = _passage_key(p)
            if h not in seen_hashes:
                seen_hashes.add(h)
                contexts.append(p)
    elif "error" not in result and "text" in result:
        p = {"text": result["text"], "source": name}
        h = _passage_key(p)
        if h not in seen_hashes:
            seen_hashes.add(h)
            contexts.append(p)
    log.append({"tool": name, "args": args, "latency_ms": latency_ms})


def tool_executor_node(state: AgentState) -> dict:
    tool_calls = state.get("tool_calls_requested", [])
    contexts = list(state.get("retrieved_contexts", []))
    log = list(state.get("tool_calls_log", []))

    # Seed the dedup set from passages already in state so reflexion loops
    # never re-add the same passages fetched in earlier iterations.
    seen_hashes = {_passage_key(p) for p in contexts}

    if len(tool_calls) > _MAX_PARALLEL_TOOLS:
        logger.warning(
            f"[ToolExecutor] Truncating {len(tool_calls)} tool calls to {_MAX_PARALLEL_TOOLS}"
        )
        tool_calls = tool_calls[:_MAX_PARALLEL_TOOLS]

    if len(tool_calls) <= 1:
        for call in tool_calls:
            name, args, result, latency_ms = _run_tool(call["name"], call.get("args", {}))
            _collect_result(name, args, result, latency_ms, contexts, seen_hashes, log)
    else:
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            futures = {
                pool.submit(_run_tool, call["name"], call.get("args", {})): call
                for call in tool_calls
            }
            for future in as_completed(futures):
                name, args, result, latency_ms = future.result()
                _collect_result(name, args, result, latency_ms, contexts, seen_hashes, log)

    return {
        "retrieved_contexts": contexts,
        "tool_calls_log": log,
        "tool_calls_requested": [],
    }
