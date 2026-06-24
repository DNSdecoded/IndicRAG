import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.state import AgentState
from agent.tool_executor import TOOL_DISPATCH
from cache import tool_cache, make_key

logger = logging.getLogger(__name__)

_CACHEABLE_TOOLS = {"indicrag_retrieval", "arxiv_search", "open_access_search", "web_search"}


def _run_tool(name: str, args: dict) -> tuple[str, dict, dict, float]:
    start = time.monotonic()
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        logger.warning(f"[ToolExecutor] Unknown tool: {name}")
        return name, args, {"passages": []}, 0.0

    cache_key = make_key(name, args) if name in _CACHEABLE_TOOLS else None
    result = tool_cache.get(cache_key) if cache_key else None

    if result is not None:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(f"[ToolExecutor] {name} cache hit ({latency_ms:.0f}ms)")
    else:
        try:
            result = fn(args)
        except Exception as e:
            logger.error(f"[ToolExecutor] {name} failed: {e}")
            result = {"passages": [], "text": str(e)}
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        if cache_key and result:
            tool_cache.put(cache_key, result)
        logger.info(f"[ToolExecutor] {name} completed in {latency_ms:.0f}ms")

    return name, args, result, latency_ms


def tool_executor_node(state: AgentState) -> dict:
    tool_calls = state.get("tool_calls_requested", [])
    contexts = list(state.get("retrieved_contexts", []))
    log = list(state.get("tool_calls_log", []))

    if len(tool_calls) <= 1:
        for call in tool_calls:
            name, args, result, latency_ms = _run_tool(call["name"], call.get("args", {}))
            if "passages" in result:
                contexts.extend(result["passages"])
            elif "text" in result:
                contexts.append({"text": result["text"], "source": name})
            log.append({"tool": name, "args": args, "latency_ms": latency_ms})
    else:
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            futures = {
                pool.submit(_run_tool, call["name"], call.get("args", {})): call
                for call in tool_calls
            }
            for future in as_completed(futures):
                name, args, result, latency_ms = future.result()
                if "passages" in result:
                    contexts.extend(result["passages"])
                elif "text" in result:
                    contexts.append({"text": result["text"], "source": name})
                log.append({"tool": name, "args": args, "latency_ms": latency_ms})

    return {
        "retrieved_contexts": contexts,
        "tool_calls_log": log,
        "tool_calls_requested": [],
    }
