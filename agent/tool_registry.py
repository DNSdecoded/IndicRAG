"""Agent tool registry (v3 Phase 1): one entry per tool, everything about it in one place.

Each Tool declares its JSON schema (offered to the model), its executor, its trust
level (§15: T3 owned corpus, T4 external web/API, T5 tool output), its timeout and
whether results are cacheable. `run()` is the only way the agent executes a tool:

- disabled tools (config.AGENT_DISABLED_TOOLS) are refused;
- arguments are validated against the schema and coerced or rejected before the
  executor sees them (a malformed call used to surface as a KeyError string);
- each call is bounded by its timeout (a hung tool used to block the whole run);
- every returned passage carries the tool's trust level, for trust-labelled context.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Callable

import config
import contextvars
from agent import tool_executor as ex
from agent.tool_declarations import FUNCTION_DECLARATIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[[dict], dict]
    trust: str            # "T3" corpus | "T4" external | "T5" tool output
    cacheable: bool

    @property
    def timeout_s(self) -> float:
        return config.TOOL_TIMEOUTS.get(self.name, 30.0)

    @property
    def enabled(self) -> bool:
        return self.name not in config.AGENT_DISABLED_TOOLS


# name -> (executor, trust, cacheable). Executors receive validated args only.
_EXECUTORS = {
    "indicrag_retrieval": (lambda a: ex.execute_indicrag(
        a["query"], a.get("expand_query", False), a.get("year_from"), a.get("year_to"),
        a.get("tags"), a.get("top_k")), "T3", True),
    "web_search": (lambda a: ex.execute_web_search(a["query"], a.get("num_results", 5)), "T4", True),
    "calculate": (lambda a: ex.execute_calculate(a["expression"]), "T5", False),
    "execute_python": (lambda a: ex.execute_python(a["code"]), "T5", False),
    "arxiv_search": (lambda a: ex.execute_arxiv_search(
        a.get("query", ""), a.get("max_results", 5), a.get("sort_by", "relevance"),
        a.get("year_from"), a.get("arxiv_id")), "T4", True),
    "open_access_search": (lambda a: ex.execute_open_access_search(
        a["query"], a.get("max_results", 5), a.get("year_range", ""),
        a.get("open_access_only", True)), "T4", True),
}

_declared = {d["name"] for d in FUNCTION_DECLARATIONS}
if _declared != set(_EXECUTORS):
    # A schema without an executor (or the reverse) is a model-visible tool that
    # cannot run; refuse to import rather than fail per call.
    raise RuntimeError(f"tool schemas and executors disagree: {_declared ^ set(_EXECUTORS)}")

REGISTRY: dict = {
    d["name"]: Tool(d["name"], d["description"], d["parameters"], *_EXECUTORS[d["name"]])
    for d in FUNCTION_DECLARATIONS
}


def openai_tools() -> list:
    """Schemas of the enabled tools, in OpenAI function-tool form (for LLMRequest.tools)."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.parameters}}
            for t in REGISTRY.values() if t.enabled]


_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


def _coerce(value, kind: str):
    """Value of JSON-schema `kind`, coercing the string forms models often send."""
    if kind == "string":
        if isinstance(value, str):
            return value
    elif kind == "integer":
        if isinstance(value, bool):
            raise ValueError("expected integer, got boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
    elif kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
    else:
        # Fail closed on a schema feature this validator does not implement.
        raise ValueError(f"unsupported schema type {kind!r}")
    raise ValueError(f"expected {kind}, got {type(value).__name__}")


def validate(tool: Tool, args) -> dict:
    """Validated, coerced copy of `args`; raises ValueError naming the problem.

    Unknown keys are dropped (logged): models add stray fields, and nothing the
    schema does not declare reaches an executor either way.
    """
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    props = tool.parameters.get("properties", {})
    out = {}
    for key, value in args.items():
        if key not in props:
            logger.info("[ToolRegistry] %s: dropping undeclared argument %r", tool.name, key)
            continue
        if value is None:
            continue
        try:
            out[key] = _coerce(value, props[key].get("type"))
        except ValueError as e:
            raise ValueError(f"{key}: {e}") from None
    missing = [k for k in tool.parameters.get("required", []) if k not in out]
    if missing:
        raise ValueError(f"missing required argument(s): {', '.join(missing)}")
    return out


# Bounded pool: a timed-out call's thread cannot be killed (Python threads are not
# cancellable), so the cap keeps hung calls from accumulating without limit.
_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tool")


def _error(message: str) -> dict:
    return {"passages": [], "error": message}


def run(name: str, args) -> dict:
    """Execute a registered tool: enabled check, validation, timeout, trust labels."""
    tool = REGISTRY.get(name)
    if tool is None:
        return _error(f"Unknown tool: {name}")
    if not tool.enabled:
        return _error(f"Tool {name} is disabled on this server")
    try:
        clean = validate(tool, args)
    except ValueError as e:
        logger.warning("[ToolRegistry] %s rejected arguments: %s", name, e)
        return _error(f"invalid arguments for {name}: {e}")
    future = _POOL.submit(contextvars.copy_context().run, tool.run, clean)   # keep the trace
    try:
        result = future.result(timeout=tool.timeout_s)
    except FutureTimeout:
        # shortcut: the worker thread keeps running until the call returns on its own;
        # only a subprocess-backed tool can be truly cancelled. Fine while executors
        # carry their own socket/subprocess timeouts, as all six do.
        logger.error("[ToolRegistry] %s timed out after %.0fs", name, tool.timeout_s)
        return _error(f"{name} timed out after {tool.timeout_s:.0f}s")
    if not isinstance(result, dict):
        return _error(f"{name} returned {type(result).__name__}, not a result object")
    for p in result.get("passages") or []:
        if isinstance(p, dict):
            p.setdefault("trust", tool.trust)
    return result
