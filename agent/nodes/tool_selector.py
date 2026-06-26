import logging

from google.genai import types
from google.genai import errors as genai_errors

import rag
import config
from agent.state import AgentState
from agent.tool_declarations import TOOLS

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a tool selector for a multilingual RAG agent.

Decision rules (follow in order):
1. ALWAYS call indicrag_retrieval first for document questions about the indexed corpus.
2. Call arxiv_search for questions about specific research papers, recent preprints,
   or when the user asks about arXiv papers by topic, author, or ID.
3. Call open_access_search for broad academic literature search across all disciplines,
   when citation counts matter, or when looking for open-access PDFs beyond arXiv.
4. Call web_search ONLY if the query needs current events or non-academic information.
   Do NOT combine web_search with indicrag_retrieval for purely corpus-based questions.
5. Call calculate for numeric computations.
6. Call execute_python for complex data manipulation.

Retry rules:
7. On retrieve_more retry: use indicrag_retrieval with expand_query=true,
   OR add arxiv_search / open_access_search for supplementary academic context.
8. On reformulate retry: use missing_aspects from feedback to craft sharper queries.
9. On regenerate retry: do NOT call any retrieval tools — the existing passages are
   adequate. Return an empty tool list so the answer generator runs again directly.

Multi-tool: Combine indicrag_retrieval with arxiv_search or open_access_search when
the query spans both local corpus and broader literature. For single-source questions,
call only the most relevant tool."""


def tool_selector_node(state: AgentState) -> dict:
    queries = state.get("query_plan") or [state["original_query"]]
    n_ctx = len(state.get("retrieved_contexts", []))
    history = state.get("reflexion_history", [])

    feedback_note = ""
    if history:
        last = history[-1]
        feedback_note = (
            f"\n\nPrevious attempt: faithfulness={last['faithfulness_score']:.2f}, "
            f"completeness={last['completeness_score']:.2f}, "
            f"action={last['action']}, "
            f"missing={last.get('missing_aspects', [])}."
        )

    user_content = (
        f"Queries to address: {queries}\n"
        f"Already retrieved: {n_ctx} passages.{feedback_note}"
    )

    gen_cfg = types.GenerateContentConfig(
        temperature=0,
        system_instruction=_SYSTEM,
        tools=[TOOLS],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
    )

    try:
        resp = rag.generate_with_failover(
            model=config.LLM_MODEL_NAME,
            contents=user_content,
            gen_config=gen_cfg,
        )

        tool_calls = []
        for part in resp.candidates[0].content.parts:
            fc = getattr(part, "function_call", None)
            if fc:
                tool_calls.append({"name": fc.name, "args": dict(fc.args)})

        if not tool_calls:
            last_action = history[-1]["action"] if history else None
            if last_action != "regenerate":
                tool_calls = [{"name": "indicrag_retrieval",
                               "args": {"query": queries[0], "expand_query": False}}]

    except Exception as exc:
        # All keys exhausted or unrecoverable error — fall back to default tool
        # so the agent can still attempt retrieval rather than returning 500.
        logger.warning(
            f"[ToolSelector] LLM unavailable after key failover ({exc!s:.200}). "
            "Falling back to default indicrag_retrieval."
        )
        tool_calls = [{"name": "indicrag_retrieval",
                       "args": {"query": queries[0], "expand_query": False}}]

    logger.info(f"[ToolSelector] tool_calls={[t['name'] for t in tool_calls]}")
    return {"tool_calls_requested": tool_calls}
