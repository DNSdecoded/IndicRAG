import logging

from google.genai import types
from google.genai import errors as genai_errors

import rag
import config
from agent.state import AgentState
from agent.tool_declarations import TOOLS

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a routing broker for a multilingual scientific RAG agent. \
Select tools to retrieve context that satisfies the pending sub-queries.

REGISTERED TOOLS — use EXACT names as listed:
  "indicrag_retrieval"  Hybrid BM25 + dense retrieval from the locally indexed \
corpus. Supports: query (str), expand_query (bool, default false). \
Use for corpus-specific questions. expand_query=true for ambiguous or broad queries.
  "arxiv_search"  arXiv preprint search by topic keywords or author. \
Supports: query (str), max_results (int, default 5), \
year_from (int, e.g. 2022 to filter papers from 2022 onwards), \
sort_by ("relevance"|"submitted_date"). \
Use for cutting-edge research, recent preprints, or when the user requests arXiv.
  "open_access_search"  Broad academic literature via Semantic Scholar + OpenAlex. \
Supports: query (str), max_results (int), year_range (str "YYYY-YYYY" or "YYYY-"), \
open_access_only (bool, default true). \
Use for citation counts, peer-reviewed papers, or wide literature surveys.
  "web_search"  Live web search. Use ONLY for current events, news, or \
non-academic queries. Do NOT use for scientific literature.
  "calculate"  Evaluate a mathematical expression.
  "execute_python"  Run sandboxed Python for data manipulation.

ROUTING RULES — apply in order, stop at first match:
0. USER OVERRIDE: If the user's message explicitly names tools \
   (e.g. "use arxiv", "search open access", "use open search"), call ONLY \
   those named tools. Skip rules 1–4.
1. CORPUS FIRST: For document/corpus questions call indicrag_retrieval.
2. ACADEMIC EXTERNAL: For research questions beyond the local corpus, \
   call arxiv_search and/or open_access_search.
3. TEMPORAL FORWARDING: If year_from is present in state, ALWAYS pass it \
   as year_from to arxiv_search AND as year_range "YYYY-" to open_access_search. \
   Never omit it on retry.
4. COMBINED: For questions spanning local + external literature, combine \
   indicrag_retrieval with arxiv_search or open_access_search.

RETRY RULES:
7. retrieve_more: Craft SHARPER queries using missing_aspects from the evaluator. \
   Never repeat the original query verbatim. Re-use year_from from state.
8. reformulate: The query was misunderstood — build a corrected query \
   from missing_aspects before selecting tools.
9. regenerate: Context is adequate; answer needs rewriting. \
   Return an EMPTY tool list so the answer generator runs without re-retrieval.\
"""


def tool_selector_node(state: AgentState) -> dict:
    queries = state.get("query_plan") or [state["original_query"]]
    n_ctx = len(state.get("retrieved_contexts", []))
    history = state.get("reflexion_history", [])
    year_from = state.get("year_from")
    domain_hints = state.get("domain_hints", [])

    temporal_note = (
        f"\nTemporal constraint: year_from={year_from} "
        f"(pass to arxiv_search and as year_range '{year_from}-' to open_access_search)."
        if year_from else ""
    )
    domain_note = (
        f"\nDomain hints (arXiv categories): {domain_hints}."
        if domain_hints else ""
    )

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
        f"Already retrieved: {n_ctx} passages."
        f"{temporal_note}{domain_note}{feedback_note}"
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
