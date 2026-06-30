import re
import logging

from google.genai import types

import rag
import config
import lang_utils
from agent.state import AgentState
from agent.json_utils import extract_json

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = """\
You are a query decomposition engine for academic retrieval. \
Begin your response IMMEDIATELY with the opening brace `{`. \
Output ONLY a raw JSON object — no markdown, no prose, no explanation, \
no text before or after the JSON object.\
"""

_DECOMPOSE_PROMPT = """\
Analyse this query for vector and keyword retrieval across academic databases.

Extract:
1. sub_queries — Search-optimised phrases (max 4). Split by TOPIC AXIS, not \
   by sentence count. Each phrase must be self-contained and optimised for \
   keyword or semantic search in arXiv or OpenAlex.
2. year_from — Integer year if the query contains temporal language \
   ("after YYYY", "since YYYY", "post-YYYY", "proposed in YYYY+"). \
   null if no temporal constraint exists.
3. domain_hints — arXiv category codes if the domain is clear. \
   Common codes: eess.AP=antennas/RF, eess.SP=signal processing, \
   cs.LG=machine learning, cs.AI=AI, cs.CV=computer vision, \
   cs.CL=NLP, cs.RO=robotics, physics.optics, q-bio.NC=neuroscience. \
   Empty list [] if domain is ambiguous or multi-disciplinary.

LANGUAGE RULE: Generate sub_queries in the EXACT same language as the input \
query. Do not translate Indic or non-English terms to English.

OUTPUT: Raw JSON only. No markdown fences. No text before or after the object.

<schema>
{{
  "sub_queries": ["search phrase 1", "search phrase 2"],
  "year_from": 2022,
  "domain_hints": ["eess.AP", "eess.SP"]
}}
</schema>

<examples>
Input: "What antenna designs for sub-6GHz IoT were proposed after 2021 \
that were NOT feasible before deep learning-assisted optimization?"
Output: {{"sub_queries": ["deep learning sub-6GHz antenna design IoT", \
"antenna optimization infeasible without neural networks", \
"reconfigurable intelligent surface IoT antenna machine learning"], \
"year_from": 2022, "domain_hints": ["eess.AP", "eess.SP"]}}

Input: "एचडीएफसी और एसबीआई की वर्तमान एफडी ब्याज दरें क्या हैं?"
Output: {{"sub_queries": ["एचडीएफसी वर्तमान एफडी ब्याज दरें", \
"एसबीआई वर्तमान एफडी ब्याज दरें"], "year_from": null, "domain_hints": []}}

Input: "What transformer attention mechanisms were introduced after 2020 \
for long document understanding?"
Output: {{"sub_queries": ["transformer attention long document understanding", \
"efficient attention mechanism sparse linear 2021 2022 2023"], \
"year_from": 2021, "domain_hints": ["cs.CL", "cs.LG"]}}
</examples>

<query>
{query}
</query>\
"""

_MAX_SUB_QUERIES = 4


def query_planner_node(state: AgentState) -> dict:
    query = state["original_query"]
    language = lang_utils.detect_language(query) or "en"

    sub_queries = [query]
    year_from = None
    domain_hints = []
    raw_resp = ""

    try:
        resp = rag.generate_with_failover(
            model=config.LLM_MODEL_NAME,
            contents=_DECOMPOSE_PROMPT.format(query=query),
            gen_config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1024,
                system_instruction=_DECOMPOSE_SYSTEM,
            ),
        )
        raw_resp = resp.text or ""
        parsed = extract_json(raw_resp)

        raw_queries = parsed.get("sub_queries")
        sub_queries = (
            [q.strip() for q in raw_queries if isinstance(q, str) and q.strip()]
            if isinstance(raw_queries, list) else [query]
        )
        sub_queries = (sub_queries or [query])[:_MAX_SUB_QUERIES]

        year_from = parsed.get("year_from")
        if year_from is not None:
            try:
                year_from = int(year_from)
            except (TypeError, ValueError):
                year_from = None

        domain_hints = parsed.get("domain_hints", [])
        if not isinstance(domain_hints, list):
            domain_hints = []

    except Exception as exc:
        logger.warning(f"[QueryPlanner] decomposition failed: {exc}")
        # Fallback: extract quoted strings from markdown if model ignored JSON format
        if raw_resp:
            extracted = [s for s in re.findall(r'"([^"]{10,})"', raw_resp)
                         if not s.startswith(('sub_queries', 'year_from', 'domain_hints'))]
            if extracted:
                sub_queries = extracted[:_MAX_SUB_QUERIES]

    logger.info(
        f"[QueryPlanner] lang={language}, sub_queries={sub_queries}, "
        f"year_from={year_from}, domain_hints={domain_hints}"
    )
    update = {
        "detected_language": language,
        "query_plan": sub_queries,
        "year_from": year_from,
        "domain_hints": domain_hints,
        "tool_calls_requested": [],
    }
    # Only clear contexts on the very first pass; reformulation loops should
    # keep prior context so good passages aren't discarded.
    if state.get("reflexion_count", 0) == 0:
        update["retrieved_contexts"] = []
    return update
