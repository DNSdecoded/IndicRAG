import json
import re
import logging

from google.genai import types

import rag
import config
import verify
from agent.state import AgentState, ReflexionFeedback

logger = logging.getLogger(__name__)
MAX_REFLEXION = 3

_COMPLETENESS_PROMPT = """\
Evaluate if this answer completely addresses all parts of the query.

Query: {query}
Answer: {answer}

Score completeness (0.0-1.0). If < 0.75, list what is missing.

The system will auto-accept if both completeness and faithfulness are >= 0.75.
Your action choice only matters when at least one score is below threshold.
Pick the action that best fixes the deficit:
- "regenerate":    retrieved passages seem adequate but the answer is poorly written
- "retrieve_more": answer is incomplete because needed context was NOT retrieved
- "reformulate":   the original query was misunderstood at the planning stage

Return JSON only:
{{"completeness_score": 0.0, "action": "retrieve_more", "missing_aspects": []}}"""


def reflexion_evaluator_node(state: AgentState) -> dict:
    count = state.get("reflexion_count", 0)

    if count >= MAX_REFLEXION:
        logger.info(f"[Reflexion] Max iterations reached ({MAX_REFLEXION}), finalising.")
        return {
            "final_answer": state.get("draft_answer", "Unable to produce a satisfactory answer."),
            "reflexion_count": count,
        }

    answer = state.get("draft_answer", "")
    chunks = [c.get("text", "") for c in state.get("retrieved_contexts", [])]

    claims = verify.check_claims(answer, chunks)
    if claims:
        faithfulness_score = sum(r["support"] for r in claims) / len(claims)
    else:
        faithfulness_score = 0.0

    try:
        resp = rag.generate_with_failover(
            model=config.LLM_MODEL_NAME,
            contents=_COMPLETENESS_PROMPT.format(
                query=state["original_query"], answer=answer[:1500]
            ),
            gen_config=types.GenerateContentConfig(temperature=0, max_output_tokens=256),
        )
        raw = resp.text or ""
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        completeness_score = float(parsed.get("completeness_score", 0.5))
        missing = parsed.get("missing_aspects", [])

        if not claims:
            action = "regenerate"
        elif faithfulness_score >= 0.75 and completeness_score >= 0.75:
            action = "accept"
        else:
            action = parsed.get("action", "retrieve_more")
    except Exception as e:
        logger.warning(f"[Reflexion] Completeness check failed: {e} — defaulting to accept")
        completeness_score, action, missing = 0.6, "accept", []

    feedback = ReflexionFeedback(
        faithfulness_score=faithfulness_score,
        completeness_score=completeness_score,
        action=action,
        missing_aspects=missing,
    )
    history = list(state.get("reflexion_history", [])) + [feedback]

    # Detect stuck loop: if completeness didn't improve, accept what we have
    prev = state.get("reflexion_history", [])
    if prev and action != "accept":
        prev_complete = prev[-1].get("completeness_score", 0.0)
        if completeness_score <= prev_complete + 0.05:
            action = "accept"
            logger.info(
                f"[Reflexion] iter={count + 1}/{MAX_REFLEXION} "
                f"faith={faithfulness_score:.2f} complete={completeness_score:.2f} "
                f"action=accept (stuck — no improvement over prior {prev_complete:.2f})"
            )
            return {
                "final_answer": answer,
                "reflexion_count": count + 1,
                "reflexion_history": history,
            }

    logger.info(
        f"[Reflexion] iter={count + 1}/{MAX_REFLEXION} "
        f"faith={faithfulness_score:.2f} complete={completeness_score:.2f} "
        f"action={action}"
    )

    if action == "accept":
        return {
            "final_answer": answer,
            "reflexion_count": count + 1,
            "reflexion_history": history,
        }

    return {"reflexion_count": count + 1, "reflexion_history": history}
