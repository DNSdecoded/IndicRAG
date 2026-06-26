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


def _truncate_at_sentence(text: str, limit: int) -> str:
    cut = text[:limit]
    pos = cut.rfind(". ")
    return cut[:pos + 1] if pos > limit // 2 else cut

_COMPLETENESS_PROMPT = """\
Evaluate if this answer completely addresses all parts of the query.

Query: {query}
Answer: {answer}

Score completeness (0.0-1.0). If < 0.75, list what is missing.

The system auto-accepts when both completeness and faithfulness are >= 0.75; your action is ignored then.
When below threshold, choose the action that fixes the actual deficit:
- "regenerate":    faithfulness is low but completeness is adequate — passages are fine, rewrite the answer
- "retrieve_more": completeness is low because needed context was NOT retrieved
- "reformulate":   the original query was misunderstood at the planning stage

Return JSON only (example — use real values, not these):
{{"completeness_score": 0.85, "action": "accept", "missing_aspects": ["methodology details"]}}"""


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
        # ponytail: min not mean — one hallucinated claim can hide in a high average (BUG-013)
        faithfulness_score = min(r["support"] for r in claims)
    else:
        faithfulness_score = 0.0

    try:
        resp = rag.generate_with_failover(
            model=config.LLM_MODEL_NAME,
            contents=_COMPLETENESS_PROMPT.format(
                query=state["original_query"], answer=_truncate_at_sentence(answer, 4000)
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
        logger.warning(f"[Reflexion] Completeness check failed: {e}")
        completeness_score, missing = 0.0, []
        action = "regenerate" if faithfulness_score >= 0.75 else "retrieve_more"

    feedback = ReflexionFeedback(
        faithfulness_score=faithfulness_score,
        completeness_score=completeness_score,
        action=action,
        missing_aspects=missing,
    )
    history = list(state.get("reflexion_history", [])) + [feedback]

    # Detect stuck loop: if completeness didn't improve, stop looping
    prev = state.get("reflexion_history", [])
    if prev and action != "accept":
        prev_complete = prev[-1].get("completeness_score", 0.0)
        if completeness_score <= prev_complete + 0.05:
            if faithfulness_score < 0.75:
                logger.info(
                    f"[Reflexion] iter={count + 1}/{MAX_REFLEXION} "
                    f"faith={faithfulness_score:.2f} complete={completeness_score:.2f} "
                    f"action=safe_stop (stuck with low faithfulness)"
                )
                missing_str = (", ".join(missing) or "the requested details") if missing else "the requested details"
                return {
                    "final_answer": (
                        f"The retrieved context does not fully support answering this question. "
                        f"The available sources do not contain sufficient information about: {missing_str}."
                    ),
                    "reflexion_count": count + 1,
                    "reflexion_history": history,
                }
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
