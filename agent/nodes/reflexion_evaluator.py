import logging

from google.genai import types

import rag
import config
import verify
from agent.state import AgentState, ReflexionFeedback
from agent.json_utils import extract_json

logger = logging.getLogger(__name__)
MAX_REFLEXION = 3


def _truncate_at_sentence(text: str, limit: int) -> str:
    cut = text[:limit]
    pos = cut.rfind(". ")
    return cut[:pos + 1] if pos > limit // 2 else cut


_COMPLETENESS_PROMPT = """\
You are a quality-control evaluator in a retrieval-augmented generation pipeline. \
Assess whether the generated answer fully satisfies the original query.

EVALUATION STEPS:

STEP 1 — SOURCE RELEVANCE:
Examine the retrieved source titles. If the majority are clearly off-topic \
relative to the query, the problem is a retrieval failure, not a writing failure.

STEP 2 — COMPLETENESS SCORE:
Score 0.0 (answer completely missing) to 1.0 (fully addresses every aspect).

STEP 3 — ACTION (choose the one action that fixes the actual deficit):
  "accept"       Score >= 0.75 AND sources are relevant.
  "regenerate"   Score < 0.75 BUT sources are relevant and adequate — \
                 the answer is poorly written; rewrite without re-retrieving.
  "retrieve_more" Score < 0.75 AND sources are on-topic but incomplete — \
                 fetch additional context with a sharper query.
  "reformulate"  Majority of source titles are OFF-TOPIC — the retrieval \
                 query was wrong; replanning needed.

OUTPUT FORMAT: Begin your response IMMEDIATELY with the opening brace `{{`. \
Raw JSON only — no markdown fences, no prose before or after the object. \
Keep missing_aspects strings SHORT (max 8 words each) to avoid truncation.

<schema>
{{
  "completeness_score": 0.85,
  "action": "accept",
  "missing_aspects": ["short description of gap"]
}}
</schema>

<original_query>
{query}
</original_query>

<retrieved_source_titles>
{source_titles}
</retrieved_source_titles>

<generated_answer>
{answer}
</generated_answer>\
"""


def reflexion_evaluator_node(state: AgentState) -> dict:
    count = state.get("reflexion_count", 0)

    if count >= MAX_REFLEXION:
        logger.info(f"[Reflexion] Max iterations ({MAX_REFLEXION}), finalising.")
        return {
            "final_answer": state.get("draft_answer", "Unable to produce a satisfactory answer."),
            "reflexion_count": count,
        }

    answer = state.get("draft_answer", "")
    chunks = [c.get("text", "") for c in state.get("retrieved_contexts", [])]

    try:
        claims = verify.check_claims(answer, chunks)
        if claims:
            # ponytail: min not mean — one hallucinated claim can hide in a high average
            faithfulness_score = min(r["support"] for r in claims)
        else:
            faithfulness_score = 1.0  # absence of citable claims ≠ hallucination
    except Exception as e:
        logger.warning(f"[Reflexion] check_claims failed ({type(e).__name__}): {e}; failing closed")
        claims = []
        faithfulness_score = 0.0  # fail closed: NLI crash forces regeneration

    titles = [c.get("title", "Unknown") for c in state.get("retrieved_contexts", [])]
    source_titles = "\n".join(f"- {t}" for t in titles[:12]) or "None retrieved"

    raw_text = ""
    try:
        resp = rag.generate_with_failover(
            model=config.LLM_MODEL_NAME,
            contents=_COMPLETENESS_PROMPT.format(
                query=state["original_query"],
                source_titles=source_titles,
                answer=_truncate_at_sentence(answer, 4000),
            ),
            gen_config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1024,
            ),
        )
        raw_text = rag.safe_extract_text(resp)
        parsed = extract_json(raw_text)
        completeness_score = float(parsed.get("completeness_score", 0.5))
        missing = parsed.get("missing_aspects", [])

        if not claims:
            action = parsed.get("action", "retrieve_more")
        elif faithfulness_score >= 0.75 and completeness_score >= 0.75:
            action = "accept"
        else:
            action = parsed.get("action", "retrieve_more")

    except Exception as e:
        logger.warning(
            f"[Reflexion] Completeness check failed ({type(e).__name__}): {e} "
            f"| raw={raw_text[:300]!r}"
        )
        completeness_score, missing = 0.5, []
        action = "regenerate" if faithfulness_score >= 0.75 else "retrieve_more"

    feedback = ReflexionFeedback(
        faithfulness_score=faithfulness_score,
        completeness_score=completeness_score,
        action=action,
        missing_aspects=missing,
    )
    history = list(state.get("reflexion_history", [])) + [feedback]

    # Stuck-loop detection: fires from iteration 2 onwards, not just the last one
    prev = state.get("reflexion_history", [])
    if prev and action != "accept":
        prev_complete = prev[-1].get("completeness_score", 0.0)
        if completeness_score <= prev_complete + 0.05 and count >= 1:
            if faithfulness_score < 0.75:
                missing_str = ", ".join(missing) or "the requested details"
                logger.info(
                    f"[Reflexion] iter={count + 1}/{MAX_REFLEXION} "
                    f"faith={faithfulness_score:.2f} complete={completeness_score:.2f} "
                    f"action=safe_stop (stuck with low faithfulness)"
                )
                return {
                    "final_answer": (
                        "The retrieved context does not fully support answering this question. "
                        f"Missing: {missing_str}."
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
