import logging
import time

from google.genai import types

import rag
import config
import verify
import llm_client
from agent.state import AgentState, ReflexionFeedback
from agent.json_utils import extract_json_with_gemini_retry

logger = logging.getLogger(__name__)
MAX_REFLEXION = 3
# AGENT_MAX_TOKENS=8192 → answers up to ~32k chars. The old 4000-char cut made the
# completeness evaluator judge a stub and report "truncated answer" on every long answer.
EVAL_ANSWER_CHARS = 30000


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

    # Time budget: stop looping once we've spent the reflexion budget. Finalises the
    # current draft rather than paying another NLI + completeness pass and a
    # retrieve→generate→verify cycle that would blow past AGENT_TIMEOUT and 504.
    # Checked on the first entry too (as long as there IS a draft): a slow first pass
    # already ate the budget, and the evaluation itself is what pushes it over.
    start = state.get("start_time")
    if start is not None and state.get("draft_answer"):
        elapsed = time.monotonic() - start
        if elapsed > config.AGENT_REFLEXION_BUDGET_S:
            logger.info(
                f"[Reflexion] iter={count + 1}/{MAX_REFLEXION} elapsed={elapsed:.0f}s "
                f"> budget {config.AGENT_REFLEXION_BUDGET_S:.0f}s → finalising best draft"
            )
            return {
                "final_answer": state.get("draft_answer", "Unable to produce a satisfactory answer."),
                "reflexion_count": count + 1,
            }

    answer = state.get("draft_answer", "")
    _contexts = state.get("retrieved_contexts", [])
    chunks = [c.get("text", "") for c in _contexts]
    # Same per-paper numbering the answer generator's format_context used, so [N]
    # resolves to the right paper's chunk(s) instead of the Nth chunk.
    chunk_metas = [{"title": c.get("title", "Unknown"), "section": c.get("section", "body")}
                   for c in _contexts]

    _nli_t0 = time.monotonic()
    try:
        claims = verify.check_claims(answer, chunks, chunk_metas)
        logger.info(
            "[Reflexion] NLI scored %d claims in %.1fs",
            len(claims), time.monotonic() - _nli_t0,
        )
        if claims:
            # Grounded fraction (RAGAS-style): min() collapsed to ~0 on any long
            # multi-claim answer because one synthesized/comparative sentence
            # scoring low entailment nuked the whole score.
            faithfulness_score = sum(1 for r in claims if r["grounded"]) / len(claims)
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
        _model = state.get("requested_model") or config.LLM_MODEL_NAME
        _provider = state.get("requested_provider")
        _completeness_prompt = _COMPLETENESS_PROMPT.format(
            query=state["original_query"],
            source_titles=source_titles,
            answer=_truncate_at_sentence(answer, EVAL_ANSWER_CHARS),
        )
        resp = rag.generate_with_failover(
            model=_model,
            contents=_completeness_prompt,
            gen_config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1024,
                # JSON completeness verdict — thinking off by default (config knob).
                thinking_config=types.ThinkingConfig(thinking_budget=config.AGENT_THINKING_BUDGET),
            ),
            provider=_provider,
        )
        raw_text = rag.safe_extract_text(resp)

        active_provider = llm_client.resolve_provider(_model, _provider)

        def _gemini_retry(p, s):
            r = rag.generate_with_failover(
                model=config.LLM_MODEL_NAME, contents=p,
                gen_config=types.GenerateContentConfig(
                    temperature=0, max_output_tokens=1024, system_instruction=s,
                    thinking_config=types.ThinkingConfig(thinking_budget=config.AGENT_THINKING_BUDGET),
                ),
                provider="gemini",
            )
            return rag.safe_extract_text(r)

        parsed = extract_json_with_gemini_retry(
            raw_text, active_provider, _gemini_retry, _completeness_prompt, "",
        )
        completeness_score = float(parsed.get("completeness_score", 0.5))
        missing = parsed.get("missing_aspects", [])

        if not claims:
            action = parsed.get("action", "retrieve_more")
        elif faithfulness_score >= config.AGENT_FAITHFULNESS_ACCEPT and completeness_score >= 0.75:
            action = "accept"
        else:
            action = parsed.get("action", "retrieve_more")

    except Exception as e:
        logger.warning(
            f"[Reflexion] Completeness check failed ({type(e).__name__}): {e} "
            f"| raw={raw_text[:300]!r}"
        )
        completeness_score, missing = 0.5, []
        action = ("regenerate" if faithfulness_score >= config.AGENT_FAITHFULNESS_ACCEPT
                  else "retrieve_more")

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
            if faithfulness_score < config.AGENT_FAITHFULNESS_ACCEPT:
                missing_str = ", ".join(missing) or "the requested details"
                logger.info(
                    f"[Reflexion] iter={count + 1}/{MAX_REFLEXION} "
                    f"faith={faithfulness_score:.2f} complete={completeness_score:.2f} "
                    f"action=safe_stop (stuck with low faithfulness)"
                )
                # Keep the draft — 200s of retrieval/generation must not be discarded;
                # surface the verification caveat instead.
                return {
                    "final_answer": (
                        f"{answer}\n\n---\n"
                        "*Note: some statements above could not be fully verified against "
                        f"the retrieved sources. Gaps: {missing_str}.*"
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
