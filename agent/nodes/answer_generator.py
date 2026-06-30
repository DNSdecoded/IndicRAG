import logging

from google.genai import types

import rag
import config
from agent.state import AgentState

logger = logging.getLogger(__name__)


def answer_generator_node(state: AgentState) -> dict:
    contexts = state.get("retrieved_contexts", [])

    chunks = [c.get("text", "") for c in contexts]
    metas = [{"title": c.get("title", "Unknown"), "section": c.get("section", "body")}
             for c in contexts]

    formatted_context, chunks_used = rag.format_context(chunks, metas)

    if chunks_used == 0:
        return {"draft_answer": config.NO_DOCUMENTS_RESPONSE}

    prompt = rag.build_prompt(
        user_query=state["original_query"],
        context=formatted_context,
        target_lang=state.get("detected_language", "en"),
        strategy=state.get("strategy", "A"),
    )

    # Build structured multi-turn contents to preserve role separation and prevent injection
    history = state.get("conversation_history", [])
    contents = []
    for m in history[-6:]:  # last 3 turns
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"][:500])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    gen_config = types.GenerateContentConfig(
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=config.AGENT_MAX_TOKENS,
        system_instruction=config.AGENT_SYSTEM_PROMPT,
        safety_settings=config.SAFETY_SETTINGS,
    )
    try:
        resp = rag.generate_with_failover(config.LLM_MODEL_NAME, contents, gen_config)
        answer = rag.safe_extract_text(resp)
    except Exception as e:
        logger.error(f"[AnswerGenerator] LLM call failed: {e}", exc_info=True)
        return {"draft_answer": "The AI model is temporarily unavailable. Please try again."}

    logger.info(f"[AnswerGenerator] chunks_used={chunks_used}, ans_len={len(answer)}")
    return {"draft_answer": answer}
