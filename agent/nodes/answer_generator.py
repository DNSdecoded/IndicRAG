import logging

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

    history = state.get("conversation_history", [])
    if history:
        lines = [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
            for m in history[-6:]  # last 3 turns
        ]
        prompt = "Prior conversation:\n" + "\n".join(lines) + "\n\n---\n\n" + prompt

    answer = rag.llm_generate(prompt, max_tokens=config.AGENT_MAX_TOKENS)

    logger.info(f"[AnswerGenerator] chunks_used={chunks_used}, ans_len={len(answer)}")
    return {"draft_answer": answer}
