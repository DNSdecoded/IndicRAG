from agent.state import AgentState


def finalizer_node(state: AgentState) -> dict:
    answer = (
        state.get("final_answer")
        or state.get("draft_answer")
        or "Unable to generate an answer. Please try rephrasing."
    )
    return {"final_answer": answer.strip()}
