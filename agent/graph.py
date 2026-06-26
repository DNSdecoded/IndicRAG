from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.query_planner import query_planner_node
from agent.nodes.tool_selector import tool_selector_node
from agent.nodes.tool_executor_node import tool_executor_node
from agent.nodes.answer_generator import answer_generator_node
from agent.nodes.reflexion_evaluator import reflexion_evaluator_node
from agent.nodes.finalizer import finalizer_node


def _route_reflexion(state: AgentState) -> str:
    count = state.get("reflexion_count", 0)
    if count >= 3 or state.get("final_answer"):
        return "finalizer"

    history = state.get("reflexion_history", [])
    if not history:
        return "finalizer"

    return {
        "accept": "finalizer",
        "regenerate": "answer_generator",
        "retrieve_more": "tool_selector",
        "reformulate": "query_planner",
    }.get(history[-1].get("action", "accept"), "finalizer")


def build_agent_graph():
    wf = StateGraph(AgentState)

    wf.add_node("query_planner", query_planner_node)
    wf.add_node("tool_selector", tool_selector_node)
    wf.add_node("tool_executor", tool_executor_node)
    wf.add_node("answer_generator", answer_generator_node)
    wf.add_node("reflexion_evaluator", reflexion_evaluator_node)
    wf.add_node("finalizer", finalizer_node)

    wf.set_entry_point("query_planner")
    wf.add_edge("query_planner", "tool_selector")
    wf.add_edge("tool_selector", "tool_executor")
    wf.add_edge("tool_executor", "answer_generator")
    wf.add_edge("answer_generator", "reflexion_evaluator")

    wf.add_conditional_edges(
        "reflexion_evaluator",
        _route_reflexion,
        {
            "finalizer": "finalizer",
            "answer_generator": "answer_generator",
            "tool_selector": "tool_selector",
            "query_planner": "query_planner",
        },
    )

    wf.add_edge("finalizer", END)
    return wf.compile()
