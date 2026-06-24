from typing import TypedDict, List, Optional, Literal


class ReflexionFeedback(TypedDict):
    faithfulness_score: float
    completeness_score: float
    action: Literal["accept", "regenerate", "retrieve_more", "reformulate"]
    missing_aspects: List[str]


class AgentState(TypedDict):
    original_query: str
    detected_language: str
    query_plan: List[str]

    tool_calls_requested: List[dict]
    retrieved_contexts: List[dict]

    draft_answer: Optional[str]
    final_answer: Optional[str]

    reflexion_count: int
    reflexion_history: List[ReflexionFeedback]

    tool_calls_log: List[dict]

    session_id: str
    strategy: Literal["A", "B"]
