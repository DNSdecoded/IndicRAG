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
    year_from: Optional[int]
    domain_hints: List[str]

    tool_calls_requested: List[dict]
    retrieved_contexts: List[dict]

    draft_answer: Optional[str]
    final_answer: Optional[str]

    reflexion_count: int
    reflexion_history: List[ReflexionFeedback]

    tool_calls_log: List[dict]
    conversation_history: List[dict]

    session_id: str
    user_id: str  # owner of this run; agent-created watches are stamped with it
    strategy: Literal["A", "B"]

    # monotonic() timestamp stamped at run start, used by the reflexion loop to
    # honour AGENT_REFLEXION_BUDGET_S. Optional so older callers still type-check.
    start_time: Optional[float]

    # Phase 2: set by finalizer_node. answer_confidence is a directional 0..1 blend
    # of faithfulness, completeness, and citation coverage; abstained flags an
    # explicit insufficient-evidence answer.
    answer_confidence: Optional[float]
    abstained: bool

    # Phase 8: user-selected model/provider for this request. When set, the tool
    # selector gates the choice against tool-capability before using it.
    requested_model: Optional[str]
    requested_provider: Optional[str]
