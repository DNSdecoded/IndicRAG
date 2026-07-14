import re

import config
from agent.state import AgentState

_CITE_RE = re.compile(r"\[\d+\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Sentences shorter than this are headers/list fragments — excluded from the
# citation-coverage denominator so they don't dilute the figure.
_MIN_SENT_CHARS = 20

# Faithfulness bar above which we trust what the answer *does* say. Matches the
# reflexion evaluator's "accept" faithfulness gate.
_ABSTAIN_FAITH_MIN = 0.75

_ABSTAIN_PREFIX = (
    "**Insufficient evidence in the corpus to fully answer this question.** "
    "Here is what the retrieved sources *do* support:\n\n"
)


def citation_coverage(text: str) -> float:
    """Fraction of substantive sentences carrying a [N] citation marker."""
    sents = [s for s in _SENT_SPLIT.split(text.strip()) if len(s) > _MIN_SENT_CHARS]
    if not sents:
        return 0.0
    return sum(1 for s in sents if _CITE_RE.search(s)) / len(sents)


def _confidence(feedback: dict, answer: str) -> float:
    """Directional 0..1 confidence. Weights are uncalibrated until Phase 1 eval."""
    faith = feedback.get("faithfulness_score", 0.0)
    comp = feedback.get("completeness_score", 0.0)
    cov = citation_coverage(answer)
    # ponytail: fixed weights; replace with the Phase 1 reliability curve once it exists.
    return round(0.5 * faith + 0.3 * comp + 0.2 * cov, 3)


def finalizer_node(state: AgentState) -> dict:
    answer = (
        state.get("final_answer")
        or state.get("draft_answer")
        or "Unable to generate an answer. Please try rephrasing."
    ).strip()

    history = state.get("reflexion_history", [])
    last = history[-1] if history else None

    if not (config.ANSWER_CONFIDENCE_ENABLE and last):
        return {"final_answer": answer, "answer_confidence": None, "abstained": False}

    base_answer = answer  # score before any abstention wrapper is prepended
    faith = last.get("faithfulness_score", 0.0)
    comp = last.get("completeness_score", 0.0)
    abstained = False

    # Abstention: grounded but incomplete after the reflexion budget is spent — the
    # answer we have is trustworthy, the corpus just doesn't cover the rest. (Low
    # faithfulness is a different failure, already caveated by the reflexion node.)
    if faith >= _ABSTAIN_FAITH_MIN and comp < config.ABSTAIN_COMPLETENESS_FLOOR:
        missing = last.get("missing_aspects") or ["parts of the question"]
        gaps = "\n".join(f"- {m}" for m in missing)
        answer = f"{_ABSTAIN_PREFIX}{base_answer}\n\n---\n*Not supported by the corpus:*\n{gaps}"
        abstained = True

    return {
        "final_answer": answer,
        "answer_confidence": _confidence(last, base_answer),
        "abstained": abstained,
    }
