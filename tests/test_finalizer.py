"""Phase 2 — finalizer confidence + abstention."""
import config
from agent.nodes.finalizer import finalizer_node, citation_coverage, _confidence


def test_citation_coverage_counts_cited_sentences():
    txt = "The antenna resonates at 2.4 GHz as reported [1]. This other sentence carries no citation at all."
    assert abs(citation_coverage(txt) - 0.5) < 1e-9


def test_citation_coverage_empty_is_zero():
    assert citation_coverage("") == 0.0


def test_confidence_full_signals_is_one():
    fb = {"faithfulness_score": 1.0, "completeness_score": 1.0}
    ans = "Every substantive sentence here is grounded in a source [1]."
    # 0.5*1 + 0.3*1 + 0.2*coverage(1.0) = 1.0
    assert _confidence(fb, ans) == 1.0


def test_abstains_when_grounded_but_incomplete():
    state = {
        "final_answer": "The supported fact is stated here with a citation [1].",
        "reflexion_history": [{
            "faithfulness_score": 0.9,
            "completeness_score": 0.2,
            "missing_aspects": ["efficiency numbers"],
        }],
    }
    out = finalizer_node(state)
    assert out["abstained"] is True
    assert "Insufficient evidence" in out["final_answer"]
    assert "efficiency numbers" in out["final_answer"]
    assert out["answer_confidence"] is not None


def test_no_abstain_when_complete():
    state = {
        "final_answer": "This is a complete and well cited answer to the question [1].",
        "reflexion_history": [{
            "faithfulness_score": 0.9,
            "completeness_score": 0.9,
            "missing_aspects": [],
        }],
    }
    out = finalizer_node(state)
    assert out["abstained"] is False
    assert out["answer_confidence"] is not None


def test_low_faithfulness_does_not_abstain():
    # low faith is a hallucination problem, handled elsewhere — not an evidence gap
    state = {
        "final_answer": "Ungrounded claim stated here without real support [1].",
        "reflexion_history": [{
            "faithfulness_score": 0.3,
            "completeness_score": 0.2,
            "missing_aspects": ["x"],
        }],
    }
    out = finalizer_node(state)
    assert out["abstained"] is False


def test_no_history_returns_answer_no_confidence():
    out = finalizer_node({"draft_answer": "hello"})
    assert out["final_answer"] == "hello"
    assert out["abstained"] is False
    assert out["answer_confidence"] is None
