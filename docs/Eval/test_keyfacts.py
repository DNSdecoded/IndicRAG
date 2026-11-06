import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import keyfacts  # noqa: E402


def test_normalization_matches_unicode_variants():
    answer = "Return loss of **−52.2 dB** and bandwidth 8.93 %, on FR‑4."
    assert keyfacts.missing_facts(answer, ["-52.2 dB", "8.93%", "FR-4"]) == []


def test_case_insensitive_and_reports_what_is_missing():
    assert keyfacts.missing_facts("Uses LightGBM and an MLP.", ["lightgbm", "RidgeCV"]) == ["RidgeCV"]


def test_full_answer_preferred_over_claims():
    r = {"answer": "SAC agent; 72% fewer CST runs.", "answer_claims": [{"claim": "Fewer CST runs."}]}
    assert keyfacts.answer_text(r) == r["answer"]


def test_claims_fallback_for_old_files():
    r = {"answer_claims": [{"claim": "First."}, {"claim": "Second."}]}
    assert keyfacts.answer_text(r) == "First. Second."


def test_score_skips_unanswered_and_factless_queries():
    golden = {"queries": [
        {"id": "a", "key_facts": ["28.12 GHz"]},
        {"id": "b", "key_facts": ["x"]},
        {"id": "c", "key_facts": []},
        {"id": "d", "key_facts": ["y"]},
    ]}
    answers = {"results": [
        {"query_id": "a", "answer": "It resonates at 28.12 GHz."},
        {"query_id": "b", "answer": "nothing"},
        {"query_id": "c", "answer": "n/a"},
        {"query_id": "d", "error": "answer generation failed after retries"},
    ]}
    res = keyfacts.score(golden, answers)
    assert (res["passed"], res["total"]) == (1, 2)
    assert [p["id"] for p in res["per_query"]] == ["a", "b"]
