"""Minimal self-check for evaluate.py. Run: python test_evaluate.py

No framework — plain asserts. Guards the Phase 1 additions:
graded nDCG, Recall@20, semantic-judge threading, per-language grouping,
and the per-query regression gate.
"""
from evaluate import (
    ndcg_at_k, recall_at_k, precision_at_k,
    citation_grounding, evaluate, jaccard, make_grounding_scorer,
    write_history_snapshot,
)


def test_ndcg_perfect_order_is_one():
    assert ndcg_at_k(["a", "b"], {"a": 3, "b": 2}, 10) == 1.0


def test_ndcg_penalises_bad_order():
    good = ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, 10)
    bad = ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, 10)
    assert bad < good


def test_ndcg_never_exceeds_one_with_duplicate_run():
    # run repeats the top doc 5x — must not double-count gain past 1.0
    assert ndcg_at_k(["a", "a", "a", "a", "a"], {"a": 3, "b": 2}, 10) <= 1.0


def test_recall20_counts_beyond_k():
    assert recall_at_k(["a", "x", "y"], {"a"}, 20) == 1.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "x"], {"a", "b"}, 3) == 2 / 3


def test_grounding_absent_excluded_from_denominator():
    r = citation_grounding([{"claim": "x", "cited_chunk_text": None, "cited_paper": None}])
    assert r["skipped"] == 1 and r["total"] == 0 and r["score"] == 1.0


def test_grounding_threshold_respected():
    r = citation_grounding(
        [{"claim": "policy gradient reward", "cited_chunk_text": "policy gradient reward", "cited_paper": "p"}],
        jaccard, 0.9,
    )
    assert r["grounded"] == 1


def test_make_scorer_jaccard():
    fn, thr, kind = make_grounding_scorer("jaccard")
    assert fn is jaccard and thr == 0.15 and kind == "jaccard"


def test_metrics_record_grounding_judge():
    j = {"queries": [{"id": "1", "text": "q", "relevant_papers": ["a"]}]}
    res = {"results": [{"query_id": "1", "retrieved_papers": ["a"], "answer_claims": []}]}
    m = evaluate(j, res, 5, jaccard, 0.15, judge="jaccard")
    assert m["grounding_judge"] == "jaccard"
    assert m["grounding_threshold"] == 0.15


def test_per_language_grouping():
    j = {"queries": [
        {"id": "1", "text": "q", "relevant_papers": ["a"], "lang": "hi"},
        {"id": "2", "text": "q", "relevant_papers": ["a"], "lang": "en"},
    ]}
    res = {"results": [
        {"query_id": "1", "retrieved_papers": ["a"], "answer_claims": []},
        {"query_id": "2", "retrieved_papers": ["a"], "answer_claims": []},
    ]}
    m = evaluate(j, res, 5)
    assert set(m["per_language"]) == {"hi", "en"}
    assert m["per_language"]["hi"]["num_queries"] == 1


def test_lang_defaults_to_en():
    j = {"queries": [{"id": "1", "text": "q", "relevant_papers": ["a"]}]}
    res = {"results": [{"query_id": "1", "retrieved_papers": ["a"], "answer_claims": []}]}
    m = evaluate(j, res, 5)
    assert m["per_language"]["en"]["num_queries"] == 1


def test_per_query_overall_present():
    j = {"queries": [{"id": "1", "text": "q", "relevant_papers": ["a"]}]}
    res = {"results": [{"query_id": "1", "retrieved_papers": ["a"], "answer_claims": []}]}
    m = evaluate(j, res, 5)
    assert "overall" in m["per_query"][0]


def test_write_history_snapshot_creates_timestamped_file():
    import json as _json
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = write_history_snapshot({"overall": 0.9}, history_dir=d)
        assert path.exists()
        assert path.name.endswith("_eval_report.json")
        assert _json.loads(path.read_text())["overall"] == 0.9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASS")
