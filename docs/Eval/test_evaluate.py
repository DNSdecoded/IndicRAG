"""Minimal self-check for evaluate.py. Run: python test_evaluate.py

No framework — plain asserts. Guards the Phase 1 additions:
graded nDCG, Recall@20, semantic-judge threading, per-language grouping,
and the per-query regression gate.
"""
import json as _json
import subprocess as _sp
import sys as _sys
from pathlib import Path as _Path

import pytest as _pytest

from evaluate import regression_gate
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

# --- Bootstrap regression gate (P0-EVL-1) ---------------------------------



def _run(per_query_vals, metric="ndcg_10"):
    return {"per_query": [{"id": f"q{i}", metric: v, "recall_10": v} for i, v in enumerate(per_query_vals)]}


def _base_vals():
    return [0.3 + 0.6 * ((i * 37) % 11) / 10 for i in range(40)]


def test_gate_passes_identical_run():
    rows = regression_gate(_run(_base_vals()), _run(_base_vals()), ["ndcg_10", "recall_10"])
    assert [r["regressed"] for r in rows] == [False, False]


def test_gate_fails_planted_regression():
    # Every query loses relevance, as when RRF is flipped to dense-only.
    worse = [max(0.0, v - 0.2) for v in _base_vals()]
    rows = regression_gate(_run(worse), _run(_base_vals()), ["ndcg_10"])
    assert rows[0]["regressed"] and rows[0]["hi"] < 0


def test_gate_ignores_single_query_noise():
    vals = _base_vals()
    noisy = list(vals)
    noisy[3] -= 0.25   # one query wobbles; the old eps rule failed on this
    assert not regression_gate(_run(noisy), _run(vals), ["ndcg_10"])[0]["regressed"]


def test_gate_refuses_disjoint_runs():
    a = {"per_query": [{"id": "x", "ndcg_10": 1.0}]}
    b = {"per_query": [{"id": "y", "ndcg_10": 1.0}]}
    with _pytest.raises(ValueError):
        regression_gate(a, b, ["ndcg_10"])


def _cli(tmp_path, retrieved, baseline_path):
    here = _Path(__file__).resolve().parent
    judg = {"queries": [{"id": f"q{i}", "text": "t", "relevant_papers": [f"p{i}"]} for i in range(30)]}
    res = {"results": [{"query_id": f"q{i}", "retrieved_papers": retrieved(i), "answer_claims": []}
                       for i in range(30)]}
    (tmp_path / "j.json").write_text(_json.dumps(judg))
    (tmp_path / "r.json").write_text(_json.dumps(res))
    return _sp.run([_sys.executable, str(here / "evaluate.py"), "--judgments", str(tmp_path / "j.json"),
                    "--results", str(tmp_path / "r.json"), "--json", str(tmp_path / "m.json"),
                    "--baseline", str(baseline_path)],
                   capture_output=True, text=True, cwd=tmp_path)


def test_cli_clean_passes_and_planted_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    good = lambda i: [f"p{i}", "z1", "z2"]  # noqa: E731
    bad = lambda i: ["z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", f"p{i}"]  # noqa: E731
    base = tmp_path / "base.json"
    first = _cli(tmp_path, good, tmp_path / "missing.json")
    assert first.returncode == 1 and "baseline" in first.stdout and "not found" in first.stdout
    base.write_text((tmp_path / "m.json").read_text())
    clean = _cli(tmp_path, good, base)
    assert clean.returncode == 0 and "CI PASS: no regression" in clean.stdout
    planted = _cli(tmp_path, bad, base)
    assert planted.returncode == 1 and "REGRESSED" in planted.stdout


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASS")
