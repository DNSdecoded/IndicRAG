#!/usr/bin/env python3
"""Phase 0 baseline measurements behind BASELINE_v2.6.md (master plan §20, Phase 0).

    python docs/Eval/baseline_study.py --out docs/Eval/baselines/study_v2.6.json [--query-log sessions.db]

Runs on the fixture index built by live_gate.py and measures, per suite query:
  * per-stage retrieval latency (dense, BM25, rerank) and total
  * nDCG@10 with and without the cross-encoder rerank  -> rerank gain
  * the BM25 top-2 score margin                         -> does it predict "rerank adds nothing"?
and, from a query_log (read-only copy), the share of queries a lexical_exact
rule would route (a proxy: the router does not exist yet).

Only aggregate numbers are written for query_log; no query text leaves the DB.
"""

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import live_gate  # noqa: E402  (sets nothing on import; used for the index env)
import stats  # noqa: E402

# Lookup-style token: letters and digits mixed (RO4003C, n257, S11) or an
# all-caps acronym of 3+ letters (SPIBB, MESA).
_IDENT = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9./_-]+$|^[A-Z]{3,}$")


def lexical_exact_proxy(q: str) -> bool:
    """Phase 0 stand-in for the router's lexical_exact rule (P0-RTR-1 defines the real one).

    A quoted phrase, or a short query (<= 8 tokens) carrying at least one
    identifier-like token. Deliberately narrow: arXiv 2606.21553 found an
    entity/number rule that sent 72% of queries to BM25 lost to fixed hybrid.
    """
    if re.search(r"[\"“”'‘’].{2,}?[\"“”'‘’]", q):
        return True
    toks = q.split()
    return len(toks) <= 8 and any(_IDENT.match(t.strip("?,.;:()")) for t in toks)


def query_log_share(db_path: Path) -> dict:
    """lexical_exact proxy share over query_log, from a temp copy (never the live DB)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(db_path) + suffix)
            if src.exists():
                shutil.copy2(src, tmp / (db_path.name + suffix))
        con = sqlite3.connect(tmp / db_path.name)  # NOSONAR S8706: temp COPY of an operator-chosen db file
        cols = {r[1] for r in con.execute("PRAGMA table_info(query_log)")}
        col = "question" if "question" in cols else "query"
        qs = [r[0] for r in con.execute(f"SELECT {col} FROM query_log") if r[0]]
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    flags = [1.0 if lexical_exact_proxy(q) else 0.0 for q in qs]
    if not flags:
        return {"n": 0}
    ci = stats.bootstrap_ci(flags)
    return {"n": len(flags), "share": ci["mean"], "lo": ci["lo"], "hi": ci["hi"]}


def _auc(pos: list, neg: list) -> float:
    """P(score of a positive > score of a negative); ties count half (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def run_study(suite_path: Path) -> list:
    import config
    import metrics
    import rag
    import bm25_search
    import vector_store
    from cache import retrieval_cache
    from evaluate import ndcg_at_k
    from run_live import _paper_ids

    timings = {}
    real_stage = metrics.stage

    @contextlib.contextmanager
    def timed_stage(name):
        t0 = time.perf_counter()
        with real_stage(name):
            yield
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0

    metrics.stage = timed_stage
    suite = json.loads(suite_path.read_text("utf-8"))
    collection = vector_store.get_or_create_collection()
    index = bm25_search.get_or_build_index(collection)

    rag.retrieve_context("warm-up query to load models", top_k=live_gate.TOP_K)   # cold load excluded
    rows = []
    for q in suite["queries"]:
        grades = q["relevance_grades"]
        out = {"id": q["id"], "lang": q["lang"], "type": q["type"]}
        for rerank in (True, False):
            config.USE_RERANKER = rerank
            retrieval_cache.invalidate()
            timings.clear()
            t0 = time.perf_counter()
            ctx = rag.retrieve_context(q["text"], top_k=live_gate.TOP_K)
            total = time.perf_counter() - t0
            tag = "rerank" if rerank else "norerank"
            out[f"ndcg_{tag}"] = ndcg_at_k(_paper_ids(ctx.get("metadatas", [])), grades, 10)
            out[f"t_{tag}"] = total
            if rerank:
                out["stages"] = dict(timings)
        config.USE_RERANKER = True
        _, scores = index.search(q["text"], top_k=30)
        s = sorted(scores, reverse=True) + [0.0, 0.0]
        out["bm25_margin"] = (s[0] - s[1]) / s[0] if s[0] > 0 else 0.0
        out["rerank_gain"] = out["ndcg_rerank"] - out["ndcg_norerank"]
        rows.append(out)
        print(f"  {q['id']}: gain {out['rerank_gain']:+.3f} margin {out['bm25_margin']:.3f} "
              f"t {out['t_rerank']:.1f}s", flush=True)
    return rows


def summarize(rows: list) -> dict:
    def pct(vals, p):
        v = sorted(vals)
        return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]

    stage_names = sorted({k for r in rows for k in r["stages"]})
    stage = {n: {"p50": pct([r["stages"].get(n, 0.0) for r in rows], 0.5),
                 "p95": pct([r["stages"].get(n, 0.0) for r in rows], 0.95)} for n in stage_names}
    total = [r["t_rerank"] for r in rows]
    rerank_share = (sum(r["stages"].get("rerank_cross_encoder", 0.0) for r in rows) / sum(total)) if total else 0.0
    helped = [r["bm25_margin"] for r in rows if r["rerank_gain"] > 0]
    not_helped = [r["bm25_margin"] for r in rows if r["rerank_gain"] <= 0]
    # Separation: does a HIGH margin mark the queries rerank did not help? AUC > 0.5 = yes.
    auc = _auc(not_helped, helped)
    rng_aucs = []
    import random
    rng = random.Random(stats.SEED)
    for _ in range(stats.RESAMPLES):
        sample = rng.choices(rows, k=len(rows))  # NOSONAR S2245: seeded bootstrap, not security
        a = _auc([r["bm25_margin"] for r in sample if r["rerank_gain"] <= 0],
                 [r["bm25_margin"] for r in sample if r["rerank_gain"] > 0])
        if not math.isnan(a):   # a resample with an empty class has no AUC
            rng_aucs.append(a)
    rng_aucs.sort()
    gain_ci = stats.bootstrap_ci([r["rerank_gain"] for r in rows])
    return {
        "n_queries": len(rows),
        "retrieval_total_s": {"mean": sum(total) / len(total), "p50": pct(total, 0.5), "p95": pct(total, 0.95)},
        "retrieval_norerank_s": {"mean": sum(r["t_norerank"] for r in rows) / len(rows)},
        "stages_s": stage,
        "rerank_share_of_retrieval": rerank_share,
        "rerank_ndcg10_gain": gain_ci,
        "rerank_helped": len(helped), "rerank_not_helped": len(not_helped),
        "margin_separates_auc": auc,
        "margin_separates_auc_ci": [stats._percentile(rng_aucs, 0.025), stats._percentile(rng_aucs, 0.975)]
        if rng_aucs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--query-log", default=None, help="sessions.db to measure the lexical_exact share on")
    args = ap.parse_args()

    result = {}
    if args.query_log:
        result["query_log_lexical_exact"] = query_log_share(Path(args.query_log))
        print(f"query_log lexical_exact proxy: {result['query_log_lexical_exact']}")

    index_dir = live_gate.INDEX_ROOT / live_gate.index_key()
    if not (index_dir / "READY.json").exists():
        raise SystemExit("fixture index not built: run docs/Eval/live_gate.py --write-baseline first")
    os.environ.update(live_gate._index_env(index_dir))   # before the app modules import config
    rows = run_study(live_gate.SUITE)
    result["summary"] = summarize(rows)
    result["per_query"] = rows
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")  # NOSONAR S2083,S8707: operator-supplied CLI path in a developer eval tool
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
