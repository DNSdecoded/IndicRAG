#!/usr/bin/env python3
"""Key-fact scorer (master plan §16.2, Generation: key-fact correctness).

A query's answer passes when it states every required fact listed in the golden
set's `key_facts`, checked after Unicode normalization. Promoted from the
2026-10-03 scratch scorer behind RESULTS_2026-10-03.md.

Usage:
    python keyfacts.py --golden golden_2026086448.json --answers answers_2026086448.json
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

# Minus-like and dash-like code points that answers and papers use
# interchangeably for "-52.2 dB".
_DASHES = dict.fromkeys(map(ord, "−‐‑‒–—﹣－"), "-")


def normalize(text: str) -> str:
    """NFKC, unified dashes, no markdown emphasis, casefolded, no whitespace.

    Whitespace is dropped, not collapsed: answers write "8.93 %" and "−52.2 dB"
    with narrow no-break spaces where the fact says "8.93%".
    """
    t = unicodedata.normalize("NFKC", text or "").translate(_DASHES)
    t = re.sub(r"[*_`]+", "", t)
    return re.sub(r"\s+", "", t).casefold()


def answer_text(result: dict) -> str:
    """The full answer when run_live.py recorded it; else its claim sentences.

    Files written before run_live.py kept `answer` hold only claims >20 chars,
    which lose short sentences and table rows, so they under-count.
    """
    if result.get("answer"):
        return result["answer"]
    return " ".join(c.get("claim", "") for c in result.get("answer_claims") or [])


def missing_facts(answer: str, facts: list) -> list:
    norm = normalize(answer)
    return [f for f in facts if normalize(f) not in norm]


def score(golden: dict, answers: dict) -> dict:
    """Per-query pass/fail over golden queries that have key facts and an answer."""
    by_id = {r["query_id"]: r for r in answers["results"]}
    per_query = []
    for q in golden["queries"]:
        facts = q.get("key_facts") or []
        r = by_id.get(q["id"])
        if not facts or r is None or r.get("error"):
            continue
        miss = missing_facts(answer_text(r), facts)
        per_query.append({"id": q["id"], "lang": q.get("lang", "en"),
                          "passed": not miss, "missing": miss})
    return {
        "passed": sum(p["passed"] for p in per_query),
        "total": len(per_query),
        "per_query": per_query,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--json", default=None, help="write per-query results here")
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))  # NOSONAR S8707: operator-supplied CLI path in a developer eval tool
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))  # NOSONAR S8707: operator-supplied CLI path in a developer eval tool
    if not answers["results"] or not any(r.get("answer_claims") for r in answers["results"]):
        print("ERROR: no answers in the results file; run run_live.py --with-answers", file=sys.stderr)
        return 2
    res = score(golden, answers)
    for p in res["per_query"]:
        if not p["passed"]:
            print(f"  {p['id']} missing: {', '.join(p['missing'])}")
    print(f"key facts {res['passed']}/{res['total']}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")  # NOSONAR S2083,S8707: operator-supplied CLI path in a developer eval tool
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
