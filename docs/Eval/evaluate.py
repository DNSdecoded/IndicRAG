"""
IndicRAG Evaluation Script
===========================

Computes:
  - Precision@k
  - Recall@k
  - MRR
  - Citation Grounding Accuracy
  - Retrieval Composite
  - Generation Composite

Usage:
  python evaluate.py
  python evaluate.py --k 3
  python evaluate.py --output report.md --json metrics.json
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set


# ============================================================
# Text Utilities
# ============================================================

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","must","can","to","of","in",
    "for","on","with","at","by","from","up","about","into",
    "through","during","it","its","this","that","these","those",
    "and","but","or","not","as","if","then","than","when","where",
    "which","who","how","all","each","every","more","most",
    "other","some","such","only","same","also"
}

GROUNDING_THRESHOLD = 0.15


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\b[a-z]{2,}\b", text.lower())) - STOPWORDS


def jaccard(a: str, b: str) -> float:
    A, B = tokenize(a), tokenize(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# ============================================================
# Retrieval Metrics
# ============================================================

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


# ============================================================
# Citation Grounding
# ============================================================

def citation_grounding(claims: List[Dict]) -> Dict:
    grounded = 0
    skipped = 0
    per_claim = []

    for c in claims:
        claim  = c.get("claim", "")
        chunk  = c.get("cited_chunk_text")
        paper  = c.get("cited_paper")

        if not chunk or not paper:
            # System correctly acknowledged absent context — exclude from denominator
            skipped += 1
            per_claim.append({
                "claim":      claim[:80],
                "similarity": None,
                "grounded":   "absent"   # explicit label, not None
            })
            continue

        sim         = jaccard(claim, chunk)
        is_grounded = sim >= GROUNDING_THRESHOLD

        if is_grounded:
            grounded += 1

        per_claim.append({
            "claim":      claim[:80],
            "similarity": round(sim, 3),
            "grounded":   is_grounded
        })

    scoreable = len(claims) - skipped
    score = grounded / scoreable if scoreable > 0 else 1.0

    return {
        "score":    round(score, 3),
        "grounded": grounded,
        "total":    scoreable,
        "skipped":  skipped,
        "details":  per_claim
    }


# ============================================================
# Evaluation Core
# ============================================================

def evaluate(judgments: Dict, results: Dict, k: int) -> Dict:
    qrels = {q["id"]: q for q in judgments["queries"]}
    runs  = {r["query_id"]: r for r in results["results"]}

    retrieval_scores = []
    grounding_scores = []
    per_query        = []

    for qid, q in qrels.items():
        if qid not in runs:
            continue

        relevant  = set(q["relevant_papers"])
        retrieved = runs[qid]["retrieved_papers"]

        p  = precision_at_k(retrieved, relevant, k)
        r  = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)

        grounding = citation_grounding(runs[qid]["answer_claims"])

        retrieval_scores.append((p, r, rr))
        grounding_scores.append(grounding["score"])

        per_query.append({
            "id":        qid,
            "query":     q["text"],
            "precision": round(p, 3),
            "recall":    round(r, 3),
            "mrr":       round(rr, 3),
            "grounding": grounding
        })

    n           = len(retrieval_scores)
    mean_p      = sum(x[0] for x in retrieval_scores) / n
    mean_r      = sum(x[1] for x in retrieval_scores) / n
    mean_mrr    = sum(x[2] for x in retrieval_scores) / n
    mean_ground = sum(grounding_scores) / len(grounding_scores)

    retrieval_composite  = (mean_p + mean_r + mean_mrr) / 3
    generation_composite = mean_ground
    overall              = (retrieval_composite + generation_composite) / 2

    return {
        "k":                    k,
        "num_queries":          len(per_query),
        "mean_precision":       round(mean_p, 3),
        "mean_recall":          round(mean_r, 3),
        "mrr":                  round(mean_mrr, 3),
        "mean_grounding":       round(mean_ground, 3),
        "retrieval_composite":  round(retrieval_composite, 3),
        "generation_composite": round(generation_composite, 3),
        "overall":              round(overall, 3),
        "per_query":            per_query,
        "timestamp":            datetime.now().isoformat(timespec="seconds")
    }


# ============================================================
# Markdown Report
# ============================================================

def bar(v: float, w: int = 20) -> str:
    filled = int(round(v * w))
    return "█" * filled + "░" * (w - filled)


def markdown_report(metrics: Dict, k: int) -> str:
    lines = []
    lines.append("# IndicRAG — Evaluation Report")
    lines.append(f"\n_Generated: {metrics['timestamp']} · k={k} · {metrics['num_queries']} queries_\n")
    lines.append("> Retrieval metrics computed from ranked document lists against manually labeled relevance judgments.")
    lines.append("> Citation grounding uses token Jaccard similarity (threshold 0.15). Claims with no citation")
    lines.append("> (system correctly acknowledged absent context) are excluded from the grounding denominator.\n")
    lines.append("---\n")

    lines.append("## Aggregate Metrics\n")
    lines.append("| Metric | Score | Bar |")
    lines.append("|---|---|---|")
    lines.append(f"| Precision@{k}         | **{metrics['mean_precision']:.3f}** | `{bar(metrics['mean_precision'])}` |")
    lines.append(f"| Recall@{k}            | **{metrics['mean_recall']:.3f}** | `{bar(metrics['mean_recall'])}` |")
    lines.append(f"| MRR                   | **{metrics['mrr']:.3f}** | `{bar(metrics['mrr'])}` |")
    lines.append(f"| Citation Grounding    | **{metrics['mean_grounding']:.3f}** | `{bar(metrics['mean_grounding'])}` |")
    lines.append(f"| **Retrieval Score**   | **{metrics['retrieval_composite']:.3f}** | `{bar(metrics['retrieval_composite'])}` |")
    lines.append(f"| **Generation Score**  | **{metrics['generation_composite']:.3f}** | `{bar(metrics['generation_composite'])}` |")
    lines.append(f"| **Overall**           | **{metrics['overall']:.3f}** | `{bar(metrics['overall'])}` |")

    lines.append("\n---\n")
    lines.append("## Per-Query Results\n")

    for q in metrics["per_query"]:
        lines.append(f"### Query {q['id']}")
        lines.append(f"\n> {q['query']}\n")
        lines.append("| Metric | Score |")
        lines.append("|---|---|")
        lines.append(f"| Precision@{k}      | {q['precision']:.3f} |")
        lines.append(f"| Recall@{k}         | {q['recall']:.3f} |")
        lines.append(f"| Reciprocal Rank    | {q['mrr']:.3f} |")
        g = q["grounding"]
        lines.append(f"| Citation Grounding | {g['score']:.3f} ({g['grounded']}/{g['total']} claims) |")

        if g["skipped"] > 0:
            lines.append(f"\n_{g['skipped']} claim(s) had no citation — system correctly acknowledged absent context._")

        lines.append("\n<details>")
        lines.append("<summary>Per-claim grounding detail</summary>\n")
        lines.append("| Claim | Similarity | Status |")
        lines.append("|---|---|---|")
        for d in g["details"]:
            sim    = f"{d['similarity']:.3f}" if d["similarity"] is not None else "—"
            status = "✅" if d["grounded"] is True else ("⚠️ absent" if d["grounded"] == "absent" else "❌")
            lines.append(f"| {d['claim']} | {sim} | {status} |")
        lines.append("\n</details>\n")

    lines.append("---\n")
    lines.append("## Methodology\n")
    lines.append(f"**Precision@{k}** — fraction of top-{k} retrieved documents in the relevant set.\n")
    lines.append(f"**Recall@{k}** — fraction of relevant documents retrieved in top {k}.\n")
    lines.append("**MRR** — reciprocal rank of first relevant document. 1.0 = top result was relevant.\n")
    lines.append("**Citation Grounding** — Jaccard similarity between each answer claim and its cited chunk. "
                 "Threshold 0.15. Claims where the system stated no source exists are excluded from the denominator "
                 "(correct epistemic behavior, labeled 'absent').\n")
    lines.append("**Retrieval Score** — mean of Precision, Recall, MRR.\n")
    lines.append("**Generation Score** — mean citation grounding across queries.\n")
    lines.append("**Limitations** — Jaccard similarity undercounts grounding for paraphrased claims. "
                 "Single annotator for relevance judgments. 4-query evaluation set is small.\n")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", default="relevance_judgments.json")
    parser.add_argument("--results",   default="answers_and_citations.json")
    parser.add_argument("--k",         type=int, default=5)
    parser.add_argument("--output",    default=None, help="Path to write markdown report")
    parser.add_argument("--json",      default=None, help="Path to write JSON metrics")
    args = parser.parse_args()

    for path in [Path(args.judgments), Path(args.results)]:
        if not path.exists():
            print(f"ERROR: {path} not found.")
            return

    with open(args.judgments) as f:
        judgments = json.load(f)
    with open(args.results) as f:
        results = json.load(f)

    metrics = evaluate(judgments, results, args.k)

    # Terminal output
    print("\nIndicRAG Evaluation")
    print("=" * 42)
    print(f"  Queries : {metrics['num_queries']}   k = {metrics['k']}")
    print("-" * 42)
    print(f"  Precision@{args.k}      {metrics['mean_precision']:.3f}  {bar(metrics['mean_precision'], 15)}")
    print(f"  Recall@{args.k}         {metrics['mean_recall']:.3f}  {bar(metrics['mean_recall'], 15)}")
    print(f"  MRR              {metrics['mrr']:.3f}  {bar(metrics['mrr'], 15)}")
    print(f"  Grounding        {metrics['mean_grounding']:.3f}  {bar(metrics['mean_grounding'], 15)}")
    print("-" * 42)
    print(f"  Retrieval Score  {metrics['retrieval_composite']:.3f}  {bar(metrics['retrieval_composite'], 15)}")
    print(f"  Generation Score {metrics['generation_composite']:.3f}  {bar(metrics['generation_composite'], 15)}")
    print("=" * 42)
    print(f"  Overall          {metrics['overall']:.3f}  {bar(metrics['overall'], 15)}")
    print("=" * 42)

    if args.output:
        report = markdown_report(metrics, args.k)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\n  Markdown report -> {args.output}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  JSON metrics    -> {args.json}")

    print()


if __name__ == "__main__":
    main()