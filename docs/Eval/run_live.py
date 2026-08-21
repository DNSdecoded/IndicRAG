#!/usr/bin/env python3
"""Regenerate answers_and_citations.json by running the live pipeline.

Why this exists
---------------
evaluate.py scores whatever is in answers_and_citations.json. That file was
produced by hand — its own `_instructions` field said "Paste your system's
output for each query" — so the CI eval gate scored a frozen snapshot and never
called retrieval. No change to rag.py, bm25_search.py, embeddings.py or any
retrieval env knob could move the number. The threshold was calibrated correctly
(0.85 against a measured 0.94) and was still unreachable, because the input only
changed when a human edited it.

This closes that loop: run the judged queries through the real pipeline, emit the
file evaluate.py already knows how to score, and the gate becomes able to fail.

Usage
-----
    python run_live.py                      # retrieval only (no LLM, no API key)
    python run_live.py --with-answers       # also generate answers for grounding
    python run_live.py --top-k 10 --out /tmp/live.json

Then:
    python evaluate.py --results <out> --ci --threshold 0.85

Retrieval-only is the default deliberately: retrieval metrics (precision,
recall, MRR, nDCG) are the ones that regress silently when a knob changes, and
they need no API key, no tokens and no non-determinism. Grounding metrics need
generated answers, so they cost money and vary run to run — opt in.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Import the application from the repo root, not this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HERE = Path(__file__).resolve().parent


def _split_claims(answer: str) -> list:
    """Split an answer into claim-sized sentences.

    Deliberately crude and dependency-free: the grounding judge compares a claim
    against the chunk it cites, so sentence boundaries only need to be roughly
    right. Anything smarter here would be a second thing to keep in sync with
    verify.py.
    """
    parts = re.split(r"(?<=[.!?])\s+", (answer or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _paper_ids(metadatas: list) -> list:
    """Retrieved paper ids in rank order, de-duplicated.

    Judgments are per paper and one paper contributes several chunks — leaving
    the duplicates in would let a single paper fill top-k and inflate precision.
    """
    seen, out = set(), []
    for m in metadatas or []:
        pid = m.get("paper_id")
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _corpus_mismatch(judgments_path: Path, collection) -> set:
    """Judged paper ids that are not in the indexed corpus.

    The judgments were written alongside hand-pasted answers, using readable
    aliases ("tandem_nn") rather than the filename-derived ids the ingest
    pipeline actually assigns ("2608_06352v1"). Nothing ever compared the two,
    because nothing ever ran the judged queries against the live index — which
    is exactly why the eval gate could never be wired up.

    Caught here rather than in evaluate.py: a total id mismatch produces a
    uniform 0.000, which is indistinguishable from "retrieval is completely
    broken" unless something says otherwise.
    """
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    judged = set(judgments.get("corpus") or [])
    if not judged:
        return set()
    got = collection.get(include=["metadatas"])
    live = {m.get("paper_id") for m in got.get("metadatas", []) if m.get("paper_id")}
    return judged - live


def run(judgments_path: Path, out_path: Path, top_k: int, with_answers: bool,
        strategy: str) -> dict:
    import config
    import rag

    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    queries = judgments["queries"]

    results = []
    for q in queries:
        qid, text = q["id"], q["text"]
        print(f"[{qid}] {text[:70]}...", flush=True)

        ctx = rag.retrieve_context(text, top_k=top_k)
        metas = ctx.get("metadatas", [])
        entry = {
            "query_id": qid,
            "retrieved_papers": _paper_ids(metas),
            "answer_claims": [],
        }
        if ctx.get("degraded"):
            # A degraded run scores as if the dense leg were simply worse. Record
            # it so a surprising score arrives with its explanation attached.
            entry["degraded"] = ctx["degraded"]
            print(f"    WARNING: retrieval degraded ({ctx['degraded']})", flush=True)

        if with_answers:
            answer_data = rag.answer_question(text, top_k=top_k, strategy=strategy)
            answer = answer_data.get("answer", "")
            chunks = ctx.get("chunks", [])
            # Citation numbers are per paper in first-seen order — the same
            # mapping format_context uses, so resolve through it rather than
            # inventing a second numbering scheme here.
            num_to_meta = rag.citation_number_map(metas)
            for claim in _split_claims(answer):
                nums = [int(n) for n in re.findall(r"\[(\d+)\]", claim)]
                meta = num_to_meta.get(nums[0]) if nums else None
                cited_paper = (meta or {}).get("paper_id")
                chunk_text = ""
                if cited_paper:
                    for m, c in zip(metas, chunks):
                        if m.get("paper_id") == cited_paper:
                            chunk_text = c
                            break
                entry["answer_claims"].append({
                    "claim": re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", claim).strip(),
                    "cited_paper": cited_paper,
                    "cited_chunk_text": chunk_text,
                })

        results.append(entry)

    payload = {
        "_instructions": (
            "GENERATED by run_live.py against the live pipeline — do not hand-edit. "
            "Re-run it after any retrieval change so the eval gate scores current "
            "behavior rather than a snapshot."
        ),
        "_generated": {
            "top_k": top_k,
            "with_answers": with_answers,
            "strategy": strategy,
            "embed_model": config.EMBEDDING_MODEL_NAME,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} results to {out_path}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", default=str(HERE / "relevance_judgments.json"))
    ap.add_argument("--out", default=str(HERE / "answers_and_citations.json"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--strategy", default="A")
    ap.add_argument("--with-answers", action="store_true",
                    help="also generate answers (needs an LLM key; costs tokens)")
    ap.add_argument("--skip-corpus-check", action="store_true",
                    help="run even when the judged papers are not indexed (scores will be 0)")
    args = ap.parse_args()

    import config
    import vector_store

    collection = vector_store.get_or_create_collection()
    count = collection.count()
    if count == 0:
        # Scoring an empty corpus reports 0.0 across the board and reads as a
        # catastrophic regression, when the real problem is that nothing is indexed.
        print("ERROR: the corpus is empty — ingest documents before running the eval.",
              file=sys.stderr)
        return 2
    print(f"Corpus: {count} chunks in '{config.COLLECTION_NAME}'")

    problem = vector_store.check_index_compatibility(collection)
    if problem:
        print(f"WARNING: {problem}", file=sys.stderr)

    if not args.skip_corpus_check:
        missing = _corpus_mismatch(Path(args.judgments), collection)
        if missing:
            print(
                "\nERROR: the judged papers are not in the indexed corpus.\n"
                f"  judged but missing: {', '.join(sorted(missing))}\n"
                "\nEvery retrieval metric would score 0.000 — not because retrieval is\n"
                "broken, but because the judgments describe a different corpus than the\n"
                "one indexed. Scoring that would read as a catastrophic regression.\n"
                "\nFix one of:\n"
                "  - ingest the papers the judgments refer to, or\n"
                "  - rewrite relevance_judgments.json against the corpus you actually have\n"
                "    (paper_id values must match the indexed ids exactly), or\n"
                "  - pass --skip-corpus-check to run anyway and inspect the raw output.\n",
                file=sys.stderr)
            return 3

    run(Path(args.judgments), Path(args.out), args.top_k, args.with_answers, args.strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
