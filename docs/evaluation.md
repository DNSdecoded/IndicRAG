# Evaluation

IndicRAG's quality gate scores the **live pipeline** on a corpus it ingests
itself, so it can never again grade a corpus that is not indexed. Through v2.6
the CI gate scored `answers_and_citations.json`, a hand-pasted snapshot about
8 papers that were absent from the live index; no retrieval change could move
it. The measured v2.6 numbers are in
[`Eval/BASELINE_v2.6.md`](Eval/BASELINE_v2.6.md). The manual qualitative report
this page used to hold is in git history (before v3 Phase 0).

## Pieces

| Path | What it is |
| --- | --- |
| `Eval/fixtures/manifest.json` | 41 CC BY 4.0 / CC0 arXiv papers, pinned by exact version and sha256 |
| `Eval/fixtures/fetch.py` | Downloads missing fixtures; `--verify` checks every hash (no network) |
| `Eval/suites/fixture_v1.json` | Judged queries over the fixtures (schema below) |
| `Eval/live_gate.py` | Builds an isolated index, runs retrieval, applies the gate |
| `Eval/run_live.py` | Runs a suite through `rag.retrieve_context` (and optionally answers) |
| `Eval/evaluate.py` | Scores a run: P/R@k, MRR, nDCG@10, recall@10/20, grounding; `--baseline` gate |
| `Eval/stats.py` | Seeded bootstrap CIs and the paired regression / non-inferiority tests |
| `Eval/keyfacts.py` | Key-fact scorer: an answer passes when it states every `key_facts` entry |
| `Eval/baseline_study.py` | Stage latency, rerank gain vs BM25 margin, `query_log` route share |
| `Eval/baselines/fixture_v1.json` | The committed v2.6 metrics the gate compares against |

## Running it

```bash
python docs/Eval/fixtures/fetch.py                 # once; verifies sha256 of every PDF
python docs/Eval/live_gate.py                      # the CI gate: exit 1 on a regression
python docs/Eval/live_gate.py --selftest           # proves a clean run passes and a planted one fails
python docs/Eval/live_gate.py --write-baseline     # re-measure the baseline (commit the JSON)
pytest -q docs/Eval                                # harness self-tests
```

The first `live_gate.py` run ingests the fixtures (cold build, tens of minutes
on a 4-core CPU); later runs reuse the index. The index lives in
`docs/Eval/.fixture_index/<fixture checksum>_<embed model>_c<CHUNKER_VERSION>/`
and points `CHROMA_DB_DIR`, `BM25_CACHE_DIR` and `SESSIONS_DB_PATH` there, so it
never touches `chroma_db/` or `sessions.db`. Ingest runs with arXiv enrichment,
multimodal captions, HyDE and ColBERT off: nothing after `fetch.py` touches the
network, and titles stay deterministic.

## The gate

Per query, `evaluate.py` records recall@10 and nDCG@10 (paper level; a paper's
chunks are de-duplicated so one paper cannot fill the top k). `--baseline`
compares them to the committed run with a **paired bootstrap** over queries
(1,000 resamples, fixed seed): a metric regresses when the 95% CI upper bound of
mean(new − baseline) is below 0. One noisy query cannot fail the gate; a
pipeline-wide drop always does. A missing baseline fails rather than skips.

`--selftest` runs the suite twice: as is, and with a planted fault
(`run_live.py --plant embedder_down`: the query embedder raises, so retrieval
takes the real sparse-only degraded path). The metric gate must pass the first
and fail the second on its own, otherwise the gate itself is broken. Separately,
a normal gate run fails whenever any result comes back `degraded`.

Two lessons from building it:

- **A saturated suite detects nothing.** On the first 14-paper corpus, 70 of 74
  queries scored perfectly, and the CI upper bound sat at exactly 0. The corpus
  grew to 41 papers with near-neighbour distractors and terminology queries.
- **"Flip to dense-only" is not a regression on this corpus.** The plan's
  example perturbation scored recall@10 +0.012 (CI −0.016 to +0.047) against
  hybrid. The gate was right to pass it: the BM25 leg shows no measured
  paper-level gain here (see BASELINE_v2.6.md). A planted regression must be
  a real one.

In CI the `live-gate` job caches `models/`, the fixture PDFs and the index (its
cache key covers the manifest and the chunking and embedding code). The old
snapshot `Eval gate` step stays until the live gate has two green weeks.

## Suite schema

`relevance_judgments.json`'s schema, plus four fields:

```json
{"id": "H01r", "lang": "hi", "type": "transliterated",
 "text": "pixel aadharit millimeter-tarang patch antenna ke ...",
 "relevant_papers": ["2608.23469v1"], "relevance_grades": {"2608.23469v1": 3},
 "key_facts": [], "parallel_of": "E01", "reviewed": false}
```

- `relevance_grades`: 3 = answered by this paper, 2 = substantially relevant, 1 = related. Every query has a grade-3 paper.
- `type`: `native`, `terminology`, `broad` (several relevant papers), `cross_lingual` (Indic script, English corpus), `transliterated` (Latin script), `code_mixed`.
- `parallel_of`: the English query this one restates, so per-form recall can be compared on the same questions.
- `reviewed`: `false` until a reviewer signs off. An Indic language gates a release only with ≥ 40 queries reviewed by a native speaker (plan §16.3); otherwise it is published as "unmeasured".

`docs/Eval/test_suites.py` enforces this schema and that every fixture paper is targeted by some query.

## Adding a language

1. Write ≥ 40 queries in the §9 mix (10 native, 10 cross-lingual, 8 code-mixed, 6 transliterated, 6 terminology), each with `parallel_of` where it restates an English query.
2. Have a native speaker review the wording and relevance; set `reviewed: true` only after that.
3. Run `live_gate.py --write-baseline`, then commit the suite and the new baseline together.

## Key facts

`keyfacts.py` checks each `key_facts` string against the answer after NFKC,
unified dashes, removed markdown emphasis, casefolding and **removed whitespace**
(answers write `8.93 %` with a narrow no-break space). It needs the full answer:
`run_live.py --with-answers` now stores `answer`. Older files hold only claim
sentences longer than 20 characters, which lose short sentences and table rows
and under-count; see BASELINE_v2.6.md for the effect.

## Generation runs

Answer metrics cost tokens and vary run to run, so they run nightly, not per PR:
`run_live.py --with-answers` pins one model with failover disabled so every
answer comes from it.
