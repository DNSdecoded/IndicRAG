# v2.6 Baseline (v3 Phase 0)

Measured 2026-11-06 on the v2.6 pipeline (branch `dev-v3.0`, base `a49e7dc`) with
the Phase 0 harness. Every later v3 change is measured against these numbers.
Raw data: `baselines/fixture_v1.json` (gate baseline, per query) and
`baselines/study_v2.6.json` (latency, rerank and margin study).

## Setup

| | |
| --- | --- |
| Hardware | Intel i5-6600T, 4 cores, 15 GB RAM, no GPU |
| Corpus | `fixtures/manifest.json` v2: 41 CC BY 4.0 / CC0 arXiv papers, corpus checksum `df932bc0ff9d1b73`, 2,861 chunks |
| Suite | `suites/fixture_v1.json`: 84 queries (60 English, 24 Hindi). **All unreviewed** (`reviewed: false`) |
| Models | BGE-M3 (int8 ONNX, CPU) dense; bge-reranker-v2-m3 cross-encoder; BM25 + RRF (k=60) |
| Retrieval | `top_k=10` chunks, rerank pool ×1, HyDE and ColBERT off, enrichment off |
| Relevance | Paper level: a run's chunks are de-duplicated into a ranked paper list |

CIs are 95% percentile bootstrap over queries (1,000 resamples, seed 20261003).
With 6–8 queries per Indic cell, those CIs are wide; read them as orders of
magnitude, not rankings.

## Retrieval quality (hybrid, the v2.6 default)

| lang | type | n | recall@10 | nDCG@10 | MRR | nDCG@10, dense-only |
| --- | --- | --- | --- | --- | --- | --- |
| all | all | 84 | 0.912 [0.863, 0.952] | 0.937 [0.901, 0.967] | 0.973 | 0.937 |
| en | broad | 6 | 0.518 [0.396, 0.648] | 0.708 [0.581, 0.837] | 1.000 | 0.702 |
| en | native | 36 | 0.958 [0.917, 1.000] | 0.975 [0.948, 0.995] | 0.986 | 0.963 |
| en | terminology | 18 | 0.917 [0.778, 1.000] | 0.935 [0.814, 1.000] | 0.944 | 0.970 |
| hi | code_mixed | 8 | 0.938 [0.812, 1.000] | 0.971 [0.927, 1.000] | 1.000 | 0.988 |
| hi | cross_lingual | 8 | 0.938 [0.812, 1.000] | 0.954 [0.880, 1.000] | 1.000 | 0.954 |
| hi | transliterated | 8 | 0.938 [0.812, 1.000] | 0.897 [0.736, 0.992] | 0.906 | 0.856 |

Weakest queries (nDCG@10): E19 (0.00, a terminology query on the metasurface
paper's "8 ring-shaped patterns"), H06r (0.36, romanized Hindi on PINN
magnetostatics), E50 (0.48, broad PINN-for-EM, 9 relevant papers), E47 (0.55).
The broad queries carry most of the headroom: their many relevant papers do not
fit in 10 chunks.

## Findings that change v3 decisions

**1. The BM25 leg shows no measured gain here.** Dense-only against hybrid, paired
over all 84 queries: recall@10 **+0.012** [−0.016, +0.047], nDCG@10 **+0.000**
[−0.019, +0.022]. Dense-only is non-inferior at the 2 pp margin and is *better*
on terminology queries (nDCG 0.970 vs 0.935), the class BM25 is meant to serve.
The plan's planted regression ("flip RRF to dense-only") is therefore not a
regression on this corpus. The live gate plants a real fault instead (embedder
down → sparse-only), which it catches: recall@10 −0.071 [−0.138, −0.014],
nDCG@10 −0.098 [−0.158, −0.047]. *Consequence for P0-RTR-1:* the case for
`lexical_exact` and for keeping BM25 on the hot path rests on corpora other than
this one. Re-test on a larger fixture before relying on it.

**2. Rerank is nearly all the latency and earns a small, real gain.** Warm
retrieval: mean 5.22 s, p50 5.07 s, p95 7.02 s per query. The cross-encoder is
**97.3%** of it (p50 4.92 s); dense search is 4 ms and BM25 2 ms (the query
embedding is outside these stages). Rerank lifts nDCG@10 by **+0.032**
[+0.011, +0.059], but helps only 9 of 84 queries and changes nothing for the
other 75.

**3. The BM25 top-2 margin does not reliably mark "rerank adds nothing".** AUC
of the margin for separating the queries rerank did not help: **0.69**
[0.48, 0.88]. The CI includes 0.5, so per §20 Phase 0 **the `lexical_exact`
rerank skip does not ship on this evidence.** Matching the literature, the
signal is corpus-dependent (arXiv 2606.25249).

**4. Little real traffic would route to `lexical_exact`.** On the dev box's
`query_log` (61 queries), the Phase 0 proxy rule (`baseline_study.lexical_exact_proxy`:
a quoted phrase, or ≤ 8 tokens with an identifier) matches **4.9%**
[0.0%, 9.8%]. Even a perfect rerank skip on that route cuts mean latency by
at most ~5%. The bigger lever is the rerank cost on every route.

**5. Romanized Hindi does not trail native script at recall.** Paired on the
same 8 questions, romanized − Devanagari: recall@10 **+0.000** [−0.188, +0.188],
nDCG@10 −0.057 [−0.241, +0.060]. Code-mixed − Devanagari: recall +0.000,
nDCG +0.017. The decision rule (build back-transliteration only if romanized
trails native by > 10 pp) is not met, so **do not build back-transliteration
for 3.0**. The CI is ±19 pp on 8 unreviewed questions, so re-decide on the
reviewed ≥ 40-query suite (P0-EVL-2).

**6. The documented key-fact counts were not reproducible.** The 2026-10-03
results (`RESULTS_2026-10-03.md`) report 17/26 at k5. `keyfacts.py` on the
committed answer files gets 14/26 (and 15, 19, 17, 18 for the other configs,
against the documented 17, 20, 19, 20). The missing facts (`8.93%`, `72%`,
`SAC`, `4.4`) are absent from the files entirely: `run_live.py` saved only claim
sentences longer than 20 characters, which drop short sentences and table rows.
`run_live.py` now saves the full `answer`, so future counts are reproducible.
The documented numbers should be treated as unverified.

**7. The `execute_python` sandbox was escapable (fixed in Phase 0).** Before the
fix, 6/6 probes reached `os` or read files. Routes included public module
aliases (`fractions.sys`, `re.enum.sys`), private ones (`collections._sys`),
generator frame walking to the wrapper's real `__import__`, and an uncalled
`str.format`. The validator also evaluated `9**99999999` *in the server
process*. It now rejects a 31-payload escape corpus and runs 16 legitimate
programs (`tests/test_sandbox.py`). Limits now cover CPU, memory, file writes,
process creation and output size.

**8. With the embedder down, a query with no BM25 hits fails the request.**
`rag._sparse_only_retrieval` returns `None` when BM25 finds nothing, and
`retrieve_context` then re-raises the embedding error. Under the planted fault,
3 of 8 Devanagari queries failed this way (an English corpus has no Devanagari
tokens). This is existing, deliberate behaviour ("nothing to degrade to"); it
belongs to the chaos suite (G-REL), which should decide between an error and an
empty degraded result.

## Ingest

Cold build of the 41-paper fixture index: about 42 minutes for 2,861 chunks,
**~0.9 s per chunk** (the 50-minute wall clock minus the ~7-minute suite run;
derived, not instrumented). The 2026-10-03 figure was ~1.3 s per chunk on the
same class of machine.

## Not measured in Phase 0

Each of these is required by §16/§17 and is still open. None can be measured
without what is listed.

| Metric | Needs |
| --- | --- |
| Key facts, groundedness, unsupported-claim rate, citation precision/recall on the fixture suite | An answer run (`run_live.py --with-answers`, one pinned model, LLM key); nightly job |
| NLI verdict accuracy, false-verification rate, ECE of the v2.6 heuristic | The labelled claim set (§8.3, Phase 3) |
| NLI per-pair cost, to size `EVIDENCE_MAX_CLAIMS` × `EVIDENCE_PER_CLAIM` | A verification timing run; do it with the first answer run |
| Time to first token, agent end-to-end p50, RSS per worker, p99 under concurrency | `docs/Eval/load.py` (§17) |
| Telugu, Tamil, Bengali | Queries plus native-speaker reviewers; published as **unmeasured** |
| Any *gated* Indic number | Hindi here is unreviewed: it reports but gates nothing (§16.3) |
