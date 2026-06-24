# IndicRAG v2.0 — Agentic Scientific RAG: Release Roadmap

**Status:** Approved direction (brainstorm complete)
**Created:** 2026-06-18
**Owner:** sanjay sakhinala

---

## Vision

Transform IndicRAG from a **single-shot retrieve→generate pipeline** into an **agentic scientific research assistant** that decomposes complex questions, uses tools, verifies its own answers against sources, and runs autonomous multi-paper research — all while preserving the multilingual Indic-language focus.

## North-Star Architecture

A **custom agent loop** built on Gemini native function-calling (no heavy framework), following the **Planner → Executor → Verifier → Synthesizer** pattern. Each phase emits streaming events so the UI shows the agent's reasoning live.

```
User query
   │
   ▼
[Planner]      decompose into a sub-question plan
   │
   ▼
[Executor]     for each step: pick tool(s), call, collect evidence   ◄──┐
   │                                                                     │ re-plan / re-retrieve
   ▼                                                                     │
[Verifier]     every claim grounded? citations valid? ── insufficient ──┘
   │ sufficient
   ▼
[Synthesizer]  compose final answer in user's language + citations
```

The existing `rag.py` retrieval/generation becomes the **first tools** the executor can call. Nothing is thrown away — it is wrapped.

---

## Release Phases

### v2.0 — Agentic Core (shippable agentic Q&A)

**Goal:** A working Planner→Executor→Verifier→Synthesizer loop over the *existing corpus*, with streaming, that measurably beats v1.5 on multi-hop questions.

**Scope**
- Agent orchestration loop (custom, Gemini function-calling)
- Query decomposition (Planner)
- Corpus-only tool tier: sub-query retrieval, figure/table extraction, metadata filter
- Self-verification loop (citation grounding + targeted re-retrieval)
- Streaming agent steps over SSE
- Both multilingual modes (A native / B English-pivot) preserved as a per-request toggle
- UI: live plan, tool-call, and verification panels

**Out of scope (deferred):** external search, code execution, autonomous workflows.

**Acceptance criteria**
- Multi-hop question that v1.5 answers incompletely is answered with correct cross-paper synthesis.
- Every factual sentence in the final answer carries a citation that the verifier confirmed against retrieved text.
- Agent steps stream to the UI in < 1s to first event.
- Both A and B modes pass the existing eval set with no regression on single-hop questions.

---

### v2.1 — External Knowledge

**Goal:** The agent can reach beyond the uploaded corpus when local evidence is insufficient.

**Scope**
- Scholarly tool tier: arXiv, Semantic Scholar, web search
- Provenance tracking: every chunk tagged `corpus` vs `external`; final answer distinguishes them
- On-the-fly ingestion: agent can pull a fetched paper into ChromaDB mid-task (dedup by DOI/arXiv id)
- Rate limiting + caching for external APIs
- UI: source-origin badges (local vs external) on citations

**Acceptance criteria**
- For a question with no local answer, the agent retrieves a relevant external paper, cites it with origin marked, and (optionally) ingests it.
- No external call is made when the corpus already answers the question (verifier gates escalation).
- External provenance is never silently merged with corpus citations.

---

### v2.2 — Execution + Autonomous Workflows

**Goal:** The agent can compute/verify derivations and run long-horizon research tasks.

**Scope**
- Sandboxed code/math execution tool (isolated, resource-capped, no network)
- Autonomous research workflows: multi-paper literature review, structured report generation
- Async job model for long tasks (reuse the existing `/ingest/all` job-store pattern)
- Optional multi-agent decomposition for very long tasks (planner spawns sub-researchers)
- UI: long-running task dashboard with progress + downloadable report

**Acceptance criteria**
- A derivation/calculation question is verified by executing code, with the execution trace shown.
- "Review the literature on X across my corpus" produces a sectioned, cited report as an async job.
- Sandbox cannot reach network or host filesystem; resource limits enforced.

---

## Dependency Order (what unblocks what)

```
v2.0 agent loop + tool registry  (foundation — everything depends on it)
        │
        ├──► v2.1 external tools  (new Tool implementations, provenance)
        │
        └──► v2.2 execution tool + async workflows  (sandbox + job runner + optional multi-agent)
```

The **tool registry and agent state model** built in v2.0 are the contract every later phase plugs into. Get them right first.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Agent loops are slow / expensive (many LLM calls) | Cap plan depth & total steps; cache sub-query retrievals; fast-path trivial questions straight to v1.5 pipeline |
| Verifier rejects valid answers (over-strict) | Tune grounding threshold against eval set; allow "single-source answer" path |
| Multilingual tool-calling unreliable in mode A | Keep mode B (English-internal) as the safe default for tool-heavy tasks |
| External APIs flaky/rate-limited (v2.1) | Cache + retry/backoff; degrade gracefully to corpus-only |
| Code-exec security (v2.2) | Hard sandbox: no network, capped CPU/mem/time, ephemeral container |

---

## Success Metric (program-level)

On a held-out set of **multi-hop, cross-paper scientific questions** in English + ≥3 Indic languages, v2.0 should show a clear lift in answer completeness and citation-grounding accuracy over v1.5, with no regression on single-hop questions and acceptable streaming latency.

See `planning.md` for architecture detail and `instruction.md` for the build sequence.
