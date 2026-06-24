# IndicRAG v2.0 — Implementation Instructions

**Read first:** `roadmap.md` (what & when) and `planning.md` (architecture).
This file is the **build sequence** — the order to implement v2.0, with conventions and a definition of done per milestone.

---

## Conventions (match the existing codebase)

- **Minimal dependencies.** No agent framework. Custom loop on `google-genai` (already a dependency). Justify any new pin in `requirements.txt` with a comment, as the repo already does.
- **Module style.** Plain functions + small dataclasses, module-level `logger = logging.getLogger(__name__)`, type hints, docstrings matching the existing `rag.py` style.
- **Config.** Constants and model names go in `config.py`; agent *prompts* go in `agent/prompts.py` to avoid bloating config.
- **No secrets in code.** Reuse `config.LLM_API_KEY` / env loading. External API keys (v2.1) follow the same `os.getenv` + `.env` pattern.
- **Back-compat.** `/query` and `/chat` keep working unchanged. The agent is additive.
- **Commits.** Do not add `Co-Authored-By: Claude` trailers (project preference). Branch off `main`; PR for review.

---

## Milestone sequence — v2.0

Build in this order; each step is independently testable.

### Step 1 — Tool registry foundation (`agent/tools/base.py`)
- Define `Tool` protocol, `ToolResult`, `ToolRegistry`.
- `as_gemini_declarations()` converts tools → Gemini function declarations.
- `dispatch(name, args)` routes a model tool-call to the right tool.
- **DoD:** unit tests register a dummy tool, produce a valid declaration, dispatch a call.

### Step 2 — Agent state & events (`agent/state.py`, `agent/events.py`)
- `AgentState`, `PlanStep`, `Evidence`, `Citation` dataclasses.
- Event types: `PLAN, STEP, TOOL_CALL, TOOL_RESULT, VERIFY, ANSWER, DONE, ERROR`, each JSON-serializable for SSE.
- **DoD:** round-trip serialize every event type; state mutates cleanly across iterations.

### Step 3 — Corpus tools (`agent/tools/corpus.py`)
- `sub_query_retrieve(query, top_k, filter)` → wraps `rag.retrieve_context`.
- `extract_figure_table(paper_id, ref)` → pull a specific figure/table region from parsed PDF.
- `metadata_filter(...)` → constrain retrieval by paper/section/year.
- First, refactor `rag.py` to expose pure retrieval/format helpers (leave `answer_question` intact).
- **DoD:** each tool runs against a seeded ChromaDB fixture and returns grounded chunks.

### Step 4 — Planner (`agent/planner.py`)
- Prompt Gemini to decompose a question into an ordered sub-question plan (with tool hints).
- Trivial questions → single-step plan (fast path).
- **DoD:** multi-hop question yields ≥2 sub-questions; trivial question yields 1.

### Step 5 — Executor (`agent/executor.py`)
- For each plan step, call Gemini with tool declarations; dispatch tool-calls; append `Evidence`.
- Emit `TOOL_CALL` / `TOOL_RESULT` events.
- **DoD:** integration test with mocked Gemini drives ≥1 tool call per step and collects evidence.

### Step 6 — Verifier (`agent/verifier.py`)
- Check each prospective claim against its supporting chunk text (grounding).
- Emit `VERIFY` event; if gaps and `iterations < max_iterations`, return refined sub-queries to the Executor.
- **DoD:** ungrounded claim triggers exactly one re-retrieval; cap stops infinite loops.

### Step 7 — Synthesizer (`agent/synthesizer.py`)
- Compose the final answer in the target language with only verified citations.
- Reuse the structured-output discipline from `config.SYSTEM_PROMPT`.
- **DoD:** final answer cites only verifier-confirmed sources; A and B modes both produce correct-language output.

### Step 8 — Orchestrator (`agent/loop.py`)
- Wire Planner→Executor→Verifier→Synthesizer over `AgentState`; handle A/B mode at the boundary (translate in/out for B).
- **DoD:** end-to-end loop answers a multi-hop question over a seeded corpus.

### Step 9 — API + streaming (`api_server.py`)
- `POST /agent/stream` (SSE over the event stream) and `POST /agent` (non-streaming).
- Extend `/chat` with optional `agentic` flag.
- **DoD:** SSE client receives ordered events; first event < 1s; non-streaming returns final result + event log.

### Step 10 — UI (`static/index.html`)
- Live panels for plan, tool calls, and verification; final answer with citations.
- Mode toggle (A/B) and an "agentic" switch alongside the existing strategy cards.
- **DoD:** a user watches the agent decompose, retrieve, verify, and answer in real time.

### Step 11 — Eval & version bump
- Add a multi-hop / cross-paper eval set (English + ≥3 Indic languages); run alongside the existing eval.
- Confirm no single-hop regression. Bump `config.VERSION` to `2.0.0`.
- **DoD:** roadmap v2.0 acceptance criteria all pass.

---

## v2.1 (after v2.0 ships)

- `agent/tools/scholarly.py`: `arxiv_search`, `semantic_scholar_search`, `web_search` implementing the same `Tool` protocol.
- Add `origin: "corpus"|"external"` to `Evidence`/`Citation`; surface origin badges in UI.
- On-the-fly ingest: dedup by DOI/arXiv id before adding to ChromaDB.
- Verifier gates external escalation (only when corpus is insufficient). Add caching + backoff.

## v2.2 (after v2.1)

- `agent/tools/execution.py`: sandboxed `run_python` — no network, capped CPU/mem/time, ephemeral.
- Async research-job runner reusing the `/ingest/all` job-store + `/ingest/status/{job_id}` pattern.
- Report generation (sectioned, cited); optional multi-agent planner spawning sub-researchers for long horizons.
- Long-task dashboard in UI.

---

## Definition of Done (program)

A user asks a complex, multi-part scientific question in an Indic language; the agent decomposes it, retrieves and (v2.1+) searches externally, verifies grounding, optionally (v2.2) executes code to check a derivation, and streams a fully-cited answer in the user's language — with every step visible.

---

## Next action

This planning set is the design artifact. To proceed to a concrete, step-by-step implementation plan for **v2.0 Step 1–11**, invoke the `writing-plans` skill against this folder.
