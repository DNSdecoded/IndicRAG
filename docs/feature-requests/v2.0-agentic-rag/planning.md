# IndicRAG v2.0 — Agentic Architecture & Design

**Companion to:** `roadmap.md` · **Build guide:** `instruction.md`
**Created:** 2026-06-18

---

## 1. Where we start (v1.5 today)

| Module | Responsibility |
|--------|----------------|
| `config.py` | All constants, prompts, model names, version |
| `rag.py` | `retrieve_context`, `format_context`, `build_prompt`, `llm_generate`, `answer_question` (A/B), `answer_with_history` |
| `vector_store.py` | ChromaDB: collection, `add_documents`, `search`, stats, delete |
| `embeddings.py` | multilingual-e5-base query/passage embedding |
| `lang_utils.py` | language detection + names |
| `translation.py` | NLLB-200 to/from English (Strategy B) |
| `pdf_utils.py` / `ingest.py` | PDF parse, chunk, ingest |
| `api_server.py` | FastAPI: `/query`, `/chat`, `/ingest*`, `/upload`, `/papers`, `/stats`, `/purge*`, `/health` |

**Flow today:** detect lang → embed → ChromaDB top-k → format context → one Gemini call → regex-extract citations. One pass, no iteration.

---

## 2. Target architecture (v2.0)

A new `agent/` package sits *above* the existing modules. The current `rag.py` functions are not rewritten — they become **tools** the agent calls.

```
agent/
  loop.py          # orchestrator: runs Planner→Executor→Verifier→Synthesizer
  state.py         # AgentState dataclass (plan, steps, evidence, citations, lang, mode)
  events.py        # streaming event types (PLAN, STEP, TOOL_CALL, TOOL_RESULT, VERIFY, ANSWER, DONE)
  planner.py       # decompose query → ordered sub-question plan
  executor.py      # for each step, let Gemini pick + call tools, gather evidence
  verifier.py      # grounding check per claim; decide sufficient / re-retrieve
  synthesizer.py   # compose final answer in target language with citations
  prompts.py       # agent-specific prompt templates (kept out of config.py bloat)
  tools/
    base.py        # Tool protocol + ToolRegistry + Gemini function-decl conversion
    corpus.py      # v2.0: sub_query_retrieve, extract_figure_table, metadata_filter
    scholarly.py   # v2.1: arxiv_search, semantic_scholar_search, web_search
    execution.py   # v2.2: run_python (sandboxed)
```

### 2.1 The agent loop (Planner → Executor → Verifier → Synthesizer)

1. **Planner** receives the (language-normalized) question and produces a `Plan`: an ordered list of sub-questions, each with a hint about which tool(s) likely apply. Trivial questions yield a single-step plan (fast path).
2. **Executor** walks the plan. For each step it calls Gemini with the registered tool function-declarations; Gemini decides which tool(s) to invoke. Tool results become `Evidence` entries in `AgentState`.
3. **Verifier** inspects the accumulated evidence against the plan. For each prospective claim it checks the supporting chunk actually contains it. If coverage is insufficient, it emits a re-retrieval directive (back to Executor with refined sub-queries), bounded by a max-iteration cap.
4. **Synthesizer** composes the final answer in the user's language (respecting A/B mode), attaching only verifier-confirmed citations.

### 2.2 Tool registry contract

`tools/base.py` defines a `Tool` protocol every tool implements:

```python
class Tool(Protocol):
    name: str
    description: str
    parameters: dict          # JSON schema for Gemini function-declaration
    def run(self, **kwargs) -> ToolResult: ...
```

A `ToolRegistry` collects tools and exposes:
- `as_gemini_declarations()` → list of function declarations for the Gemini call
- `dispatch(name, args)` → `ToolResult`

This contract is **the stable interface** for all later phases. v2.1 and v2.2 add new `Tool` implementations without touching the loop.

### 2.3 AgentState

```python
@dataclass
class AgentState:
    question: str
    language: str            # detected ISO code
    mode: str                # "A" (native) or "B" (english-pivot)
    plan: list[PlanStep]
    evidence: list[Evidence] # chunk text + metadata + origin (corpus|external)
    citations: list[Citation]
    iterations: int
    max_iterations: int
    events: list[Event]      # for streaming + audit trail
```

### 2.4 Multilingual handling (both modes preserved)

The A/B toggle from v1.5 is lifted to the agent boundary:
- **Mode A (native):** planner/executor/verifier prompts run in the user's language; tools that need English (e.g. embeddings already handle multilingual) receive the query as-is. Best nuance.
- **Mode B (english-pivot):** question is translated to English on entry (`translation.translate_to_english`), the entire agent loop runs in English, the final synthesized answer is translated back (`translation.translate_from_english`). Most reliable tool-calling.

Mode is a per-request field, defaulting from config. Tool-heavy/external tasks should prefer B.

### 2.5 Streaming

A new SSE endpoint streams `events.py` event objects as the loop runs. Each Planner/Executor/Verifier/Synthesizer transition and every tool call/result is an event. The UI renders these as live panels. Non-streaming callers get the final consolidated response (back-compat with `/query`).

---

## 3. API surface changes

| Endpoint | Change |
|----------|--------|
| `POST /agent/stream` | **New.** SSE stream of agent events for a question. Body: `{question, mode, session_id, max_iterations}` |
| `POST /agent` | **New.** Non-streaming agentic answer (runs loop, returns final result + event log) |
| `POST /query` | **Kept.** Routes to v1.5 single-shot pipeline (fast path / back-compat) |
| `POST /chat` | **Extended.** Optional `agentic: bool` flag to route through the agent loop with history |
| `/ingest*`, `/papers`, `/stats`, `/purge*` | Unchanged in v2.0 |

v2.1 adds provenance fields to citation objects. v2.2 adds async research-job endpoints mirroring `/ingest/all` + `/ingest/status/{job_id}`.

---

## 4. Data flow (v2.0, mode A, streaming)

```
POST /agent/stream {question, mode:"A"}
  → detect language
  → AgentState init
  → Planner: emit PLAN event (sub-questions)
  → loop:
      Executor step → emit TOOL_CALL / TOOL_RESULT events
      (corpus tool → rag.retrieve_context under the hood)
      Verifier → emit VERIFY event (grounded? gaps?)
        if gaps and iterations < max → refine sub-queries, continue
        else → break
  → Synthesizer → emit ANSWER event (final text + citations)
  → emit DONE
```

---

## 5. What stays, what's new

**Reused as-is:** `vector_store.py`, `embeddings.py`, `lang_utils.py`, `translation.py`, `pdf_utils.py`, ingestion, ChromaDB schema.

**Wrapped as tools:** `rag.retrieve_context`, `rag.format_context` (corpus tools call these).

**New:** the entire `agent/` package, SSE streaming, two new endpoints, agent prompt templates, UI step panels.

**Refactor (targeted):** `rag.py` is large and mixes retrieval + generation + orchestration. Extract the pure retrieval/format helpers so corpus tools import them cleanly; leave `answer_question`/`answer_with_history` intact as the fast-path v1.5 entry points.

---

## 6. Testing strategy

- **Unit:** each tool (`run` with fixtures), planner output shape, verifier grounding logic, A/B language routing.
- **Loop integration:** mocked Gemini + mocked tools to assert the Planner→Executor→Verifier→Synthesizer transitions and the re-retrieval cap.
- **Eval regression:** existing eval set must not regress on single-hop; new multi-hop eval set added for cross-paper synthesis, run in English + ≥3 Indic languages.
- **Streaming:** assert event ordering and first-event latency.

---

## 7. Open questions to resolve during v2.0 build

- Exact max-iteration / max-plan-depth caps (tune on eval set).
- Verifier grounding threshold (precision vs recall of claim-checking).
- Whether figure/table extraction needs a vision call or can reuse parsed PDF structure.
