# v2.0.0 — Agentic RAG Pipeline with Reflexion, arXiv & Open Access Search

The biggest release since v1.0. IndicRAG v2.0 adds a full **agentic pipeline** — a LangGraph state machine that plans queries, selects and executes tools, generates answers, and **self-corrects through reflexion loops** — while keeping the proven Standard RAG mode available side-by-side.

---

## Highlights

### 🤖 Agentic RAG Pipeline
A multi-step pipeline powered by [LangGraph](https://github.com/langchain-ai/langgraph) that goes far beyond single-pass retrieval:

**Query Planner** → **Tool Selector** → **Tool Executor** → **Answer Generator** → **Reflexion Evaluator** → loop or finalize

- The **Reflexion Evaluator** checks every answer for faithfulness (via `verify.check_claims()` using the existing bge-reranker-v2-m3) and completeness (via a Gemini Flash judge). If either score falls below 0.75, the agent can:
  - **Regenerate** the answer from existing context
  - **Retrieve more** using additional tools
  - **Reformulate** the original query at the planning stage
- Hard-capped at **3 reflexion iterations** and **45-second wall-clock timeout** to prevent runaway loops.
- Uncited answers (no `[n]` markers) are force-regenerated regardless of completeness score.

### 🔧 6 Agent Tools
All available to the LLM via google-genai native function calling (`FunctionCallingConfig(mode="ANY")`):

| Tool | Source | API Key? | What It Does |
|------|--------|----------|-------------|
| `indicrag_retrieval` | Local corpus | No | Hybrid BM25 + dense retrieval with cross-encoder reranking. Supports query expansion (3 LLM-generated variants). |
| `arxiv_search` | [arXiv API](https://arxiv.org/) | No | Search by topic, author, or paper ID. Returns titles, abstracts, authors, dates, and PDF links. |
| `open_access_search` | [Semantic Scholar](https://www.semanticscholar.org/) + [OpenAlex](https://openalex.org/) fallback | No | Broad academic search across all disciplines. Returns citation counts and open-access PDFs. Automatic fallback to OpenAlex when Semantic Scholar rate-limits (429). |
| `web_search` | [Tavily](https://tavily.com/) | Yes | General web search for current events and non-academic information. |
| `calculate` | [numexpr](https://github.com/pydata/numexpr) | No | Math expression evaluation (supports sqrt, log, trig, exponentiation). |
| `execute_python` | Process isolation | No | Python execution in a subprocess sandbox with string blocklist, stripped environment, and 10s timeout. |

### 📄 Rich Source Display
Agent responses now include full paper metadata — not just citation numbers:
- Paper title (linked to source)
- Authors, publication year, citation count
- Open-access PDF download links
- Color-coded badges by provider (arXiv, Semantic Scholar, OpenAlex, Corpus, Web)

### ⚡ Multi-Key Load Balancing
The agentic pipeline makes 4-8+ Gemini API calls per query. New support for multiple API keys with round-robin load balancing distributes the load and avoids per-key rate limits:
```ini
LLM_API_KEYS=key-one,key-two,key-three
```

### 🖥️ Redesigned Web UI
- **Pipeline Mode toggle** — switch between Standard RAG and Agentic RAG in the sidebar
- **Agent progress stepper** — animated 5-step indicator (Planning → Selecting tools → Executing → Generating → Evaluating) with live elapsed timer, replacing the static typing dots
- **Source cards** — each retrieved paper rendered as a card with metadata and PDF links
- **Tool call log** — shows which tools were invoked, their arguments, and execution latency

### 🗄️ Three-Layer TTL Cache
New `cache.py` module with thread-safe LRU caches across the entire pipeline:

| Cache | What It Saves | Max Size | TTL | Impact |
|-------|---------------|----------|-----|--------|
| **LLM response cache** | Identical prompts skip Gemini API call entirely | 128 | 10 min | Saves cost + latency on repeated/similar queries |
| **Retrieval cache** | Same query skips embedding + ChromaDB + BM25 + reranking | 64 | 5 min | Eliminates ~200ms per repeated retrieval |
| **Tool result cache** | arXiv, Semantic Scholar, web search results cached across reflexion loops | 64 | 3 min | Prevents redundant API calls during agent retries |

- All sizes and TTLs are **env-configurable** (`LLM_CACHE_SIZE`, `LLM_CACHE_TTL`, etc.)
- Retrieval cache **auto-invalidates on document ingest** (single and bulk)
- `GET /cache/stats` endpoint for observability (hit rate, size, config)
- `DELETE /cache` endpoint to manually clear all caches
- Calculator and Python sandbox are intentionally NOT cached (cheap + deterministic)

---

## New Endpoints

### `GET /cache/stats`
Returns hit/miss stats for all three caches:
```json
{
  "llm": {"hits": 12, "misses": 5, "hit_rate": 0.706, "size": 5, "max_size": 128, "ttl_seconds": 600},
  "retrieval": {"hits": 8, "misses": 3, "hit_rate": 0.727, "size": 3, "max_size": 64, "ttl_seconds": 300},
  "tool": {"hits": 4, "misses": 6, "hit_rate": 0.4, "size": 6, "max_size": 64, "ttl_seconds": 180}
}
```

### `DELETE /cache`
Clears all caches. Returns `{"status": "cleared"}`.

### `POST /agent/query`

```json
// Request
{
  "question": "What are the latest advances in antenna optimization using ML?",
  "strategy": "A",
  "session_id": null
}

// Response
{
  "answer": "...",
  "language": "en",
  "session_id": "uuid",
  "reflexion_iterations": 1,
  "sources": [
    {
      "title": "Design and Comparative Analysis of THz Antenna...",
      "authors": "Rachit Jain, Vandana Vikas Thakare, P. K. Singhal",
      "year": "2024",
      "citations": 58,
      "source": "https://doi.org/...",
      "pdf_url": "https://...",
      "section": "openalex"
    }
  ],
  "tool_calls": [
    {"tool": "open_access_search", "args": {"query": "..."}, "latency_ms": 655.0},
    {"tool": "arxiv_search", "args": {"query": "..."}, "latency_ms": 19651.0}
  ],
  "processing_time": 313.9
}
```

---

## New Files (15)

```
agent/                          # Agentic RAG pipeline
├── __init__.py
├── state.py                    # AgentState TypedDict + ReflexionFeedback
├── tool_declarations.py        # google-genai FunctionDeclaration objects (6 tools)
├── tool_executor.py            # Python implementations: corpus, arXiv, S2/OpenAlex, web, calc, sandbox
├── graph.py                    # LangGraph StateGraph with conditional reflexion edges
└── nodes/
    ├── __init__.py
    ├── query_planner.py        # Language detection + query decomposition
    ├── tool_selector.py        # Gemini function calling (mode=ANY)
    ├── tool_executor_node.py   # Tool dispatch + context accumulation + audit log
    ├── answer_generator.py     # Reuses rag.format_context / build_prompt / llm_generate
    ├── reflexion_evaluator.py  # Faithfulness (check_claims) + completeness (Gemini judge)
    └── finalizer.py            # Terminal node

cache.py                            # Thread-safe TTL LRU cache (LLM, retrieval, tool instances)

tests/
└── test_agent.py               # 16 unit tests + 1 integration test
```

## Modified Files (8)

| File | Change |
|------|--------|
| `requirements.txt` | +4 deps: `langgraph`, `tavily-python`, `numexpr`, `arxiv` |
| `config.py` | Added `AGENT_MAX_TOKENS` (4096), `LLM_API_KEY_POOL` (multi-key), 6 cache config vars |
| `rag.py` | Round-robin client pool + LLM response cache + retrieval result cache |
| `api_server.py` | `POST /agent/query`, `GET /cache/stats`, `DELETE /cache`, cache invalidation on ingest |
| `start_server.py` | Pre-flight check now recognizes `LLM_API_KEYS` (multi-key) |
| `static/index.html` | Pipeline mode toggle, agent progress stepper, source cards, tool call display, XSS-safe URL rendering |
| `.env.example` | Documented `LLM_API_KEYS`, `TAVILY_API_KEY`, `AGENT_MAX_TOKENS` |
| `README.md` | Full rewrite for v2.0: architecture diagram, agent API docs, new project structure |

---

## New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | >=0.2.0 | State machine for the agent pipeline |
| `tavily-python` | >=0.3.0 | Web search tool |
| `numexpr` | >=2.8.0 | Math expression evaluation tool |
| _(subprocess sandbox)_ | _(stdlib)_ | Process-isolated Python execution (replaced RestrictedPython) |
| `arxiv` | >=2.1.0 | arXiv paper search tool |

---

## New Environment Variables

| Variable | Default | Required? | Description |
|----------|---------|-----------|-------------|
| `LLM_API_KEYS` | — | No | Comma-separated Gemini keys for round-robin load balancing. Overrides `LLM_API_KEY` when set. |
| `TAVILY_API_KEY` | — | Only for `web_search` tool | Tavily API key. Agent works without it (other 5 tools are free). |
| `AGENT_MAX_TOKENS` | `4096` | No | Max output tokens for agent answer generation (vs 2048 for standard RAG). |
| `LLM_CACHE_SIZE` | `128` | No | Max entries in LLM response cache. |
| `LLM_CACHE_TTL` | `600` | No | LLM cache TTL in seconds (10 min). Set to `0` to effectively disable. |
| `RETRIEVAL_CACHE_SIZE` | `64` | No | Max entries in retrieval result cache. |
| `RETRIEVAL_CACHE_TTL` | `300` | No | Retrieval cache TTL in seconds (5 min). Auto-invalidated on ingest. |
| `TOOL_CACHE_SIZE` | `64` | No | Max entries in agent tool result cache. |
| `TOOL_CACHE_TTL` | `180` | No | Tool cache TTL in seconds (3 min). |

---

## Design Decisions

**Why google-genai native function calling, not LangChain tools?**
The codebase already uses `google-genai` for LLM calls. Using native `types.FunctionDeclaration` + `FunctionCallingConfig(mode="ANY")` avoids adding LangChain LLM wrappers as a dependency, keeps the tool declarations type-safe, and reuses the existing `rag._get_client()`.

**Why OpenAlex as fallback?**
Semantic Scholar's free tier rate-limits at 100 requests per 5 minutes. The agentic pipeline can hit this during heavy use. [OpenAlex](https://openalex.org/) is fully open (CC0 data), has effectively no rate limits, covers 250M+ works, and returns the same metadata (title, authors, year, citations, open-access PDFs).

**Why `AGENT_MAX_TOKENS=4096` instead of just increasing `LLM_MAX_TOKENS`?**
The agent retrieves context from multiple sources (corpus + arXiv + open access) — substantially more than standard single-pass RAG. A separate config lets you tune agent verbosity without affecting the leaner standard responses.

**Why round-robin, not least-loaded?**
Gemini API keys don't expose quota usage in response headers. Round-robin is simple, stateless, thread-safe (using `itertools.cycle`), and distributes calls evenly — which is optimal when all keys have the same quota.

---

## Backward Compatibility

- **Zero changes to existing endpoints.** `/query`, `/chat`, `/ingest`, `/upload`, `/papers`, `/stats`, `/health`, `/purge/*` all work exactly as before.
- **Zero changes to existing modules.** `rag.py`, `verify.py`, `rerank.py`, `bm25_search.py`, `embeddings.py`, `vector_store.py`, `translation.py`, `lang_utils.py`, `pdf_utils.py`, `ingest.py` — none modified in their public APIs.
- **Single `LLM_API_KEY` still works.** Multi-key is opt-in via `LLM_API_KEYS`.
- **`TAVILY_API_KEY` is optional.** The agent functions without it — 5 of 6 tools require no API key.
- **Standard RAG remains the default** in the web UI. Agentic mode is an explicit toggle.

---

## Testing

```bash
# Unit tests (no API keys needed, no network for most)
pytest tests/test_agent.py -v -m "not integration and not network"

# Network tests (requires internet, may hit S2 rate limits)
pytest tests/test_agent.py -v -m "network"

# Full integration test (requires running IndicRAG + API keys)
pytest tests/test_agent.py -v -m "integration"
```

**11 unit tests** covering: state schema, tool dispatch, answer generator (verifies it reuses `rag.format_context`/`build_prompt`/`llm_generate`), reflexion hard-stop, routing logic, finalizer fallback, arXiv search, tool registry.

---

## Upgrade Guide

```bash
# 1. Pull the latest code
git pull origin main

# 2. Install new dependencies
pip install -r requirements.txt

# 3. Update .env (optional — for agent web search and multi-key)
# Add to your existing .env:
# TAVILY_API_KEY=your-tavily-key
# LLM_API_KEYS=key1,key2,key3
# AGENT_MAX_TOKENS=4096

# 4. Start the server
python start_server.py
```

No database migrations. No re-ingestion required. Your existing ChromaDB data and papers work as-is.

---

## What's Next

- Streaming agent progress via SSE (replace client-side step simulation with real server events)
- Threshold calibration tooling against `docs/Eval/relevance_judgments.json`
- Agent conversation memory (multi-turn agentic chat)

---

**Full Changelog:** v1.5.0...v2.0.0

**Stats:** 15 new files, 8 modified files, 4 new dependencies, 3 new API endpoints, 6 agent tools, 3 cache layers, 16 unit tests
