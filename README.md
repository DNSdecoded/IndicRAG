# 🌐 IndicRAG — Multilingual Agentic Scientific RAG

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DNSdecoded/IndicRAG)
[![Code Wiki](https://img.shields.io/badge/Code%20Wiki-Documentation-blue)](https://codewiki.google/github.com/dnsdecoded/indicrag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.130+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.6%20Flash-blueviolet.svg)](https://ai.google.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent--pipeline-orange.svg)](https://github.com/langchain-ai/langgraph)
![Version](https://img.shields.io/badge/version-2.4-blue.svg)

![INDICRAG.png](https://cdn.jsdelivr.net/gh/free-whiteboard-online/Free-Erasorio-Alternative-for-Collaborative-Design@3a5f22554411d3d6df27ee788c2df99d583f2c91/uploads/2025-12-03T05-25-45-007Z-3i36rbzio.png)

A **production-ready** Retrieval-Augmented Generation system with an **agentic pipeline**, multilingual support for 10+ Indian languages, and tools for searching arXiv, Semantic Scholar, OpenAlex, and the web — alongside your own indexed document corpus.

Two pipelines ship side-by-side: **Standard RAG** (single-pass hybrid retrieval) and **Agentic RAG** (multi-tool planning with reflexion self-correction). Answers stream token-by-token over SSE, sessions survive restarts, and every retrieval knob is env-configurable. Now with **multi-provider LLM** support (Gemini + OpenRouter), **topic watches**, and **literature review reports**.

---

## What's New in v2.4

| Area | v2.3 | v2.4 |
|------|------|------|
| **Default model** | gemini-3.5-flash | **gemini-3.6-flash** — latest Gemini Flash, faster and more capable |
| **Multi-user support** | Single-user API key | **Per-user data isolation** — sessions, watches, feedback, preferences scoped to API key |
| **User preferences** | None | **GET/PUT /prefs/{user_id}** — per-user settings with opt-in `ENABLE_USER_PREFS` |
| **Admin API key** | Falls back to API_KEYS | **Dedicated ADMIN_API_KEY** for destructive /purge/* operations |
| **Session management** | Basic eviction | **Configurable** — `SESSION_MAX_AGE_HOURS`, `CHAT_HISTORY_MAX_TURNS` |
| **Rate limiting** | Per-IP only | **Per-API-key** — each user gets their own rate-limit bucket |
| **Linting** | CI failures on unused imports | **Clean CI** — fixed F401/E401 linting errors |

### v2.4 patch — correctness fixes

* **Faithfulness verification restored** — `verify.check_claims` was passing `(index, text)` tuples to the NLI model instead of the chunk text, so every call raised and every answer reported `confidence: 0.0` with no evidence.
* **Generation restored on `gemini-3.6-flash`** — models that reject `thinking_budget=0` returned `400 INVALID_ARGUMENT`. The Gemini backend now remembers per-model rejections and retries once without `thinking_config`.
* **Retrieval cache actually populates** — the cacheability check ran after the collection was materialized, so nothing was ever stored. Cached entries are now copied on both read and write so callers can't mutate them.
* **Tags filter returns results** — tags are stored as one comma-joined metadata string and must be filtered in Python, which needs an over-fetch (`TAGS_OVERFETCH`); without it a tag query returned nothing.
* **Cross-vendor failover is really cross-vendor** — the OpenRouter fallback was handed a bare Gemini model name, which OpenRouter rewrites to `google/<model>`, routing back to the vendor that just failed. It now picks a `/`-shaped slug from `LLM_SELECTABLE_MODELS`.
* **Selected model honoured in agentic answers** — the answer generator used the configured default instead of the requested model.
* **Web UI** — new Retrieval view, Library row actions (dry-run, edit metadata, delete, bulk delete), Depth + tags controls, BibTeX/Markdown evidence exports, saved reports, and an index-health diagnostics panel.

---

## ✨ Key Features

### 🤖 Agentic RAG Pipeline

* **LangGraph state machine** — query planner → tool selector → tool executor → answer generator → reflexion evaluator, with conditional loops
* **6 agent tools:**
  * **indicrag_retrieval** — hybrid BM25 + dense search with cross-encoder reranking on your indexed corpus, with optional **year-range filter**
  * **arxiv_search** — search arXiv by topic, author, or paper ID; returns abstracts, authors, PDF links
  * **open_access_search** — Semantic Scholar with automatic OpenAlex fallback (free, no API key); returns citation counts and open-access PDFs
  * **web_search** — Tavily web search for current events and non-academic info
  * **calculate** — numexpr math evaluation (identifier-whitelisted)
  * **execute_python** — process-isolated Python with AST-based validation (import whitelist, dunder + dangerous-builtin blocking) + 10s timeout
* **Reflexion loops with layered budgets** — the evaluator checks faithfulness (fraction of claims grounded by NLI entailment) and completeness (Gemini Flash). Below threshold it can regenerate, retrieve more, or reformulate — bounded by an iteration cap (3), a loop budget (`AGENT_REFLEXION_BUDGET_S`, iteration 2+), a deadline reserve (`AGENT_EVAL_RESERVE_S`), and a per-call LLM timeout (`LLM_REQUEST_TIMEOUT_S`). Stuck-loop detection auto-accepts when completeness stops improving.
* **Contradiction detection** — NLI-based cross-source contradiction flagging in the answer generator; surfaces both sides with citations when sources disagree
* **Confidence & abstention** — finalizer surfaces a confidence score; low-confidence answers get an explicit abstention prefix with partial sourcing
* **Multi-turn conversations** — session history threaded through `AgentState` so follow-ups resolve pronouns and references
* **Parallel tool execution** — multiple selected tools run concurrently via `ThreadPoolExecutor`
* **Sub-query cap** — `AGENT_MAX_SUB_QUERIES` limits per-cycle retrievals to bound latency and cost
* **Model failover + circuit breaker** — per-(provider, model) circuit keys; one model's 429 doesn't block others. Cross-provider failover (Gemini → OpenRouter, or reverse), with a **guaranteed Gemini backstop** so any selected OpenRouter model still resolves when its free-tier endpoint 429s
* **Multi-provider LLM** — Gemini + OpenRouter (Claude, GPT, Llama, etc.) with user-selectable models, capability gating, and non-Gemini JSON fallback
* **google-genai native function calling** — no LangChain LLM wrappers; `mode=AUTO` lets the model return an empty tool list on `regenerate` actions

### 🔍 Hybrid Retrieval Pipeline

* **Dense + sparse** — BGE-M3 (1024d) fused with BM25 via Reciprocal Rank Fusion (RRF)
* **Two-stage reranking** — `BAAI/bge-reranker-v2-m3` cross-encoder, with optional **ColBERT** multi-vector MaxSim rerank on the narrowed candidate set
* **Optional HyDE** — generate a hypothetical answer, embed it, and retrieve against it for recall on sparse queries
* **Faithfulness verification** — a multilingual NLI cross-encoder (`NLI_MODEL_NAME`, int8 ONNX on CPU) scores entailment per claim against its cited chunks; unsupported assertions flagged, stripped, or regenerated (`FAITHFULNESS_ENFORCE`). The threshold is **model-specific and calibrated**, not a taste setting — see `FAITHFULNESS_THRESHOLD`
* **HNSW tuning knobs** — `ef_search`, `ef_construction`, `M` all env-configurable

### 📥 Smart Ingestion

* **Section-aware chunking** — per-section chunk sizes (abstract, methods, results, …) instead of uniform splits
* **Multimodal figure/table indexing** — extract figure/table crops from PDFs, generate captions, and embed alongside text chunks
* **Metadata enrichment** — auto-fetch authors, year, DOI from arXiv by fuzzy title match at ingest time
* **Title dedup** — near-duplicate papers rejected by `SequenceMatcher` ratio (`DEDUP_TITLE_THRESHOLD`)
* **MD5 content dedup** + parallel extraction, Indic-aware chunking

### 🌍 True Multilingual Support

* **10+ Indian languages** + English (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu)
* Unicode script-based language detection with Devanagari hi/mr disambiguation
* **Two RAG strategies:** Direct multilingual reasoning (A, recommended) or Translation-enhanced with NLLB-200 (B, sentence-batched)
* Cross-lingual semantic search via BGE-M3

### 🛡️ Production-Ready Infrastructure

* **SQLite session/job persistence** — restarts don't drop in-flight state (`SESSIONS_DB_PATH`)
* **SSE streaming** — token-by-token answers and live ingest progress
* Thread-safe model init (double-checked locking on all singletons)
* Startup warm-up via FastAPI lifespan (embeddings, vector store, reranker, BM25) — no cold first request
* Request-ID correlation across log lines; Prometheus metrics
* API-key auth, env-driven CORS, Pydantic v2 validation, path-traversal + URL-scheme guards
* **int8 quantized ONNX** cross-encoders for CPU — lower memory, faster inference

### 📡 Topic Watches & Literature Reports

* **Topic watches** (`/watch/*`) — persistent monitoring with daily/weekly/monthly cadence; background digest loop fetches new papers, summarizes, and stores results
* **Literature review reports** (`/report/*`) — async decomposition of a topic into sections, cited synthesis from the corpus, and downloadable Markdown artifact
* **Model selection** (`GET /models`) — curated allowlist enriched with OpenRouter tool-capability metadata; UI dropdown with badges
* **Chat history** — endpoints + UI panels for browsing past conversations and topic watches

### 🗄️ Three-Layer Caching + Gemini Prompt Cache

* **LLM response cache** (128 entries, 10 min TTL) — identical prompts skip the API
* **Retrieval cache** (64 entries, 5 min TTL) — auto-invalidated on ingest
* **Tool result cache** (64 entries, 3 min TTL) — shared across reflexion loops
* **Explicit Gemini prompt caching** (opt-in) — caches the stable system-instruction prefix per API key
* All sizes/TTLs env-configurable; `GET /cache/stats` for observability

---

## 🚀 Quick Start

### Prerequisites

* Python 3.11+
* Google Gemini API key ([Get one here](https://aistudio.google.com/api-keys))
* 8GB+ RAM recommended
* (Optional) Tavily API key for agent web search ([Get one here](https://app.tavily.com))

### Installation

```bash
git clone https://github.com/DNSdecoded/IndicRAG.git
cd IndicRAG

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# Required
LLM_API_KEY=your_gemini_api_key_here

# Optional — enables multi-provider mode (OpenRouter)
# OPENROUTER_API_KEY=your_openrouter_key_here
# LLM_PROVIDER=gemini          # gemini|openrouter
# LLM_FALLBACK_PROVIDER=openrouter

# Optional — enables agent web search tool
TAVILY_API_KEY=your_tavily_key_here

# Optional — higher token limit for agent answers (default 8192)
AGENT_MAX_TOKENS=8192

# Optional — agent thinking tokens: 0=off (cheapest), -1=dynamic, N=cap (default 0)
AGENT_THINKING_BUDGET=0

# Optional — retrieval quality boosters (off by default, cost more compute)
USE_COLBERT_RERANK=false
USE_HYDE=false

# Optional — topic watches and literature reports (off by default)
# WATCH_ENABLE=true
# REPORT_ENABLE=true
```

### Ingest Documents

```bash
# Place PDFs in papers/ directory, then:
python ingest.py

# Or specify a directory:
python ingest.py path/to/pdfs
```

### Start Server

```bash
python start_server.py

# Development mode with auto-reload
python start_server.py --dev
```

Access at:
* **Web Interface:** http://localhost:8080
* **API Docs:** http://localhost:8080/api/docs

---

## 📖 Usage

### 🖥️ Web UI

Open http://localhost:8080:

**Ask** — pick a pipeline mode (Standard RAG or Agentic RAG), a strategy (Direct Multilingual `A` or English Pivot `B`), a model from the allowlist, and ask in English or any supported Indic language. The composer also exposes **Depth** (`top_k` = 4/8/12/16/20), a **scope** selector, and a **tags** filter. Agentic mode ignores scope, tags, and depth — the UI says so inline.

**Retrieval** — retrieval-only view over `POST /search`: query the corpus, the web, or both at k = 5/10/20/30 and inspect the raw passages with no LLM in the loop. Results export to BibTeX.

**Library** — upload and ingest PDFs, plus per-row actions: ingest, **dry-run** (preview chunking/dedup before writing), **edit metadata** (title, authors, year, tags via `PATCH /papers/{id}`), and delete. Multi-select supports bulk delete.

**Evidence exports** — answers export their cited sources as **BibTeX** or **Markdown**.

**History** — sessions persist across restarts and can be reopened or deleted individually.

**Reports** — saved literature reviews are listed and reopenable, rendered as Markdown with a download button.

**Index health** — a diagnostics panel showing deep health, cache stats, index quality signals, and submitted feedback, with a one-click cache clear.

In Agentic mode the UI shows an animated progress stepper with elapsed timer, color-coded source cards (title, authors, year, citation count, PDF link), a tool-call log with latencies, and copy buttons on answers and LaTeX equations.

### 🔌 REST API

#### Streaming Query — `POST /query/stream`

```python
import requests

with requests.post('http://localhost:8080/query/stream',
                   json={"question": "What are the latest advances in antenna optimization using ML?"},
                   stream=True) as r:
    for line in r.iter_lines():
        if line:
            print(line.decode())  # Server-Sent Events
```

#### Standard Chat — `POST /chat`

```python
r = requests.post('http://localhost:8080/chat', json={
    "message": "యాంటెన్నాతో ml ను ఎలా అమలు చేయవచ్చు?",
    "strategy": "A"
})
print(r.json()['answer'])
```

#### Agentic Query — `POST /agent/query`

```python
r = requests.post('http://localhost:8080/agent/query', json={
    "question": "What are the latest advances in antenna optimization using ML?",
    "strategy": "A"
})

data = r.json()
print(data['answer'])
print(f"Sources: {len(data['sources'])}  Reflexion iterations: {data['reflexion_iterations']}")
for src in data['sources']:
    print(f"  [{src['section']}] {src['title']} ({src['year']}) — {src['citations']} citations")
    if src.get('pdf_url'):
        print(f"    PDF: {src['pdf_url']}")
```

**Agent response fields:** `answer`, `language`, `sources` (title/authors/year/citations/pdf_url/url), `tool_calls` (name/args/latency_ms), `reflexion_iterations` (0–3), `processing_time`.

---

## 🔧 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Single-turn question answering |
| `/query/stream` | POST | Same, streamed token-by-token (SSE) |
| `/chat` | POST | Multi-turn chat with persisted session history |
| `/chat/stream` | POST | Streamed multi-turn chat (SSE) |
| `/chat/{session_id}` | DELETE | Clear a chat session |
| `/chat` | GET | List persisted chat sessions |
| `/chat/{session_id}` | GET | Fetch one session's history |
| `/agent/query` | POST | Agentic pipeline with reflexion loops (timeout → 504) |
| `/agent/stream` | POST | Agentic pipeline, streamed step-by-step (SSE) |
| `/compare` | POST | Kick off a multi-model answer comparison (async, returns `job_id`) |
| `/compare/status/{job_id}` | GET | Comparison job status / results |
| `/models` | GET | Curated model allowlist with tool-capability metadata |
| `/search` | POST | Retrieval-only — corpus, web, or both (no LLM) |
| `/search/export` | GET | Export retrieved papers as BibTeX (`format=bibtex` only) |
| `/export/bibtex` | POST | Export an answer's cited sources as BibTeX |
| `/upload` | POST | Upload PDF (multipart form) |
| `/ingest` | POST | Ingest one PDF into the vector store |
| `/ingest/all` | POST | Bulk ingest all PDFs (async, returns `job_id`) |
| `/ingest/status/{job_id}` | GET | Bulk ingest job status |
| `/ingest/stream/{job_id}` | GET | Live ingest progress (SSE) |
| `/ingest/dry-run` | POST | Preview chunking/dedup without writing |
| `/ingest/reindex` | POST | Rebuild the index from stored papers |
| `/ingest/from-url` | POST | Fetch a PDF by URL and ingest it |
| `/ingest/health` | GET | Deep ingest-path health (store, embeddings, disk) |
| `/papers` | GET | List uploaded PDFs |
| `/papers/{paper_id}` | PATCH | Edit paper metadata (title, authors, year, tags) |
| `/papers/{paper_id}` | DELETE | Delete a single paper |
| `/watch` | GET/POST | List or create topic watches |
| `/watch/{id}` | GET/DELETE | Read or delete a watch |
| `/watch/{id}/run` | POST | Trigger a watch run immediately |
| `/watch/{id}/digest` | GET | Fetch persisted digest for a watch |
| `/report` | POST | Kick off a literature review report |
| `/report/status/{job_id}` | GET | Report generation status |
| `/report/{job_id}/download` | GET | Download completed report (Markdown) |
| `/reports` | GET | List saved reports |
| `/reports/{report_id}` | GET | Fetch one saved report |
| `/feedback` | POST | Submit answer feedback |
| `/feedback` | GET | List submitted feedback |
| `/feedback/stats` | GET | Aggregate feedback counts |
| `/prefs/{user_id}` | GET / PUT | Read / update user preferences |
| `/stats` | GET | Vector store statistics |
| `/quality` | GET | Index quality signals (chunk/metadata coverage) |
| `/cache/stats` | GET | Cache hit rates, sizes, TTL config |
| `/cache` | DELETE | Clear all caches |
| `/health` | GET | Health check |
| `/purge/papers` | DELETE | Delete all PDFs (admin key) |
| `/purge/database` | DELETE | Clear vector database (admin key) |

---

## 📁 Project Structure

> Full annotated tree: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

```
IndicRAG/
│
├── 📄 Root
│   ├── requirements.txt             # dependencies
│   ├── .env.example                 # LLM_API_KEY(S), TAVILY, AGENT_MAX_TOKENS, ...
│   ├── start_server.py              # Launcher with pre-flight checks
│   └── patterns.json                # Regex patterns for PDF cleaning
│
├── 🐍 Core Modules
│   ├── config.py                    # Configuration + env parsing (VERSION = 2.4.0-dev)
│   ├── api_server.py                # FastAPI app: lifespan warm-up + router mounting
│   ├── deps.py                      # Shared deps: auth, rate limit, session/job state
│   ├── middleware.py                # Request-ID propagation
│   ├── persistence.py               # SQLite session/job/watch/report persistence
│   ├── llm_client.py                # Multi-provider dispatcher: Gemini + OpenRouter
│   ├── gemini_cache.py              # Explicit Gemini prompt caching (per client)
│   ├── rag.py                       # RAG pipeline orchestration
│   ├── sse_utils.py                 # Shared SSE streaming bridge
│   ├── embeddings.py                # BGE-M3 embeddings (thread-safe)
│   ├── vector_store.py              # ChromaDB wrapper (HNSW knobs)
│   ├── bm25_search.py               # BM25 + RRF fusion
│   ├── rerank.py                    # Cross-encoder reranker (int8 ONNX)
│   ├── onnx_ce.py                   # int8 quantized ONNX cross-encoder inference
│   ├── colbert_rerank.py            # ColBERT multi-vector MaxSim rerank (opt-in)
│   ├── verify.py                    # NLI faithfulness verification
│   ├── contradiction.py             # Cross-source contradiction detection (Phase 5)
│   ├── lang_utils.py                # Unicode script + langdetect
│   ├── pdf_utils.py                 # PDF extraction, Indic-aware chunking
│   ├── figure_captioner.py          # Figure/table crop extraction + captioning (Phase 3)
│   ├── metadata_enrich.py           # arXiv metadata auto-fetch at ingest
│   ├── ingest.py                    # Section-aware parallel ingestion + dedup
│   ├── translation.py               # NLLB-200 sentence-batched (Strategy B)
│   ├── cache.py                     # Thread-safe TTL LRU cache (LLM/retrieval/tool)
│   ├── watch_runner.py              # Background topic-watch digest loop
│   ├── report_runner.py             # Async literature-review report generation
│   └── purge.py                     # CLI cleanup (papers, db, models)
│
├── 🔌 providers/                     # LLM provider backends
│   ├── base.py                      # LLMBackend interface
│   ├── gemini.py                    # GeminiBackend — google-genai native
│   └── openrouter.py                # OpenRouterBackend — OpenAI-compatible API
│
├── 🌐 routes/                       # FastAPI routers
│   ├── query.py                     # /query, /query/stream, /health, /
│   ├── chat.py                      # /chat, /chat/stream, /chat/{id}
│   ├── agent.py                     # /agent/query
│   ├── ingest.py                    # /ingest*, /upload
│   ├── management.py                # /search, /papers, /stats, /cache, /purge
│   ├── feedback.py                  # /feedback, /prefs/{user_id}
│   ├── models.py                    # /models — curated model allowlist
│   ├── watch.py                     # /watch — topic monitoring CRUD + run + digest
│   └── report.py                    # /report — literature review generation
│
├── 🤖 agent/                        # Agentic RAG Pipeline
│   ├── state.py                     # AgentState + ReflexionFeedback schemas
│   ├── tool_declarations.py         # 6 google-genai FunctionDeclarations
│   ├── tool_executor.py             # Tool impls: corpus, arXiv, S2/OpenAlex, web, calc, sandbox
│   ├── graph.py                     # LangGraph StateGraph + reflexion routing
│   ├── json_utils.py                # Robust LLM JSON parsing
│   └── nodes/
│       ├── query_planner.py         # Language detection + decomposition (+ HyDE)
│       ├── tool_selector.py         # Gemini function calling
│       ├── tool_executor_node.py    # Dispatch + context accumulation + audit log
│       ├── answer_generator.py      # Reuses rag context/prompt/generate + contradiction detect
│       ├── reflexion_evaluator.py   # check_claims() + Gemini completeness judge
│       └── finalizer.py             # Terminal node — confidence + abstention
│
├── 🌐 static/index.html             # SPA: mode toggle, stepper, source cards, model dropdown, copy buttons
├── 📚 docs/                         # ARCHITECTURE, DEPLOYMENT, evaluation, Eval/, ...
├── 💡 examples/                     # example_ingest.py, example_query.py
├── 🔧 deploy/                       # nginx.example.conf
│
└── 📊 Data (git-ignored)
    ├── papers/                      # PDF documents
    ├── figures/                     # Extracted figure/table crops (Phase 3)
    ├── chroma_db/                   # Vector database
    ├── sessions.db                  # Persisted sessions/jobs/watches
    └── models/                      # Cached ML models
```

---

## ⚙️ Configuration

Key settings (all overridable via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | (required) | Google Gemini API key (comma-separate for a round-robin pool) |
| `LLM_PROVIDER` | `gemini` | Primary LLM provider: `gemini` or `openrouter` |
| `LLM_FALLBACK_PROVIDER` | `openrouter` | Cross-provider failover when the primary is down |
| `OPENROUTER_API_KEY` | (none) | OpenRouter API key for multi-provider mode |
| `LLM_MODEL_NAME` | `gemini-3.6-flash` | Gemini model for generation |
| `LLM_FALLBACK_MODEL` | `gemma-4-26b-a4b-it` | Fallback when primary is overloaded (503/429) |
| `LLM_SELECTABLE_MODELS` | `gemini-3.6-flash,gemini-3.5-flash,anthropic/claude-haiku,openai/gpt-5.4-nano` | Curated model dropdown (comma-separated; first entry is the default). `.env.example` ships a wider free-tier list. Bare name → Gemini, `/` slug → OpenRouter — **keep at least one `/` slug**, since cross-vendor failover picks the first one here |
| `LLM_MAX_TOKENS` | `8192` | Max tokens for standard RAG (covers thinking + answer) |
| `AGENT_MAX_TOKENS` | `8192` | Max tokens for agentic pipeline |
| `AGENT_TIMEOUT` | `120` | Agent pipeline timeout (seconds) → 504 |
| `LLM_REQUEST_TIMEOUT_S` | `60` | Per-request HTTP timeout for every LLM call. Without it the SDK defaults apply (OpenAI: 600s × 2 retries) and one stalled request outlives the whole agent budget — multiplied by the 3-attempt failover chain |
| `AGENT_REFLEXION_BUDGET_S` | `90` | Wall-clock budget for reflexion **loops** — blocks starting another cycle, from iteration 2 onwards |
| `AGENT_EVAL_RESERVE_S` | `90` | Room that must remain under `AGENT_TIMEOUT` to attempt an evaluation at all. Can skip iteration 1, but only when finishing would overrun the timeout and discard the draft |
| `AGENT_THINKING_BUDGET` | `0` | Agent thinking tokens: `0`=off, `-1`=dynamic, `N`=cap |
| `AGENT_MAX_SUB_QUERIES` | `3` | Cap per-cycle retrievals to bound latency |
| `CONTRADICTION_DETECT_ENABLE` | `false` | NLI-based cross-source contradiction flagging |
| `CONTRADICTION_NLI_THRESHOLD` | `0.6` | NLI score threshold for contradiction detection |
| `TAVILY_API_KEY` | (optional) | Enables agent web search tool |
| `WATCH_ENABLE` | `false` | Enable topic watch endpoints (`/watch/*`) |
| `WATCH_DEFAULT_CADENCE` | `weekly` | Default watch cadence: `daily`, `weekly`, or `monthly` |
| `WATCH_MAX_RESULTS` | `10` | Papers fetched per watch run |
| `REPORT_ENABLE` | `false` | Enable literature review report endpoints (`/report/*`) |
| `REPORT_MAX_SECTIONS` | `6` | Cap report sections to bound cost/latency |
| `USE_HYBRID_SEARCH` | `true` | BM25 + dense fusion |
| `USE_RERANKER` | `true` | Cross-encoder reranking |
| `USE_COLBERT_RERANK` | `false` | ColBERT multi-vector rerank layer |
| `COLBERT_WEIGHT` | `0.5` | Dense-vs-ColBERT fusion weight |
| `USE_HYDE` | `false` | Hypothetical document embeddings |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Sentence-transformer used for dense vectors (1024d) |
| `TAGS_OVERFETCH` | `10` | Multiplier applied to `top_k` when a tags filter is active — tags live in one comma-joined metadata string, so filtering happens in Python after retrieval and the search must over-fetch to avoid returning nothing |
| `TAGS_OVERFETCH_MAX` | `300` | Hard ceiling on that widened fetch |
| `AGENT_MAX_CONTEXT_CHUNKS` | `20` | Chunk cap for the agentic answer prompt |
| `AGENT_MAX_CONTEXT_LENGTH` | `80000` | Character cap for the agentic answer prompt |
| `SCOPED_MAX_CHUNKS` | `50` | Chunk cap when a query is scoped to specific papers |
| `SCOPED_MAX_CONTEXT_LENGTH` | `120000` | Character cap for scoped queries |
| `ENRICH_METADATA` | `true` | Auto-fetch arXiv metadata at ingest |
| `DEDUP_PAPERS` | `true` | Reject near-duplicate titles |
| `DEDUP_TITLE_THRESHOLD` | `0.9` | Title similarity cutoff for dedup |
| `HNSW_EF_SEARCH` | `100` | ChromaDB HNSW query-time search breadth |
| `HNSW_EF_CONSTRUCTION` | `100` | HNSW build-time breadth (index quality vs. ingest speed) |
| `HNSW_M` | `16` | HNSW graph connectivity |
| `FAITHFULNESS_ENFORCE` | `warn` | `warn`, `strip`, or `regen` |
| `FAITHFULNESS_THRESHOLD` | `0.15` | Per-claim entailment probability above which a claim counts as grounded. **Model-specific — recalibrate if you change `NLI_MODEL_NAME`.** Measured for the default model: a sentence copied verbatim out of its own chunk scores median 0.226 / max 0.428; an unrelated paper's sentence median 0.099 / p90 0.158. A 0.5 bar sits above every positive, so faithfulness reads ~0 for every answer |
| `AGENT_FAITHFULNESS_ACCEPT` | `0.6` | Fraction of claims that must be grounded for reflexion to accept and for the finalizer to abstain on completeness alone. Below 1.0 by design: per-claim recall is ~0.70, so a fully grounded answer lands near 0.70 |
| `NLI_MODEL_NAME` | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | NLI model backing faithfulness + contradiction checks |
| `NLI_ENTAILMENT_INDEX` | `0` | Index of the entailment label in that model's output — change if you swap models |
| `NLI_MAX_SEQ_LENGTH` | `256` | Premise truncation. NLI cost is linear in length: 1.15s/pair at 512 tokens vs 0.4s at 256 on CPU, with no measured quality cost |
| `NLI_MAX_CHUNKS_PER_CITATION` | `2` | Chunks scored per cited paper. Uncapped, one answer cost ~275 pairs = 318s in a single reflexion pass |
| `GEMINI_CACHE_ENABLED` | `false` | Explicit Gemini prompt caching |
| `GEMINI_CACHE_TTL` | `3600` | Prompt cache lifetime (seconds) |
| `SESSIONS_DB_PATH` | `sessions.db` | SQLite path for session/job/watch persistence |
| `ENABLE_USER_PREFS` | `false` | Enable `/prefs` user preferences |
| `SESSION_MAX_AGE_HOURS` | `24` | Max session age before eviction (hours) |
| `CHAT_HISTORY_MAX_TURNS` | `20` | Max conversation turns per session |
| `ADMIN_API_KEY` | (none) | Required for `/purge/*` endpoints — fail-closed: with no value set, those routes return 403 for everyone, and an ordinary `API_KEYS` entry never authorizes them |
| `API_KEYS` | (none) | Comma-separated keys for request auth (each key = isolated user) |
| `CORS_ORIGINS` | localhost | Comma-separated allowed origins |
| `LLM_CACHE_SIZE` / `LLM_CACHE_TTL` | `128` / `600` | LLM response cache |
| `RETRIEVAL_CACHE_SIZE` / `RETRIEVAL_CACHE_TTL` | `64` / `300` | Retrieval cache |
| `TOOL_CACHE_SIZE` / `TOOL_CACHE_TTL` | `64` / `180` | Agent tool cache |

---

## 🎯 Supported Languages

| Language | Code | Native Name | Language | Code | Native Name |
|----------|------|-------------|----------|------|-------------|
| English | en | English | Kannada | kn | ಕನ್ನಡ |
| Hindi | hi | हिंदी | Malayalam | ml | മലയാളം |
| Telugu | te | తెలుగు | Punjabi | pa | ਪੰਜਾਬੀ |
| Tamil | ta | தமிழ் | Odia | or | ଓଡ଼ିଆ |
| Bengali | bn | বাংলা | Urdu | ur | اردو |
| Marathi | mr | मराठी | Gujarati | gu | ગુજરાતી |

---

## 🏗️ Architecture

The system runs in two modes: a full **Agentic RAG** loop (below) and a lightweight **Standard RAG** fast path.

```mermaid
flowchart TD
    Q([💬 User Query]) --> P

    subgraph AGENT["🤖  Agentic RAG · LangGraph state machine"]
        direction TB
        P["<b>① Query Planner</b><br/>language detection · decomposition · HyDE <i>(optional)</i>"]
        S["<b>② Tool Selector</b><br/>Gemini function calling · mode = AUTO"]
        E["<b>③ Tool Executor</b><br/>parallel dispatch · ThreadPoolExecutor"]
        G["<b>④ Answer Generator</b><br/>format context → build prompt → generate"]
        R{"<b>⑤ Reflexion Evaluator</b><br/>claim check · completeness judge"}

        P --> S --> E --> G --> R
        R -. regenerate .-> G
        R -. retrieve more .-> S
        R -. reformulate .-> P
    end

    subgraph TOOLS["🧰  Tool Belt"]
        direction LR
        T1["📚 IndicRAG Corpus<br/><i>BM25 + dense → cross-encoder → ColBERT (opt.)</i>"]
        T2["📄 arXiv"]
        T3["🎓 S2 / OpenAlex"]
        T4["🌐 Web Search"]
        T5["🧮 Calculator"]
        T6["🐍 Python Sandbox"]
    end

    E -.->|invokes| TOOLS
    R ==>|✅ accept| F([Finalizer])
    F --> A([📦 Answer · sources · tool log])

    classDef term fill:#bae6fd,stroke:#0284c7,stroke-width:2px,color:#082f49
    classDef eval fill:#fde68a,stroke:#d97706,stroke-width:2px,color:#451a03
    classDef stage fill:#e2e8f0,stroke:#64748b,color:#0f172a
    classDef tool fill:#dcfce7,stroke:#16a34a,color:#052e16

    class Q,A,F term
    class R eval
    class P,S,E,G stage
    class T1,T2,T3,T4,T5,T6 tool
```

### 🔒 Loop guardrails

| Guard | Behavior |
|---|---|
| **Iteration cap** | Max **3** reflexion cycles |
| **Loop budget** | `AGENT_REFLEXION_BUDGET_S` — blocks starting another cycle, from iteration 2 onwards. Deliberately does *not* gate the first evaluation: on a CPU-only box the first pass alone runs past it, so gating iteration 1 would ship every answer with no faithfulness score and no confidence |
| **Deadline reserve** | `AGENT_EVAL_RESERVE_S` — skips the evaluation entirely when too little remains under `AGENT_TIMEOUT` to finish it, since being killed mid-evaluation discards the draft and 504s |
| **Per-call timeout** | `LLM_REQUEST_TIMEOUT_S` bounds every LLM request so one stalled call can't consume the run |
| **Stuck-loop detection** | Auto-accepts once completeness stops improving |

### ⚡ Standard RAG mode

Skips the agent graph entirely — a single pass:

```
Query ──▶ Hybrid retrieval (BM25 + dense → rerank) ──▶ Generate
```

---

## 📊 Performance

Typical query latency (on CPU):

| Mode | Latency | Notes |
|------|---------|-------|
| Standard RAG (Strategy A) | ~1–2s | Single-pass |
| Standard RAG (Strategy B) | ~3–6s | Includes NLLB translation |
| Agentic RAG (1 reflexion) | ~2–4 min | Measured on a CPU-only box: ~45s retrieval (3 sub-queries, embed + rerank), ~50s generation, ~25s evaluation (NLI 20s / ~23 claims + completeness call). Tools run in parallel but contend for the same CPU |
| Agentic RAG (max reflexions) | bounded by `AGENT_TIMEOUT` | Loop budget stops further cycles; the draft is returned rather than discarded |

On CPU the agent is dominated by the cross-encoders, not the LLM. The levers, cheapest first: lower `AGENT_MAX_SUB_QUERIES` (near-linear — the parallel retrievals thrash one CPU), add keys to `LLM_API_KEYS` so an exhausted model doesn't cost a failed call plus fallback on every step, and drop `NLI_MAX_CHUNKS_PER_CITATION` to `1`. A GPU removes most of this.

Memory: base ~500MB · +BGE-M3 ~2.5GB · +reranker ~3.5GB · +NLLB (Strategy B) ~6GB. ColBERT rerank adds ~1GB when enabled.

---

## 📈 KPI Metrics

| Metric | Score | Metric | Score |
|--------|-------|--------|-------|
| Retrieval Precision | 0.93 | Technical Depth | 0.88 |
| Retrieval Recall | 0.91 | Mechanistic Reasoning | 0.86 |
| Faithfulness (Grounding) | 0.98 | Cross-Document Discipline | 0.95 |
| Attribution Accuracy | 0.97 | Hallucination Rate | < 2% |

See [docs/evaluation.md](docs/evaluation.md) for methodology.

---

## 🐛 Troubleshooting

**"API key not configured"** — check `.env`: `grep LLM_API_KEY .env`

**"No documents indexed"** — run `python ingest.py`

**Agent web search fails** — ensure `TAVILY_API_KEY` is set in `.env`

**Agent answers truncated** — raise `AGENT_MAX_TOKENS` (e.g. `16384`)

**"Agent pipeline timed out"** — every node logs its wall time as `[Graph] <node> took Ns`; read those before changing knobs. On CPU the usual culprit is retrieval or the NLI pass, not the LLM. If a single LLM call hangs, lower `LLM_REQUEST_TIMEOUT_S` so failover fires sooner; raising `AGENT_TIMEOUT` only delays the error

**Faithfulness reads ~0 on every answer** — the threshold is calibrated per NLI model. If you changed `NLI_MODEL_NAME`, recalibrate `FAITHFULNESS_THRESHOLD`: score a sentence copied verbatim out of its own chunk (positive) against one from an unrelated paper (negative) over ~20 chunks, and put the threshold between the two distributions. A bar above every positive silently reports everything as ungrounded

**"Translation model gated"** — NLLB-200 needs no auth; first use downloads ~2.4GB automatically

**Sessions lost on restart** — check `SESSIONS_DB_PATH` is writable; SQLite persistence is on by default

---

## 🧹 Maintenance

```bash
python purge.py --papers      # Delete all PDFs
python purge.py --db          # Clear vector database
python purge.py --models      # Remove cached models
python purge.py --all --yes   # Clear everything
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md).

**v2.4 highlights:** gemini-3.6-flash default model · multi-user data isolation with per-API-key scoping · configurable session management · dedicated admin API key · clean CI linting. Full history in the git log and `docs/`.

---

## 🙏 Acknowledgments

Built with [Google Gemini](https://ai.google.dev/) · [LangGraph](https://github.com/langchain-ai/langgraph) · [Sentence Transformers](https://www.sbert.net/) (BGE-M3, reranker) · [arXiv API](https://arxiv.org/) · [Semantic Scholar](https://www.semanticscholar.org/) · [OpenAlex](https://openalex.org/) · [Tavily](https://tavily.com/) · [ChromaDB](https://www.trychroma.com/) · [FastAPI](https://fastapi.tiangolo.com/).

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🆘 Support

* 📖 [Documentation](docs/)
* 💬 [GitHub Discussions](https://github.com/DNSdecoded/IndicRAG/discussions)
* 🐛 [Issue Tracker](https://github.com/DNSdecoded/IndicRAG/issues)

---

<div align="center">

**Built with ❤️ for multilingual scientific accessibility**

⭐ Star this repo if you find it useful!

[Report Bug](https://github.com/DNSdecoded/IndicRAG/issues) · [Request Feature](https://github.com/DNSdecoded/IndicRAG/issues) · [Documentation](docs/)

</div>
