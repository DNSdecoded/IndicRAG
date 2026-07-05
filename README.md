# 🌐 IndicRAG — Multilingual Agentic Scientific RAG

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DNSdecoded/IndicRAG)
[![Code Wiki](https://img.shields.io/badge/Code%20Wiki-Documentation-blue)](https://codewiki.google/github.com/dnsdecoded/indicrag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.130+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash-blueviolet.svg)](https://ai.google.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent--pipeline-orange.svg)](https://github.com/langchain-ai/langgraph)
![Version](https://img.shields.io/badge/version-2.2-blue.svg)

![INDICRAG.png](https://cdn.jsdelivr.net/gh/free-whiteboard-online/Free-Erasorio-Alternative-for-Collaborative-Design@3a5f22554411d3d6df27ee788c2df99d583f2c91/uploads/2025-12-03T05-25-45-007Z-3i36rbzio.png)

A **production-ready** Retrieval-Augmented Generation system with an **agentic pipeline**, multilingual support for 10+ Indian languages, and tools for searching arXiv, Semantic Scholar, OpenAlex, and the web — alongside your own indexed document corpus.

Two pipelines ship side-by-side: **Standard RAG** (single-pass hybrid retrieval) and **Agentic RAG** (multi-tool planning with reflexion self-correction). Answers stream token-by-token over SSE, sessions survive restarts, and every retrieval knob is env-configurable.

---

## 🆕 What's New in v2.2

| Area | v2.1 | v2.2 |
|------|------|------|
| **Codebase** | Monolithic `api_server.py` | Split into `routes/` (query, chat, ingest, agent, management, feedback) + shared `deps.py` |
| **Streaming** | Full-response only | Token-by-token **SSE streaming** on `/query/stream`, `/chat/stream`, `/ingest/stream` |
| **State durability** | In-memory (lost on restart) | **SQLite-backed** session + job persistence (`sessions.db`) |
| **Reranking** | Cross-encoder only | Optional **ColBERT** multi-vector MaxSim rerank layered on cross-encoder |
| **Query expansion** | Sub-query decomposition | Optional **HyDE** (hypothetical document embeddings) |
| **Ingestion** | Fixed-size chunks | **Section-aware chunking** + auto **metadata enrichment** (arXiv authors/year/DOI by title match) + title dedup |
| **Corpus tool** | Topic only | **Year-range metadata filter** on agent corpus retrieval |
| **Gemini cost** | LRU response cache | Opt-in **explicit Gemini prompt caching** for the stable system-instruction prefix |
| **Reflexion** | Iteration cap only | Iteration cap **+ wall-clock budget** (`AGENT_REFLEXION_BUDGET_S`) |
| **Feedback** | None | `/feedback` + `/prefs/{user_id}` endpoints; opt-in user preferences |
| **Observability** | Prometheus metrics | Metrics + **request-ID correlation** across all log lines |
| **Answer tokens** | `AGENT_MAX_TOKENS=4096` | `AGENT_MAX_TOKENS=8192` default; LaTeX equations returned verbatim with copy buttons |

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
* **Reflexion loops with dual budget** — the evaluator checks faithfulness (NLI entailment, minimum across claims) and completeness (Gemini Flash). Below threshold it can regenerate, retrieve more, or reformulate — bounded by **both** an iteration cap (3) and a wall-clock budget (`AGENT_REFLEXION_BUDGET_S`). Stuck-loop detection auto-accepts when completeness stops improving.
* **Multi-turn conversations** — session history threaded through `AgentState` so follow-ups resolve pronouns and references
* **Parallel tool execution** — multiple selected tools run concurrently via `ThreadPoolExecutor`
* **Model failover + circuit breaker** — on 503/429, `gemini-3.5-flash` falls back to `gemma-4-26b-a4b-it`; the breaker skips the primary for 60s after failure
* **google-genai native function calling** — no LangChain LLM wrappers; `mode=AUTO` lets the model return an empty tool list on `regenerate` actions

### 🔍 Hybrid Retrieval Pipeline

* **Dense + sparse** — BGE-M3 (1024d) fused with BM25 via Reciprocal Rank Fusion (RRF)
* **Two-stage reranking** — `BAAI/bge-reranker-v2-m3` cross-encoder, with optional **ColBERT** multi-vector MaxSim rerank on the narrowed candidate set
* **Optional HyDE** — generate a hypothetical answer, embed it, and retrieve against it for recall on sparse queries
* **Faithfulness verification** — `cross-encoder/nli-deberta-v3-base` scores entailment per claim; unsupported assertions flagged, stripped, or regenerated (`FAITHFULNESS_ENFORCE`)
* **HNSW tuning knobs** — `ef_search`, `ef_construction`, `M` all env-configurable

### 📥 Smart Ingestion

* **Section-aware chunking** — per-section chunk sizes (abstract, methods, results, …) instead of uniform splits
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

# Optional — enables agent web search tool
TAVILY_API_KEY=your_tavily_key_here

# Optional — higher token limit for agent answers (default 8192)
AGENT_MAX_TOKENS=8192

# Optional — agent thinking tokens: 0=off (cheapest), -1=dynamic, N=cap (default 0)
AGENT_THINKING_BUDGET=0

# Optional — retrieval quality boosters (off by default, cost more compute)
USE_COLBERT_RERANK=false
USE_HYDE=false
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

1. **Select Pipeline Mode** — Standard RAG (single-pass) or Agentic RAG (multi-tool + reflexion)
2. **Select Strategy** — Direct Multilingual (A) or English Pivot (B)
3. **Ask Questions** — in English or any supported Indic language
4. **Manage Documents** — upload PDFs, ingest, view stats

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
| `/agent/query` | POST | Agentic pipeline with reflexion loops (timeout → 504) |
| `/search` | POST | Retrieval-only — corpus, web, or both (no LLM) |
| `/search/export` | GET | Export search results as plain text |
| `/upload` | POST | Upload PDF (multipart form) |
| `/ingest` | POST | Ingest one PDF into the vector store |
| `/ingest/all` | POST | Bulk ingest all PDFs (async, returns `job_id`) |
| `/ingest/status/{job_id}` | GET | Bulk ingest job status |
| `/ingest/stream/{job_id}` | GET | Live ingest progress (SSE) |
| `/ingest/dry-run` | POST | Preview chunking/dedup without writing |
| `/ingest/reindex` | POST | Rebuild the index from stored papers |
| `/papers` | GET | List uploaded PDFs |
| `/papers/{paper_id}` | DELETE | Delete a single paper |
| `/feedback` | POST | Submit answer feedback |
| `/prefs/{user_id}` | GET / PUT | Read / update user preferences |
| `/stats` | GET | Vector store statistics |
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
│   ├── config.py                    # Configuration + env parsing (VERSION = 2.2.0)
│   ├── api_server.py                # FastAPI app: lifespan warm-up + router mounting
│   ├── deps.py                      # Shared deps: auth, rate limit, session/job state
│   ├── middleware.py                # Request-ID propagation
│   ├── persistence.py               # SQLite session/job persistence
│   ├── llm_client.py                # Gemini client pool: round-robin, failover, breaker
│   ├── gemini_cache.py              # Explicit Gemini prompt caching (per client)
│   ├── rag.py                       # RAG pipeline orchestration
│   ├── sse_utils.py                 # Shared SSE streaming bridge
│   ├── embeddings.py                # BGE-M3 embeddings (thread-safe)
│   ├── vector_store.py              # ChromaDB wrapper (HNSW knobs)
│   ├── bm25_search.py               # BM25 + RRF fusion
│   ├── rerank.py                    # Cross-encoder reranker
│   ├── colbert_rerank.py            # ColBERT multi-vector MaxSim rerank (opt-in)
│   ├── verify.py                    # NLI faithfulness verification
│   ├── lang_utils.py                # Unicode script + langdetect
│   ├── pdf_utils.py                 # PDF extraction, Indic-aware chunking
│   ├── metadata_enrich.py           # arXiv metadata auto-fetch at ingest
│   ├── ingest.py                    # Section-aware parallel ingestion + dedup
│   ├── translation.py               # NLLB-200 sentence-batched (Strategy B)
│   ├── cache.py                     # Thread-safe TTL LRU cache (LLM/retrieval/tool)
│   └── purge.py                     # CLI cleanup (papers, db, models)
│
├── 🌐 routes/                       # FastAPI routers
│   ├── query.py                     # /query, /query/stream, /health, /
│   ├── chat.py                      # /chat, /chat/stream, /chat/{id}
│   ├── agent.py                     # /agent/query
│   ├── ingest.py                    # /ingest*, /upload
│   ├── management.py                # /search, /papers, /stats, /cache, /purge
│   └── feedback.py                  # /feedback, /prefs/{user_id}
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
│       ├── answer_generator.py      # Reuses rag context/prompt/generate
│       ├── reflexion_evaluator.py   # check_claims() + Gemini completeness judge
│       └── finalizer.py             # Terminal node
│
├── 🌐 static/index.html             # SPA: mode toggle, stepper, source cards, copy buttons
├── 📚 docs/                         # ARCHITECTURE, DEPLOYMENT, evaluation, Eval/, ...
├── 💡 examples/                     # example_ingest.py, example_query.py
├── 🔧 deploy/                       # nginx.example.conf
│
└── 📊 Data (git-ignored)
    ├── papers/                      # PDF documents
    ├── chroma_db/                   # Vector database
    ├── sessions.db                  # Persisted sessions/jobs
    └── models/                      # Cached ML models
```

---

## ⚙️ Configuration

Key settings (all overridable via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | (required) | Google Gemini API key (comma-separate for a round-robin pool) |
| `LLM_MODEL_NAME` | `gemini-3.5-flash` | Gemini model for generation |
| `LLM_FALLBACK_MODEL` | `gemma-4-26b-a4b-it` | Fallback when primary is overloaded (503/429) |
| `LLM_MAX_TOKENS` | `2048` | Max tokens for standard RAG |
| `AGENT_MAX_TOKENS` | `8192` | Max tokens for agentic pipeline |
| `AGENT_TIMEOUT` | `120` | Agent pipeline timeout (seconds) → 504 |
| `AGENT_REFLEXION_BUDGET_S` | `90` | Wall-clock budget for reflexion loops |
| `AGENT_THINKING_BUDGET` | `0` | Agent thinking tokens: `0`=off, `-1`=dynamic, `N`=cap |
| `TAVILY_API_KEY` | (optional) | Enables agent web search tool |
| `USE_HYBRID_SEARCH` | `true` | BM25 + dense fusion |
| `USE_RERANKER` | `true` | Cross-encoder reranking |
| `USE_COLBERT_RERANK` | `false` | ColBERT multi-vector rerank layer |
| `COLBERT_WEIGHT` | `0.5` | Dense-vs-ColBERT fusion weight |
| `USE_HYDE` | `false` | Hypothetical document embeddings |
| `ENRICH_METADATA` | `true` | Auto-fetch arXiv metadata at ingest |
| `DEDUP_PAPERS` | `true` | Reject near-duplicate titles |
| `DEDUP_TITLE_THRESHOLD` | `0.9` | Title similarity cutoff for dedup |
| `HNSW_EF_SEARCH` | `100` | ChromaDB HNSW query-time search breadth |
| `FAITHFULNESS_ENFORCE` | `warn` | `warn`, `strip`, or `regen` |
| `FAITHFULNESS_THRESHOLD` | `0.5` | NLI support score threshold |
| `GEMINI_CACHE_ENABLED` | `false` | Explicit Gemini prompt caching |
| `GEMINI_CACHE_TTL` | `3600` | Prompt cache lifetime (seconds) |
| `SESSIONS_DB_PATH` | `sessions.db` | SQLite path for session/job persistence |
| `ENABLE_USER_PREFS` | `false` | Enable `/prefs` user preferences |
| `ADMIN_API_KEY` | (none) | Required for `/purge/*` endpoints |
| `API_KEYS` | (none) | Comma-separated keys for request auth |
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
| **Wall-clock budget** | `AGENT_REFLEXION_BUDGET_S` |
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
| Agentic RAG (1 reflexion) | ~15–30s | Multi-tool + evaluation (parallel tools) |
| Agentic RAG (max reflexions) | ~60–90s | Bounded by timeout + reflexion budget |

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

**v2.2 highlights:** modular `routes/` split · SSE streaming (query/chat/ingest) · SQLite session+job persistence · optional ColBERT rerank + HyDE · section-aware ingestion with arXiv metadata enrichment + title dedup · year-range corpus filter · explicit Gemini prompt caching · reflexion wall-clock budget · `/feedback` + `/prefs` endpoints · request-ID log correlation · HNSW tuning knobs · LaTeX equation rendering + copy buttons. Full history in the git log and `docs/`.

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
