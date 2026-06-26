# 🌐 IndicRAG — Multilingual Agentic Scientific RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash-blueviolet.svg)](https://ai.google.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent--pipeline-orange.svg)](https://github.com/langchain-ai/langgraph)
![Version](https://img.shields.io/badge/version-2.0-blue.svg)

![INDICRAG.png](https://cdn.jsdelivr.net/gh/free-whiteboard-online/Free-Erasorio-Alternative-for-Collaborative-Design@3a5f22554411d3d6df27ee788c2df99d583f2c91/uploads/2025-12-03T05-25-45-007Z-3i36rbzio.png)

A **production-ready** Retrieval-Augmented Generation system with an **agentic pipeline**, multilingual support for 10+ Indian languages, and tools for searching arXiv, Semantic Scholar, OpenAlex, and the web — alongside your own indexed document corpus.

---

## 🆕 What's New in v2.0

| Feature | v1.5 (Standard) | v2.0 (Agentic) |
|---------|-----------------|-----------------|
| Pipeline | Single-pass retrieve → generate | Multi-step: plan → select tools → execute → generate → **reflexion loop** |
| External search | Corpus only | Corpus + **arXiv** + **Semantic Scholar / OpenAlex** + **Tavily web search** |
| Tools | None | 6 tools: corpus retrieval, arxiv search, open-access search, web search, calculator, sandboxed Python |
| Self-correction | None | Up to 3 reflexion iterations (faithfulness + completeness checks) |
| Answer quality | `LLM_MAX_TOKENS=2048` | `AGENT_MAX_TOKENS=4096` for richer synthesis |
| Source display | Citation numbers | Full paper cards with authors, year, citation count, PDF links |
| Progress UI | Typing dots | Animated 5-step pipeline stepper with elapsed timer |
| Caching | Query embedding LRU only | 3-layer TTL cache: LLM responses, retrieval results, tool results |
| API key management | Single key | Multi-key round-robin load balancing |

Both modes are available side-by-side — toggle in the web UI.

---

## ✨ Key Features

### 🤖 Agentic RAG Pipeline (v2.0)

* **LangGraph state machine** — query planner → tool selector → tool executor → answer generator → reflexion evaluator, with conditional loops
* **6 agent tools:**
  * **indicrag_retrieval** — hybrid BM25 + dense search with cross-encoder reranking on your indexed corpus
  * **arxiv_search** — search arXiv by topic, author, or paper ID; returns abstracts, authors, PDF links
  * **open_access_search** — Semantic Scholar with automatic OpenAlex fallback (free, no API key); returns citation counts and open-access PDFs
  * **web_search** — Tavily web search for current events and non-academic info
  * **calculate** — numexpr math evaluation
  * **execute_python** — process-isolated Python execution with AST-based validation (import whitelist, dunder blocking, dangerous builtin blocking) + 10s timeout
* **Reflexion loops with stuck-loop detection** — after generating an answer, the evaluator checks faithfulness (NLI entailment score, minimum across all claims) and completeness (Gemini Flash). If either score < 0.75, the agent can regenerate, retrieve more, or reformulate — up to 3 iterations. On `regenerate`, tool selection is skipped and the answer generator reruns with existing context. If completeness doesn't improve >0.05 between iterations, the evaluator auto-accepts to break stuck loops
* **Multi-turn agent conversations** — session history is passed to the answer generator so follow-up questions resolve pronouns and references correctly
* **Parallel tool execution** — when multiple tools are selected (e.g., corpus + arXiv), they run concurrently via `ThreadPoolExecutor`
* **Model failover with circuit breaker** — if `gemini-3.5-flash` is overloaded (503/429), automatically falls back to `gemma-4-26b-a4b-it` (Gemma 4). Circuit breaker skips the primary model for 60s after failure, so subsequent calls go directly to the fallback
* **google-genai native function calling** — no LangChain LLM wrappers; uses `types.FunctionCallingConfig(mode="AUTO")` so the model can return an empty tool list on `regenerate` reflexion actions

### 🔍 Hybrid Retrieval Pipeline

* **Dense + sparse search** — BGE-M3 dense vectors fused with BM25 lexical search via Reciprocal Rank Fusion (RRF)
* **Cross-encoder reranking** — BAAI/bge-reranker-v2-m3 scores the top candidates for precision
* **Faithfulness verification** — `cross-encoder/nli-deberta-v3-base` NLI model scores entailment probability per claim; unsupported assertions are flagged or rejected
* Retrieves 30 candidates, reranks to top 12, verifies citations against source chunks

### 🌍 True Multilingual Support

* **10+ Indian languages** + English (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu)
* Unicode script-based language detection with Devanagari hi/mr disambiguation (no misclassification of short Indic queries)
* **Two RAG strategies:**
  * **Strategy A:** Direct multilingual reasoning (recommended)
  * **Strategy B:** Translation-enhanced reasoning with NLLB-200 (sentence-batched)
* Cross-lingual semantic search with BGE-M3 embeddings (1024d)

### 🛡️ Production-Ready Infrastructure

* Thread-safe model initialization (double-checked locking on all singletons)
* Warm-up at startup via FastAPI lifespan (embeddings, vector store, reranker, BM25 index) — first request is never cold
* Configurable agent pipeline timeout (default 120s) with graceful 504 response
* Session TTL eviction, admin-gated destructive ops, Prometheus metrics
* API key authentication, env-driven CORS, Pydantic v2 validation
* Path traversal protection, URL scheme validation on rendered links

### 🗄️ Three-Layer Caching

* **LLM response cache** (128 entries, 10 min TTL) — identical prompts skip Gemini API entirely
* **Retrieval cache** (64 entries, 5 min TTL) — same query skips embedding + ChromaDB + BM25 + reranking; auto-invalidated on document ingest
* **Tool result cache** (64 entries, 3 min TTL) — arXiv, Semantic Scholar, web search results cached across reflexion loops
* All sizes/TTLs configurable via env vars; `GET /cache/stats` for observability

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

# Optional — higher token limit for agent answers (default 4096)
AGENT_MAX_TOKENS=4096
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

In Agentic mode the UI shows:
* Animated progress stepper with elapsed timer while the agent works
* Color-coded source cards with paper titles, authors, year, citation counts, and PDF links
* Tool call log with execution latencies

### 🔌 REST API

#### Standard Chat — `POST /chat`

```python
import requests

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
print(f"Sources: {len(data['sources'])}")
print(f"Reflexion iterations: {data['reflexion_iterations']}")
for src in data['sources']:
    print(f"  [{src['section']}] {src['title']} ({src['year']}) — {src['citations']} citations")
    if src['pdf_url']:
        print(f"    PDF: {src['pdf_url']}")
```

**Agent response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Generated answer |
| `language` | string | Detected language code |
| `sources` | list | Papers/passages with title, authors, year, citations, pdf_url, source URL |
| `tool_calls` | list | Tool execution log with name, args, latency_ms |
| `reflexion_iterations` | int | Number of reflexion loops executed (0-3) |
| `processing_time` | float | Total seconds |

---

## 🔧 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Multi-turn chat with session history |
| `/agent/query` | POST | Agentic pipeline with reflexion loops (configurable timeout, default 120s → 504) |
| `/query` | POST | Single-turn question answering |
| `/upload` | POST | Upload PDF file (multipart form) |
| `/ingest` | POST | Ingest a PDF into the vector store |
| `/ingest/all` | POST | Bulk ingest all PDFs (async, returns job_id) |
| `/ingest/status/{job_id}` | GET | Check bulk ingest job status |
| `/papers` | GET | List uploaded PDFs |
| `/stats` | GET | Vector store statistics |
| `/cache/stats` | GET | Cache hit rates, sizes, and TTL config |
| `/search` | POST | Retrieval-only search — corpus, web, or both (no LLM generation) |
| `/cache` | DELETE | Clear all caches (LLM, retrieval, tool) |
| `/health` | GET | Health check |
| `/purge/papers` | DELETE | Delete all PDFs (requires admin key) |
| `/purge/database` | DELETE | Clear vector database (requires admin key) |

---

## 📁 Project Structure

> Full annotated tree: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

```
IndicRAG/
│
├── 📄 Root
│   ├── requirements.txt             # 31 packages
│   ├── .env.example                 # LLM_API_KEYS, TAVILY, AGENT_MAX_TOKENS, etc.
│   ├── start_server.py              # Launcher with pre-flight checks (multi-key aware)
│   └── patterns.json                # Regex patterns for PDF cleaning
│
├── 🐍 Core Modules (13)
│   ├── config.py                    # Configuration + env parsing (LLM_API_KEY_POOL)
│   ├── api_server.py                # FastAPI — /chat, /query, /agent/query + 10 more
│   ├── rag.py                       # RAG pipeline + round-robin client pool
│   ├── embeddings.py                # BGE-M3 embeddings (thread-safe)
│   ├── vector_store.py              # ChromaDB wrapper
│   ├── bm25_search.py               # BM25 + RRF fusion
│   ├── rerank.py                    # Cross-encoder reranker (bge-reranker-v2-m3)
│   ├── verify.py                    # NLI faithfulness verification
│   ├── lang_utils.py                # Unicode script + langdetect
│   ├── pdf_utils.py                 # PDF extraction, Indic-aware chunking
│   ├── ingest.py                    # Parallel ingestion with MD5 dedup
│   ├── translation.py               # NLLB-200 sentence-batched (Strategy B)
│   ├── cache.py                     # Thread-safe TTL LRU cache (LLM, retrieval, tool)
│   └── purge.py                     # CLI cleanup (papers, db, models)
│
├── 🤖 agent/                        # Agentic RAG Pipeline (v2.0)
│   ├── state.py                     # AgentState + ReflexionFeedback schemas
│   ├── tool_declarations.py         # 6 google-genai FunctionDeclarations
│   ├── tool_executor.py             # Tool impls: corpus, arXiv, S2/OpenAlex, web, calc, sandbox
│   ├── graph.py                     # LangGraph StateGraph + reflexion routing
│   └── nodes/
│       ├── query_planner.py         # Language detection + query decomposition
│       ├── tool_selector.py         # Gemini function calling (mode=ANY)
│       ├── tool_executor_node.py    # Dispatch + context accumulation + audit log
│       ├── answer_generator.py      # Reuses rag.format_context/build_prompt/llm_generate
│       ├── reflexion_evaluator.py   # check_claims() + Gemini completeness judge
│       └── finalizer.py             # Terminal node
│
├── 🧪 tests/
│   └── test_agent.py               # 37 unit + 1 integration test
│
├── 🌐 static/
│   └── index.html                   # SPA: mode toggle, progress stepper, source cards
│
├── 📚 docs/                         # 13 documentation files
│   ├── QUICKSTART.md                # evaluation.md, ARCHITECTURE.md, CONTRIBUTING.md, ...
│   ├── Eval/                        # nDCG@10, Recall@20, relevance judgments
│   ├── RELEASE_v2.0.0.md           # v2.0 release notes
│   └── feature-requests/            # v2.0 planning docs
│
├── 💡 examples/                     # example_ingest.py, example_query.py
├── 🔧 deploy/                       # nginx.example.conf
│
└── 📊 Data (git-ignored)
    ├── papers/                      # PDF documents
    ├── chroma_db/                   # Vector database
    └── models/                      # Cached ML models
```

---

## ⚙️ Configuration

Key settings (all overridable via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | (required) | Google Gemini API key |
| `LLM_MODEL_NAME` | `gemini-3.5-flash` | Gemini model for generation |
| `LLM_FALLBACK_MODEL` | `gemma-4-26b-a4b-it` | Fallback model when primary is overloaded (503/429) |
| `LLM_MAX_TOKENS` | `2048` | Max tokens for standard RAG |
| `AGENT_MAX_TOKENS` | `4096` | Max tokens for agentic pipeline |
| `AGENT_TIMEOUT` | `120` | Agent pipeline timeout in seconds |
| `TAVILY_API_KEY` | (optional) | Tavily key for agent web search |
| `ADMIN_API_KEY` | (none) | Required for `/purge/*` endpoints |
| `API_KEYS` | (none) | Comma-separated keys for auth |
| `CORS_ORIGINS` | localhost | Comma-separated allowed origins |
| `USE_RERANKER` | `true` | Enable cross-encoder reranking |
| `USE_HYBRID_SEARCH` | `true` | Enable BM25 + dense fusion |
| `FAITHFULNESS_ENFORCE` | `warn` | `warn`, `strip`, or `regen` |
| `FAITHFULNESS_THRESHOLD` | `0.5` | NLI support score threshold |
| `LLM_CACHE_SIZE` | `128` | Max entries in LLM response cache |
| `LLM_CACHE_TTL` | `600` | LLM cache TTL in seconds (10 min) |
| `RETRIEVAL_CACHE_SIZE` | `64` | Max entries in retrieval cache |
| `RETRIEVAL_CACHE_TTL` | `300` | Retrieval cache TTL in seconds (5 min) |
| `TOOL_CACHE_SIZE` | `64` | Max entries in agent tool cache |
| `TOOL_CACHE_TTL` | `180` | Tool cache TTL in seconds (3 min) |

---

## 🎯 Supported Languages

| Language | Code | Native Name |
|----------|------|-------------|
| English | en | English |
| Hindi | hi | हिंदी |
| Telugu | te | తెలుగు |
| Tamil | ta | தமிழ் |
| Bengali | bn | বাংলা |
| Marathi | mr | मराठी |
| Gujarati | gu | ગુજરાતી |
| Kannada | kn | ಕನ್ನಡ |
| Malayalam | ml | മലയാളം |
| Punjabi | pa | ਪੰਜਾਬੀ |
| Odia | or | ଓଡ଼ିଆ |
| Urdu | ur | اردو |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Query Planner                                                   │
│  detect_language() → decompose into sub-queries                  │
└───────┬─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tool Selector                                                   │
│  Gemini function calling (mode=AUTO) picks from 6 tools          │
└───────┬─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tool Executor                                                   │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────┐ ┌──────┐ ┌──┐ │
│  │IndicRAG  │ │  arXiv   │ │  S2 /   │ │ Web  │ │ Calc │ │Py│ │
│  │ Corpus   │ │  Search  │ │OpenAlex │ │Search│ │      │ │  │ │
│  └──────────┘ └──────────┘ └─────────┘ └──────┘ └──────┘ └──┘ │
└───────┬─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Answer Generator                                                │
│  rag.format_context() → rag.build_prompt() → rag.llm_generate() │
└───────┬─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Reflexion Evaluator                                             │
│  verify.check_claims() + Gemini completeness judge               │
│  → accept | regenerate | retrieve_more | reformulate             │
└───────┬────────────────────────────────┬────────────────────────┘
        │ accept                         │ retry (max 3)
        ▼                                └──→ back to Tool Selector
┌──────────────┐                              or Query Planner
│   Finalizer  │
└──────────────┘
```

---

## 📊 Performance

Typical query latency (on CPU):

| Mode | Latency | Notes |
|------|---------|-------|
| Standard RAG (Strategy A) | ~1-2s | Single-pass |
| Standard RAG (Strategy B) | ~3-6s | Includes NLLB translation |
| Agentic RAG (1 reflexion) | ~15-30s | Multi-tool + evaluation (parallel tool execution) |
| Agentic RAG (max reflexions) | ~60-90s | Configurable timeout (default 120s) → 504 |

Memory usage:
* Base system: ~500MB
* With BGE-M3 embeddings: ~2.5GB
* With reranker: ~3.5GB
* With NLLB translation: ~6GB (Strategy B only)

---

## 📈 KPI Metrics

| Metric | Score |
|--------|-------|
| Retrieval Precision | 0.93 |
| Retrieval Recall | 0.91 |
| Faithfulness (Grounding Accuracy) | 0.98 |
| Attribution Accuracy | 0.97 |
| Technical Depth | 0.88 |
| Convergence / Mechanistic Reasoning | 0.86 |
| Cross-Document Discipline | 0.95 |
| Hallucination Rate | < 2% |

See [docs/evaluation.md](docs/evaluation.md) for detailed methodology.

---

## 🐛 Troubleshooting

**"API key not configured"**
```bash
cat .env | grep LLM_API_KEY
```

**"No documents indexed"**
```bash
python ingest.py
```

**Agent web search fails**
```bash
# Ensure TAVILY_API_KEY is set in .env
cat .env | grep TAVILY_API_KEY
```

**Agent answers truncated**
```bash
# Increase token limit in .env
AGENT_MAX_TOKENS=8192
```

**"Translation model gated"**
- The system uses NLLB-200 which requires no authentication
- First use downloads ~2.4GB automatically

---

## 🧹 Maintenance

```bash
# Delete all PDFs
python purge.py --papers

# Clear vector database
python purge.py --db

# Remove cached models
python purge.py --models

# Clear everything
python purge.py --all --yes
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md)

**v2.0 Changelog:**
* Agentic RAG pipeline with LangGraph state machine and reflexion loops
* 6 agent tools: corpus retrieval, arXiv, Semantic Scholar/OpenAlex, web search, calculator, process-isolated Python
* `POST /agent/query` endpoint with source metadata and tool call audit log
* `POST /search` retrieval-only endpoint (corpus, web, or both)
* Three-layer TTL cache: LLM responses, retrieval results, agent tool results — all env-configurable
* `GET /cache/stats` and `DELETE /cache` endpoints for cache observability and management
* Multi-key Gemini API load balancing via `LLM_API_KEYS` with model-level failover (Gemma 4 fallback) and circuit breaker
* Web UI: pipeline mode toggle, agent progress stepper, paper source cards with PDF links
* OpenAlex fallback for Semantic Scholar 429 rate limits (single attempt, no retries)
* Parallel tool execution via ThreadPoolExecutor
* Reflexion stuck-loop detection (auto-accept when completeness stops improving)
* AST-based Python sandbox (replaces string denylist)
* Persistent error bubble in UI (replaces vanishing toast)
* Citation numbers on agentic source cards
* URL scheme validation (XSS prevention) on rendered source links
* `AGENT_MAX_TOKENS` config for longer agent answers

**v2.0 Patch (2026-06-27):**
* **Faithfulness verifier** switched from relevance reranker to `cross-encoder/nli-deberta-v3-base` NLI model — score is now entailment probability, not relevance; faithfulness uses minimum across all claims (not average) so a single hallucinated claim is caught
* **Multi-turn agent** — `conversation_history` threaded through `AgentState` and prepended to the answer generator's prompt; follow-up questions in `/agent/query` now resolve prior context
* **Regenerate reflexion** fixed — tool selector now returns an empty tool list (was being overridden to `indicrag_retrieval` even when the model correctly chose no tools)
* **BM25 index** keyed per collection — custom-collection queries no longer search the wrong index
* **Client pool race** fixed — `next(_client_index)` now runs under `_client_lock` in both `_get_client()` and `generate_with_failover()`
* **Error responses** sanitised — internal paths and exception details no longer leak in 500 responses
* **Agent graph** lazy-loaded — import errors in agent nodes no longer prevent standard RAG endpoints from starting
* **Section header regex** broadened; 6 additional academic headers added (`limitations`, `future work`, `appendix`, etc.)
* **Translation** `max_length` raised 512 → 1024 to prevent silent truncation of long Indic answers
* **numexpr** input validated against an identifier whitelist before evaluation
* **Query cache**, **session eviction**, **job store** — thread-safety and memory-leak fixes
* **Completeness evaluator** — truncates at sentence boundary; JSON example uses non-zero values to prevent models echoing back the placeholder; prompt clarifies faithfulness vs. completeness action mapping
* **System prompt** — outside-knowledge contradiction resolved; backslash continuations removed; medical disclaimer narrowed to specific treatment/dosage recommendations
* 26 additional unit tests covering hybrid search, translation, cache invalidation, and concurrency

**v1.5 Features:**
* Hybrid retrieval pipeline (BGE-M3 dense + BM25 + RRF)
* Cross-encoder reranking (bge-reranker-v2-m3)
* NLI-based faithfulness verification
* Thread-safe model initialization
* Sentence-batched translation (Strategy B)
* Evaluation framework (nDCG@10, Recall@20, CI gating)

---

## 🙏 Acknowledgments

Built with:

* [Google Gemini](https://ai.google.dev/) — Multilingual LLM
* [LangGraph](https://github.com/langchain-ai/langgraph) — Agent state machine
* [Sentence Transformers](https://www.sbert.net/) — BGE-M3 embeddings & reranking
* [arXiv API](https://arxiv.org/) — Preprint search
* [Semantic Scholar](https://www.semanticscholar.org/) — Academic paper search
* [OpenAlex](https://openalex.org/) — Open scholarly metadata
* [Tavily](https://tavily.com/) — Web search for AI agents
* [ChromaDB](https://www.trychroma.com/) — Vector database
* [FastAPI](https://fastapi.tiangolo.com/) — API framework
* Python subprocess sandbox — Process-isolated code execution

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file for details.

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
