# IndicRAG — Project Instructions

## Overview

IndicRAG is a multilingual Retrieval-Augmented Generation system for scientific papers, supporting English + 11 Indic languages. It has two pipelines:
- **Standard RAG** (`/query`, `/chat`) — retrieve, rerank, generate
- **Agentic RAG** (`/agent/query`) — LangGraph state machine with 6 tools, reflexion loop, multi-source retrieval

Current version: **v2.0** (unreleased, targeting July 5 2025)

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn
- **LLM:** Google Gemini (gemini-3.5-flash) via `google-genai` SDK — NOT LangChain LLM wrappers
- **Embeddings:** BGE-M3 (`BAAI/bge-m3`, 1024d) via sentence-transformers
- **Vector Store:** ChromaDB (local persistent)
- **Reranker:** bge-reranker-v2-m3 (cross-encoder)
- **Translation:** NLLB-200 (facebook/nllb-200-distilled-600M)
- **Agent Framework:** LangGraph (state machine with conditional reflexion edges)
- **Frontend:** Single-page HTML (`static/index.html`)

## Architecture

```
start_server.py          Entry point (pre-flight checks)
config.py                All configuration + env var parsing
api_server.py            FastAPI routes (16 endpoints)
rag.py                   Core pipeline: retrieve → rerank → format → prompt → generate
cache.py                 Thread-safe TTL LRU cache (3 instances: LLM, retrieval, tool)
embeddings.py            BGE-M3 singleton
vector_store.py          ChromaDB wrapper
bm25_search.py           BM25 lexical index + RRF fusion
rerank.py                Cross-encoder reranker
verify.py                NLI faithfulness verification
lang_utils.py            Language detection (Unicode script + langdetect)
translation.py           NLLB-200 translation (sentence-batched)
pdf_utils.py             PDF extraction + Indic-aware chunking
ingest.py                PDF ingestion (parallel, MD5 dedup)
purge.py                 CLI cleanup utility

agent/                   Agentic RAG pipeline
  state.py               AgentState TypedDict
  tool_declarations.py   google-genai FunctionDeclaration (6 tools)
  tool_executor.py       Tool implementations
  graph.py               LangGraph StateGraph
  nodes/                 Pipeline nodes (query_planner → tool_selector → tool_executor_node → answer_generator → reflexion_evaluator → finalizer)
```

## Key Design Decisions

- **google-genai native function calling** — tool declarations use `types.FunctionDeclaration` + `FunctionCallingConfig(mode="ANY")`. Do NOT introduce LangChain LLM wrappers.
- **Reuse existing rag.py functions** — agent answer_generator calls `rag.format_context()`, `rag.build_prompt()`, `rag.llm_generate()`. Don't duplicate.
- **Reuse existing verify.py** — reflexion_evaluator uses `verify.check_claims()` for faithfulness scoring. Don't duplicate.
- **Round-robin API key load balancing** — `itertools.cycle` over `config.LLM_API_KEY_POOL`. Thread-safe. Don't switch to least-loaded (Gemini doesn't expose quota in headers).
- **Model-level failover with circuit breaker** — `generate_with_failover()` tries all keys on the primary model (`gemini-3.5-flash`), then falls back to `LLM_FALLBACK_MODEL` (`gemma-4-26b-a4b-it`). A circuit breaker skips the primary for 60s after all keys fail, avoiding wasted retry time on subsequent calls.
- **AGENT_MAX_TOKENS (4096) separate from LLM_MAX_TOKENS (2048)** — agent gets more room because it has multi-source context.
- **OpenAlex as Semantic Scholar fallback** — S2 gets a single attempt (8s timeout, no retries). OpenAlex is CC0, no rate limits, 250M+ works.
- **AST-based sandbox for code execution** — `execute_python` parses code into an AST and validates against an import whitelist, blocks dunder attribute access and dangerous builtins. Runs in a child process with stripped env and 10s timeout. Never use `exec()` or `eval()` directly in the main process.
- **Parallel tool execution** — when multiple tools are selected, `tool_executor_node` runs them concurrently via `ThreadPoolExecutor`.
- **Reflexion stuck-loop detection** — if completeness score doesn't improve by >0.05 between iterations, the evaluator auto-accepts instead of looping uselessly.

## API Endpoints (17)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve frontend SPA |
| GET | `/health` | Health check |
| POST | `/query` | Standard RAG query |
| POST | `/chat` | Multi-turn chat |
| DELETE | `/chat/{session_id}` | Delete chat session |
| POST | `/search` | Retrieval-only search (corpus, web, or both) — no LLM generation |
| POST | `/ingest` | Ingest single PDF |
| POST | `/ingest/all` | Bulk ingest all PDFs |
| GET | `/ingest/status/{job_id}` | Check ingest job status |
| POST | `/upload` | Upload PDF file |
| GET | `/papers` | List ingested papers |
| GET | `/stats` | Collection statistics |
| GET | `/cache/stats` | Cache hit/miss stats |
| DELETE | `/cache` | Clear all caches |
| DELETE | `/purge/papers` | Delete paper files |
| DELETE | `/purge/database` | Wipe vector store |
| POST | `/agent/query` | Agentic RAG query (configurable timeout via `AGENT_TIMEOUT`, default 120s → 504) |

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Start server (runs pre-flight checks)
python start_server.py

# Dev mode
python start_server.py --dev

# Run tests (no API keys needed)
pytest tests/test_agent.py -v -m "not integration and not network"
```

## Environment Variables

Required: `LLM_API_KEY` (or `LLM_API_KEYS` for multi-key load balancing)
Optional: `LLM_FALLBACK_MODEL` (default `gemma-4-26b-a4b-it`), `AGENT_TIMEOUT` (default `120`), `TAVILY_API_KEY` (only for web_search tool), `API_KEYS` (endpoint auth)

All config is in `config.py` — read from `.env` via `python-dotenv`. See `.env.example` for the full list.

## Development Rules

- **Never add `Co-Authored-By: Claude` to git commits** — user preference.
- **Thread safety matters** — embeddings, vector store, BM25 index, and cache are all shared singletons. Use locks where needed.
- **Cache invalidation on ingest** — `retrieval_cache.invalidate()` is called in both `/ingest` and `/ingest/all` handlers.
- **XSS prevention** — all URLs rendered in the frontend go through `safeHref()` (http/https only). Keep this pattern.
- **No LangChain dependency** — the codebase uses google-genai directly. LangGraph is the only LangChain-adjacent dep (for the state machine only).
- **Version string** — lives in `config.py` as `VERSION`. Update it for releases.
- **Indic language support** — 11 languages defined in `config.INDIC_LANGUAGES`. Translation uses NLLB-200 with sentence-level batching.

## Testing

Tests are in `tests/test_agent.py`. Markers:
- (default) — unit tests, no network/API needed
- `@pytest.mark.network` — requires internet (S2/OpenAlex)
- `@pytest.mark.integration` — requires running server + API keys

## Git-Ignored Directories

- `papers/` — uploaded PDFs
- `chroma_db/` — ChromaDB persistent storage
- `models/` — cached HuggingFace models (BGE-M3, reranker, NLLB)
