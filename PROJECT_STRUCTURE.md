# 📁 Project Structure — IndicRAG v2.4

```
IndicRAG/
│
├── 📄 Root Files
│   ├── README.md                    # Main documentation
│   ├── PROJECT_STRUCTURE.md         # This file
│   ├── PRODUCTION.md                # Production hardening notes (root copy)
│   ├── LICENSE                      # MIT License
│   ├── requirements.txt             # Python dependencies
│   ├── pyproject.toml               # Build/tooling config
│   ├── .env.example                 # Annotated environment template
│   ├── .gitignore                   # Git ignore rules
│   ├── patterns.json                # Regex patterns for PDF cleaning
│   ├── Dockerfile                   # python:3.11-slim image, non-root, EXPOSE 8080
│   ├── docker-compose.yml           # Single-node stack (service `indicrag`, 8080)
│   └── start_server.py              # Server launcher with pre-flight checks
│
├── 🐍 Core Modules
│   ├── config.py                    # Configuration constants + env var parsing
│   ├── api_server.py                # FastAPI app — mounts the routers, docs at /api/docs
│   ├── deps.py                      # Shared dependencies (API-key auth, rate limiter)
│   ├── middleware.py                # Request middleware
│   ├── rag.py                       # RAG pipeline — retrieve, rerank, format, prompt, generate
│   ├── llm_client.py                # Provider dispatch, circuit breaker, model failover chain
│   ├── embeddings.py                # BGE-M3 multilingual embeddings (thread-safe singleton)
│   ├── vector_store.py              # ChromaDB wrapper (thread-safe)
│   ├── bm25_search.py               # BM25 lexical index + RRF fusion with dense scores
│   ├── rerank.py                    # Cross-encoder reranker (bge-reranker-v2-m3)
│   ├── colbert_rerank.py            # Optional ColBERT MaxSim late-interaction rerank
│   ├── onnx_ce.py                   # ONNX cross-encoder runtime path
│   ├── verify.py                    # NLI-based faithfulness verification (claim-level)
│   ├── contradiction.py             # Cross-source contradiction detection
│   ├── lang_utils.py                # Unicode script + langdetect language detection
│   ├── pdf_utils.py                 # PDF extraction, Indic-aware chunking
│   ├── ingest.py                    # PDF ingestion pipeline (parallel, MD5 dedup)
│   ├── metadata_enrich.py           # Title/author/year/DOI enrichment
│   ├── figure_captioner.py          # Figure caption extraction
│   ├── download_utils.py            # Safe remote PDF fetching (used by /ingest/from-url)
│   ├── translation.py               # Translation for Strategy B
│   ├── cache.py                     # Thread-safe TTL LRU cache (LLM, retrieval, tool instances)
│   ├── cache_refresh.py             # Background cache refresh
│   ├── gemini_cache.py              # Gemini context caching
│   ├── persistence.py               # SQLite (WAL) — sessions, jobs, feedback, watches, reports
│   ├── sse_utils.py                 # Server-sent-events streaming helper
│   ├── report_runner.py             # Saved-report generation
│   ├── watch_runner.py              # Scheduled corpus watches
│   └── purge.py                     # CLI cleanup utility (papers, db, models)
│
├── 🛣️ routes/                       # FastAPI routers (~51 endpoints total)
│   ├── query.py                     # /query, /query/stream, /search, /health, /stats, /
│   ├── agent.py                     # /agent/query, /agent/stream
│   ├── chat.py                      # /chat session persistence
│   ├── ingest.py                    # /ingest, /ingest/all, /ingest/from-url, /upload, /ingest/health
│   ├── management.py                # /papers, /cache, /quality, /purge/*, exports
│   ├── models.py                    # /models — the selectable-model dropdown
│   ├── feedback.py                  # /feedback, /feedback/stats
│   ├── report.py                    # /reports
│   └── watch.py                     # /watch
│
├── 🔌 providers/                    # LLM backends behind llm_client.py
│   ├── base.py                      # LLMBackend interface
│   ├── gemini.py                    # Google Gemini (bare model names)
│   └── openrouter.py                # OpenRouter (`vendor/model` slugs)
│
├── 🤖 agent/                        # Agentic RAG Pipeline
│   ├── state.py                     # AgentState TypedDict + ReflexionFeedback schema
│   ├── tool_declarations.py         # google-genai FunctionDeclaration objects
│   ├── tool_executor.py             # Tool implementations — corpus, arXiv, S2/OpenAlex, web, calc, AST-validated Python sandbox
│   ├── json_utils.py                # Tolerant JSON parsing of model output
│   ├── graph.py                     # LangGraph StateGraph with conditional reflexion edges
│   └── nodes/
│       ├── query_planner.py         # Language detection + query decomposition
│       ├── tool_selector.py         # Function calling — picks tools
│       ├── tool_executor_node.py    # Parallel tool dispatch, context accumulation, audit logging
│       ├── answer_generator.py      # Reuses rag.format_context / build_prompt / llm_generate
│       ├── reflexion_evaluator.py   # Faithfulness + completeness judge + stuck-loop detection
│       └── finalizer.py             # Terminal node — selects final_answer or draft_answer
│
├── 🧪 tests/                        # 27 test modules + shared conftest.py
│   ├── conftest.py                  # Fixtures — points the vector store at a temp dir
│   ├── test_agent.py                # Agent pipeline
│   ├── test_rag.py                  # Retrieval, fusion, prompt assembly, caching
│   ├── test_ingest.py               # Ingestion, dedup, collision locking
│   ├── test_llm_client_dispatch.py  # Attempt chain, fallbacks, circuit breaker
│   ├── test_providers_*.py          # Gemini / OpenRouter / base backend
│   ├── test_api.py                  # Route-level tests
│   └── ...                          # verify, colbert, hyde, watch, report, feedback, embeddings, …
│
├── 🌐 static/                       # Web Frontend
│   └── index.html                   # SPA — Ask, Retrieval, Library, History, Reports, Index health
│
├── 📚 docs/                         # Documentation
│   ├── QUICKSTART.md                # 5-minute setup guide
│   ├── ARCHITECTURE.md              # Technical deep dive
│   ├── DEPLOYMENT.md                # Deployment guide (canonical)
│   ├── DEPLOY.md                    # Pointer to DEPLOYMENT.md
│   ├── PRODUCTION.md                # Docker/production hardening
│   ├── GEMINI_SETUP.md              # LLM provider configuration
│   ├── CONTRIBUTING.md              # Contribution guide
│   ├── evaluation.md                # Evaluation methodology + KPI metrics
│   ├── RELEASE_v2.0.0.md            # v2.0 release notes (historical)
│   ├── Eval/                        # Evaluation framework
│   │   ├── evaluate.py              # Automated eval runner
│   │   ├── relevance_judgments.json # Ground truth judgments
│   │   ├── answers_and_citations.json
│   │   ├── eval_report.json         # Latest eval results
│   │   └── eval_report.md           # Human-readable eval report
│   ├── feature-requests/            # Feature planning docs (historical)
│   └── superpowers/                 # Plans + specs (historical)
│
├── 💡 examples/                     # Example Scripts
│   ├── example_ingest.py            # PDF ingestion example
│   └── example_query.py             # Query examples (single-turn + multi-turn)
│
├── 🔧 deploy/                       # Deployment Configs
│   └── nginx.example.conf           # Nginx reverse proxy config
│
├── 🛠️ Utilities
│   ├── check_db.py                  # ChromaDB inspection utility
│   └── test_gen.py                  # Generation test script
│
└── 📊 Data Directories (git-ignored)
    ├── papers/                      # Your PDF documents
    ├── chroma_db/                   # ChromaDB vector database
    ├── figures/                     # Extracted figures
    ├── models/                      # Cached ML models (BGE-M3, reranker, NLI)
    └── sessions.db                  # SQLite state (sessions, jobs, feedback, watches, reports)
```

---

## 📊 Stats

| Category | Count |
|----------|-------|
| Route modules | 9 |
| API endpoints | ~51 |
| LLM providers | 2 (Gemini, OpenRouter) |
| Agent nodes | 6 |
| Agent tools | 6 |
| Test modules | 27 |
| Example scripts | 2 |
| Frontend | 1 SPA |
| Supported languages | 12 (English + 11 Indic) |
| Dependencies | 35 packages |

---

## 🚀 Quick Commands

```bash
# Start server
python start_server.py

# Development mode (auto-reload; downgrades the API_KEYS check to a warning)
python start_server.py --dev

# Ingest documents
python ingest.py papers/

# Run tests
pytest tests/ -m "not integration and not network"

# Cleanup (destructive)
python purge.py --all --yes
```

**Access:**
- 🌐 Web UI: http://localhost:8080
- 📖 API Docs: http://localhost:8080/api/docs
- ❤️ Health: http://localhost:8080/health (add `?deep=true` for component checks)
