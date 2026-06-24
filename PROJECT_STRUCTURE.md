# 📁 Project Structure — IndicRAG v2.0

```
IndicRAG/
│
├── 📄 Root Files
│   ├── README.md                    # Main documentation (v2.0)
│   ├── PROJECT_STRUCTURE.md         # This file
│   ├── PRODUCTION.md                # Production deployment notes
│   ├── LICENSE                      # MIT License
│   ├── requirements.txt             # Python dependencies (31 packages)
│   ├── .env.example                 # Environment template (LLM_API_KEYS, TAVILY, etc.)
│   ├── .gitignore                   # Git ignore rules
│   ├── patterns.json                # Regex patterns for PDF cleaning
│   └── start_server.py              # Server launcher with pre-flight checks
│
├── 🐍 Core Modules
│   ├── config.py                    # Configuration constants + env var parsing
│   ├── api_server.py                # FastAPI server — /chat, /query, /agent/query + management endpoints
│   ├── rag.py                       # RAG pipeline — retrieve, rerank, format, prompt, generate (multi-key client pool)
│   ├── embeddings.py                # BGE-M3 multilingual embeddings (thread-safe singleton)
│   ├── vector_store.py              # ChromaDB wrapper (thread-safe)
│   ├── bm25_search.py               # BM25 lexical index + RRF fusion with dense scores
│   ├── rerank.py                    # Cross-encoder reranker (bge-reranker-v2-m3)
│   ├── verify.py                    # NLI-based faithfulness verification (claim-level)
│   ├── lang_utils.py                # Unicode script + langdetect language detection
│   ├── pdf_utils.py                 # PDF extraction, Indic-aware chunking
│   ├── ingest.py                    # PDF ingestion pipeline (parallel, MD5 dedup)
│   ├── translation.py               # NLLB-200 translation, sentence-batched (Strategy B)
│   ├── cache.py                     # Thread-safe TTL LRU cache (LLM, retrieval, tool instances)
│   └── purge.py                     # CLI cleanup utility (papers, db, models)
│
├── 🤖 agent/                        # Agentic RAG Pipeline (v2.0)
│   ├── __init__.py
│   ├── state.py                     # AgentState TypedDict + ReflexionFeedback schema
│   ├── tool_declarations.py         # google-genai FunctionDeclaration objects (6 tools)
│   ├── tool_executor.py             # Tool implementations — corpus, arXiv, S2/OpenAlex, web, calc, AST-validated Python sandbox
│   ├── graph.py                     # LangGraph StateGraph with conditional reflexion edges
│   └── nodes/
│       ├── __init__.py
│       ├── query_planner.py         # Language detection + query decomposition via Gemini
│       ├── tool_selector.py         # Gemini function calling (mode=ANY) — picks tools
│       ├── tool_executor_node.py    # Parallel tool dispatch via ThreadPoolExecutor, context accumulation, audit logging
│       ├── answer_generator.py      # Reuses rag.format_context / build_prompt / llm_generate (with model failover)
│       ├── reflexion_evaluator.py   # Faithfulness (verify.check_claims) + completeness (Gemini judge) + stuck-loop detection
│       └── finalizer.py             # Terminal node — selects final_answer or draft_answer
│
├── 🧪 tests/
│   ├── __init__.py
│   └── test_agent.py               # 11 unit tests + 1 integration test for agent pipeline
│
├── 🌐 static/                       # Web Frontend
│   └── index.html                   # SPA — pipeline mode toggle, agent progress stepper, source cards
│
├── 📚 docs/                         # Documentation
│   ├── QUICKSTART.md                # 5-minute setup guide
│   ├── ARCHITECTURE.md              # Technical deep dive
│   ├── DEPLOYMENT.md                # Deployment guide
│   ├── DEPLOY.md                    # Simple deploy reference
│   ├── PRODUCTION.md                # Production hardening
│   ├── GEMINI_SETUP.md              # Gemini API configuration
│   ├── CONTRIBUTING.md              # Contribution guide
│   ├── PDF_UPLOAD_NOTE.md           # Upload notes
│   ├── evaluation.md                # Evaluation methodology + KPI metrics
│   ├── RELEASE_v2.0.0.md           # v2.0 release notes draft
│   ├── Eval/                        # Evaluation framework
│   │   ├── evaluate.py              # Automated eval runner (nDCG@10, Recall@20)
│   │   ├── relevance_judgments.json # Ground truth judgments
│   │   ├── answers_and_citations.json
│   │   ├── eval_report.json         # Latest eval results
│   │   └── eval_report.md           # Human-readable eval report
│   └── feature-requests/            # Feature planning docs
│       └── v2.0-agentic-rag/
│           ├── instruction.md
│           ├── planning.md
│           └── roadmap.md
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
    └── models/                      # Cached ML models (BGE-M3, reranker, NLLB)
```

---

## 📊 Stats

| Category | Count |
|----------|-------|
| Core Python modules | 14 |
| Agent modules | 10 |
| Test files | 1 (16 unit tests + 1 integration) |
| Documentation files | 13 |
| Example scripts | 2 |
| Frontend | 1 SPA |
| Agent tools | 6 |
| API endpoints | 15 |
| Supported languages | 12 (English + 11 Indic) |
| Dependencies | 31 packages |

---

## 🚀 Quick Commands

```bash
# Start server
python start_server.py

# Development mode
python start_server.py --dev

# Ingest documents
python ingest.py

# Run agent tests
pytest tests/test_agent.py -v -m "not integration and not network"

# Cleanup
python purge.py --all --yes
```

**Access:**
- 🌐 Web UI: http://localhost:8080
- 📖 API Docs: http://localhost:8080/api/docs
- ❤️ Health: http://localhost:8080/health
