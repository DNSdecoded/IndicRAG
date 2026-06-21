# 🌐 Multilingual Scientific RAG System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash-blueviolet.svg)](https://ai.google.dev/)
![Production Ready](https://img.shields.io/badge/status-production--ready-green.svg)

![INDICRAG.png](https://cdn.jsdelivr.net/gh/free-whiteboard-online/Free-Erasorio-Alternative-for-Collaborative-Design@3a5f22554411d3d6df27ee788c2df99d583f2c91/uploads/2025-12-03T05-25-45-007Z-3i36rbzio.png)

A **production-ready** Retrieval-Augmented Generation (RAG) system with multilingual support for scientific research and knowledge exploration. Built with robust error handling, structured logging, and enterprise-grade features.

---

## ✨ Key Features

### 🧠 **Advanced Document Processing**
* PDF extraction with PyMuPDF (context managers for resource safety)
* Intelligent text cleaning (preserves structure, removes noise)
* Semantic chunking with Indic script-aware sentence splitting
* Persistent vector storage via ChromaDB

### 🔍 **Hybrid Retrieval Pipeline**
* **Dense + sparse search** — BGE-M3 dense vectors fused with BM25 lexical search via Reciprocal Rank Fusion (RRF)
* **Cross-encoder reranking** — BAAI/bge-reranker-v2-m3 scores the top candidates for precision
* **Faithfulness verification** — NLI-based claim-level grounding check flags unsupported assertions
* Retrieves 30 candidates, reranks to top 12, verifies citations against source chunks

### 🌍 **True Multilingual Support**
* **10+ Indian languages** + English (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia)
* Unicode script-based language detection (no misclassification of short Indic queries)
* **Two RAG strategies:**
  * **Strategy A:** Direct multilingual reasoning (recommended)
  * **Strategy B:** Translation-enhanced reasoning with NLLB-200 (sentence-batched to prevent truncation)
* Cross-lingual semantic search with BGE-M3 embeddings (1024d, strong on Indic scripts)

### 🤖 **LLM Integration**
* Google Gemini 3.5 Flash integration with automatic retry (tenacity, 3 attempts with exponential backoff)
* Optimized system prompt — grounding-first, no mandatory section padding
* Smart citation extraction with range validation
* Low temperature (0.1) for deterministic grounded responses

### 🛡️ **Production-Ready Infrastructure**
* **Thread-safe model initialization** — double-checked locking on all singletons
* **Warm-up at startup** — models loaded via FastAPI lifespan, first request is never cold
* **Session TTL eviction** — stale chat sessions cleaned automatically
* **Admin-gated destructive ops** — purge endpoints require `ADMIN_API_KEY`
* API key authentication, Prometheus metrics, env-driven CORS
* Pydantic v2 validation and type safety

### 🧹 **Operational Tools**
* **`purge.py`** - CLI utility to safely clear PDFs, database, or model cache
* **Web-based document management** - Upload, ingest, and purge via UI
* Comprehensive ingestion pipeline with progress tracking
* Evaluation framework with nDCG@10, Recall@20, and CI gating

---

## 🚀 Quick Start

### Prerequisites

* Python 3.11+
* Google Gemini API key ([Get one here](https://aistudio.google.com/api-keys))
* 8GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/DNSdecoded/IndicRAG.git
cd IndicRAG

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
# LLM_API_KEY=your_gemini_api_key_here

# Optional: Configure API authentication
# API_KEYS=key1,key2,key3
```

### Ingest Documents

```bash
# Place PDFs in papers/ directory
# Then ingest them:
python ingest.py

# Or specify a directory:
python ingest.py path/to/pdfs
```

### Start Server

```bash
# With pre-flight checks
python start_server.py

# Skip checks (for production)
python start_server.py --skip-checks

# Development mode with auto-reload
python start_server.py --dev
```

🎉 **That's it!** Access the API at:
* **Interactive docs:** http://localhost:8080/api/docs
* **Web Interface:** http://localhost:8080

---

## 📖 Usage

### Via Web UI

Open http://localhost:8080 and:

1. **Ask Questions** - Enter queries in any supported language
2. **Manage Documents** - Expand the panel to:
   - Upload PDFs via drag-and-drop
   - View uploaded papers list
   - Ingest papers into the vector store
   - Purge papers or database (with confirmation)

### Via REST API

```python
import requests

response = requests.post('http://localhost:8080/query', json={
    "question": "యాంటెన్నాతో ml ను ఎలా అమలు చేయవచ్చు?",  # Telugu
    "strategy": "A",
    "top_k": 5
})

result = response.json()
print(result['answer'])
print(f"Citations: {len(result['citations'])}")
```

### Via Python

```python
import rag

result = rag.answer_question(
    "मधुमेह का इलाज क्या है?",  # Hindi: diabetes treatment
    strategy="B",
    top_k=8
)

print(f"Answer ({result['language_name']}): {result['answer']}")
print(f"Used {result['chunks_used']} document chunks")
```

---

## 🔧 API Reference

### `POST /query`

Ask a question and get an AI-powered answer with citations.

**Request:**
```json
{
  "question": "What is quantum computing?",
  "strategy": "A",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Quantum computing is...",
  "language": "en",
  "language_name": "English",
  "chunks_used": 4,
  "citations": [
    {"number": "1", "title": "Quantum Computing Basics", "section": "Introduction"}
  ],
  "processing_time": 1.23
}
```

### `POST /ingest`

Ingest a PDF document (returns extracted title).

### `GET /stats`

Get vector store statistics.

### `GET /health`

Health check endpoint.

### `POST /upload`

Upload a PDF file (multipart form).

### `GET /papers`

List all uploaded PDFs with sizes.

### `DELETE /purge/papers`

Delete all uploaded PDF files.

### `DELETE /purge/database`

Clear the vector database (all chunks).

---

## 🧹 Maintenance Tools

### Purge Utility

Safely clear indexed data:

```bash
# Delete all PDFs
python purge.py --papers

# Clear vector database
python purge.py --db

# Remove cached models (will re-download)
python purge.py --models

# Clear everything (with confirmation)
python purge.py --all

# Non-interactive mode
python purge.py --all --yes
```

### Examples

```bash
# Test with example queries
python examples/example_query.py
```

---

## 📁 Project Structure

```
IndicRAG/
├── api_server.py          # FastAPI app with auth, lifespan warm-up, session TTL
├── config.py              # All configuration constants and prompts
├── rag.py                 # Core RAG pipeline (retrieval, rerank, generate, verify)
├── embeddings.py          # BGE-M3 multilingual embeddings (thread-safe)
├── rerank.py              # Cross-encoder reranker (bge-reranker-v2-m3)
├── bm25_search.py         # BM25 lexical index + RRF fusion
├── verify.py              # NLI-based faithfulness verification
├── vector_store.py        # ChromaDB wrapper (thread-safe)
├── translation.py         # NLLB-200 translation, sentence-batched
├── lang_utils.py          # Unicode script + langdetect detection
├── pdf_utils.py           # PDF extraction, Indic-aware chunking
├── ingest.py              # PDF ingestion pipeline
├── start_server.py        # Server launcher with pre-flight checks
├── purge.py               # CLI cleanup utility
│
├── static/                # Web frontend
│   └── index.html
│
├── docs/                  # Documentation
│   ├── Eval/              # Evaluation framework (nDCG, Recall@20, CI gate)
│   ├── QUICKSTART.md
│   ├── ARCHITECTURE.md
│   └── ...
│
├── examples/              # Example scripts
├── papers/                # Your PDF documents
├── chroma_db/             # Vector database
└── models/                # Cached ML models
```

---

## ⚙️ Configuration

Key settings in `config.py` (all overridable via environment variables):

```python
# Embedding model (BGE-M3: dense + sparse, Indic-strong)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

# Retrieval pipeline
USE_RERANKER = True                 # cross-encoder reranking
USE_HYBRID_SEARCH = True            # dense + BM25 fusion
DEFAULT_TOP_K = 30                  # retrieve wide
MAX_CONTEXT_CHUNKS = 12             # keep after rerank
MAX_CONTEXT_LENGTH = 48000          # ~12k tokens

# Faithfulness verification
FAITHFULNESS_THRESHOLD = 0.5
FAITHFULNESS_ENFORCE = "warn"       # warn | strip | regen

# LLM
LLM_MODEL_NAME = "gemini-3.5-flash"
LLM_TEMPERATURE = 0.1              # low for grounded citation tasks
LLM_MAX_TOKENS = 2048
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | (required) | Google Gemini API key |
| `ADMIN_API_KEY` | (none) | Required for `/purge/*` endpoints |
| `API_KEYS` | (none) | Comma-separated keys for general auth |
| `CORS_ORIGINS` | localhost | Comma-separated allowed origins |
| `USE_RERANKER` | `true` | Enable cross-encoder reranking |
| `USE_HYBRID_SEARCH` | `true` | Enable BM25 + dense fusion |
| `FAITHFULNESS_ENFORCE` | `warn` | `warn`, `strip`, or `regen` |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Sentence-transformers model |

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

## 📈 Final KPI Metrics

For detailed evaluation methodology, automated metrics, and per-query qualitative reports, see [docs/evaluation.md](docs/evaluation.md).

| Metric | Final Score |
|--------|-------------|
| Retrieval Precision | 0.93 |
| Retrieval Recall | 0.91 |
| Faithfulness (Grounding Accuracy) | 0.98 |
| Attribution Accuracy | 0.97 |
| Technical Depth | 0.88 |
| Convergence / Mechanistic Reasoning | 0.86 |
| Cross-Document Discipline | 0.95 |
| Hallucination Rate | < 2% |
| Formatting & Structural Compliance | 0.98 |

---

## 📊 Performance

Typical query latency (on CPU):
* **Strategy A** (direct multilingual): ~1-2s
* **Strategy B** (with translation): ~3-6s (includes NLLB translation time)

ChromaDB retrieval: <100ms for 1000s of documents

Memory usage:
* Base system: ~500MB
* With BGE-M3 embeddings: ~2.5GB
* With reranker: ~3.5GB
* With NLLB translation: ~6GB (Strategy B only)

---

## 🔒 Production Features

### Security
* API key authentication with secure parsing
* Admin key gating for destructive operations (`ADMIN_API_KEY`)
* Input validation with Pydantic v2
* Env-driven CORS (`CORS_ORIGINS`)
* Path traversal protection on ingest endpoints

### Observability
* Structured logging across all modules
* Prometheus metrics at `/metrics`
* Processing time tracking
* Faithfulness warnings logged for ungrounded claims

### Robustness
* Thread-safe model singletons (double-checked locking)
* Warm-up at startup via FastAPI lifespan
* LLM retry with exponential backoff (tenacity)
* Session TTL eviction
* Graceful empty collection handling

### Quality
* Cross-encoder reranking + faithfulness verification
* Hybrid dense+lexical retrieval
* Citation range validation (caps [2020-2023] false positives)
* Sentence-batched translation prevents truncation

---

## 🐛 Common Issues & Solutions

**"API key not configured"**
```bash
# Check .env file
cat .env | grep LLM_API_KEY
```

**"No documents indexed"**
```bash
# Ingest PDFs
python ingest.py
```

**"Translation model gated/authentication required"**
- The system now uses **NLLB-200** which requires no authentication
- First use will download ~2.4GB automatically
- See documentation for manual download if needed

**"Out of memory"**
```python
# Edit config.py to reduce memory usage
CHUNK_SIZE = 512  # Smaller chunks
MAX_CONTEXT_CHUNKS = 3  # Fewer chunks in context
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md)

**Recent improvements:**
* ✅ Hybrid retrieval pipeline (BGE-M3 dense + BM25 lexical + RRF fusion)
* ✅ Cross-encoder reranking (bge-reranker-v2-m3)
* ✅ NLI-based faithfulness verification with configurable enforcement
* ✅ Thread-safe model initialization across all modules
* ✅ Sentence-batched translation (fixes Strategy B truncation)
* ✅ Unicode script-based language detection for short Indic queries
* ✅ LLM retry with exponential backoff (tenacity)
* ✅ Optimized system prompt — grounding-first, no section padding
* ✅ Expanded evaluation framework (nDCG@10, Recall@20, CI gating)
* ✅ Admin key gating for destructive purge endpoints
* ✅ Env-driven CORS, warm-up at startup, session TTL eviction
* ✅ Query embedding LRU cache, Indic-aware chunking

---

## 🙏 Acknowledgments

Built with excellent open-source tools:

* [Google Gemini](https://ai.google.dev/) - Multilingual LLM
* [Sentence Transformers](https://www.sbert.net/) - BGE-M3 embeddings & reranking
* [Facebook NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) - Translation
* [ChromaDB](https://www.trychroma.com/) - Vector database
* [FastAPI](https://fastapi.tiangolo.com/) - API framework
* [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
* [Tenacity](https://github.com/jd/tenacity) - Retry logic

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

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
