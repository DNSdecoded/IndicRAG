# 🌐 Multilingual Scientific RAG System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Production Ready](https://img.shields.io/badge/status-production--ready-green.svg)]()
[![INDICRAG.png](https://i.postimg.cc/vTcp9wtY/INDICRAG.png)

A **production-ready** Retrieval-Augmented Generation (RAG) system with multilingual support for scientific research and knowledge exploration. Built with robust error handling, structured logging, and enterprise-grade features.

---

## ✨ Key Features

### 🧠 **Advanced Document Processing**
* PDF extraction with PyMuPDF (context managers for resource safety)
* Intelligent text cleaning (preserves structure, removes noise)
* Semantic chunking with configurable overlap
* Persistent vector storage via ChromaDB

### 🌍 **True Multilingual Support**
* **10+ Indian languages** + English (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia)
* Automatic language detection
* **Two RAG strategies:**
  * **Strategy A:** Direct multilingual reasoning (recommended)
  * **Strategy B:** Translation-enhanced reasoning with NLLB-200
* Cross-lingual semantic search with E5 embeddings

### 🤖 **LLM Integration**
* Google Gemini 2.5 Flash integration
* Configurable safety settings and generation parameters
* Smart citation extraction from retrieved context
* Empty collection handling with graceful degradation

### 🛡️ **Production-Ready Infrastructure**
* Structured logging throughout (no `print()` statements)
* Robust error handling with detailed error shapes
* API key authentication with secure parsing
* Health checks and monitoring endpoints
* Pydantic v2 validation and type safety
* Safe directory creation with permission checks

### 🧹 **Operational Tools**
* **`purge.py`** - CLI utility to safely clear PDFs, database, or model cache
* Comprehensive ingestion pipeline with progress tracking
* Pre-flight checks before server startup
* Test suite for pipeline validation

---

## 🚀 Quick Start

### Prerequisites

* Python 3.11+
* Google Gemini API key ([Get one here](https://ai.google.dev/))
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

Open http://localhost:8080 and ask questions in any supported language!

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

### Testing

```bash
# Run integration tests
python test_pipeline.py

# Test with example queries
python examples/example_query.py
```

---

## 📁 Project Structure

```
multilingual-rag/
├── api_server.py          # FastAPI app with authentication
├── config.py              # Configuration with ensure_directories()
├── embeddings.py          # E5 multilingual embeddings
├── ingest.py              # PDF ingestion pipeline
├── lang_utils.py          # Language detection
├── pdf_utils.py           # PDF processing
├── rag.py                 # Core RAG logic
├── translation.py         # NLLB-200 translation (Strategy B)
├── vector_store.py        # ChromaDB wrapper
├── start_server.py        # Server launcher with pre-flight checks
├── purge.py               # Cleanup utility (NEW!)
├── test_pipeline.py       # Integration tests
│
├── static/                # Web frontend
│   └── index.html         # Modern Ocean UI
│
├── docs/                  # Documentation
│   ├── QUICKSTART.md
│   ├── DEPLOYMENT.md
│   ├── ARCHITECTURE.md
│   └── CONTRIBUTING.md
│
├── examples/              # Example scripts
│   ├── example_ingest.py
│   └── example_query.py
│
├── papers/                # Your PDF documents
├── chroma_db/             # Vector database
└── models/                # Cached ML models
```

---

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Embedding model (multilingual E5)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSION = 768

# Translation models (Strategy B)
TRANSLATION_MODEL_EN_TO_INDIC = "facebook/nllb-200-distilled-600M"
TRANSLATION_MODEL_INDIC_TO_EN = "facebook/nllb-200-distilled-600M"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# RAG
DEFAULT_TOP_K = 8
MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_LENGTH = 4000

# LLM
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048
```

---

## 🎯 Supported Languages

| Language | Code | Native Name |
|----------|------|-------------|
| English | en | English |
| Hindi | hi | हिंदी |
| Tamil | ta | தமிழ் |
| Telugu | te | తెలుగు |
| Bengali | bn | বাংলা |
| Marathi | mr | मराठी |
| Gujarati | gu | ગુજરાતી |
| Kannada | kn | ಕನ್ನಡ |
| Malayalam | ml | മലയാളം |
| Punjabi | pa | ਪੰਜਾਬੀ |
| Odia | or | ଓଡ଼ିଆ |

---

## 📊 Performance

Typical query latency (on CPU):
* **Strategy A** (direct multilingual): ~1-2s
* **Strategy B** (with translation): ~3-6s (includes NLLB translation time)

ChromaDB retrieval: <100ms for 1000s of documents

Memory usage:
* Base system: ~500MB
* With E5 embeddings: ~2GB
* With NLLB translation: ~4.5GB

---

## 🔒 Production Features

### Security
* API key authentication with secure parsing
* Input validation with Pydantic v2
* CORS configuration
* Environment-based secrets

### Observability
* Structured logging across all modules
* Request/response logging in API
* Processing time tracking
* Health check endpoint

### Robustness
* Graceful empty collection handling
* Resource safety (context managers)
* Permission-aware directory creation
* Comprehensive error responses

### Quality
* Citation extraction from English answers (Strategy B)
* Context length enforcement
* Top-k bounds validation
* Newline preservation in PDF processing

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
* ✅ Production logging migration
* ✅ Robust error handling
* ✅ Pydantic v2 compatibility
* ✅ Resource safety improvements
* ✅ Empty collection handling
* ✅ Citation extraction fixes
* ✅ Purge utility addition

---

## 🙏 Acknowledgments

Built with excellent open-source tools:

* [Google Gemini](https://ai.google.dev/) - Multilingual LLM
* [Sentence Transformers](https://www.sbert.net/) - E5 embeddings
* [Facebook NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) - Translation
* [ChromaDB](https://www.trychroma.com/) - Vector database
* [FastAPI](https://fastapi.tiangolo.com/) - API framework
* [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing

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
