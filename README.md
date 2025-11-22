# 🌐 Multilingual RAG API Server
[![INDICRAG.png](https://i.postimg.cc/vTcp9wtY/INDICRAG.png)](https://postimg.cc/GTn7wNfV)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready Retrieval-Augmented Generation (RAG) service with multilingual support, built for scientific research and knowledge exploration.

---

## ✨ Features

### 🧠 **Semantic Search over PDFs**
* Extracts and indexes text using PyMuPDF
* Cleans noisy content (page numbers, line art, etc.)
* Splits documents into overlapping chunks for optimal retrieval
* Persistent vector storage with ChromaDB

### 🌍 **Multilingual Support**
* **10+ Indian languages** + English support (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia)
* Language detection for queries
* **Two RAG strategies:**
  * **Strategy A:** Direct multilingual LLM reasoning (recommended)
  * **Strategy B:** Translate → English reasoning → translate back
* Optional IndicTrans2 for high-quality Indic language translation

### 🤖 **LLM Integration**
* Google Gemini 2.5 Flash integration (fast & accurate)
* Configurable model selection
* Citation extraction from retrieved context
* Streaming support (future enhancement)

### 🌐 **Production-Ready HTTP API**
* FastAPI with automatic OpenAPI docs
* Optional API key authentication
* CORS support
* Health checks and metrics
* Static file serving for web UI

---

## 🚀 Quick Start

### Prerequisites

* Python 3.11+
* Google Gemini API key ([Get one here](https://ai.google.dev/))
* 8GB+ RAM recommended (for embeddings + translation models)

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
```

### Ingest Documents

```bash
# Place PDFs in papers/ directory
# Then ingest them:
python ingest.py --dir papers
```

### Start Server

```bash
python start_server.py
```

🎉 **That's it!** Access the API at:
* **Swagger UI:** http://localhost:8080/docs
* **Web Interface:** http://localhost:8080

---

## 📖 Usage

### Via Web UI

Open http://localhost:8080 and ask questions in any supported language!

### Via REST API

```python
import requests

response = requests.post('http://localhost:8080/query', json={
    "question": "मधुमेह का इलाज क्या है?",  # Hindi: diabetes treatment
    "strategy": "A",
    "top_k": 5
})

result = response.json()
print(result['answer'])
print(result['citations'])
```

### Via Python

```python
import rag

result = rag.answer_question(
    "What are the effects of climate change?",
    strategy="A",
    top_k=8
)

print(f"Answer ({result['language_name']}): {result['answer']}")
print(f"Sources: {len(result['citations'])} citations")
```

---

## 📁 Project Structure

```
multilingual-rag/
├── api_server.py          # FastAPI app & HTTP endpoints
├── config.py              # Global configuration
├── embeddings.py          # E5 embedding model
├── ingest.py              # PDF ingestion pipeline
├── lang_utils.py          # Language detection
├── pdf_utils.py           # PDF extraction & chunking
├── rag.py                 # Core RAG logic
├── translation.py         # IndicTrans2 translation
├── vector_store.py        # ChromaDB wrapper
├── start_server.py        # Server launcher with checks
│
├── static/                # Web frontend
│   └── index.html         # Modern Ocean UI
│
├── docs/                  # Documentation
│   ├── QUICKSTART.md      # 5-minute setup
│   ├── DEPLOYMENT.md      # Production deployment
│   ├── ARCHITECTURE.md    # Technical deep dive
│   └── CONTRIBUTING.md    # Contribution guide
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
  "chunks_used": 5,
  "citations": [
    {"number": "1", "title": "Quantum Computing Basics", "section": "Introduction"}
  ],
  "processing_time": 1.23
}
```

### `POST /ingest`

Ingest a new PDF document.

**Request:**
```json
{
  "pdf_path": "my_paper.pdf"
}
```

### `GET /stats`

Get vector store statistics (document count, metadata, etc.)

### `GET /health`

Health check endpoint for monitoring.

---

## 🎯 Supported Languages

| Language | Code | Example Query |
|----------|------|---------------|
| English | en | What is machine learning? |
| Hindi | hi | मशीन लर्निंग क्या है? |
| Tamil | ta | இயந்திர கற்றல் என்றால் என்ன? |
| Telugu | te | మెషిన్ లెర్నింగ్ అంటే ఏమిటి? |
| Bengali | bn | মেশিন লার্নিং কি? |
| Marathi | mr | मशीन लर्निंग म्हणजे काय? |
| Gujarati | gu | મશીન લર્નિંગ શું છે? |
| Kannada | kn | ಯಂತ್ರ ಕಲಿಕೆ ಎಂದರೇನು? |
| Malayalam | ml | മെഷീൻ ലേണിംഗ് എന്താണ്? |
| Punjabi | pa | ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਕੀ ਹੈ? |
| Odia | or | ମେସିନ୍ ଲର୍ଣ୍ଣିଂ କ'ଣ? |

---

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Embedding model
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# RAG
DEFAULT_TOP_K = 8
MAX_CONTEXT_CHUNKS = 10

# LLM
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3
```

---

## 🐳 Docker Deployment (Optional)

```bash
# Build image
docker build -t indicrag .

# Run container
docker run -p 8080:8080 \
  -e LLM_API_KEY=your_key \
  -v $(pwd)/papers:/app/papers \
  -v $(pwd)/chroma_db:/app/chroma_db \
  indicrag
```

---

## 🧪 Testing

```bash
# Run end-to-end test
python test_pipeline.py

# Test specific query
python examples/example_query.py
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

**Areas for improvement:**
* Additional document formats (HTML, EPUB, Word)
* Advanced reranking with cross-encoders
* Query expansion techniques
* Better citation extraction
* Performance optimizations

---

## 📊 Performance

Typical query latency:
* **Strategy A** (direct multilingual): ~1-2s
* **Strategy B** (with translation): ~3-5s

ChromaDB retrieval: <100ms for 1000 documents

---

## 🤝 Acknowledgments

This project builds on excellent open-source work:

* [Google Gemini](https://ai.google.dev/) — Multilingual LLM
* [Sentence Transformers](https://www.sbert.net/) — E5 Embeddings
* [AI4Bharat](https://ai4bharat.org/) — IndicTrans2 Translation
* [ChromaDB](https://www.trychroma.com/) — Vector Database
* [FastAPI](https://fastapi.tiangolo.com/) — API Framework

---

## 🪪 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🆘 Support

### Common Issues

**"API key not configured"**
```bash
# Check .env file exists and has valid key
cat .env | grep LLM_API_KEY
```

**"No documents found"**
```bash
# Verify PDFs are ingested
python ingest.py --dir papers
```

**"Out of memory"**
```bash
# Reduce chunk size or use lighter embedding model
# Edit config.py: CHUNK_SIZE = 256
```

### Getting Help

* 📖 [Documentation](docs/)
* 💬 [GitHub Discussions](https://github.com/DNSdecoded/IndicRAG/discussions)
* 🐛 [Issue Tracker](https://github.com/DNSdecoded/IndicRAG/issues)

---

<div align="center">

**Built with ❤️ for multilingual scientific accessibility**

⭐ Star this repo if you find it useful!

[Report Bug](https://github.com/DNSdecoded/IndicRAG/issues) · [Request Feature](https://github.com/DNSdecoded/IndicRAG/issues) · [Documentation](docs/)

</div>
