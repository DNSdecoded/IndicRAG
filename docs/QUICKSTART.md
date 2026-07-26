# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies (2 min)

```bash
git clone https://github.com/DNSdecoded/IndicRAG.git
cd IndicRAG
pip install -r requirements.txt
```

### 2. Configure Your LLM API Key (2 min)

**Get a Gemini API key:**
1. Go to https://aistudio.google.com/app/apikey
2. Sign in and click "Create API Key"
3. Copy your API key (starts with `AIza...`)

**Set up the key:**

```bash
# Copy the example env file
copy .env.example .env      # Windows
cp .env.example .env        # Linux/Mac
```

Edit `.env`:

```
LLM_API_KEYS=AIzaSyYourActualAPIKeyHere
LLM_MODEL_NAME=gemini-3.6-flash

# Required before the server will start in production mode
API_KEYS=pick-a-long-random-string
ADMIN_API_KEY=pick-another-long-random-string
```

`LLM_API_KEYS` accepts a comma-separated list — extra keys are load-balanced.
`LLM_API_KEY` (singular) is still read as a fallback.

For OpenRouter models, also set `OPENROUTER_API_KEY`. See
[GEMINI_SETUP.md](GEMINI_SETUP.md) for provider details.

### 3. Start the Server (30 sec)

```bash
python start_server.py
```

Pre-flight checks run first (Python version, `.env`, API key, dependencies,
endpoint auth, indexed documents). On success:

- 🌐 Web UI: http://localhost:8080
- 📖 API docs: http://localhost:8080/api/docs
- ❤️ Health: http://localhost:8080/health

Development mode with auto-reload: `python start_server.py --dev`
(also downgrades the missing-`API_KEYS` check from a hard failure to a warning).

### 4. Add Papers

**Easiest — the web UI:** open http://localhost:8080, go to the **Library**
view, and upload PDFs there (or paste a PDF URL to ingest by link).

**Or from the command line:**

```bash
# Single PDF
python ingest.py path/to/paper.pdf

# Whole directory
python ingest.py papers/

# Guided example script
python examples/example_ingest.py
```

**Quick paper sources:**
- https://arxiv.org/
- https://www.ncbi.nlm.nih.gov/pmc/

### 5. Ask Questions!

Use the web UI, or:

```bash
python examples/example_query.py
```

Try:
- Hindi: `मधुमेह का इलाज क्या है?`
- Tamil: `நீரிழிவு நோய்க்கான சிகிச்சை என்ன?`
- English: `What is the treatment for diabetes?`

---

## 📝 Common Commands

```bash
# Start server (production / dev / custom port)
python start_server.py
python start_server.py --dev
python start_server.py --port 9000

# Ingest a PDF or a directory
python ingest.py path/to/paper.pdf
python ingest.py path/to/directory

# Example scripts
python examples/example_ingest.py
python examples/example_query.py

# Run the test suite (skips network/integration tests)
pytest tests/ -m "not integration and not network"

# Inspect the vector store
python check_db.py

# Cleanup (destructive)
python purge.py --all --yes
```

---

## 🔧 Troubleshooting

### "No module named 'fitz'"
```bash
pip install pymupdf
```

### "✗ API_KEYS not set — server would bind 0.0.0.0 with NO endpoint auth."
Production mode refuses to start without endpoint auth. Either set `API_KEYS`
in `.env`, or opt in to anonymous access on a private host with
`ALLOW_UNAUTHENTICATED=1`.

### "No PDFs found"
Add PDFs to the `papers/` directory, or upload them from the Library view.

### "API key not configured"
Set `LLM_API_KEYS` (or `LLM_API_KEY`) in `.env` or as an environment variable.

### "API key not valid"
- Check the key was copied in full (Gemini keys start with `AIza`)
- Get a new key from https://aistudio.google.com/app/apikey

### "Quota exceeded"
You've hit the free-tier limit. Wait for the reset, add more keys to
`LLM_API_KEYS`, or upgrade to a paid tier.

### "CUDA out of memory"
Models auto-detect and fall back to CPU. To cut memory further, disable
`USE_RERANKER` / `USE_COLBERT_RERANK` in `.env`.

---

## 📚 Full Documentation

- **Setup & Usage**: [../README.md](../README.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md) · [PRODUCTION.md](PRODUCTION.md)
- **LLM providers**: [GEMINI_SETUP.md](GEMINI_SETUP.md)

---

**That's it! You're ready to ask scientific questions in Indian languages! 🎉**
