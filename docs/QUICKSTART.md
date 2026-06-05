# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies (2 min)

```bash
cd d:/RAG
pip install -r requirements.txt
```

### 2. Configure Google Gemini API (2 min)

**Get your API key:**
1. Go to https://makersuite.google.com/app/apikey
2. Sign in and click "Create API Key"
3. Copy your API key (starts with `AIza...`)

**Set up the key:**

```bash
# Copy the example env file
copy .env.example .env
```

Edit `.env` and add your API key:
```
LLM_API_KEY=AIzaSyYourActualAPIKeyHere
LLM_MODEL_NAME=gemini-3-flash-preview
```

**Or use environment variable:**
```bash
set LLM_API_KEY=AIzaSyYourActualAPIKeyHere
```

**Install Gemini package:**
```bash
pip install google-genai
```

✅ **Done!** The system is already configured to use Gemini in `rag.py`

### 3. Add Papers (30 sec)

Download 2-3 scientific PDFs and place in `d:/RAG/papers/`

**Quick sources:**
- https://arxiv.org/ (search "diabetes treatment")
- https://www.ncbi.nlm.nih.gov/pmc/ (search "diabetes")

### 4. Ingest Papers (1 min)

```bash
python example_ingest.py
```

Choose option 1 or 2.

### 5. Ask Questions! (30 sec)

```bash
python example_query.py
```

Try:
- Hindi: `मधुमेह का इलाज क्या है?`
- Tamil: `நீரிழிவு நோய்க்கான சிகிச்சை என்ன?`
- English: `What is the treatment for diabetes?`

---

## 📝 Common Commands

```bash
# Ingest PDFs
python example_ingest.py

# Run queries
python example_query.py

# Run tests
python test_pipeline.py

# Ingest single PDF
python ingest.py path/to/paper.pdf

# Ingest directory
python ingest.py path/to/directory
```

---

## 🔧 Troubleshooting

### "No module named 'fitz'"
```bash
pip install pymupdf
```

### "No PDFs found"
Add PDFs to `d:/RAG/papers/` directory

### "API key not configured"
Make sure you've set `LLM_API_KEY` in `.env` file or environment variable

### "API key not valid"
- Check your API key is correct (starts with `AIza`)
- Get a new key from https://makersuite.google.com/app/apikey

### "Quota exceeded"
You've hit Gemini's free tier limit. Wait or upgrade to paid tier.

### "CUDA out of memory"
Use CPU instead - models will auto-detect

---

## 📚 Full Documentation

- **Setup & Usage**: [README.md](file:///d:/RAG/README.md)
- **Architecture**: [ARCHITECTURE.md](file:///d:/RAG/ARCHITECTURE.md)
- **Walkthrough**: [walkthrough.md](file:///C:/Users/sanjay/.gemini/antigravity/brain/ee5eb13d-3474-43bc-abe1-b7e49c6ea3ce/walkthrough.md)

---

**That's it! You're ready to ask scientific questions in Indian languages! 🎉**
