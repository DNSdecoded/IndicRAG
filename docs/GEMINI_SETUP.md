# LLM Provider Setup Guide

IndicRAG dispatches generation through `llm_client.py`, which supports two
backends:

- **Gemini** (`providers/gemini.py`) — any bare model name, e.g. `gemini-3.6-flash`
- **OpenRouter** (`providers/openrouter.py`) — any `vendor/model` slug, e.g. `anthropic/claude-haiku`

Everything else in the pipeline (embeddings, vector store, reranking, NLI
verification) runs locally. Only generation is remote.

## 🚀 Quick Setup (5 minutes)

### Step 1: Get Your Gemini API Key (2 min)

1. Go to **[Google AI Studio](https://aistudio.google.com/app/apikey)**
2. Sign in with your Google account
3. Click **"Get API Key"** or **"Create API Key"**
4. Copy your API key (starts with `AIza...`)

### Step 2: Configure the API Key (1 min)

**Option A: Using .env file (Recommended)**

1. Copy the example file:
   ```bash
   cp .env.example .env      # copy .env.example .env  on Windows
   ```

2. Edit `.env` and add your API key:
   ```
   LLM_API_KEYS=AIzaSyYourActualAPIKeyHere
   LLM_MODEL_NAME=gemini-3.6-flash
   ```

   `LLM_API_KEYS` takes a comma-separated list and load-balances across the
   keys, which is the simplest way to raise the free-tier ceiling. The older
   singular `LLM_API_KEY` is still read as a fallback.

   For OpenRouter models, also set:
   ```
   OPENROUTER_API_KEY=sk-or-...
   ```

**Option B: Using environment variable**

Windows:
```bash
set LLM_API_KEYS=AIzaSyYourActualAPIKeyHere
```

Linux/Mac:
```bash
export LLM_API_KEYS=AIzaSyYourActualAPIKeyHere
```

### Step 3: Install Gemini Package (1 min)

```bash
pip install google-genai
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Step 4: Test the Setup (1 min)

```bash
python -c "import rag, config; print(rag.llm_generate('Say hello', model=config.LLM_MODEL_NAME))"
```

If this prints a response, you're all set! ✅

---

## 📊 Choosing a Model

Model choice is not hard-coded. Two settings control it:

| Setting | Meaning |
|---|---|
| `LLM_MODEL_NAME` | Default model used when a request doesn't name one |
| `LLM_SELECTABLE_MODELS` | Comma-separated dropdown exposed to the UI (`GET /models`); **first entry is the default** |

Routing rule: a **bare name** (`gemini-3.6-flash`) goes to Gemini; anything
containing a **`/`** (`anthropic/claude-haiku`) goes to OpenRouter.

**Keep at least one `/` slug in `LLM_SELECTABLE_MODELS`.** Cross-vendor
failover picks the first slug in that list, and OpenRouter silently rewrites a
bare Gemini name to `google/<model>` — so a list with no slug would fail back
onto the same vendor that just failed.

Rough guidance rather than a fixed list, since model names change often:

- **Flash-class** models: fastest and cheapest, good default for high volume.
- **Pro-class** models: better reasoning on multi-paper synthesis, several times
  the cost and latency.
- **Non-Gemini via OpenRouter**: useful as a genuinely independent failover leg.

### Thinking level (and the legacy budget)

Gemini 3.x models take a **thinking level**: `minimal`, `low`, `medium`, or
`high`. Set it with `LLM_THINKING_LEVEL` (standard RAG) and
`AGENT_THINKING_LEVEL` (agentic pipeline); both default to `minimal`.

```bash
LLM_THINKING_LEVEL=minimal    # minimal | low | medium | high, or empty for the model default
AGENT_THINKING_LEVEL=minimal
```

Leaving these empty is a real choice, not a neutral one: with no level sent,
`gemini-3.6-flash` thinks at `medium`. Those thought tokens are billed and are
drawn from `LLM_MAX_TOKENS`, so the answer gets less room.

The older `AGENT_THINKING_BUDGET` (`0` off, `-1` model decides, `N` caps) still
works on models that accept budgets. Gemini 3.x rejects it with a 400 —
`providers/gemini.py` detects that per model and retries once with the budget
translated to the closest level (`0` → `MINIMAL`, `<=1024` → `LOW`, higher →
`MEDIUM`, `-1` → omit the field so the model decides). Streaming only retries if
nothing was emitted yet.

---

## 💰 Pricing

Prices change too often to pin here. Check:

- Gemini — https://ai.google.dev/pricing (also lists free-tier RPM/TPD limits)
- OpenRouter — https://openrouter.ai/models

Per-query cost scales with retrieved context, so `top_k` and
`AGENT_MAX_CONTEXT_CHUNKS` are the main levers. The agentic pipeline costs
several times a classic query because it makes multiple LLM turns.

---

## 🔧 Configuration Options

Set these in `.env` (defaults live in `config.py`):

```bash
LLM_MODEL_NAME=gemini-3.6-flash
LLM_SELECTABLE_MODELS=gemini-3.6-flash,gemini-3.5-flash,anthropic/claude-haiku

LLM_MAX_TOKENS=8192    # Maximum response length (thinking + answer share this)
LLM_TEMPERATURE=0.3    # Lower = more factual, higher = more creative
```

---

## 🐛 Troubleshooting

### "API key not configured"
- Make sure you've set `LLM_API_KEYS` (or `LLM_API_KEY`) in `.env` or the environment
- Check that `.env` sits in the repo root, next to `start_server.py`
- Verify the API key starts with `AIza`

### "API key not valid"
- Double-check you copied the full API key
- Make sure there are no extra spaces
- Try generating a new API key

### "Quota exceeded"
- You've hit the free-tier limit
- Wait for the quota to reset (per minute / daily)
- Add more keys to `LLM_API_KEYS` — requests are load-balanced across them
- Or upgrade to a paid tier

### Requests keep landing on the wrong vendor
`llm_client` tries, in order: the requested `(provider, model)`, a
same-provider fallback, a cross-provider fallback, then a guaranteed Gemini
backstop. A `(provider, model)` pair that keeps failing is tripped by a circuit
breaker and skipped until it resets. Check the logs for the attempt chain.

### "Prompt was blocked"
- Gemini has safety filters that may block certain content
- Try rephrasing your question
- Check the prompt feedback for details

### "Module 'google.genai' not found"
```bash
pip install google-genai
```

---

## ✅ Verification

Test your setup:

```bash
# Test 1: Check a key is loaded
python -c "import config; print('Keys configured:', len(config.LLM_API_KEY_POOL))"

# Test 2: Test generation through the real dispatch path
python -c "import rag, config; print(rag.llm_generate('Say hello', model=config.LLM_MODEL_NAME))"

# Test 3: Run the provider/dispatch tests
pytest tests/test_llm_client_dispatch.py tests/test_providers_gemini.py -q
```

---

## 🎯 Next Steps

Once your provider is configured:

1. **Add PDFs** to `papers/`, or upload from the web UI's Library view
2. **Ingest documents**: `python ingest.py papers/`
3. **Run queries**: open http://localhost:8080, or `python examples/example_query.py`

---

## 📚 Additional Resources

- **Gemini API Docs**: https://ai.google.dev/docs
- **Get API Key**: https://aistudio.google.com/app/apikey
- **Pricing**: https://ai.google.dev/pricing
- **Safety Settings**: https://ai.google.dev/docs/safety_setting_gemini

---

## 🔐 Security Best Practices

1. **Never commit `.env` file** to git
   - Add `.env` to `.gitignore`

2. **Use environment variables** in production
   - Don't hardcode API keys in code

3. **Rotate keys regularly**
   - Generate new keys periodically

4. **Monitor usage**
   - Check Google Cloud Console for API usage

---

**You're all set to use Google Gemini with your multilingual RAG system! 🎉**
