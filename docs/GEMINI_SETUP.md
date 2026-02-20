# Google Gemini API Setup Guide

## 🚀 Quick Setup (5 minutes)

### Step 1: Get Your Gemini API Key (2 min)

1. Go to **[Google AI Studio](https://makersuite.google.com/app/apikey)**
2. Sign in with your Google account
3. Click **"Get API Key"** or **"Create API Key"**
4. Copy your API key (starts with `AIza...`)

### Step 2: Configure the API Key (1 min)

**Option A: Using .env file (Recommended)**

1. Copy the example file:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```
   LLM_API_KEY=AIzaSyYourActualAPIKeyHere
   LLM_MODEL_NAME=gemini-3-flash-preview
   ```

**Option B: Using environment variable**

Windows:
```bash
set LLM_API_KEY=AIzaSyYourActualAPIKeyHere
```

Linux/Mac:
```bash
export LLM_API_KEY=AIzaSyYourActualAPIKeyHere
```

### Step 3: Install Gemini Package (1 min)

```bash
pip install google-generativeai
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Step 4: Test the Setup (1 min)

```bash
python -c "import google.generativeai as genai; import os; genai.configure(api_key=os.getenv('LLM_API_KEY')); model = genai.GenerativeModel('gemini-3-flash-preview'); print(model.generate_content('Hello!').text)"
```

If this prints a response, you're all set! ✅

---

## 📊 Gemini Model Options

### gemini-3-flash-preview (Recommended)
- **Speed**: ⚡ Very fast
- **Cost**: 💰 Most affordable
- **Quality**: ✅ Good for most use cases
- **Best for**: Production use, high volume queries

### gemini-1.5-pro
- **Speed**: 🐌 Slower
- **Cost**: 💰💰 More expensive
- **Quality**: ⭐ Higher quality, better reasoning
- **Best for**: Complex scientific questions, highest accuracy

### gemini-pro (Legacy)
- **Speed**: ⚡ Fast
- **Cost**: 💰 Affordable
- **Quality**: ✅ Good
- **Note**: Older model, use 1.5-flash instead

---

## 💰 Pricing (as of Nov 2024)

### gemini-3-flash-preview
- **Input**: $0.075 per 1M tokens
- **Output**: $0.30 per 1M tokens
- **Free tier**: 15 requests/minute, 1M tokens/day

### gemini-1.5-pro
- **Input**: $1.25 per 1M tokens
- **Output**: $5.00 per 1M tokens
- **Free tier**: 2 requests/minute, 50 requests/day

**Typical query cost** (with 5 chunks):
- gemini-3-flash-preview: ~$0.0001 per query
- gemini-1.5-pro: ~$0.002 per query

---

## 🔧 Configuration Options

Edit `config.py` or set environment variables:

```python
# Model selection
LLM_MODEL_NAME = "gemini-3-flash-preview"  # or "gemini-1.5-pro"

# Generation parameters
LLM_MAX_TOKENS = 2048  # Maximum response length
LLM_TEMPERATURE = 0.3  # Lower = more factual, Higher = more creative
```

---

## 🐛 Troubleshooting

### "API key not configured"
- Make sure you've set `LLM_API_KEY` in `.env` or environment variable
- Check that `.env` file is in the project root (`d:/RAG/`)
- Verify the API key starts with `AIza`

### "API key not valid"
- Double-check you copied the full API key
- Make sure there are no extra spaces
- Try generating a new API key

### "Quota exceeded"
- You've hit the free tier limit
- Wait for the quota to reset (daily/per minute)
- Or upgrade to paid tier in Google Cloud Console

### "Prompt was blocked"
- Gemini has safety filters that may block certain content
- Try rephrasing your question
- Check the prompt feedback for details

### "Module 'google.generativeai' not found"
```bash
pip install google-generativeai
```

---

## ✅ Verification

Test your setup:

```bash
# Test 1: Check API key is set
python -c "import config; print('API Key configured:', bool(config.LLM_API_KEY))"

# Test 2: Test Gemini connection
python -c "import google.generativeai as genai; import config; genai.configure(api_key=config.LLM_API_KEY); model = genai.GenerativeModel(config.LLM_MODEL_NAME); print(model.generate_content('Say hello').text)"

# Test 3: Test RAG pipeline (requires ingested documents)
python test_pipeline.py
```

---

## 🎯 Next Steps

Once Gemini is configured:

1. **Add PDFs** to `papers/` directory
2. **Ingest documents**: `python example_ingest.py`
3. **Run queries**: `python example_query.py`

---

## 📚 Additional Resources

- **Gemini API Docs**: https://ai.google.dev/docs
- **Get API Key**: https://makersuite.google.com/app/apikey
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
