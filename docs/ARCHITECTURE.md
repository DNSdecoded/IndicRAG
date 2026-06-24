# Architecture Deep Dive

## System Overview

This document provides a technical deep dive into the design decisions and architecture of the multilingual scientific Q&A RAG system.

## Table of Contents

1. [Core Design Principles](#core-design-principles)
2. [Component Architecture](#component-architecture)
3. [Cross-lingual Retrieval](#cross-lingual-retrieval)
4. [Strategy Comparison](#strategy-comparison)
5. [Performance Considerations](#performance-considerations)
6. [Future Improvements](#future-improvements)

---

## Core Design Principles

### 1. Modularity

Each component is isolated and can be swapped independently:
- **Embedding model**: Change `EMBEDDING_MODEL_NAME` in config
- **Vector store**: Replace ChromaDB with FAISS or Pinecone
- **LLM**: Implement `llm_generate()` for any LLM
- **Translation**: Use different translation models

### 2. Language Agnostic Pipeline

The pipeline doesn't hardcode language-specific logic. Instead:
- Language detection is automatic
- Embeddings handle all languages uniformly
- Prompts adapt based on detected language

### 3. Offline-First

Can run completely offline:
- Local embedding model (sentence-transformers)
- Local vector store (ChromaDB)
- Local LLM (Ollama)
- Optional: Local translation (IndicTrans2)

---

## Component Architecture

### 1. PDF Processing Pipeline

```
PDF File
  ↓
PyMuPDF Extraction → Raw text with formatting artifacts
  ↓
Text Cleaning → Remove headers, footers, page numbers
  ↓
Section Detection → Split into Introduction, Methods, etc.
  ↓
Chunking → Overlapping chunks (~1000 chars, Indic sentence terminators)
  ↓
Metadata Attachment → {paper_id, title, section, chunk_index}
```

#### Design Decisions

**Why PyMuPDF?**
- Fast: 10-100x faster than pdfminer
- Accurate: Preserves text layout
- Lightweight: No Java dependencies

**Why overlapping chunks?**
- Prevents splitting important sentences
- Improves retrieval recall
- Default overlap: 200 chars (~20% of chunk size)
- Overlap computed on whole-word boundaries to avoid slicing tokens mid-word

**Why section detection?**
- Enables filtering (e.g., "only search Methods sections")
- Improves citation quality
- Helps with relevance scoring

### 2. Embedding System

```
Text Input
  ↓
Tokenization → BERT tokenizer
  ↓
Model Forward Pass → BAAI/bge-m3
  ↓
Dense embedding → 1024-dim vector
  ↓
L2 Normalization → Unit vector for cosine similarity
```

#### Why BGE-M3?

**Alternatives considered:**
1. **multilingual-e5-base**: Good for retrieval (768 dim) but dense-only
2. **mBERT**: Older, worse performance on retrieval
3. **XLM-RoBERTa**: Good but not optimized for retrieval
4. **LaBSE**: Google's model, but larger and slower

**BGE-M3 wins because:**
- Supports dense + sparse + ColBERT retrieval in one model
- Strong on Indic scripts (Devanagari, Tamil, Telugu, etc.)
- 1024-dim embeddings for higher quality
- Supports 100+ languages
- Enables hybrid search (dense + BM25 fusion) without a separate sparse model

### 3. Vector Store

```
Document Ingestion:
  Embeddings + Metadata → ChromaDB
    ↓
  HNSW Index (Hierarchical Navigable Small World)
    ↓
  Persisted to disk (DuckDB + Parquet)

Query Time:
  Query Embedding → HNSW Search
    ↓
  Top-K Results (cosine similarity)
    ↓
  Return: {ids, documents, metadatas, distances}
```

#### Why ChromaDB?

**Alternatives:**
1. **FAISS**: Faster but requires manual metadata management
2. **Pinecone**: Cloud-only, costs money
3. **Weaviate**: Overkill for local use
4. **Qdrant**: Good alternative, but less Python-native

**ChromaDB advantages:**
- Pure Python, easy to install
- Built-in metadata filtering
- Persistent storage (DuckDB backend)
- Good performance for <1M documents
- Active development and community

#### HNSW Index

ChromaDB uses HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search:

- **Build time**: O(n log n) - slower ingestion
- **Query time**: O(log n) - fast retrieval
- **Accuracy**: 95%+ recall@10 for most datasets

Trade-off: Slower ingestion for faster queries (acceptable for RAG use case).

### 4. Language Detection

```
Input Text
  ↓
Unicode Script Detection (Devanagari, Tamil, Telugu, etc.)
  ↓ (if Devanagari)
Marathi Word-List Disambiguation (आहे, होते, केले, etc.)
  ↓ (if Latin or ambiguous)
langdetect (Google's language-detection library)
  ↓
ISO 639-1 Code (e.g., "hi", "ta", "mr", "en")
  ↓
Map to Native Name (e.g., "हिंदी", "தமிழ்", "मराठी")
```

#### Design

- **Script-first detection**: Unicode script ranges (`\p{Devanagari}`, `\p{Tamil}`, etc.) are checked before statistical detection — unambiguous for most Indic scripts and reliable even on very short text
- **Devanagari disambiguation**: Both Hindi and Marathi use Devanagari script. A word-list of Marathi-specific markers (`आहे`, `होते`, `केले`, `आणि`, etc.) distinguishes the two
- **Short text handling**: Texts <15 chars with no Indic script default to English
- **Fallback**: `langdetect` handles Latin-script and mixed text

#### Limitations

- **Code-mixing**: May misdetect Hinglish as English
- **Rare Devanagari languages**: Sanskrit, Nepali text may be classified as Hindi

---

## Cross-lingual Retrieval

### How It Works

```
User Query (Hindi): "मधुमेह का इलाज क्या है?"
  ↓
Embed with BGE-M3: [0.23, -0.15, 0.87, ...]  (1024-dim)
  ↓
Cosine Similarity with English chunks:
  - "Diabetes treatment includes..." → 0.78
  - "Machine learning for disease..." → 0.45
  - "Insulin therapy is effective..." → 0.82
  ↓
Return top-k chunks (sorted by similarity)
```

### Why This Works

**Multilingual embeddings create a shared semantic space:**

```
Semantic Space (simplified to 2D):

    "diabetes"
        ↓
    [English]
        ↓
    "मधुमेह" (Hindi)
        ↓
    "நீரிழிவு" (Tamil)
        ↓
    [Shared concept cluster]
```

BGE-M3 is trained with:
1. **Parallel corpora**: Aligned translations
2. **Contrastive learning**: Pull similar meanings together
3. **Hard negatives**: Push different meanings apart

Result: Queries in Hindi retrieve English documents about the same topic.

### Retrieval Quality

**Factors affecting quality:**

1. **Domain coverage**: E5 trained on general text, may miss domain-specific terms
2. **Language balance**: Better for Hindi/Bengali than Odia/Konkani
3. **Query length**: Longer queries → better retrieval
4. **Chunk size**: Too small → lacks context; too large → dilutes relevance

**Typical performance** (based on MIRACL benchmark, improved with BGE-M3 + hybrid search):
- Hindi → English: ~75-80% recall@10
- Tamil → English: ~70-75% recall@10
- English → English: ~85% recall@10

---

## Strategy Comparison

### Strategy A: Direct Multilingual LLM

```
User Query (Hindi)
  ↓
Retrieve English chunks
  ↓
Prompt: "Answer in Hindi using context"
  ↓
LLM generates Hindi answer directly
```

**Prompt structure:**
```
System: You are a scientific assistant...

Context: [English chunks]

Question: मधुमेह का इलाज क्या है?

Answer in हिंदी using the context.
```

**Pros:**
- Simple pipeline (fewer components)
- Faster (no translation overhead)
- Better context understanding (LLM sees original query)
- Natural language mixing (can use English terms in Hindi answer)

**Cons:**
- LLM quality varies by language
- May hallucinate in less-supported languages
- Inconsistent formatting across languages
- Requires multilingual LLM (limits model choice)

**Best LLMs for Strategy A:**
- Gemma-2-9B-IT (good Hindi/Bengali support)
- Aya-23-8B (optimized for Indic languages)
- GPT-4 (best quality but expensive)
- Mistral-7B-Instruct (decent multilingual)

### Strategy B: English Reasoning + Translation

```
User Query (Hindi)
  ↓
Translate to English (IndicTrans2)
  ↓
Retrieve English chunks
  ↓
Prompt: "Answer in English using context"
  ↓
LLM generates English answer
  ↓
Translate to Hindi (IndicTrans2)
```

**Pros:**
- Consistent reasoning quality (English LLMs are best)
- High-quality translation (IndicTrans2 is SOTA for Indic)
- Can use English-only LLMs (more options)
- Better for complex scientific content

**Cons:**
- Slower (2 translation steps)
- Larger models (~5GB for translation)
- Potential translation errors
- May lose nuance in translation

**Best LLMs for Strategy B:**
- GPT-4 (best reasoning)
- Claude-3 (good scientific understanding)
- Llama-3-70B (strong open-source option)
- Mixtral-8x7B (good quality/speed trade-off)

### When to Use Each

| Use Case | Strategy A | Strategy B |
|----------|-----------|-----------|
| Simple factual queries | ✅ | ✅ |
| Complex reasoning | ❌ | ✅ |
| Low latency required | ✅ | ❌ |
| Hindi/Bengali queries | ✅ | ✅ |
| Less common languages (Odia, Konkani) | ❌ | ✅ |
| Limited compute | ✅ | ❌ |
| Highest accuracy | ❌ | ✅ |

---

## Performance Considerations

### Ingestion Performance

**Bottlenecks:**
1. **PDF extraction**: ~1-5 seconds per paper (CPU-bound)
2. **Embedding**: ~0.1 seconds per chunk (GPU-accelerated)
3. **Vector store insertion**: ~0.01 seconds per chunk

**Optimization strategies:**
```python
# 1. Batch embedding
embeddings = embed_passages(all_chunks, batch_size=64)  # vs batch_size=1

# 2. Parallel PDF processing
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(process_pdf, pdf_files)

# 3. GPU acceleration
# Ensure PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Expected throughput:**
- CPU-only: ~10-20 papers/minute
- GPU (RTX 3060): ~50-100 papers/minute

### Query Performance

**Latency breakdown** (typical):
1. Language detection: ~10ms
2. Query embedding: ~50ms (CPU) / ~10ms (GPU)
3. Vector search: ~20ms (for 10k chunks)
4. LLM generation: ~2-10 seconds (depends on model)
5. Translation (Strategy B): ~500ms per direction

**Total latency:**
- Strategy A: ~2-10 seconds
- Strategy B: ~3-11 seconds

**Optimization:**
```python
# 1. Reduce top_k
retrieve_context(query, top_k=5)  # vs top_k=10

# 2. Use faster LLM
# Gemma-2-9B: ~2s per response
# GPT-4: ~5-10s per response

# 3. Cache embeddings
# Embedding model is cached after first load

# 4. Parallel translation (Strategy B)
# Translate query while retrieving context
```

### Memory Usage

**Components:**
1. Embedding model: ~2GB RAM
2. Translation models: ~5GB RAM (Strategy B)
3. LLM: Varies (Gemma-2-9B: ~18GB, GPT-4: API call)
4. ChromaDB: ~100MB per 10k chunks

**Total RAM required:**
- Strategy A (local): ~20-25GB (with local LLM)
- Strategy A (API): ~2-3GB
- Strategy B (local): ~25-30GB
- Strategy B (API): ~7-8GB

**Optimization:**
```python
# 1. Use smaller embedding model
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # 384 dim vs 768

# 2. Quantize LLM (if local)
# Use GGUF quantized models with Ollama

# 3. Unload translation models when not in use
_en_to_indic_model = None  # Free memory
```

---

## Implemented Improvements (v1.5–v2.0)

### Reranking ✅
Cross-encoder reranking with `BAAI/bge-reranker-v2-m3`. Retrieves 30 candidates via dense+BM25, reranks to top 12. See `rerank.py`.

### Hybrid Search ✅
BGE-M3 dense vectors fused with BM25 lexical search via Reciprocal Rank Fusion (RRF). BM25 tokenizer uses the `regex` module for correct Indic script handling (vowel matras, conjuncts). See `bm25_search.py`.

### Query Expansion ✅ (Agent only)
Agent tool `indicrag_retrieval` supports `expand_query=True`: generates 3 LLM variants, retrieves for each, deduplicates by SHA-256 hash. Gated behind a minimum word-count heuristic to avoid expanding trivial queries.

### Faithfulness Verification ✅
NLI-based claim-level grounding check via `verify.check_claims()`. Batched cross-encoder scoring. Reflexion evaluator forces regeneration when no citations are found (default score = 0.0, not 0.5).

### Citation Extraction ✅
Robust parser handles `[1]`, `[1, 2]`, `[1-3]` with a range-width cap (>50 skipped to avoid matching year ranges). See `rag.extract_citations()`.

### Model Failover with Circuit Breaker ✅
`generate_with_failover()` tries all API keys on the primary model, then falls back to `LLM_FALLBACK_MODEL` (default: `gemma-4-26b-a4b-it`). After all keys fail, a circuit breaker skips the primary for 60s so subsequent calls go directly to the fallback — avoids ~15s of wasted retries per LLM call during outages.

### Parallel Tool Execution ✅
When the tool selector picks multiple tools (e.g., `indicrag_retrieval` + `arxiv_search`), `tool_executor_node` runs them concurrently via `ThreadPoolExecutor`. Saves 3-10s per query vs sequential execution.

### Reflexion Stuck-Loop Detection ✅
If completeness score doesn't improve by >0.05 between iterations, the evaluator auto-accepts instead of looping to timeout. Prevents wasting time on broad queries where retrieval can't improve.

### AST-Based Python Sandbox ✅
Replaced string denylist with `ast.parse()` + tree walk: whitelisted safe imports only, blocks dunder attribute access and dangerous builtins (`eval`, `exec`, `getattr`, `open`, etc.). Subprocess env remains scrubbed (no credentials).

## Future Improvements

### 1. Streaming Responses

Stream LLM output via SSE for better perceived latency — the user sees partial answers as tokens arrive. Gemini Flash supports streaming natively.

### 2. Multi-modal Support

Add support for figures and tables using a vision-language model to describe extracted images, then index the descriptions alongside text.

### 3. Domain-Specific Embeddings

Fine-tune BGE-M3 on scientific QA datasets (SciQ, PubMedQA) for improved domain-specific retrieval.

### 4. Token-Aware Chunking

Replace character-based `CHUNK_SIZE=1000` with token-based sizing (~350 tokens) for consistent chunk sizes across scripts.

### 5. Redis-Backed Sessions

Move `_sessions` and `_jobs` from process-local dicts to Redis for multi-worker / multi-replica correctness.

### 6. Comprehensive Eval Suite

Expand from 4 queries / single-paper to 50–100+ queries across 10+ papers with graded relevance judgments and CI-gated nDCG@10 / Recall@20.

---

## Conclusion

This architecture balances:
- **Simplicity**: Easy to understand and modify
- **Performance**: Fast enough for interactive use
- **Quality**: SOTA models for each component
- **Flexibility**: Modular design allows component swapping

The system is production-ready for:
- Educational tools (students asking science questions in native language)
- Research assistants (multilingual literature review)
- Public health (medical information in local languages)

Already implemented for production:
1. API key authentication + admin key for destructive ops
2. Three-layer TTL caching (LLM, retrieval, tool results)
3. Prometheus metrics instrumentation
4. Env-driven CORS, session TTL eviction

Remaining considerations for scale:
1. Redis-backed sessions for multi-worker deployments
2. Managed vector DB (Pinecone, Weaviate) for >1M documents
3. SSE streaming for perceived latency reduction
4. Feedback mechanism (thumbs up/down) for quality monitoring

---

**Questions or suggestions? Open an issue or contribute!**
