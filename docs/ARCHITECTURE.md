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
Chunking → Overlapping chunks (~1000 chars)
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
- Default overlap: 200 chars (20% of chunk size)

**Why section detection?**
- Enables filtering (e.g., "only search Methods sections")
- Improves citation quality
- Helps with relevance scoring

### 2. Embedding System

```
Text Input
  ↓
Add E5 Prefix → "query: " or "passage: "
  ↓
Tokenization → BERT tokenizer
  ↓
Model Forward Pass → multilingual-e5-base
  ↓
Mean Pooling → 768-dim vector
  ↓
L2 Normalization → Unit vector for cosine similarity
```

#### Why multilingual-e5-base?

**Alternatives considered:**
1. **mBERT**: Older, worse performance on retrieval
2. **XLM-RoBERTa**: Good but not optimized for retrieval
3. **LaBSE**: Google's model, but larger and slower
4. **multilingual-e5-small**: Smaller (384 dim) but lower quality
5. **multilingual-e5-large**: Better quality but 3x slower

**multilingual-e5-base wins because:**
- SOTA on multilingual retrieval benchmarks (MIRACL, Mr. TyDi)
- Balanced size/performance (768 dim, ~1GB)
- Explicit query/passage prefixes improve retrieval
- Supports 100+ languages including all major Indic languages

#### E5 Prefix Mechanism

E5 models use asymmetric encoding:
```python
query_embedding = encode("query: " + user_question)
passage_embedding = encode("passage: " + document_chunk)
```

This improves retrieval by:
- Distinguishing query intent from document content
- Optimizing for semantic matching (not just lexical)
- Reducing false positives from keyword overlap

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
langdetect (Google's language-detection library)
  ↓
ISO 639-1 Code (e.g., "hi", "ta", "en")
  ↓
Map to Native Name (e.g., "हिंदी", "தமிழ்")
```

#### Limitations

- **Short texts**: Unreliable for <20 characters
- **Code-mixing**: May misdetect Hinglish as English
- **Similar scripts**: May confuse Hindi/Marathi (both Devanagari)

**Mitigation:**
- Encourage longer queries
- Allow manual language override
- Use language priors (e.g., assume Hindi for Devanagari script)

---

## Cross-lingual Retrieval

### How It Works

```
User Query (Hindi): "मधुमेह का इलाज क्या है?"
  ↓
Embed with E5: [0.23, -0.15, 0.87, ...]  (768-dim)
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

E5 is trained with:
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

**Typical performance** (based on MIRACL benchmark):
- Hindi → English: ~70% recall@10
- Tamil → English: ~65% recall@10
- English → English: ~80% recall@10

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

## Future Improvements

### 1. Reranking

Add a cross-encoder reranker after initial retrieval:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Rerank top-k results
scores = reranker.predict([(query, chunk) for chunk in chunks])
reranked_chunks = [chunks[i] for i in np.argsort(scores)[::-1]]
```

**Expected improvement**: +10-15% recall@5

### 2. Query Expansion

Expand queries with synonyms/related terms:

```python
# Use LLM to generate related queries
expanded_queries = llm_generate(f"Generate 3 related questions to: {query}")

# Retrieve for each and merge results
all_results = [retrieve_context(q) for q in expanded_queries]
merged_results = deduplicate_and_rank(all_results)
```

**Expected improvement**: +5-10% recall, especially for short queries

### 3. Citation Extraction

Extract exact citations from PDFs:

```python
# Parse references section
references = extract_references(pdf_text)

# Match chunks to references
for chunk in chunks:
    chunk_metadata['references'] = find_matching_references(chunk, references)
```

**Benefit**: More accurate citations in answers

### 4. Domain-Specific Embeddings

Fine-tune E5 on scientific corpora:

```python
# Fine-tune on (query, relevant_passage) pairs from scientific QA datasets
# E.g., SciQ, PubMedQA, COVID-QA
```

**Expected improvement**: +15-20% recall for scientific queries

### 5. Hybrid Search

Combine dense (embedding) and sparse (BM25) retrieval:

```python
# BM25 for keyword matching
bm25_results = bm25_search(query, top_k=20)

# Dense for semantic matching
dense_results = vector_search(query, top_k=20)

# Combine with reciprocal rank fusion
final_results = reciprocal_rank_fusion(bm25_results, dense_results)
```

**Expected improvement**: +10-15% recall, especially for entity queries

### 6. Streaming Responses

Stream LLM output for better UX:

```python
def llm_generate_stream(prompt):
    # Use streaming API
    for chunk in llm_stream(prompt):
        yield chunk

# In UI, display tokens as they arrive
```

**Benefit**: Perceived latency reduction (user sees partial answer immediately)

### 7. Multi-modal Support

Add support for figures and tables:

```python
# Extract images from PDFs
figures = extract_figures(pdf_path)

# Use vision-language model (e.g., LLaVA) to describe
descriptions = vision_model.describe(figures)

# Index descriptions alongside text
```

**Benefit**: Answer questions about charts, diagrams, molecular structures

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

For production deployment, consider:
1. Add authentication and rate limiting
2. Implement caching (query → answer)
3. Monitor and log queries for quality improvement
4. Add feedback mechanism (thumbs up/down)
5. Deploy with Docker for reproducibility
6. Use managed vector DB (Pinecone, Weaviate) for scale

---

**Questions or suggestions? Open an issue or contribute!**
