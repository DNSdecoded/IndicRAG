# IndicRAG v2.0.1 — Code Review for v2.1

**Date:** 2026-06-29
**Scope:** Full codebase audit across features, performance, architecture, and quality.

---

## 1. FEATURE GAPS

### F-01: Streaming Responses
- **What:** All endpoints return full responses after 10-30s with zero intermediate feedback.
- **Where:** `api_server.py` (routes), `rag.py` (LLM calls), `agent/nodes/answer_generator.py`
- **Fix:** Add `/query/stream` and `/chat/stream` using SSE. Gemini's `generate_content_stream()` already supports this. Thread a callback through `llm_generate()`.
- **Effort:** Medium. **Impact:** High.

### F-02: Single-Paper Delete
- **What:** `DELETE /purge/papers` destroys everything. No endpoint deletes one paper. `vector_store.delete_by_paper_id()` exists at line 225 but has no API surface and always returns `0` instead of actual count.
- **Where:** `api_server.py`, `vector_store.py:225-232`
- **Fix:** Expose `DELETE /papers/{paper_id}`. Fix the return value in `delete_by_paper_id` to count removed chunks before deletion.
- **Effort:** Low. **Impact:** High.

### F-03: Rate Limiting
- **What:** No throttling on any endpoint. Expensive LLM+embedding calls can be spammed, exhausting API keys or causing OOM on CPU.
- **Where:** `api_server.py`
- **Fix:** Add `slowapi` middleware. Set per-IP limits: `/query` 30/min, `/agent/query` 10/min, `/ingest` 5/min.
- **Effort:** Low. **Impact:** High.

### F-04: Agent Progress via WebSocket
- **What:** `/agent/query` blocks for up to 120s with no progress indication. Users cannot see which tools are running or reflexion status.
- **Where:** `api_server.py:1102-1218`, `agent/graph.py`
- **Fix:** Add `WS /agent/stream` emitting events per node transition. Use LangGraph's `astream_events` or a callback emitter on `AgentState` updates.
- **Effort:** Medium. **Impact:** Medium.

### F-05: Post-Ingest Metadata Editing
- **What:** Once a paper is ingested, its title/authors/year/tags are frozen. No PATCH endpoint exists.
- **Where:** `api_server.py`, `vector_store.py`
- **Fix:** Add `PATCH /papers/{paper_id}` that calls `collection.update()` on matching metadata.
- **Effort:** Low. **Impact:** Medium.

### F-06: Collection Export/Import
- **What:** No way to snapshot or transfer ingested data between instances.
- **Where:** `api_server.py`, `vector_store.py`
- **Fix:** Expose ChromaDB's `export`/`import` functionality via `/collections/export` and `/collections/import`.
- **Effort:** Low. **Impact:** Low.

---

## 2. PERFORMANCE BOTTLENECKS

### P-01: BM25 Full Rebuild on Every Ingest
- **What:** After each `/ingest` call, `bm25_search.invalidate()` wipes the entire index and a daemon thread rebuilds it from scratch. For large corpora this means an O(n) scan every upload. During rebuild, all hybrid queries silently degrade to dense-only.
- **Where:** `api_server.py:578-583`, `bm25_search.py`
- **Fix:** Implement `BM25Index.add_documents增量` — append new token frequencies and document counts instead of rebuilding. The `ingest_paper()` function already knows the new chunks; pass them into the index incrementally.
- **Effort:** Medium. **Impact:** High.

### P-02: Double MD5 During Bulk Ingest
- **What:** `_extract_worker()` at `ingest.py:190` computes MD5 in the subprocess, then `ingest_pdf()` at line 167 recomputes it. This doubles file I/O for large PDFs.
- **Where:** `ingest.py:167,190`
- **Fix:** Return the precomputed hash from the worker and inject it as metadata, bypassing the second call.
- **Effort:** Low. **Impact:** Low.

### P-03: Embed Query Duplicate Under Concurrency
- **What:** `embed_query()` in `embeddings.py:103-116` has no in-flight sentinel. Two concurrent requests for the same uncached query both execute the full embedding computation.
- **Where:** `embeddings.py:103-116`
- **Fix:** Use a `ComputingEntry` sentinel or a per-key `threading.Event` so the second caller waits for the first to finish rather than duplicating work.
- **Effort:** Low. **Impact:** Medium.

### P-04: `collection.count()` on Every Search
- **What:** `vector_store.search()` at line 156 calls `collection.count()` to clamp `top_k`. ChromaDB handles this internally — the guard is redundant and adds IPC overhead on every query.
- **Where:** `vector_store.py:156`
- **Fix:** Remove the `min(top_k, count)` line. Let ChromaDB return up to `top_k` results naturally. If count must be known, cache it with a 60s TTL invalidated on ingest.
- **Effort:** Low. **Impact:** Medium.

### P-05: Translation Model Pinned in Memory
- **What:** NLLB-200 (~2.4GB on CPU) loads on first Strategy B use and never frees. Even if Strategy B is rarely used, the memory stays allocated.
- **Where:** `translation.py:22-56`
- **Fix:** Implement idle-unload: after 30 minutes without a translation call, `del _translation_model; gc.collect()`. Reloading adds ~3s latency but frees 2.4GB.
- **Effort:** Low. **Impact:** Medium.

### P-06: BM25 Linear Document Scan
- **What:** `BM25Index.search()` iterates over every document per query. At 50k chunks, each search does 50k × query_term_count comparisons.
- **Where:** `bm25_search.py:49-71`
- **Fix:** For the current corpus size (≤10k papers, ~100k chunks) this is acceptable. At 500k+, switch to an inverted index with precomputed IDF lookups.
- **Effort:** Medium. **Impact:** Medium (at scale).

### P-07: `get_or_create_collection` Logs Count at Startup
- **What:** `lifespan()` calls `get_or_create_collection()` which logs `collection.count()` on every server boot. For large DBs this adds startup latency.
- **Where:** `vector_store.py:80`
- **Fix:** Remove or downgrade the count log to DEBUG. Move to `/stats` endpoint.
- **Effort:** Trivial. **Impact:** Low.

---

## 3. ARCHITECTURAL ISSUES

### A-01: Monolithic api_server.py (1230 lines)
- **What:** Request models, route handlers, session management, job tracking, and agent initialization all live in one file. Adding any endpoint increases merge conflicts and testing difficulty.
- **Where:** `api_server.py`
- **Fix:** Split into FastAPI `APIRouter` modules: `routes/query.py`, `routes/chat.py`, `routes/ingest.py`, `routes/agent.py`, `routes/management.py`. Extract session/job stores to `services/sessions.py` and `services/jobs.py`.
- **Effort:** Medium. **Impact:** High.

### A-02: Volatile In-Memory State
- **What:** `_sessions` and `_jobs` dicts are pure in-memory. Server restarts destroy all chat history and in-flight ingest job statuses.
- **Where:** `api_server.py:29-67`
- **Fix:** Persist to SQLite (via `aiosqlite`) for sessions and jobs. Alternatively, use Redis for sessions if deployment targets cloud infrastructure.
- **Effort:** Medium. **Impact:** High.

### A-03: Duplicated JSON Extraction
- **What:** `_extract_json()` is copy-pasted identically between `query_planner.py:77-115` and `reflexion_evaluator.py:70-112`.
- **Where:** `agent/nodes/query_planner.py`, `agent/nodes/reflexion_evaluator.py`
- **Fix:** Extract to `agent/json_utils.py`. Both files import from there.
- **Effort:** Low. **Impact:** Medium.

### A-04: Hardcoded Safety Settings in Two Places
- **What:** `_SAFETY_SETTINGS` list appears in both `rag.py:311-328` and `answer_generator.py:11-16` with identical values. Changing safety policy requires editing both.
- **Where:** `rag.py`, `agent/nodes/answer_generator.py`
- **Fix:** Define once in `config.py` and import everywhere.
- **Effort:** Trivial. **Impact:** Low.

### A-05: `rag.py` Module-Level Side Effects
- **What:** Lines 20-21 use `__import__('threading')` and `__import__('itertools')` to avoid circular imports. This is fragile and obscures dependencies.
- **Where:** `rag.py:20-21`
- **Fix:** Move `_client_pool`, `_client_lock`, `_client_index` to a `llm_client.py` module. `rag.py` imports from there. Breaks the circular dependency cleanly.
- **Effort:** Medium. **Impact:** Medium.

### A-06: No Health Check for Downstream Dependencies
- **What:** `/health` only checks if `LLM_API_KEY_POOL` is non-empty. It doesn't verify ChromaDB is reachable, the embedding model is loaded, or the reranker is available.
- **Where:** `api_server.py:366-374`
- **Fix:** Add optional deep health: check `collection.count()`, attempt `embed_query("health")`, verify reranker `_model is not None`. Return as `{"status": "healthy|degraded|unhealthy", "checks": {...}}`.
- **Effort:** Medium. **Impact:** Medium.

### A-07: No Graceful Shutdown Handling
- **What:** `start_server.py` uses `subprocess.run()` which doesn't handle SIGTERM. On container orchestration (Docker, K8s), the process gets killed mid-request.
- **Where:** `start_server.py:134-139`
- **Fix:** Run uvicorn directly via `uvicorn.run()` with signal handlers, or use `--timeout-graceful-shutdown` flag.
- **Effort:** Low. **Impact:** Medium.

---

## 4. CODE QUALITY & RELIABILITY

### Q-01: `delete_by_paper_id` Always Returns 0
- **What:** `vector_store.delete_by_paper_id()` at line 232 hardcodes `return 0` regardless of how many chunks were removed.
- **Where:** `vector_store.py:225-232`
- **Fix:** Call `collection.count()` before and after delete, return the difference. Or query count of matching `paper_id` before deletion.
- **Effort:** Trivial. **Impact:** Medium.

### Q-02: `_extract_json` Fallback is Fragile
- **What:** The truncation-repair logic in both `_extract_json` implementations (`query_planner.py:104-115`, `reflexion_evaluator.py:101-112`) uses regex heuristics to fix broken JSON. These can produce invalid JSON silently.
- **Where:** `agent/nodes/query_planner.py:104-115`, `agent/nodes/reflexion_evaluator.py:101-112`
- **Fix:** Use `json5` or a lenient parser. Alternatively, log the raw response when repair is attempted so debugging is possible.
- **Effort:** Low. **Impact:** Medium.

### Q-03: No Timeout on ChromaDB Operations
- **What:** `vector_store.search()` and `collection.get()` have no timeout. A corrupted or locked ChromaDB file can hang the entire request indefinitely.
- **Where:** `vector_store.py` (all functions)
- **Fix:** Wrap ChromaDB calls with a `threading.Timer` or `concurrent.futures.ThreadPoolExecutor` timeout. If ChromaDB doesn't respond in 5s, raise a clear error.
- **Effort:** Medium. **Impact:** High.

### Q-04: BM25 Tokenizer is Naive for Indic Scripts
- **What:** `BM25Index._tokenize()` at `bm25_search.py:29-30` uses `regex.findall(r'\w+', text.lower())`. The `\w+` pattern matches Latin alphanumerics and underscore but does not correctly segment Indic scripts (Devanagari, Tamil, etc.) where word boundaries are different.
- **Where:** `bm25_search.py:29-30`
- **Fix:** Use `regex.findall(r'[\p{L}\p{N}]+', text)` which matches Unicode letters/numbers across all scripts. This is critical for Hindi/Tamil/Telugu queries hitting BM25.
- **Effort:** Low. **Impact:** High.

### Q-05: Session Eviction Uses Naive Datetime Comparison
- **What:** `_evict_stale_sessions()` at `api_server.py:57-67` calls `datetime.now(timezone.utc).replace(tzinfo=None)` which creates a naive UTC datetime. If `updated_at` was stored with a different timezone assumption, comparisons break silently.
- **Where:** `api_server.py:57-67,77,96`
- **Fix:** Store all datetimes as ISO 8601 with explicit `+00:00` suffix (timezone-aware). Compare using `datetime.fromisoformat()` with timezone support. Avoid the `.replace(tzinfo=None)` antipattern throughout.
- **Effort:** Low. **Impact:** Medium.

### Q-06: Agent State TypedDict Has Optional Fields Not in Use
- **What:** `AgentState` in `state.py:15-16` declares `year_from: Optional[int]` and `domain_hints: List[str]` but `AgentState(...)` constructor calls in `api_server.py:1114-1128` omit them, relying on TypedDict's permissive defaults.
- **Where:** `agent/state.py`, `api_server.py:1114-1128`
- **Fix:** Either provide default values in all `AgentState()` calls, or add `total=False` to those fields in the TypedDict definition.
- **Effort:** Trivial. **Impact:** Low.

### Q-07: CORS Allows All Methods and Headers
- **What:** `api_server.py:149-155` sets `allow_methods=["*"]` and `allow_headers=["*"]`. This is overly permissive for a REST API that only uses GET/POST/DELETE.
- **Where:** `api_server.py:149-155`
- **Fix:** Restrict to `allow_methods=["GET", "POST", "DELETE", "OPTIONS"]` and `allow_headers=["Content-Type", "X-API-Key"]`.
- **Effort:** Trivial. **Impact:** Medium (security hygiene).

### Q-08: No Request ID Propagation
- **What:** Logs don't include a request-scoped identifier. When debugging multi-user issues, correlating logs across retrieval, LLM calls, and tool execution is impossible.
- **Where:** All modules
- **Fix:** Add FastAPI middleware that generates a UUID per request, attaches it to `contextvars`, and includes it in all log output via a custom `logging.Filter`.
- **Effort:** Medium. **Impact:** Medium.

### Q-09: Test Coverage Gaps
- **What:** Only 30 unit tests exist. No tests for: `rag.retrieve_context()` (with mocked vector store), `rag.build_prompt()`, `pdf_utils.simple_chunk()`, `pdf_utils.extract_sections()`, `lang_utils.detect_language()`, `config.py` env parsing, the `/upload` endpoint, or the `/search` endpoint.
- **Where:** `tests/test_agent.py` (only file)
- **Fix:** Add `tests/test_rag.py`, `tests/test_pdf_utils.py`, `tests/test_api.py` (using `httpx.AsyncClient` with FastAPI's TestClient). Target 80%+ coverage on core modules.
- **Effort:** Medium. **Impact:** High.

### Q-10: MD5 for Dedup is Cryptographically Weak
- **What:** `ingest.py:19-25` uses MD5 for file dedup. While not a security vulnerability here (it's not for authentication), MD5 collisions exist and `hashlib.sha256` is negligibly slower.
- **Where:** `ingest.py:19-25`
- **Fix:** Switch to `hashlib.sha256`.
- **Effort:** Trivial. **Impact:** Low.

### Q-11: `ingest_directory` Accesses Private Executor Attribute
- **What:** `ingest.py:265` reads `executor._max_workers` — a private attribute of `ProcessPoolExecutor`. This breaks across Python versions.
- **Where:** `ingest.py:265`
- **Fix:** Use `os.cpu_count() or 4` directly, or expose max_workers as a parameter.
- **Effort:** Trivial. **Impact:** Low.

### Q-12: OpenAlex User-Agent Contains Placeholder Email
- **What:** `tool_executor.py:415` sends `"mailto:indicrag@example.com"` as User-Agent. OpenAlex recommends a real contact email.
- **Where:** `agent/tool_executor.py:415`
- **Fix:** Use a real project URL or contact email in the User-Agent string.
- **Effort:** Trivial. **Impact:** Low.

---

## PRIORITIZED SUMMARY

### Quick Wins (1-2 hours each, high impact)

| ID | Title | Impact |
|----|-------|--------|
| F-02 | Single-paper delete endpoint | High |
| F-03 | Rate limiting | High |
| P-02 | Fix double MD5 in bulk ingest | Low |
| P-04 | Remove redundant `count()` in search | Medium |
| A-03 | Deduplicate `_extract_json()` | Medium |
| A-04 | Centralize safety settings | Low |
| Q-01 | Fix `delete_by_paper_id` return value | Medium |
| Q-04 | Fix BM25 tokenizer for Indic scripts | High |
| Q-05 | Fix timezone-aware datetime handling | Medium |
| Q-07 | Restrict CORS methods/headers | Medium |
| Q-10 | MD5 → SHA256 | Low |
| Q-11 | Remove private attribute access | Low |

### Medium Effort (half-day to 1 day, high impact)

| ID | Title | Impact |
|----|-------|--------|
| F-01 | Streaming responses | High |
| F-05 | Post-ingest metadata editing | Medium |
| P-01 | Incremental BM25 index updates | High |
| P-03 | Embed query deduplication | Medium |
| P-05 | Idle-unload translation model | Medium |
| A-05 | Extract LLM client pool to own module | Medium |
| A-06 | Deep health check endpoint | Medium |
| A-07 | Graceful shutdown | Medium |
| Q-03 | ChromaDB operation timeouts | High |
| Q-08 | Request ID propagation | Medium |
| Q-09 | Expand test coverage | High |

### Strategic (1-3 days, architectural impact)

| ID | Title | Impact |
|----|-------|--------|
| A-01 | Split api_server.py into router modules | High |
| A-02 | Persist sessions and jobs to SQLite/Redis | High |
| F-04 | WebSocket agent progress streaming | Medium |
| F-06 | Collection export/import | Low |
| Q-02 | Robust JSON parsing for LLM outputs | Medium |

---

## RECOMMENDED v2.1 ROADMAP

**Phase 1 — Reliability & Correctness (Week 1)**
1. Fix Q-01, Q-04, Q-05 (data correctness bugs)
2. Add F-02 + F-03 (single-paper delete, rate limiting)
3. Add Q-03 (ChromaDB timeouts)
4. Deduplicate A-03, A-04

**Phase 2 — Performance & UX (Week 2)**
1. Implement F-01 (streaming)
2. Fix P-01 (incremental BM25)
3. Fix P-03 (embed query dedup)
4. Add Q-08 (request IDs)

**Phase 3 — Architecture (Week 3)**
1. Execute A-01 (split api_server)
2. Implement A-02 (persistent sessions)
3. Add Q-09 (test coverage expansion)
4. Add A-06 (deep health checks)
