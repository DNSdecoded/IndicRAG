# IndicRAG v2.4 — Architecture Review

> Review date: 2026-08-21 · Branch: `v2.4.1-dev` · Scope: full system design
> Method: source-grounded (code read, not README-derived). File:line references are load-bearing.
> Revision 2 folds in a second review pass; items marked **[R2]** came from it, and
> §H records the recommendations that were checked and declined.
>
> Where this document and `README.md` disagree on a default, the code wins. (The README's
> stale `AGENT_TIMEOUT` "default 120" was corrected after this review; both it and the code
> now say 300.)

## Fix status

Items closed in the first implementation pass (318 unit tests green, 37 new):

| Item | Status | Where |
|---|---|---|
| **A1 / A2** cross-user data exposure | **Fixed** | `owner` on sessions/watches/feedback/reports/query_log; scoping in `routes/chat.py`, `agent.py`, `watch.py`, `feedback.py`, `report.py`; mismatches 404 |
| **D4** SQLite secondary indexes | **Fixed** | 9 indexes in `persistence.py` |
| **E5** LLM-spend rate limits | **Fixed** | `5/minute` on `POST /report` and `POST /watch/{id}/run` (verified: 429 fires) |
| **C6** session lost update | **Fixed** | `deps.session_turn_lock`, held across read→generate→append in both `/chat` handlers |
| **C10a** ChromaDB circuit breaker | **Fixed** | `vector_store` — opens after 3 consecutive timeouts, 60s cooldown |
| **C10b** embedding-failure fallback | **Fixed** | `rag._sparse_only_retrieval`, BM25-only with a `degraded` marker |
| **D5** parallel figure captioning | **Fixed** | `FIGURE_CAPTION_WORKERS` (default 4), order preserved |
| **B5** thread oversubscription | **Fixed** | `TORCH_NUM_THREADS` / `OMP_NUM_THREADS` pinned in `config.py` |
| **F5** migrations | **Partial** | `persistence._ensure_column` — idempotent `ADD COLUMN` only; a versioned runner is still needed for anything that backfills or transforms |
| **F6a** BM25 test coverage | **Fixed** | `tests/test_bm25_search.py` — lands before the B2 rewrite |

~~Known gap from this pass: the `degraded` marker is logged but not yet surfaced in any API
response.~~ **Closed:** `degraded` is on `QueryResponse` (`routes/query.py:119`),
`ChatResponse` (`routes/chat.py:68`) and the SSE `done` event on both streaming routes.

**System as built:** single uvicorn process (`workers=1`), embedded ChromaDB (`PersistentClient` on a local dir), in-memory BM25 index, three process-local TTL caches, one SQLite connection behind one global lock, in-process asyncio watch scheduler, in-process ingest jobs, ~6 GB of models resident. Everything is one node with node-local state.

That is a coherent single-node design. Nearly every finding below follows from one root fact: **state that must be shared lives inside the process**, so the system cannot be replicated, and correctness/latency guarantees are only as good as one box.

---

## A. Critical — data isolation is broken (contradicts the v2.4 multi-user claim)

### A1. Chat sessions, watches, feedback, reports are globally readable across API keys — **High**

**Change:** scope every list/read/delete by `current_owner` (SHA-256 key fingerprint), the same pattern `routes/query.py:401` already uses for jobs. Add an `owner` column to `sessions`, `feedback`, `reports`; stop trusting client-supplied `user_id`.

**Problem:** `routes/chat.py:205,228,244` — `GET /chat`, `GET /chat/{id}`, `DELETE /chat/{id}` take only `verify_api_key`. Any valid key lists and reads every user's conversation history and deletes it. `routes/watch.py:101` takes `user_id` as a **query parameter** — omit it and get all users' watches; supply someone else's and read theirs. `persistence.get_feedback_with_context` and `list_reports` have no owner column at all. Watch creation (`routes/watch.py:85`) stores the caller-declared `user_id`, so identity is self-asserted.

**Benefit:** the per-user isolation v2.4 advertises actually holds.

**Trade-off:** existing rows have no owner — migration needs a backfill (assign to admin, or leave NULL = admin-only visible). One breaking API change: `user_id` on `POST /watch` becomes derived, not supplied.

**Impact:** `persistence.py` schema + all list/get queries, `deps.py` (make `current_owner` a hard dependency), `routes/chat.py`, `routes/watch.py`, `routes/feedback.py`, `routes/report.py`.

### A2. `current_owner` returns `None` when auth is disabled, and jobs treat `owner=None` as public — **High**

**Change:** when `ALLOW_UNAUTHENTICATED=1`, force single-tenant mode explicitly — disable `/watch/*`, `/prefs/*`, and cross-session listing rather than silently making them shared.

**Problem:** `job.get("owner") is not None and ...` (`routes/query.py:401`) means unauthenticated deployments have no ownership check anywhere; the isolation code path is inert exactly where it is least obvious.

**Benefit:** no mode where multi-user routes exist but isolation is off.

**Trade-off:** dev convenience.

**Impact:** `deps.py`, route registration in `api_server.py`.

---

## B. Scalability — the single-process ceiling

### B1. Externalize shared state so `workers > 1` and replicas become possible — **High**

**Change:** move the BM25 index, the three TTL caches, `_sessions`, and `_jobs` out of process. Redis for caches + session/job state (already anticipated in the `docker-compose.yml` comment); BM25 either rebuilt per-worker from a shared snapshot or replaced (see B2).

**Problem:** `start_server.py:workers=1` is not a tuning choice, it is forced. `bm25_search._indices`, `cache.llm_cache/retrieval_cache/tool_cache`, `deps._sessions`, `deps._jobs` are module globals. Two workers = two divergent BM25 indexes, halved cache hit rate, invalidation on ingest that reaches only one worker (stale retrieval served indefinitely from the other), and job status visible only from the worker that owns it.

**Benefit:** horizontal scale; a crashed worker stops taking the whole service down.

**Trade-off:** Redis is a new dependency and a new failure mode — the cache must fail-open (miss on Redis error, never 500). Serialization cost on session read/write.

**Impact:** `cache.py` (swap `TTLCache` for a Redis-backed implementation behind the same `get/put/invalidate` interface — the interface is already right), `deps.py`, `bm25_search.py`, `cache_refresh.py`.

**DDIA framing (Ch 1, Ch 5):** you cannot replicate what you have not first made shareable. Do this before any other scaling work — it gates everything else.

### B2. BM25 is a full linear scan and a full-corpus RAM load — **High**

**Change:** build a real inverted index (`term -> [(doc_idx, tf)]`) and score only the postings for query terms. Same file, roughly 30 lines.

**Problem:** `BM25Index.search` loops `for i in range(self.n_docs)` over **every chunk** for every query — O(N·|q|) with a Python inner loop. `get_or_build_index` calls `collection.get(include=["documents"])`, materializing the entire corpus text in RAM to build, and keeps `doc_freqs` (a `Counter` per chunk) resident forever. At 10k papers ≈ 500k chunks that is multiple GB of Counters and a per-query scan in the hundreds of ms — sparse retrieval becomes the dominant term in p99, ahead of the cross-encoder.

**Benefit:** query cost drops to postings length (typically 100–1000× less work); memory drops to one dict of postings.

**Trade-off:** rebuild is slightly more code; deletions need tombstones or a rebuild.

**Impact:** `bm25_search.py` only — `build`/`search` are behind a clean interface, `rrf` and callers unchanged.

**DDIA framing (Ch 3, Ch 6):** this is exactly the scan-vs-index distinction. A term-partitioned inverted index is the standard answer.

### B3. Every ingest nukes the whole BM25 index; the next query pays the rebuild — **High**

**Change:** incremental `add_documents(ids, texts)` on the index (update `df`, `doc_lens`, `avg_dl`, postings). Keep full rebuild as a repair path. If a rebuild is unavoidable, build into a new object and swap the reference — never serve from a half-built index or block on it.

**Problem:** `bm25_search.invalidate()` sets `_indices = {}`. The first query after any ingest — including every watch digest run — synchronously rebuilds from the full corpus while holding `_lock`, and every concurrent query blocks behind it. This is a self-inflicted periodic p99 cliff that grows linearly with corpus size.

**Benefit:** ingest stops being a latency event.

**Trade-off:** incremental IDF drifts slightly from a clean rebuild (negligible; schedule a nightly rebuild if it matters).

**Impact:** `bm25_search.py`, `cache_refresh._post_ingest_refresh`.

### B4. No admission control on expensive endpoints — **High**

**Change:** a global `asyncio.Semaphore` (or bounded threadpool) around agentic query / rerank / NLI, sized to cores. Over the limit, return 503 with `Retry-After` immediately.

**Problem:** rate limiting (`deps.limiter`) counts requests, not concurrent work. An agent query can hold a thread for `AGENT_TIMEOUT` — **300s** — while running cross-encoder + NLI + contradiction detection (up to 56 NLI passes). Twenty concurrent agent queries on an 8-core box means every one of them thrashes and times out — the classic failure where the system does maximum work and returns zero successful responses. A five-minute per-thread hold makes this materially worse than the README's numbers suggest.

**Benefit:** bounded, predictable p99 under overload; some requests shed fast instead of all failing slow.

**Trade-off:** visible 503s at peak — which is the correct, honest behavior.

**Impact:** `routes/agent.py`, `routes/query.py`, `rerank.py` / `verify.py` entry points.

**DDIA framing (Ch 1):** response time is a distribution. Unbounded queueing converts a throughput problem into a total-availability problem.

### B5. Thread oversubscription — **Medium**

**Change:** pin `OMP_NUM_THREADS` / `torch.set_num_threads` and size the pools together; document the total.

**Problem:** a 32-worker `_chroma_executor` + FastAPI's default 40-thread pool + `ProcessPoolExecutor` for PDF parsing + `ThreadPoolExecutor` for parallel agent tools + torch/ONNX intra-op threads, all unbounded relative to each other. On an 8-core box the ready queue is dozens deep and every stage gets slower simultaneously.

**Trade-off:** lower peak throughput on an idle box, much better behavior under load.

**Impact:** `config.py`, `vector_store.py`, `api_server.py` startup.

### B6. BM25 index is rebuilt from ChromaDB on every process start — **Medium** **[R2]**

**Change:** persist the index (pickle, or a plain postings file once B2 lands) and reload it at startup; fall back to a rebuild when the file is missing, corrupt, or stamped with a stale corpus count.

**Problem:** `get_or_build_index` calls `collection.get(include=["documents"])` on the first query after every restart. Cold start therefore scales with corpus size, and the warm-up in `api_server.lifespan` pays it on every deploy. Combined with B3 (invalidate-on-ingest), the full rebuild is currently the system's most frequently executed expensive operation.

**Benefit:** near-instant cold start; one less full read of the vector store per deploy.

**Trade-off:** a persisted index is a second copy that can go stale — it must be invalidated on ingest and validated against the collection count on load.

**Impact:** `bm25_search.py`, `cache_refresh.py`. Do it after B2, so what gets persisted is the postings structure and not the current per-chunk `Counter` list.

---

## C. Reliability & correctness

### C1. Long-running work runs inside the web process with no durable queue — **High**

**Change:** a real work queue for ingest, reports, and watch runs. Lazy version that fits what is already here: a `tasks` table in SQLite with `status`, `lease_expires_at`, `attempts`, `fencing_token`, plus a worker loop in a separate process. Full version: Redis + RQ/Celery.

**Problem:** ingest jobs and report generation execute in the API process's threadpool. Consequences: (a) a bulk ingest starves query latency; (b) a restart mid-job leaves the job row `running` forever — nothing reaps it, so `/compare/status/{id}` polls a dead job indefinitely; (c) no retry; (d) the process cannot be scaled or restarted independently of the CPU-heavy work.

**Benefit:** restart safety, retries, ingest isolated from query latency, workers scale independently.

**Trade-off:** a second deployable unit; job handoff must be idempotent.

**Impact:** `routes/ingest.py`, `deps._jobs` / `_update_job`, `report_runner.py`, `watch_runner.py`, `persistence.py`, `docker-compose.yml`.

**DDIA framing (Ch 8):** a lease with a **fencing token** is required — a worker that hung and a worker that died are indistinguishable, and the recovered one must not write over its successor.

### C2. In-process watch scheduler duplicates work the moment there is a second replica — **High (blocking for B1)**

**Change:** either run the scheduler as its own single-replica deployment, or take a leader lease before each tick (`UPDATE watches SET next_run=? WHERE id=? AND next_run=?` — a compare-and-set claim is enough and needs no new infrastructure).

**Problem:** `api_server.lifespan` starts `watch_runner.watch_loop()` unconditionally per process. Two workers = every watch runs twice = duplicate arXiv fetches, duplicate ingests (dedup catches most, but not the API cost), duplicate digests.

**Benefit:** exactly-once-ish scheduling that survives scaling.

**Trade-off:** a claimed-but-crashed watch needs a lease timeout to be re-runnable.

**Impact:** `api_server.py`, `watch_runner.py`, `persistence.due_watches`.

### C3. No backup or restore path for ChromaDB — **High**

**Change:** a scheduled snapshot of `chroma_db/` (a plain copy while writes are in flight is unsafe — snapshot the volume, or add an export-collection job) plus a documented, *tested* restore. Add `/admin/snapshot`.

**Problem:** the vector store is a bind-mounted directory with one writer and no replication. Volume loss or partial corruption during an ingest crash = full re-ingest of the corpus (hours of PDF parsing + embedding, and remote metadata that may no longer resolve). There is no snapshot, no checksum, no restore procedure.

**Benefit:** recovery in minutes instead of hours.

**Trade-off:** disk.

**Impact:** ops + one management route.

### C4. Derived indexes have no rebuild-from-source path — **Medium/High**

**Change:** an append-only ingest event log (`doc_id`, `content_hash`, `source_path`, extracted chunks + section labels, timestamp) as the system of record. Chroma, BM25, and the figure store become **derived views** rebuildable from it.

**Problem:** today the truth is split: PDFs in `papers/`, chunks in Chroma, the lexical index in RAM, metadata in Chroma metadata, artifacts in SQLite. There is no way to deterministically rebuild the vector store — you must re-parse every PDF and re-call the VLM captioner. That makes an embedding-model upgrade or a chunking-strategy change a multi-hour, non-reproducible operation, which in practice means it never happens.

**Benefit:** reindexing becomes routine; model upgrades, chunker changes, and corruption recovery are all the same replayable operation.

**Trade-off:** storage for chunk text (small next to the PDFs), one new write on the ingest path.

**Impact:** `ingest.py`, `persistence.py`, new `reindex.py`.

**DDIA framing (Ch 11, Ch 12):** unbundle — one durable log, many derived indexes. This is the highest-leverage structural change in the list for long-term evolvability.

### C5. Embeddings and chunks carry no version stamp — **High**

**Change:** stamp every chunk with `embed_model`, `embed_dim`, `chunker_version`, `schema_version`. Refuse to query a collection whose stamp differs from the running config (fail loudly), and make reindex the migration path.

**Problem:** nothing records which model produced a vector. Swap `bge-m3` for anything else, or change per-section chunk sizes, and new chunks silently join old ones. Cosine distance between two different embedding spaces is meaningless but never errors — retrieval quality degrades quietly and is nearly undebuggable from the outside.

**Benefit:** model migrations become detectable and safe.

**Trade-off:** a few bytes of metadata per chunk; one migration to stamp existing rows.

**Impact:** `ingest.py`, `vector_store.py`, `embeddings.py`, startup check in `lifespan`.

**DDIA framing (Ch 4):** this is schema evolution. Un-versioned data written by two incompatible writers is the canonical failure.

### C6. Concurrent turns on one session lose updates — **Medium**

**Change:** a per-session lock across read-history → generate → append, or make the append conditional on the message count that was read.

**Problem:** `_get_or_create_session` returns `list(...)` — a copy. Two concurrent requests on the same session both read history at length N, both generate, and both append; the result is a history where each turn was answered without knowledge of the other. `_append_session_messages` also re-materializes an evicted session (the correct fix for a KeyError, but it silently resurrects a session with empty history mid-conversation).

**Benefit:** coherent multi-turn context.

**Trade-off:** serializes concurrent turns per session (correct — they are not independent).

**Impact:** `deps.py`.

**DDIA framing (Ch 7):** read-modify-write without isolation = lost update.

### C7. Timeouts do not cancel the work behind them — **Medium**

**Change:** cooperative cancellation — a deadline in `AgentState` checked at each graph node boundary and before each tool dispatch. For Chroma, treat a timeout as a degraded-mode signal (circuit-break the collection) rather than only leaking a thread.

**Problem:** `vector_store._chroma_call` documents that a timed-out call keeps running (the 32-worker pool is a leak budget, not a fix). `AGENT_TIMEOUT` returns 504 while the reflexion loop keeps burning LLM quota and CPU for a client that is gone. Under load, timed-out work is pure waste that makes the next request slower — a retry-storm amplifier.

**Benefit:** timeouts actually free capacity.

**Trade-off:** cancellation checks in agent nodes.

**Impact:** `agent/graph.py`, `agent/tool_executor.py`, `vector_store.py`.

### C8. Ingest dedup is check-then-write — **Medium**

**Change:** a `UNIQUE` constraint on `content_hash` in the ingest log (C4), or a per-hash lock; let the DB decide the winner.

**Problem:** the three-layer dedup (file hash, content hash, title similarity) reads current state and then writes. Two concurrent uploads of the same paper both pass all three checks and both index — duplicated chunks skew BM25 IDF and produce duplicate citations.

**Benefit:** dedup holds under concurrency.

**Trade-off:** one loser gets an "already ingested" response.

**Impact:** `ingest.py`, `routes/ingest.py`.

### C9. Health endpoint conflates liveness and readiness — **Low/Medium**

**Change:** split `/livez` (process up) from `/readyz` (models loaded, collection reachable).

**Problem:** the compose healthcheck already needs `start_period: 120s` to paper over this; an orchestrator would kill the pod during model load.

**Impact:** `routes/management.py`, `docker-compose.yml`, any k8s manifest.

### C10. No degraded retrieval mode: a sick ChromaDB or a failed embedding call fails the whole request — **High** **[R2]**

**Change:** one piece of work with two halves. (a) Wrap ChromaDB in the same circuit breaker already used per `(provider, model)` in `llm_client.py` — after N consecutive timeouts, trip and skip the dense leg for the cooldown. (b) Wrap `embeddings.embed_query` so a failure degrades to the same path. In both cases serve BM25-only results with an explicit `degraded: "sparse_only"` field in the response.

**Problem, half (a):** `_chroma_call` has a 5s timeout and no breaker. When ChromaDB is unhealthy — disk full, corrupt segment, hung compaction — *every* request pays the full 5s before failing, and each one parks a worker in the 32-thread `_chroma_executor` that cannot be cancelled. The pool is exhausted in seconds and the failure spreads to requests that would not have touched the dense path at all.

**Problem, half (b):** `rag.py:358` calls `embeddings.embed_query(user_query)` unguarded on the main retrieval path. An OOM, a corrupted model file, or a driver fault takes down every query, including ones a BM25 index sitting in the same process could have answered.

**Benefit:** the dense leg becomes a component that can fail rather than the whole system. Sparse-only answers are worse, but they exist.

**Trade-off:** degraded results must be surfaced honestly — a silently sparse-only answer is worse than an error because it looks normal. This also demands the explicit degraded-mode contract listed in §G.

**Impact:** `vector_store.py`, `rag.py` (`retrieve_context`), response models in `routes/query.py` and `routes/agent.py`. Reuse the breaker from `llm_client.py`; do not write a second one.

**DDIA framing (Ch 8):** timeouts alone are a fragile failure detector. A breaker converts "slow" into "known down" and stops the pile-up.

### C11. Agent streaming is post-hoc chunking, not streaming — **Low** **[R2]**

**Change:** stream the answer generator's tokens through the existing `sse_utils` queue bridge instead of slicing the finished string.

**Problem:** `routes/agent.py:325` sets `chunk_size = 80` and emits the completed answer in 80-character slices — the docstring at :247 says so plainly. Time-to-first-*answer*-token equals total pipeline time.

**Not as bad as it looks:** thinking events (:275) and tool-call events (:301) *do* stream live during the run, so the user sees progress throughout. This is a fidelity problem, not a blank-screen problem — which is why it stays Low.

**Trade-off:** the answer generator runs in a threadpool inside a LangGraph node; threading real tokens out means the node must yield through the queue bridge, and citation compaction currently has to complete before the first chunk goes out (:304). That ordering constraint is the actual blocker, not the streaming plumbing.

**Impact:** `agent/graph.py`, `routes/agent.py`, `sse_utils.py`.

---

## D. Performance

### D1. SQLite: one connection, one global lock, on the query hot path — **Medium**

**Change:** connection-per-thread (or a small pool) with WAL — WAL's whole point is concurrent readers, and `_db_lock` around a single connection discards that. Batch `log_query` writes (buffer + flush) instead of committing synchronously inside request handling.

**Problem:** `persistence._db_lock` serializes *all* DB access globally, and every function commits individually. Every `/query` writes a `query_log` row with an fsync-per-commit, in a threadpool thread, behind a process-global lock. The ceiling is roughly low hundreds of writes/sec and it contends with session writes and the watch scheduler.

**Benefit:** reads stop blocking on writes; the query path stops paying an fsync.

**Trade-off:** a crash can lose the last few query-log rows (acceptable — it is analytics, not answers).

**Impact:** `persistence.py` only.

### D2. Tags stored as one comma-joined string, filtered in Python with over-fetch — **Medium**

**Change:** store each tag as its own Chroma metadata key (`tag_ml: true`) and filter server-side with `$and`/`$in`, or keep an authoritative `paper_tags` table in SQLite and pre-resolve to `paper_ids`.

**Problem:** the v2.4 notes describe the current shape honestly — tags need `TAGS_OVERFETCH` because filtering happens after retrieval. That makes recall a function of a tuning constant: a rare tag on a large corpus silently returns fewer results than it should, and the failure is invisible.

**Benefit:** correct, tuning-free tag filtering; no wasted over-fetch.

**Trade-off:** a metadata migration.

**Impact:** `ingest.py`, `vector_store.py`, `rag.py`.

### D3. Contradiction detection costs up to 56 NLI passes per query — **Low**

**Change:** make it opt-in per request, or run it after the answer streams and patch it in as a follow-up SSE event.

**Trade-off:** contradiction warnings arrive slightly late.

**Impact:** `contradiction.py`, `routes/agent.py`.

### D4. SQLite has no secondary indexes at all — **Medium (cheapest item in this document)** **[R2]**

**Change:** six statements.

```sql
CREATE INDEX IF NOT EXISTS idx_watches_next_run  ON watches(next_run);
CREATE INDEX IF NOT EXISTS idx_watches_user      ON watches(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_query    ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status       ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_sessions_updated  ON sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_reports_watch     ON reports(watch_id);
```

**Problem:** all seven `CREATE TABLE` statements in `persistence.py` declare a PRIMARY KEY and nothing else. Every non-key access is a full scan: `due_watches` filters on `next_run` and runs on every scheduler tick; `get_feedback_with_context` LEFT JOINs `query_log` on `query_id`; `load_sessions` and `load_jobs` delete by timestamp on every startup; `list_reports` filters by `watch_id`. All are fine at current row counts and all degrade linearly as `query_log` and `sessions` accumulate.

**Benefit:** O(log n) instead of O(n) on the scheduler's hot path and on every feedback read.

**Trade-off:** marginal write amplification, invisible at this write rate.

**Impact:** `persistence.py`, additive only — no query or logic changes. Pairs naturally with F5 (migrations) but does not need to wait for it: `CREATE INDEX IF NOT EXISTS` is safe to run at startup alongside the existing `CREATE TABLE IF NOT EXISTS` calls.

**DDIA framing (Ch 3):** without an index the storage engine scans every row. That is the whole finding.

### D5. Figure captioning is sequential — **Low** **[R2]**

**Change:** a bounded `ThreadPoolExecutor` (3–5 workers) over the region list, sized to stay inside the Gemini rate limit.

**Problem:** `figure_captioner.py:178` iterates `for r in regions:`, one VLM round-trip at a time. A figure-heavy paper serializes a dozen network calls into ingest latency.

**Benefit:** figure-heavy ingestion drops to roughly wall-clock/N.

**Trade-off:** must respect the API rate limit — unbounded concurrency here trades a slow ingest for a 429 storm that the LLM circuit breaker will then trip.

**Impact:** `figure_captioner.py`, isolated.

---

## E. Security

### E1. API keys are static plaintext env values — **Medium**

**Change:** an `api_keys` table storing SHA-256 hashes with `owner`, `scopes`, `created_at`, `expires_at`, `revoked_at`. Keep the constant-time comparison (already correct in `_key_matches`). Add issue/revoke admin routes.

**Problem:** rotation means a redeploy; revoking one user's access means rotating everyone's; a leaked key is valid forever; there is no scoping (any key can ingest — purge is admin-only, but everything else is uniform). `API_KEYS` sits in the process environment and in `.env` on disk.

**Benefit:** rotation and revocation without downtime; per-key scopes and quotas.

**Trade-off:** one lookup per request (cache it).

**Impact:** `deps.py`, `persistence.py`, new admin routes.

### E2. `execute_python` sandbox — **Medium/High**

**Change:** default it off. When on, add hard resource limits (`RLIMIT_AS`, `RLIMIT_CPU`, no network namespace) on top of the existing AST validation and 10s timeout.

**Problem:** AST whitelisting is a deny-list-shaped defense pretending to be an allow-list — a well-known bypass surface, and the tool is reachable by any authenticated user. Process isolation without an RSS limit still allows a memory bomb that OOM-kills the whole server (which, per B1, is the entire service).

**Benefit:** a bypass costs one sandboxed process, not the node.

**Trade-off:** setup complexity; `resource` limits are POSIX-only (Windows dev needs a fallback path).

**Impact:** `agent/tool_executor.py`, deployment.

### E3. SSRF guard has a DNS TOCTOU gap — **Medium** **[R2, corrected]**

> Correction to revision 1, which asked for a size cap, timeouts, and a redirect guard on the
> download path. All three already exist — `_MAX_PDF_BYTES` (50 MB), `timeout=30`, and
> `_NoRedirectHandler` with per-hop re-validation. The real gap is narrower and is named in the
> code's own `ponytail:` comment.

**Change:** pin the resolution — resolve the hostname once, verify that address, then connect to *that IP* (custom `HTTPConnection` / connection adapter with the `Host` header preserved) instead of letting `urlopen` resolve a second time.

**Problem:** `download_utils._is_private_ip` calls `socket.getaddrinfo`, and `_NoRedirectOpener.open(req)` then performs its own independent lookup. A hostile or rebinding DNS server can answer public for the check and `127.0.0.1` / `169.254.169.254` for the fetch. Every other SSRF control on this path is sound; this one hole bypasses all of them.

**Benefit:** closes the last SSRF vector on the ingest path, including cloud metadata access.

**Trade-off:** a custom opener, and TLS SNI/cert validation must still key on the hostname rather than the pinned IP — easy to get subtly wrong, so it needs a test.

**Realistic severity:** the fetched URLs originate from arXiv / Semantic Scholar / OpenAlex records rather than raw user input, and the routes are authenticated. High impact, low likelihood — worth fixing, not worth blocking a release on.

**Impact:** `download_utils.py` only. `tests/test_download_utils.py` already exists; extend it with a rebinding case.

### E4. No per-key cost quota — **Medium**

**Change:** track LLM tokens per owner in `query_log`; enforce a daily budget, 429 over it.

**Problem:** rate limiting caps request *count*. One user running agentic queries with reflexion at 8192 max tokens can exhaust the shared Gemini quota (and bill), degrading every other user — a noisy-neighbor failure that per-IP/per-key request limits do not address at all.

**Impact:** `providers/*`, `persistence.py`, `deps.py`.

### E5. LLM-spending endpoints have no rate limit — **High** **[R2, re-prioritized]**

**Change:** `@limiter.limit(...)` on `POST /report` (`routes/report.py:97`) and `POST /watch/{watch_id}/run` (`routes/watch.py:116`) first; then the read-heavy routes.

**Problem:** rate limits are applied only in `routes/query.py`, `routes/chat.py`, `routes/agent.py`, and `routes/ingest.py`. The `management`, `feedback`, `watch`, `report`, and `models` routers have none. Two of those unprotected routes each kick off a full LLM workload — report generation decomposes a topic and synthesizes every section; a manual watch run searches, ingests, and generates a digest. An authenticated user can loop either one and drain the shared Gemini quota, which per E4 degrades every other user. `/search` and `/papers` are the milder case: ChromaDB threads and disk I/O.

**Benefit:** the two endpoints that spend money get a bound.

**Trade-off:** limits must be generous enough for legitimate bulk report generation; per-key rather than per-IP (already the behavior of `_rate_limit_key`).

**Impact:** decorators in `routes/report.py`, `routes/watch.py`, then `routes/management.py`, `routes/feedback.py`, `routes/models.py`.

---

## F. Maintainability & observability

### F1. Stage-level metrics are missing — **High** (highest observability payoff per line)

**Change:** Prometheus histograms per stage: `retrieval_dense`, `retrieval_bm25`, `rrf`, `colbert`, `cross_encoder`, `llm_generate`, `nli_verify`, `contradiction`, plus counters for cache hits per cache, LLM tokens, circuit-breaker trips, failover hops, reflexion iterations, and abstentions.

**Problem:** `Instrumentator()` gives HTTP-level latency only. When p99 doubles, nothing in the metrics distinguishes "BM25 rebuild after ingest" (B3) from "Gemini is slow" from "reranker thrashing" (B5). `GET /cache/stats` exists but is not scraped, so there is no history.

**Benefit:** every other item in this review becomes measurable instead of arguable.

**Trade-off:** a handful of decorators.

**Impact:** `rag.py`, `rerank.py`, `verify.py`, `llm_client.py`.

**DDIA framing (Ch 1):** measure percentiles per stage, not averages over the whole request.

### F2. The retrieval-quality gate exists but cannot fail — **High** **[corrected]**

> Correction to revision 1, which said "the harness exists but nothing enforces it."
> `.github/workflows/ci.yml` **does** have an "Eval gate" step running
> `python evaluate.py --ci --threshold 0.85`. The gate is real. The problem is narrower
> and worse than a missing gate: it cannot fail from a code change.

**Change:** add a CI job that **regenerates** `answers_and_citations.json` by running the
live pipeline over the fixture corpus, then feeds the existing scorer. Keep the current
snapshot job as a fast pre-check.

**Problem:** `evaluate.py:220` reads `runs[qid]["retrieved_papers"]` from the committed
`answers_and_citations.json`. It scores a frozen snapshot of past answers and never calls
retrieval, so no change to `rag.py`, `bm25_search.py`, `embeddings.py`, or any retrieval
env knob can move the number. The threshold (0.85, against a measured 0.94) is calibrated
correctly and is still unreachable, because the input never changes unless someone
regenerates the file by hand. The CI comment is honest about why — no keys, no model
downloads in CI — but the effect is a gate that passes by construction.

**Benefit:** the scoring half is already built and working; wiring a live run to it turns
an ornament into an actual gate.

**Trade-off:** needs either a small embedding model or a cached HF snapshot in CI (see F6),
plus an LLM key or a jaccard-only judge for the generation half.

**Impact:** `.github/workflows/ci.yml`, `docs/Eval/`. Effort drops from "build a quality
gate" to "wire the harness to a live run".

### F3. `rag.py` at 1221 lines is the coupling hotspot — **Medium**

**Change:** split into `retrieval.py` (hybrid + fusion + rerank), `generation.py` (prompt + LLM + failover), `pipeline.py` (orchestration). Mechanical, no behavior change.

**Problem:** it is imported by the agent, all query routes, `watch_runner`, and `report_runner`. Every change touches every consumer's blast radius, and it is the file where retrieval and generation concerns get accidentally entangled.

**Trade-off:** one large diff, temporary merge friction.

**Impact:** import sites only.

### F4. Config sprawl without a printed effective config — **Low/Medium**

**Change:** log the full resolved config (secrets redacted) at startup and expose it at `/config` behind admin auth.

**Problem:** 584 lines of `config.py` with 74 `os.getenv` call sites. Reproducing a production behavior locally currently means reading source to find which variables exist.

### F5. No schema migration framework — **Medium** **[R2]**

**Change:** a `schema_version` table plus an ordered list of migration functions, applied at startup. Alembic works but drags in SQLAlchemy; for seven tables a 40-line versioned runner is enough.

**Problem:** `persistence.py` creates schema imperatively with `CREATE TABLE IF NOT EXISTS`. That guard means an existing database silently keeps its *old* shape — add a column in a new release and every deployed instance quietly lacks it until something throws `no such column` at runtime. Which is exactly what A1 (owner columns), C4 (ingest log), and C5 (version stamps) all require. Every schema-touching item in this document is blocked on having a way to migrate.

**Benefit:** schema changes become releasable. Rollback becomes possible.

**Trade-off:** one more startup step; migrations must be tested against a populated database, not an empty one.

**Impact:** `persistence.py` + a `migrations/` directory. Do this **before** A1, C4, and C5 rather than alongside them.

**DDIA framing (Ch 4):** schema evolution needs an explicit, versioned path — `IF NOT EXISTS` is not one.

### F6. Zero integration tests — every model is mocked — **High** **[R2]**

**Change:** a small integration suite with a 2–3 paper fixture corpus, a temporary ChromaDB directory, and a real (small) embedding model, exercising ingest → retrieve → rerank → generate end to end. Mark it `@pytest.mark.integration` so the fast suite stays fast.

**Problem:** 28 test files, and not one loads a real model — no reference to `load_embedding_model()`, `BGEM3FlagModel`, or `SentenceTransformer` anywhere in `tests/`. Every LLM, vector store, and embedding call is mocked. The consequence is that the seams *between* stages are untested: prompt assembly, context formatting, citation-marker compaction, chunk truncation, and the SSE bridge are each verified against a mock's idea of the contract. This is precisely how the v2.4 patch bugs happened — `verify.check_claims` receiving `(index, text)` tuples, the retrieval cache never populating, the answer generator ignoring the selected model. All three are seam defects that mocks cannot catch and one real end-to-end run would have.

**Also:** `bm25_search.py` has no dedicated test at all (it is only reached incidentally through `test_agent.py`) — which matters now, because B2, B3, and B6 all propose rewriting it.

**Benefit:** catches the exact class of bug this codebase has actually been shipping.

**Trade-off:** seconds instead of milliseconds; needs a CI-friendly small model or a cached HF snapshot.

**Impact:** new files under `tests/`. Write `tests/test_bm25_search.py` *before* touching B2.

---

## G. Missing functionality

| Gap | Why it matters | Priority |
|---|---|---|
| **Delete consistency** | Deleting a paper leaves BM25 postings and cached retrievals stale until the next invalidate — deleted papers keep getting cited. Wire delete into `_post_ingest_refresh`. | **High** |
| **Reindex / model-migration job** | No supported path to change embedding model or chunking (see C4/C5). | **High** |
| **Stale-job reaper** | Jobs killed by a restart stay `running` forever; clients poll indefinitely. | **Medium** |
| **Per-user cost/usage view** | Cannot answer "who is spending the quota" (see E4). | **Medium** |
| **Multi-collection / tenant separation** | One Chroma collection for everyone — no path to per-org corpora. `get_or_create_collection` is already parameterized, so this is mostly routing. | **Medium** |
| **Answer-level audit trail** | `query_log` stores question + answer, but not the retrieved chunk ids, model version, or config hash — a past answer cannot be reproduced or explained. DDIA Ch 12: end-to-end auditability. | **Medium** |
| **Graceful degradation modes** | If the reranker or NLI model fails to load, behavior is undefined; declare explicit degraded modes (skip rerank, report `confidence: null` rather than `0.0`). | **Medium** |
| **HNSW params frozen at creation** | `ef_construction` / `M` are silently ignored on an existing collection — a documented footgun that needs a rebuild path to actually tune. | **Low** |

### Risk summary

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Cross-user data exposure (sessions, watches, feedback, reports) | High | **Certain** — reachable today by any valid key | A1 / A2 |
| Single-process SPOF; no worker can be added | High | Medium | B1 |
| Unbounded LLM spend via `/report`, `/watch/{id}/run` | High | Medium | E5, E4 |
| Silent retrieval degradation after a model or chunker change | High | Medium | C5, F2 |
| ChromaDB unhealthy → thread-pool exhaustion → total outage | Medium | Medium | C10 |
| Vector store loss with no restore path | High | Low | C3 |
| Seam defects shipping undetected (mock-only tests) | Medium | **High** — three already shipped in v2.4 | F6 |
| SQLite full-scan degradation as `query_log` grows | Medium | Medium | D4 |
| Schema change cannot be deployed to an existing DB | Medium | **Certain** once A1/C4/C5 start | F5 |
| BM25 p99 cliff after every ingest | Medium | High | B3, B6 |
| SSRF via DNS rebinding | High | Low (URLs come from arXiv/S2/OpenAlex) | E3 |

---

## Verified strengths — do not "fix" these

Checked in code, not taken from the README. Listed so a later pass does not propose work that is already done.

- **Layered timeouts, real values:** 5s ChromaDB (`_chroma_call`), 60s `LLM_REQUEST_TIMEOUT_S`, 300s `LLM_STREAM_TIMEOUT_S`, 300s `AGENT_TIMEOUT`, 90s `AGENT_REFLEXION_BUDGET_S` — and `config.py:321-341` documents *why* the reflexion budget does not bound the failover chain. That comment is better than most systems' incident postmortems.
- **SSRF controls, mostly complete:** scheme allow-list, DNS-resolved private/loopback/link-local rejection, 50 MB streaming cap, and manual per-hop redirect re-validation via `_NoRedirectHandler`. Only the DNS TOCTOU (E3) is open.
- **Circuit breaker + failover:** per-`(provider, model)` keys with 60s cooldown, cross-vendor failover, guaranteed backstop. C10 should reuse this, not reinvent it.
- **Concurrency discipline:** double-checked locking on every singleton, in-flight embedding dedup (`embeddings.py:104`), copy-on-read *and* copy-on-write for cached retrievals, `_ingest_lock` for URL dedup.
- **Auth primitives:** `hmac.compare_digest` with no early exit across the key set, UTF-8 encoding so a non-ASCII header fails auth instead of 500ing, fail-closed `ADMIN_API_KEY`, and a production boot guard that refuses to bind `0.0.0.0` unauthenticated. The primitives are right — A1 is a *usage* gap in four routers, not a broken mechanism.
- **Faithfulness verification:** claim-level NLI with contradiction detection and abstention is rare in production RAG and is the system's strongest differentiator.

---

## H. Considered and declined

Two recommendations were checked against the code and rejected. Recorded so they are not re-proposed.

### H1. Partition the ChromaDB collection per language — **do not do this**

The proposal: split `scientific_papers` (`config.py:242`) into per-language collections so a Hindi query searches a smaller graph.

**Why not:** BGE-M3 was chosen precisely because it embeds 100+ languages into one shared vector space. Cross-lingual retrieval — a Hindi query surfacing an English paper — is the product's headline capability, not an incidental benefit. Per-language partitioning either deletes that capability or forces a scatter-gather across every partition on every query, which is strictly worse than the single graph it replaced (§6 of the DDIA framing: local secondary indexes make reads expensive). The stated trade-off, "cross-language retrieval becomes harder," understates it: the feature stops working.

**What is defensible:** partitioning by *domain* or *tenant*, where queries genuinely do not cross the boundary. That is the multi-collection item already in §G, and `get_or_create_collection` is parameterized for it.

### H2. Replace the custom BM25 with Elasticsearch or Meilisearch

Externalizing the vector store and the caches (B1) is right. Externalizing BM25 is not, for this corpus.

**Why not:** the current index uses a Unicode-aware `[\p{L}\p{M}\p{N}]+` tokenizer specifically so Devanagari, Tamil, and the other Indic scripts tokenize correctly. Meilisearch's Indic tokenization is materially weaker; Elasticsearch needs per-language analyzer configuration to match what 20 lines of `regex` already do here. The swap risks a silent multilingual retrieval regression — the hardest kind to notice, given F2 (no quality gate) — in exchange for solving a scaling problem that B2 solves in one file with no new service.

**Revisit when:** the corpus exceeds roughly 10⁵ chunks *and* B2 has been shown insufficient by the metrics from F1.

---

## Priority ordering

**Phase 0 — unblock everything else (do these first, they gate the rest):**

1. **F5** — migration framework. A1, C4, and C5 all add columns; without this they cannot ship safely.
2. **D4** — six `CREATE INDEX` statements. Smallest diff in the document.
3. **F6 (partial)** — `tests/test_bm25_search.py` before rewriting BM25.

**Phase 1 — correctness and safety:**

4. **A1 / A2** — data isolation (security bug, contradicts a shipped claim)
5. **C5** — version-stamp embeddings (cheap now, unfixable-in-place later)
6. **E5** — rate-limit `POST /report` and `POST /watch/{id}/run` (unbounded LLM spend)
7. **B2 / B3** — BM25 inverted index + incremental update (single file, largest latency win)
8. **C10** — Chroma circuit breaker + embedding fallback → sparse-only degraded mode
9. **F1** — stage metrics (makes everything else measurable)
10. **C3** — ChromaDB backup + restore
11. **B4** — admission control (worse than it first looked: `AGENT_TIMEOUT` is 300s)

**Phase 2 — structural, enables scale:**

12. **B1** — externalize state to Redis (vectors and cache; *not* BM25 — see H2)
13. **C1 / C2** — durable job queue + scheduler lease with fencing token
14. **C4** — ingest event log, derived-view rebuilds
15. **F2 + F6 (full)** — eval gate and integration suite in CI
16. **E2 / E3** — sandbox resource limits, SSRF DNS pinning

**Then:** B6, D1, D2, D5, E1, E4, C6–C9, C11, F3, F4, and the §G table.

---

## Overall

The retrieval and agent design is genuinely strong — hybrid + RRF + two-stage rerank + claim-level NLI is the right pipeline, the per-request defenses are layered and deliberate, and the recent fix log shows failure modes being found and closed properly.

The weaknesses cluster in three places, none of them in the retrieval logic:

1. **The state layer.** Shared mutable state lives inside one process, derived indexes have no source to be rebuilt from, and the multi-user isolation exists in the job store but not in the four routers added since.
2. **The data layer's evolvability.** No migrations, no version stamps, no secondary indexes — three cheap omissions that together make every schema-touching improvement expensive.
3. **The test layer.** Everything is mocked, so defects live in the seams between stages. The v2.4 patch notes are a list of exactly that class of bug.

Fix isolation, get a migration path, version the data, and put one real end-to-end test in place. Then get state out of the process. The rest is tuning.
