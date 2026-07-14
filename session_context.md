# Session Context — 2026-07-11

**Branch:** `2.3.0-dev`  ·  **Repo:** `F:\Indicragv2\IndicRAG`  ·  **Plan:** `F:\Indicragv2\implementation_plan.md`

Handoff doc for resuming after compaction. Everything shipped this session is committed; working tree carries only the local `.env` change (gitignored).

---

## What happened this session

### 1. Phase 8 (OpenRouter) — planned only, deferred to LAST
Added **Phase 8** to `implementation_plan.md`: OpenRouter as a first-class secondary LLM provider + user-selectable models. Design settled:
- Provider abstraction behind `generate_with_failover` / `llm_generate_stream`; OpenRouter backend returns a **Gemini-shaped response shim** so the ~6 agent call sites stay unchanged.
- Curated `.env` allowlist `LLM_SELECTABLE_MODELS` (e.g. `gemini-3.5-flash,anthropic/claude-haiku,openai/gpt-5.4-nano`) drives the UI dropdown; live OpenRouter catalog only enriches capability metadata.
- `LLM_PROVIDER` / `LLM_FALLBACK_PROVIDER` make either vendor primary or fallback.
- Scoped 6 d → **8 d** after folding 6 gaps (re-export constraint at `rag.py:577`, per-backend error classifiers, `(provider,model)` circuit key, net-new streaming cross-provider failover, `google-genai` stays a hard dep, `SAFETY_SETTINGS` dropped on OpenRouter, pre-commit capability gating). **No code written — deferred behind Phases 5–7.**

### 2. Phase 5 (contradiction detection) — VALIDATED end-to-end
Ran the real mDeBERTa NLI model (offline) against the plan's stated "over-firing on paraphrase" risk. Result: opposing claims **0.724**, paraphrase **0.019**, unrelated **0.126** — threshold 0.6 well-placed, wide separation, no false positives. Ships as-is, **no code change**. Memory updated. (Real-model check kept in scratchpad; repo tests deliberately mock NLI for offline CI.)

### 3. Phase 6 (watch-a-topic) — building incrementally
Decisions: **manual-trigger first** (schedule loop deferred) + **full-PDF arXiv ingest** (no-PDF → abstract-only). Single-worker, no Redis/Celery.

- ✅ **Increment 1** — `f70ae57` — `watches` table + CRUD in `persistence.py` (`save_watch`/`get_watch`/`list_watches`/`due_watches`/`delete_watch`). Mirrors job/prefs pattern. Tests: `tests/test_watch_persistence.py` (6).
- ✅ **Increment 2** — `b5eb6dd` — `routes/watch.py` CRUD (`POST/GET/GET{id}/DELETE /watch`), gated by `config.WATCH_ENABLE` (404 when off). Router wired in `api_server.py`. Knobs `WATCH_ENABLE` + `WATCH_DEFAULT_CADENCE` in `config.py` + `.env.example`. Tests: `tests/test_watch_routes.py` (8).

18 watch tests green; no regressions.

---

## NEXT: Increment 3 — `run_watch` core + `POST /watch/{id}/run`

Design (contracts already confirmed):
1. `execute_arxiv_search(topic, max_results)` (+ `execute_open_access_search` fallback) → hits `{text,title,source,pdf_url,arxiv_id}`.
2. Filter hits whose `arxiv_id ∉ watch.seen_ids`.
3. Per new hit: download `pdf_url` → temp file → `ingest.ingest_pdf(path, paper_id=arxiv_id, metadata={title,source})` → returns `(n_chunks, title)`; `n_chunks > 0` = genuinely new (built-in paper_id/content-hash dedup). No `pdf_url` → abstract-only chunk.
4. Summarize the newly-ingested papers into a **cited** digest (LLM).
5. Update `seen_ids += new arxiv_ids`, `latest_digest`, `last_run=now`, advance `next_run = now + cadence`; `save_watch`.
6. Endpoint: `POST /watch/{id}/run` — sync via `run_in_threadpool` (ingest blocks the event loop otherwise).
7. Tests: mock `execute_arxiv_search`, `ingest_pdf`, and the LLM summarizer; assert dedup, `seen_ids` growth, digest storage, `next_run` advance.

Then Increment 4 (asyncio lifespan loop over `due_watches`) and Increment 5 (`GET /watch/{id}/digest` + UI).

---

## Gotchas / how to test
- **Restart the server** after `.env` changes — `config.py` reads env only at import.
- `.env` has `WATCH_ENABLE=true` locally (gitignored, NOT committed). `/watch` returns `{"detail":"Topic watches are not enabled"}` (404) when off — that's the gate working.
- Run tests: `python -m pytest tests/test_watch_routes.py tests/test_watch_persistence.py -q` (needs `PYTHONPATH` = repo root for standalone scripts; pytest handles it via conftest which also redirects the SQLite DB to a temp file).
- **GateGuard** fact-forcing hook fires on first-touch of each file + first Bash. Disable with `export ECC_GATEGUARD=off` or `ECC_DISABLED_HOOKS += pre:edit-write:gateguard-fact-force`.
- No Co-Authored-By Claude lines in commits (user preference).

## Watch dict schema
`{id, user_id, topic, language, cadence(daily|weekly|monthly), seen_ids:[], latest_digest, next_run(ISO-8601 UTC|null), last_run, created_at}` — stored as json in `watches.data`; `user_id`/`next_run`/`last_run`/`created_at` denormalized as columns.
