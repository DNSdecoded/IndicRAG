# IndicRAG v2.3 — CodeRabbit Review Bug Report

**Reviewed:** PR for `2.3.0-dev` branch (Phases 0–8)
**Date:** 2026-07-13
**Reviewer:** CodeRabbit automated analysis, verified manually against source

---

## Executive Summary

CodeRabbit flagged **35 findings** across the codebase. After manual verification against actual source code:

- **25 confirmed bugs** (5 critical/high, 11 medium, 9 low)
- **3 false positives**
- **2 borderline** (acknowledged gaps, low priority)
- **5 documentation/housekeeping** issues

The most impactful issues are: unguarded network calls in the ingest pipeline that can discard successfully-ingested text, a content serialization bug in OpenRouter that mangles agent payloads, and an SSRF/OOM vector in the watch runner's PDF downloader.

---

## Table of Contents

1. [Critical & High Severity](#critical--high-severity)
2. [Medium Severity](#medium-severity)
3. [Low Severity](#low-severity)
4. [Documentation & Housekeeping](#documentation--housekeeping)
5. [False Positives](#false-positives)
6. [Borderline / Acknowledged Gaps](#borderline--acknowledged-gaps)

---

## Critical & High Severity

### BUG-01: Unguarded `caption_regions()` discards text chunks on failure

**File:** `ingest.py:143-158`
**Severity:** Critical
**Category:** Stability & Availability
**CodeRabbit:** 🩺 Stability & Availability | 🔴 Critical

**Description:**
`figure_captioner.caption_regions()` is a network/VLM call with no error handling. If it raises (network timeout, VLM API error, malformed image), the exception propagates out of `_build_paper_chunks`, discarding all already-built and deduped body-text chunks for that paper. An optional multimodal enhancement failure takes down otherwise-successful core ingestion.

**Current code:**
```python
if config.ENABLE_MULTIMODAL_INGEST and figures:
    import figure_captioner
    for fig in figure_captioner.caption_regions(figures, paper_id):  # unguarded network call
        all_chunks.append(fig["text"])
        all_metadata.append({...})
        all_ids.append(f"{paper_id}_{fig['chunk_type']}_{chunk_counter}")
        chunk_counter += 1
```

**Impact:** A single failed VLM captioning request causes the entire paper to lose all its text chunks from the vector store. In a batch ingest of 100 papers, one transient network hiccup can silently drop a paper's entire content.

**Recommended fix:**
```python
if config.ENABLE_MULTIMODAL_INGEST and figures:
    import figure_captioner
    try:
        captions = figure_captioner.caption_regions(figures, paper_id)
    except Exception as cap_err:
        logger.error(f"Figure captioning failed for {paper_id}, continuing without figure chunks: {cap_err}")
        captions = []
    for fig in captions:
        all_chunks.append(fig["text"])
        all_metadata.append({...})
        all_ids.append(f"{paper_id}_{fig['chunk_type']}_{chunk_counter}")
        chunk_counter += 1
```

---

### BUG-02: Unguarded `extract_regions()` marks successful ingestion as failed

**File:** `ingest.py:302-305`
**Severity:** High
**Category:** Stability & Availability
**CodeRabbit:** 🩺 Stability & Availability | 🟠 Major

**Description:**
`figure_captioner.extract_regions()` performs CPU-side PDF figure extraction. If it fails (corrupted PDF, image processing error), the exception crashes the entire `ingest_paper` call. The outer per-paper handler in `ingest_directory` catches this and marks the paper as "failed" — even though `pdf_utils.process_pdf()` already succeeded and produced valid text sections.

**Current code (in `ingest_pdf` and `_extract_worker`):**
```python
figures = None
if config.ENABLE_MULTIMODAL_INGEST:
    import figure_captioner
    figures = figure_captioner.extract_regions(pdf_path, paper_id)  # unguarded
```

**Impact:** Papers with corrupted embedded images are marked as failed ingestion even though their text content is fully extractable. This silently reduces the usable corpus size.

**Recommended fix:**
```python
figures = None
if config.ENABLE_MULTIMODAL_INGEST:
    import figure_captioner
    try:
        figures = figure_captioner.extract_regions(pdf_path, paper_id)
    except Exception as ext_err:
        logger.warning(f"Figure extraction failed for {paper_id}, continuing without figures: {ext_err}")
        figures = None
```

---

### BUG-03: `str(contents)` mangles Gemini Content/Part objects in OpenRouter

**File:** `providers/openrouter.py:25-29`
**Severity:** High
**Category:** Functional Correctness
**CodeRabbit:** 🎯 Functional Correctness | 🟡 Minor

**Description:**
When `contents` is a list of `google.genai` `Content`/`Part` objects (which the agent pipeline sends from `answer_generator.py` and `figure_captioner.py`), `str(contents)` produces a Python repr string like `[Content(parts=[Part(text='...')], role='user')]` — not the actual prompt text. The OpenRouter backend receives a meaningless string instead of the real user query.

**Current code:**
```python
def _to_messages(contents, gen_config) -> list[dict]:
    messages = []
    sys_inst = getattr(gen_config, "system_instruction", None)
    if sys_inst:
        messages.append({"role": "system", "content": str(sys_inst)})
    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    else:
        # contents is a list of parts/strings — flatten to text
        messages.append({"role": "user", "content": str(contents)})  # BUG: repr, not text
    return messages
```

**Impact:** Any agent request routed through OpenRouter produces garbage prompts. The model receives Python object representations instead of the actual query, yielding useless answers. This is a silent data corruption — no error is raised.

**Recommended fix:**
```python
else:
    # contents is a list of Content/Part objects — extract actual text
    parts = []
    for item in contents:
        if hasattr(item, "parts"):
            for part in item.parts:
                if hasattr(part, "text") and part.text:
                    parts.append(part.text)
        elif hasattr(item, "text") and item.text:
            parts.append(item.text)
        elif isinstance(item, str):
            parts.append(item)
    messages.append({"role": "user", "content": "\n".join(parts) if parts else str(contents)})
```

---

### BUG-04: SSRF and OOM in watch runner PDF downloader

**File:** `watch_runner.py:27-37`
**Severity:** High
**Category:** Security & Privacy
**CodeRabbit:** 🔒 Security & Privacy | 🟡 Minor

**Description:**
`_download_pdf()` has two security issues:
1. **No URL scheme validation** — accepts `file://`, `ftp://`, `gopher://` and internal network URLs. `urllib.request.urlopen` will happily open `file:///etc/passwd` or internal service endpoints (SSRF).
2. **No response size limit** — `resp.read()` loads the entire response into memory with no cap. A malicious or misconfigured URL could serve a multi-gigabyte response and OOM the process.

**Current code:**
```python
def _download_pdf(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IndicRAG/2.0"})
        fd, path = tempfile.mkstemp(suffix=".pdf")
        with urllib.request.urlopen(req, timeout=30) as resp, os.fdopen(fd, "wb") as f:
            f.write(resp.read())  # unbounded read
        return path
    except Exception as e:
        logger.warning(f"[Watch] PDF download failed {url}: {e}")
        return None
```

**Impact:** A crafted watch topic pointing to a `file://` URL could read local files. A URL serving a large response could crash the server with OOM.

**Recommended fix:**
```python
_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB

def _download_pdf(url: str) -> str | None:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        logger.warning(f"[Watch] Rejected non-HTTP(S) URL: {url}")
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IndicRAG/2.0"})
        fd, path = tempfile.mkstemp(suffix=".pdf")
        with urllib.request.urlopen(req, timeout=30) as resp, os.fdopen(fd, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_PDF_BYTES:
                    logger.warning(f"[Watch] PDF too large ({total} bytes), aborting: {url}")
                    os.close(fd)
                    os.unlink(path)
                    return None
                f.write(chunk)
        return path
    except Exception as e:
        logger.warning(f"[Watch] PDF download failed {url}: {e}")
        return None
```

---

### BUG-05: Race condition in `_init_backends()` singleton init

**File:** `llm_client.py:39-42`
**Severity:** High
**Category:** Stability & Availability
**CodeRabbit:** 🩺 Stability & Availability | 🟠 Major

**Description:**
`_init_backends()` checks-then-sets the module-global `_backends` dict without any lock. Concurrent first requests can each see `_backends` as empty, construct their own `GeminiBackend()`/`OpenRouterBackend()` instances, and race to assign. Since these backends hold internal pooled state (key rotation, HTTP clients), using different instances mid-request is risky.

**Current code:**
```python
def _init_backends() -> None:
    global _backends
    if not _backends:
        _backends = {"gemini": GeminiBackend(), "openrouter": OpenRouterBackend()}
```

**Impact:** Duplicate backend construction wastes resources. If two concurrent requests each get a different backend instance, circuit breaker state and key rotation are not shared — the failover logic may not work correctly.

**Recommended fix:**
```python
_backends_lock = threading.Lock()

def _init_backends() -> None:
    global _backends
    if not _backends:
        with _backends_lock:
            if not _backends:
                _backends = {"gemini": GeminiBackend(), "openrouter": OpenRouterBackend()}
```

---

## Medium Severity

### BUG-06: Untitled chunks falsely skipped by contradiction detection

**File:** `contradiction.py:61-79`
**Severity:** Medium
**Category:** Functional Correctness
**CodeRabbit:** 🎯 Functional Correctness | 🟡 Minor

**Description:**
`_title()` falls back to the literal string `"Unknown"` when no title metadata is present. If two distinct (but both untitled) chunks are compared, `ti == tj` evaluates to `True` and the pair is skipped via the same-paper guard — even though they may be genuinely different sources that actually contradict.

**Current code:**
```python
def _title(i: int) -> str:
    m = metas[i] if i < len(metas) else None
    return ((m or {}).get("title") or "Unknown").strip() or "Unknown"

for i, j in itertools.combinations(range(len(items)), 2):
    ti, tj = _title(i), _title(j)
    if ti == tj:
        continue  # same-paper guard — false positive for two untitled papers
```

**Impact:** Contradictions between chunks from different untitled papers are never detected. Low practical impact since ingested chunks normally carry titles, but the edge case silently disables detection.

**Recommended fix:**
```python
def _title(i: int) -> str:
    m = metas[i] if i < len(metas) else None
    return ((m or {}).get("title") or f"Unknown-{i}").strip() or f"Unknown-{i}"
```

---

### BUG-07: Contradiction check runs on untrimmed chunks

**File:** `agent/nodes/answer_generator.py:37-48`
**Severity:** Medium
**Category:** Functional Correctness
**CodeRabbit:** 🎯 Functional Correctness | 🟡 Minor

**Description:**
`find_contradictions()` receives the full untrimmed `chunks` and `metas` lists, not the subset that was actually formatted into the prompt by `rag.format_context()`. The formatter can stop early on chunk/length limits (`AGENT_MAX_CONTEXT_CHUNKS`), so contradictions may be detected between sources the LLM never saw.

**Current code:**
```python
formatted_context, chunks_used = rag.format_context(
    chunks, metas,
    max_chunks=config.AGENT_MAX_CONTEXT_CHUNKS,
    max_length=config.AGENT_MAX_CONTEXT_LENGTH,
)
# ...
if config.CONTRADICTION_DETECT_ENABLE:
    cons = contradiction.find_contradictions(chunks, metas)  # ALL chunks, not chunks_used
```

**Impact:** The answer generator may add a "contradiction detected" note about sources that were truncated from the prompt. The LLM has no context to present both sides, making the note misleading.

**Recommended fix:**
```python
if config.CONTRADICTION_DETECT_ENABLE:
    cons = contradiction.find_contradictions(chunks[:chunks_used], metas[:chunks_used])
```

---

### BUG-08: IndexError on empty choices in OpenRouter streaming

**File:** `providers/openrouter.py:120-130`
**Severity:** Medium
**Category:** Stability & Availability
**CodeRabbit:** 🩺 Stability & Availability | 🟡 Minor

**Description:**
`chunk.choices[0]` will raise `IndexError` if a streaming chunk has an empty `choices` array. The OpenRouter/OpenAI streaming protocol can send chunks with empty `choices` (e.g., the final `[DONE]` signal in some implementations).

**Current code:**
```python
for chunk in client.chat.completions.create(...):
    delta = chunk.choices[0].delta  # IndexError if choices is []
```

**Impact:** Streaming responses from OpenRouter may crash with an unhandled `IndexError` instead of gracefully completing.

**Recommended fix:**
```python
for chunk in client.chat.completions.create(...):
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
```

---

### BUG-09: `model`/`provider` silently dropped in `/query` endpoint

**File:** `routes/query.py:44-45`
**Severity:** Medium
**Category:** Functional Correctness
**CodeRabbit:** 🎯 Functional Correctness | 🟡 Minor

**Description:**
`QueryRequest` validates `model` and `provider` fields from the allowlist, but both `/query` and `/query/stream` handlers silently drop them before calling `rag.answer_question` / `rag.prepare_query_for_stream`. The fields have no effect.

**Current code:**
```python
class QueryRequest(BaseModel):
    model: Optional[str] = Field(None, description="LLM model id from the /models allowlist...")
    provider: Optional[str] = Field(None, description="LLM provider override (gemini|openrouter)...")
    # ...

result = await run_in_threadpool(
    rag.answer_question,
    user_query=body.question,
    strategy=body.strategy,
    top_k=top_k,
    filter_dict=build_paper_filter(body.paper_ids),
    # body.model and body.provider are NOT passed
)
```

**Impact:** Users can set model/provider in the request body but the server ignores them entirely for standard RAG queries. The UI model dropdown has no effect outside agent mode.

**Recommended fix:** Either thread `model`/`provider` through to `rag.answer_question` (which also needs parameter additions), or remove them from `QueryRequest` with a clear comment explaining the limitation.

---

### BUG-10: UI model dropdown not sent in `/chat/stream` payload

**File:** `static/index.html:1112-1116`
**Severity:** Medium
**Category:** Functional Correctness
**CodeRabbit:** 🎯 Functional Correctness | 🟡 Minor

**Description:**
The standard `/chat/stream` request payload omits the `model` field from the `#modelSelect` dropdown. The agent endpoint includes it, but the chat endpoint does not. Users who select a model in the UI see no effect on standard chat.

**Agent payload (correct):**
```javascript
body: JSON.stringify({ question: text, session_id: sessionId, strategy,
                       model: document.getElementById('modelSelect').value || null }),
```

**Chat payload (missing model):**
```javascript
body: JSON.stringify({ message: text, session_id: sessionId, strategy, top_k: 8,
                       paper_ids: paperIds.length ? paperIds : null }),
```

**Impact:** UI model selection is silently ignored for standard chat mode. This also requires a backend change — `ChatRequest` in `routes/chat.py` doesn't define a `model` field.

---

### BUG-11: Phantom `crop_path` on write failure

**File:** `figure_captioner.py:181-195`
**Severity:** Medium
**Category:** Data Integrity & Integration
**CodeRabbit:** 🗄️ Data Integrity & Integration | 🟡 Minor

**Description:**
If `crop_path.write_bytes()` fails, the chunk still records `crop_path` as if the file exists. Consumers that later try to render or link the figure (e.g., the UI rendering `<img src="/figures/...">`) will get a 404.

**Current code:**
```python
try:
    crop_path.write_bytes(r["png"])
except Exception as e:
    logger.warning("crop write failed %s: %s", crop_path, e)
    # falls through — crop_path still recorded

chunks.append({
    "crop_path": str(crop_path),  # points to non-existent file
    # ...
})
```

**Impact:** The UI renders broken figure thumbnails with 404 errors instead of gracefully omitting the image.

**Recommended fix:**
```python
crop_written = True
try:
    crop_path.write_bytes(r["png"])
except Exception as e:
    logger.warning("crop write failed %s: %s", crop_path, e)
    crop_written = False

chunks.append({
    "crop_path": str(crop_path) if crop_written else None,
    # ...
})
```

---

### BUG-12: No `safeHref()` on figure URLs in `rag.py`

**File:** `rag.py:38-45`
**Severity:** Medium
**Category:** Security & Privacy
**CodeRabbit:** 🔒 Security & Privacy | 🟡 Minor

**Description:**
`_crop_url()` returns a URL constructed from filesystem paths with no HTML/attribute sanitization. While `Path.resolve().relative_to()` provides path traversal protection, if the URL is rendered in HTML attributes without escaping, it could be exploited via crafted `crop_path` values containing `"` or event handlers.

**Current code:**
```python
def _crop_url(crop_path: str) -> Optional[str]:
    from pathlib import Path
    try:
        rel = Path(crop_path).resolve().relative_to(config.FIGURES_DIR.resolve())
        return "/figures/" + rel.as_posix()  # no sanitization
    except (ValueError, OSError):
        return None
```

**Impact:** Potential XSS if a malicious `crop_path` value contains HTML-breaking characters that survive into `<img src="...">` or `<a href="...">` attributes.

**Recommended fix:** Apply `safeHref()` or `urllib.parse.quote()` to the URL before returning it.

---

### BUG-13: `async def get_models` blocks event loop with sync HTTP call

**File:** `routes/models.py:76-78`
**Severity:** Medium
**Category:** Stability & Availability
**CodeRabbit:** 🩺 Stability & Availability | 🟡 Minor

**Description:**
`get_models()` is declared `async` but calls synchronous `list_models()` → `model_supports_tools()` → `_catalog()` → `_fetch_openrouter_catalog()` → `httpx.get(url, timeout=10)`. This blocks the FastAPI event loop for up to 10 seconds on cache miss.

**Current code:**
```python
@router.get("/models", tags=["Models"])
async def get_models():
    return {"models": list_models(), "default": (config.LLM_SELECTABLE_MODELS or [None])[0]}
```

**Impact:** The first request to `/models` (or after `MODELS_CACHE_TTL` expiry) blocks all concurrent async requests for up to 10 seconds.

**Recommended fix:** Change to `def get_models():` (sync) so FastAPI runs it in the threadpool, or wrap the blocking call in `run_in_threadpool`.

---

### BUG-14: `language` param dead in `plan_sections()`

**File:** `report_runner.py:27-39`
**Severity:** Medium
**Category:** Functional Correctness
**CodeRabbit:** 🎯 Functional Correctness | 🟡 Minor

**Description:**
`plan_sections()` accepts a `language: str = "en"` parameter but never includes it in the LLM planning prompt. Section titles are always planned in English regardless of the requested language.

**Current code:**
```python
def plan_sections(topic: str, language: str = "en", max_sections: int = None) -> list[str]:
    prompt = (
        f"You are planning a literature-review report on: {topic!r}.\n"
        f"Propose at most {max_sections} section titles ..."
        # language parameter is never mentioned
    )
```

**Impact:** Literature review reports always have English section titles even when the user requests Hindi, Tamil, or other Indic languages.

---

### BUG-15: `make_grounding_scorer` only catches `ImportError`

**File:** `docs/Eval/evaluate.py:121-126`
**Severity:** Medium
**Category:** Stability & Availability
**CodeRabbit:** 🩺 Stability & Availability | 🟠 Major

**Description:**
The `except ImportError` only catches missing `sentence_transformers`. If `CrossEncoder()` constructor fails with network errors, disk errors, or corrupt model weights, the exception propagates and crashes the eval instead of falling back to Jaccard.

**Current code:**
```python
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("  [warn] sentence-transformers unavailable — falling back to Jaccard")
    return jaccard, GROUNDING_THRESHOLD
model = CrossEncoder("BAAI/bge-reranker-v2-m3")  # can raise OSError, RuntimeError, etc.
```

**Impact:** Eval crashes in offline/CI environments where `sentence_transformers` is installed but the model weights aren't cached.

**Recommended fix:**
```python
try:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
except Exception as e:
    print(f"  [warn] cross-encoder unavailable ({e}) — falling back to Jaccard")
    return jaccard, GROUNDING_THRESHOLD
```

---

### BUG-16: Grounding judge not recorded in eval metrics

**File:** `docs/Eval/evaluate.py:254-269`
**Severity:** Medium
**Category:** Functional Correctness
**CodeRabbit:** 🎯 Functional Correctness | 🟡 Minor

**Description:**
`evaluate()` never records which grounding judge (`jaccard` vs `cross-encoder`) or threshold was used. The `markdown_report` hardcodes "Citation grounding uses token Jaccard similarity (threshold 0.15)" regardless of actual configuration.

**Impact:** Reports generated with `--grounding-judge cross-encoder` misrepresent their own methodology.

---

## Low Severity

### BUG-17: Unsafe `m["content"]` access in chat session listing

**File:** `routes/chat.py:196`
**Severity:** Low
**Category:** Functional Correctness
**CodeRabbit:** 🎯 Functional Correctness | 🟡 Minor

**Description:**
`m["content"]` (direct dict access) is used alongside `m.get("role")` (safe access) on the same expression. If a message dict has `role == "user"` but lacks a `content` key (e.g., corrupted persistence), this raises an unhandled `KeyError`.

**Recommended fix:**
```python
first_user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
```

---

### BUG-18: `_circuit_breaker` dict has no explicit synchronization

**File:** `llm_client.py:21,103-117,160-183`
**Severity:** Low
**Category:** Stability & Availability
**CodeRabbit:** 🩺 Stability & Availability | 🔵 Trivial

**Description:**
`_circuit_breaker` is read, deleted, and written from concurrent request threads without a lock. CPython's GIL makes individual dict operations atomic, so this won't corrupt the dict, but interleavings can cause a thread to skip a path another thread just reopened (soft timing correctness issue).

**Impact:** Minor race condition — a thread may attempt a failing backend that another thread just re-opened, or vice versa. No data corruption or crashes.

---

### BUG-19: `next_client_idx()` skips `_ensure_pool()`

**File:** `providers/gemini.py:46-48`
**Severity:** Low
**Category:** Stability & Availability
**CodeRabbit:** 🩺 Stability & Availability | 🔵 Trivial

**Description:**
`next_client_idx()` calls `next(self._index)` without first calling `_ensure_pool()`. Currently safe because both call sites access `.pool` first (which triggers lazy init), but if this method is ever called standalone, it will hit an empty `itertools.cycle([])` and raise `StopIteration`.

**Recommended fix:**
```python
def next_client_idx(self) -> int:
    self._ensure_pool()
    with self._lock:
        return next(self._index)
```

---

### BUG-20: Silent exception swallowing in `_with_cache`

**File:** `providers/gemini.py:85`
**Severity:** Low
**Category:** Maintainability
**CodeRabbit:** 📐 Maintainability & Code Quality | 🔵 Trivial

**Description:**
The `except Exception` in `_with_cache` catches all errors and returns the original `gen_config` with no logging. Cache failures (network, auth, quota) are completely invisible in production.

**Recommended fix:**
```python
except Exception as exc:
    logger.debug("Gemini context caching skipped: %s", exc)
    return gen_config
```

---

### BUG-21: No done callback on `watch_task`

**File:** `api_server.py:50-63`
**Severity:** Low
**Category:** Stability & Availability
**CodeRabbit:** 🩺 Stability & Availability | 🔵 Trivial

**Description:**
The `asyncio.create_task(watch_runner.watch_loop())` has no `add_done_callback`. If `watch_loop()` crashes mid-run, the exception is silently lost until garbage collection logs a "Task exception was never retrieved" warning.

**Recommended fix:**
```python
watch_task = asyncio.create_task(watch_runner.watch_loop())

def _log_watch_task_result(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        logger.error("[Watch] schedule loop died unexpectedly", exc_info=task.exception())

watch_task.add_done_callback(_log_watch_task_result)
```

---

### BUG-22: No per-user ownership enforcement on watch endpoints

**File:** `routes/watch.py:100-146`
**Severity:** Low (UUID obscurity mitigates)
**Category:** Security & Privacy

**Description:**
None of the watch endpoints verify that the authenticated user owns the watch they're accessing. Any API key holder can view, run, read digests of, or delete any other user's watch by guessing the UUID.

**Impact:** Low in practice because watch IDs are UUIDs (hard to guess), but violates principle of least privilege. If watch IDs become predictable or are logged, cross-user data exposure becomes trivial.

---

## Documentation & Housekeeping

### DOC-01: Generated eval report committed to repo

**File:** `docs/Eval/eval_report.json`
**Issue:** Timestamped machine-generated artifact is committed. It becomes stale after every eval run and creates noisy diffs.
**Fix:** Add to `.gitignore` or replace with a static example.

### DOC-02: Missing fenced code block language tag

**File:** `docs/superpowers/specs/2026-07-12-openrouter-provider-design.md:94`
**Issue:** Bare ` ``` ` without language identifier. Should be ` ```dotenv ` for proper syntax highlighting.
**Fix:** Add `dotenv` language tag.

### DOC-03: Inaccurate eval hook description in spec

**File:** `docs/superpowers/specs/2026-07-12-openrouter-provider-design.md:148-152`
**Issue:** Spec says `--model`/`--provider` flags are "threaded through to the queries it issues" but they only label pre-generated results. The eval harness does not generate answers.
**Fix:** Revise to accurately state that flags label results for reporting.

### DOC-04: `FunctionCallingConfig` not translated to OpenRouter

**File:** `providers/openrouter.py:88-105`
**Issue:** Gemini's `FunctionCallingConfig(mode=ANY/NONE)` is silently dropped. The module docstring acknowledges this as a "stated behavior gap."
**Severity:** Informational — acknowledged limitation.

### DOC-05: `AGENT_MAX_TOKENS` default mismatch

**File:** `.env.example:51`
**Issue:** `.env.example` says `AGENT_MAX_TOKENS=4096` but `config.py` default is `8192`. The example is outdated.
**Fix:** Update `.env.example` to `8192`.

---

## False Positives

### FP-01: Missing `answer_confidence`/`abstained` in AgentState init

**File:** `routes/agent.py:96-113`
**CodeRabbit claim:** Initialize `answer_confidence=None, abstained=False` in the constructor.

**Why it's a false positive:**
`AgentState` is a `TypedDict`. Missing keys are perfectly valid — they're populated later by the finalizer node and read via `.get()` with defaults (`result.get("abstained", False)`). Adding them to the initial state is unnecessary overhead with no behavioral change. No `KeyError` is possible in current code.

---

### FP-02: Duplicate ONNX loading logic between `rerank.py` and `verify.py`

**File:** `rerank.py:19-30`
**CodeRabbit claim:** Factor duplicated loading logic into a shared helper.

**Why it's a false positive:**
The code is correct. The broad `except Exception` is intentional for ONNX fallback. The two loaders have different model names (`RERANK_MODEL_NAME` vs `NLI_MODEL_NAME`) and cache subdirectories. Factoring into a helper is a style preference, not a bug. Both callers are already thread-safe with proper double-checked locking.

---

### FP-03: VERSION still says `2.3.0-dev`

**File:** `config.py:487`
**CodeRabbit claim:** Update VERSION to `"2.3.0"` for the release.

**Why it's a false positive:**
This is a dev branch (`2.3.0-dev`). The version string is intentional pre-release notation. It should be updated to `"2.3.0"` only when the branch is merged to `main` for release — not during active development.

---

## Borderline / Acknowledged Gaps

### BG-01: `FunctionCallingConfig` not translated

**File:** `providers/openrouter.py:88-105`
**Status:** Acknowledged in module docstring as a "stated behavior gap." When the agent sets `mode=ANY` (must call tools) or `mode=NONE` (no tools), OpenRouter always defaults to AUTO. Low practical impact since the agent path primarily uses Gemini, but limits OpenRouter tool-calling reliability.

### BG-02: `onnx_ce.py` `_QFILE` path may mismatch export output

**File:** `onnx_ce.py:21-22`
**Status:** `_QFILE = "onnx/model_quint8_avx2.onnx"` assumes the sentence-transformers export function writes into an `onnx/` subdirectory. The internal code is consistent (existence check and load both use `_QFILE`), but correctness depends on the third-party export function's actual output layout. Needs validation against the installed `sentence-transformers` version.

---

## Priority Fix Order

1. **BUG-01** (ingest caption_regions) + **BUG-02** (ingest extract_regions) — highest blast radius, easiest fix
2. **BUG-03** (OpenRouter content translation) — silent data corruption
3. **BUG-04** (watch_runner SSRF/OOM) — security vulnerability
4. **BUG-05** (backend init race) — thread safety
5. **BUG-08** (OpenRouter empty choices) — streaming crash
6. **BUG-06** (contradiction untitled) + **BUG-07** (untrimmed chunks) — correctness
7. **BUG-09** + **BUG-10** (model not threaded) — feature completeness
8. **BUG-11** + **BUG-12** (crop_path, safeHref) — data integrity / security
9. **BUG-13** (async blocking) + **BUG-14** (language dead param) — correctness
10. **BUG-15** + **BUG-16** (eval grounding) — eval accuracy
11. **BUG-17–22** — low severity, quick wins
12. **DOC-01–05** — housekeeping
