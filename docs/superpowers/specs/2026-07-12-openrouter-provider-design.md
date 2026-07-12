# Phase 8 — Secondary LLM Provider (OpenRouter) — Design

**Status:** Approved
**Date:** 2026-07-12
**Basis:** `implementation_plan.md` Phase 8 (full scope, all 8 tasks)

## Goal

Add OpenRouter as a first-class second LLM provider alongside Gemini for
cross-vendor failover (a Google-wide outage no longer means total LLM
failure), model breadth, and a user-selectable model/provider per request.

## Constraints that shape the design

- `rag.py:473` re-exports `llm_generate_stream = llm_client.llm_generate_stream`
  and `rag.py:577` re-exports `generate_with_failover = llm_client.generate_with_failover`.
  ~52 tests patch `rag.generate_with_failover`. Both module-level symbols and
  their call signatures MUST be preserved.
- Every agent node that reads an LLM response expects a Gemini-shaped object:
  `resp.candidates[0].content.parts[*].function_call`, `resp.text`,
  `rag.safe_extract_text(resp)`. Confirmed call sites: `tool_selector.py:141`,
  `query_planner.py:104` (via `resp.text`), `reflexion_evaluator.py:140` (via
  `rag.safe_extract_text`).
- `agent/json_utils.extract_json` (brace-matching + truncation repair) is
  already the shared, tolerant JSON parser used by `query_planner.py` and
  `reflexion_evaluator.py` — reused, not rewritten.
- `openai`, `httpx`, and `requests` are already installed in the environment
  (confirmed via `python -c "import openai"` etc.) but none are pinned in
  `requirements.txt`. The `openai` SDK is added to `requirements.txt` and used
  for the OpenRouter backend (OpenAI-compatible Chat Completions API).

## 1. Provider abstraction & layout

New `providers/` package:
- `providers/base.py` — `LLMBackend` interface: `generate(model, contents, gen_config) -> GenResponse`,
  `generate_stream(model, contents, gen_config) -> Iterator[str]`,
  `is_transient(exc) -> bool`, `is_permanent(exc) -> bool`.
- `providers/gemini.py` — `GeminiBackend`, wraps the existing pool/circuit-breaker
  logic currently in `llm_client.py` verbatim (behavior unchanged).
- `providers/openrouter.py` — `OpenRouterBackend`, new. Uses the `openai` SDK
  client pointed at `config.OPENROUTER_BASE_URL` with `config.OPENROUTER_API_KEY`.

`llm_client.py` becomes the dispatcher: picks a backend by model-id shape
(contains `"/"` → OpenRouter slug, e.g. `anthropic/claude-haiku`; bare name →
Gemini) or an explicit `provider` argument. `generate_with_failover` and
`llm_generate_stream` keep their exact current names, module location, and
call signatures (an optional `provider: str | None = None` kwarg is additive,
not breaking).

## 2. Request/response translation + streaming

`OpenRouterBackend.generate()` calls OpenAI Chat Completions, then wraps the
result in a **Gemini-shaped response shim** exposing:
`resp.candidates[0].content.parts[0].text`,
`resp.candidates[0].content.parts[*].function_call.{name,args}`, `resp.text`.
`rag.safe_extract_text` and all agent nodes work against this shim unmodified.

Translation:
- `contents` (str/list) → OpenAI `messages`.
- `GenerateContentConfig.temperature` / `max_output_tokens` / `system_instruction`
  → OpenAI `temperature` / `max_tokens` / a `system` message.
- `types.Tool(function_declarations=[...])` → OpenAI `tools: [{"type": "function", "function": {...}}]`.
  Parameters are already plain JSON-Schema in `agent/tool_declarations.py`, so
  they pass through unchanged; only the wrapper differs.
- `thinking_config` → OpenRouter `reasoning` param when the target model's
  catalog entry supports it (checked via `supported_parameters`, task 6);
  otherwise dropped, mirroring the existing `_supports_thinking` gate for Gemma.
- `config.SAFETY_SETTINGS` has no OpenRouter equivalent — silently dropped on
  this backend. Documented as a stated behavior difference, not a bug.

Streaming: OpenAI SDK stream deltas are translated chunk-by-chunk into the
same `yield chunk.text` shape `llm_generate_stream` already produces, so
`sse_utils.py` and callers are unmodified.

## 3. Cross-provider failover + config

- **Per-backend error classifiers.** `GeminiBackend` keeps today's
  `_is_transient`/`_is_permanent` (Google-string sniffing). `OpenRouterBackend`
  gets its own classifier keyed on the `openai` SDK's exception types
  (`RateLimitError`, `APIStatusError` with `.status_code`, etc.) so OpenRouter
  429s/503s are recognized as transient instead of silently aborting failover.
- **Circuit breaker rekeyed** from bare model name to `(provider, model)` so
  a Gemini outage doesn't trip an identically-named OpenRouter model (unlikely
  in practice but the key must be provider-scoped regardless).
- **Failover order:** requested `(provider, model)` → same-provider fallback
  (existing `LLM_FALLBACK_MODEL` behavior, unchanged for Gemini) → cross-provider
  fallback using `LLM_FALLBACK_PROVIDER`'s default model, only if the same-provider
  path is fully exhausted/circuit-open.
- **Streaming cross-provider failover** is net-new work (today's streaming
  failover is same-vendor key/model rotation only). Same rule applies: failover
  only before the first chunk is emitted; a mid-stream failure re-raises.

New config (`config.py` + `.env.example`):
```
LLM_PROVIDER=gemini                             # default backend: gemini|openrouter
LLM_FALLBACK_PROVIDER=openrouter                # cross-vendor redundancy
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_SELECTABLE_MODELS=gemini-3.5-flash,anthropic/claude-haiku,openai/gpt-5.4-nano
MODELS_CACHE_TTL=3600
```

## 4. Capability gating (agent/tool-calling path)

`llm_client.model_supports_tools(provider, model) -> bool`, backed by the
`/models` catalog cache (task 6; bare Gemini names always return `True`).
`tool_selector.py` calls this once, only when a non-default model was
requested for the current request, before building `gen_cfg`. If `False`,
force the request back to the Gemini default and log a warning — this
prevents the silent-degrade failure mode the plan calls out: a tool-incapable
model falling through to the `indicrag_retrieval` default and *looking* like
it worked. The chat path (`routes/query.py`) has no such gate — chat doesn't
call functions.

## 5. Structured-output parsing robustness

`agent/json_utils.extract_json` is reused unchanged in `query_planner.py` and
`reflexion_evaluator.py`. New addition: when `extract_json` raises AND the
active provider for that call was not Gemini, retry the same prompt once
against the Gemini default before falling through to each node's existing
regex/default-value fallback path. This is an additive retry, not a rewrite
of either node's error handling.

## 6. `/models` endpoint

New `routes/models.py`: `GET /models` returns `config.LLM_SELECTABLE_MODELS`
enriched with `supported_parameters` pulled from OpenRouter's `/models`
catalog, cached in a `cache.py` `TTLCache` instance (`MODELS_CACHE_TTL`,
default 1h). Bare model names (no `/`) are returned as
`{"provider": "gemini", "tools": true}` without any network call — only
OpenRouter slugs hit the catalog. The route rejects (400) any `model`/`provider`
combination not present in the allowlist when used to validate task-7 requests.

## 7. User selection plumbing

Optional `model: Optional[str]` and `provider: Optional[str]` fields added to
`QueryRequest` (`routes/query.py`) and `AgentQueryRequest` (`routes/agent.py`),
validated against the `/models` allowlist with the same `field_validator`
pattern already used for `QueryRequest.strategy`. Values thread through to the
per-request `generate_with_failover(..., provider=...)` call — no global
config mutation, so concurrent requests with different models don't interfere.
UI: a model dropdown in `static/index.html` fed by `GET /models` on load,
defaulting to the first allowlist entry; entries without `tools: true` are
greyed out only when the UI is in agent mode.

## 8. Per-model eval hook

`docs/Eval/evaluate.py` gains an optional `--model` / `--provider` CLI flag
threaded through to the queries it issues, so running the existing eval
harness against a non-default model surfaces quality regressions in the
existing report instead of being invisible (per the plan: "unvetted catalog
is the user's risk by design").

## Testing

- New `tests/test_providers.py`: `GeminiBackend` behavior-parity tests
  (moved/duplicated from today's `llm_client.py` coverage) + `OpenRouterBackend`
  translation tests (request shape, response shim shape, streaming shim) with
  the `openai` SDK client mocked.
- New `tests/test_llm_client_dispatch.py`: model-id routing (`"/"` → OpenRouter,
  bare → Gemini), `(provider, model)` circuit-breaker keying, cross-provider
  failover order (same-provider exhausted → cross-provider fallback), explicit
  `provider` kwarg override.
- Existing `tests/test_agent.py` / `tests/test_hyde.py` (52 references to
  `rag.generate_with_failover`) must pass unmodified — this is the acceptance
  bar for task 1's re-export constraint.
- New `tests/test_models_route.py`: allowlist enrichment, bare-name no-network-call,
  400 on off-allowlist model.
- New `tests/test_capability_gating.py`: tool-incapable model on agent path
  forces Gemini fallback; chat path has no gate.

## Non-goals / explicit scope cuts

- `google-genai` remains a hard dependency even in OpenRouter-only mode
  (call sites build `types.GenerateContentConfig` objects the adapter
  translates *from*). Accepted per the plan.
- No spend-cap knob for OpenRouter usage (plan lists this as out of scope
  unless requested later).
- `GEMINI_CACHE_ENABLED` explicit caching has no OpenRouter equivalent; not
  emulated.
