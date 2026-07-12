# OpenRouter Secondary LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenRouter as a first-class second LLM provider alongside Gemini, with cross-vendor failover, per-request user model selection, and a `/models` endpoint.

**Architecture:** Extract a `providers/` package with an `LLMBackend` interface. `GeminiBackend` wraps today's `llm_client.py` pool/circuit-breaker logic unchanged; `OpenRouterBackend` speaks OpenAI Chat Completions and returns a **Gemini-shaped response shim** so agent nodes stay untouched. `llm_client.generate_with_failover` / `llm_generate_stream` become dispatchers that keep their exact current names, module location, and signatures (the ~52 tests that patch `rag.generate_with_failover` must pass unmodified).

**Tech Stack:** Python 3.13, FastAPI, `google-genai` (Gemini), `openai` SDK (OpenRouter, OpenAI-compatible), pytest.

## Global Constraints

- `rag.py:473` (`llm_generate_stream = llm_client.llm_generate_stream`) and `rag.py:577` (`generate_with_failover = llm_client.generate_with_failover`) MUST keep working — module-level symbol + signature preserved. New `provider` kwarg must be optional (`provider: str | None = None`).
- Agent nodes read a Gemini-shaped response: `resp.candidates[0].content.parts[*].function_call.{name,args}`, `resp.text`, `rag.safe_extract_text(resp)`. The OpenRouter shim must satisfy all three.
- `google-genai` stays a hard dependency even in OpenRouter-only mode. Accepted.
- `config.SAFETY_SETTINGS` is silently dropped on OpenRouter (no equivalent). Documented, not a bug.
- Every new config knob is env-gated with a safe default; add to both `config.py` and `.env.example`.
- Existing `tests/test_agent.py` + `tests/test_hyde.py` must pass unchanged after every task.
- OpenRouter model ids contain `/` (e.g. `anthropic/claude-haiku`); bare names route to Gemini.

---

### Task 1: Config knobs for OpenRouter

**Files:**
- Modify: `config.py` (LLM Configuration section, after line 298)
- Modify: `.env.example` (after the Gemini block, ~line 60)
- Test: `tests/test_openrouter_config.py`

**Interfaces:**
- Produces: `config.LLM_PROVIDER`, `config.LLM_FALLBACK_PROVIDER`, `config.OPENROUTER_API_KEY`, `config.OPENROUTER_BASE_URL`, `config.LLM_SELECTABLE_MODELS` (list[str]), `config.MODELS_CACHE_TTL` (int).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openrouter_config.py
import importlib


def test_provider_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_SELECTABLE_MODELS", raising=False)
    import config
    importlib.reload(config)
    assert config.LLM_PROVIDER == "gemini"
    assert config.LLM_FALLBACK_PROVIDER == "openrouter"
    assert config.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert config.LLM_SELECTABLE_MODELS[0] == "gemini-3.5-flash"
    assert config.MODELS_CACHE_TTL == 3600


def test_selectable_models_parsed_from_env(monkeypatch):
    monkeypatch.setenv("LLM_SELECTABLE_MODELS", "gemini-3.5-flash, anthropic/claude-haiku ,openai/gpt-5.4-nano")
    import config
    importlib.reload(config)
    assert config.LLM_SELECTABLE_MODELS == [
        "gemini-3.5-flash", "anthropic/claude-haiku", "openai/gpt-5.4-nano",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_openrouter_config.py -v`
Expected: FAIL with `AttributeError: module 'config' has no attribute 'LLM_PROVIDER'`

- [ ] **Step 3: Add config knobs**

In `config.py`, immediately after the `LLM_API_KEY = ...` block (line 298), add:

```python
# ============================================================================
# Phase 8 — Secondary LLM provider (OpenRouter)
# ============================================================================
# Default backend for LLM calls and the cross-vendor fallback. When the chosen
# provider's models are all exhausted/circuit-open, failover crosses to the
# other provider's default model so a whole-vendor outage isn't a total failure.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")                    # gemini|openrouter
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "openrouter")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Curated allowlist offered to the user in the model dropdown (comma-separated).
# Bare name → Gemini; slug with "/" → OpenRouter. First entry is the default.
_raw_selectable = os.getenv(
    "LLM_SELECTABLE_MODELS",
    "gemini-3.5-flash,anthropic/claude-haiku,openai/gpt-5.4-nano",
)
LLM_SELECTABLE_MODELS = [m.strip() for m in _raw_selectable.split(",") if m.strip()]
# How long the enriched OpenRouter /models catalog is cached (seconds).
MODELS_CACHE_TTL = int(os.getenv("MODELS_CACHE_TTL", "3600"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_openrouter_config.py -v`
Expected: PASS

- [ ] **Step 5: Update .env.example**

In `.env.example`, after the `AGENT_THINKING_BUDGET=0` block (line 60), add:

```bash
# ==============================================================================
# Phase 8 — Secondary LLM provider (OpenRouter)
# ==============================================================================
# OpenRouter is an OpenAI-compatible gateway. Adding it gives cross-vendor
# failover (a Google-wide outage no longer kills all LLM calls) and lets the
# user pick a model per request. Off unless OPENROUTER_API_KEY is set.
#   LLM_PROVIDER          default backend: gemini|openrouter
#   LLM_FALLBACK_PROVIDER cross-vendor fallback when the primary is exhausted
LLM_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=openrouter
# Get a key at https://openrouter.ai/keys . Paid, no free tier.
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# Models offered in the UI dropdown (comma-separated). Bare name → Gemini;
# slug with "/" → OpenRouter. First entry is the default. Off-allowlist
# requests are rejected server-side.
LLM_SELECTABLE_MODELS=gemini-3.5-flash,anthropic/claude-haiku,openai/gpt-5.4-nano
# How long the enriched OpenRouter model catalog is cached (seconds).
MODELS_CACHE_TTL=3600
```

- [ ] **Step 6: Commit**

```bash
git add config.py .env.example tests/test_openrouter_config.py
git commit -m "feat(llm): Phase 8 Task 1 — OpenRouter config knobs"
```

---

### Task 2: `providers/base.py` — backend interface + Gemini-shaped shim types

**Files:**
- Create: `providers/__init__.py`
- Create: `providers/base.py`
- Test: `tests/test_providers_base.py`

**Interfaces:**
- Produces:
  - `providers.base.LLMBackend` (ABC): `generate(model, contents, gen_config) -> object`, `generate_stream(model, contents, gen_config) -> Iterator[str]`, `is_transient(exc) -> bool`, `is_permanent(exc) -> bool`, property `name -> str`.
  - `providers.base.ShimResponse(text: str, function_calls: list[tuple[str, dict]])` — a Gemini-shaped response object exposing `.text`, `.candidates[0].content.parts[*].function_call.{name,args}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_base.py
from providers.base import ShimResponse, LLMBackend


def test_shim_exposes_text():
    r = ShimResponse(text="hello", function_calls=[])
    assert r.text == "hello"


def test_shim_exposes_function_calls_gemini_shape():
    r = ShimResponse(text="", function_calls=[("indicrag_retrieval", {"query": "x"})])
    parts = r.candidates[0].content.parts
    assert parts[0].function_call.name == "indicrag_retrieval"
    assert dict(parts[0].function_call.args) == {"query": "x"}


def test_shim_text_part_when_no_function_calls():
    r = ShimResponse(text="answer body", function_calls=[])
    parts = r.candidates[0].content.parts
    assert parts[0].text == "answer body"
    assert getattr(parts[0], "function_call", None) is None


def test_llmbackend_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        LLMBackend()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'providers'`

- [ ] **Step 3: Create the package**

```python
# providers/__init__.py
"""LLM provider backends (Gemini, OpenRouter) behind a common interface."""
```

```python
# providers/base.py
"""Backend interface + a Gemini-shaped response shim.

Agent nodes read responses as `resp.candidates[0].content.parts[*].function_call`
and `resp.text`. Non-Gemini backends wrap their output in ShimResponse so those
call sites never change.
"""

from abc import ABC, abstractmethod
from typing import Iterator


class _FunctionCall:
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args


class _Part:
    def __init__(self, text: str = "", function_call: _FunctionCall | None = None):
        self.text = text
        self.function_call = function_call


class _Content:
    def __init__(self, parts: list[_Part]):
        self.parts = parts


class _Candidate:
    def __init__(self, content: _Content):
        self.content = content
        self.finish_reason = "STOP"


class ShimResponse:
    """Gemini-shaped response wrapping a non-Gemini backend's output.

    Exposes `.text` and `.candidates[0].content.parts[*].function_call` so
    tool_selector / query_planner / reflexion_evaluator / safe_extract_text
    read it identically to a real google-genai response.
    """

    def __init__(self, text: str, function_calls: list[tuple[str, dict]]):
        self.text = text
        if function_calls:
            parts = [_Part(function_call=_FunctionCall(n, a)) for n, a in function_calls]
        else:
            parts = [_Part(text=text)]
        self.candidates = [_Candidate(_Content(parts))]


class LLMBackend(ABC):
    """One LLM vendor. Dispatch/failover lives in llm_client, not here."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(self, model: str, contents, gen_config): ...

    @abstractmethod
    def generate_stream(self, model: str, contents, gen_config) -> Iterator[str]: ...

    @abstractmethod
    def is_transient(self, exc: Exception) -> bool: ...

    @abstractmethod
    def is_permanent(self, exc: Exception) -> bool: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/__init__.py providers/base.py tests/test_providers_base.py
git commit -m "feat(llm): Phase 8 Task 2 — provider interface + Gemini-shaped shim"
```

---

### Task 3: `providers/gemini.py` — GeminiBackend (behavior-preserving extraction)

**Files:**
- Create: `providers/gemini.py`
- Test: `tests/test_providers_gemini.py`

**Interfaces:**
- Consumes: `providers.base.LLMBackend`.
- Produces: `providers.gemini.GeminiBackend` — `.name == "gemini"`; `generate(model, contents, gen_config, client=None)` returns the raw google-genai response (NOT a shim — Gemini is already the native shape); `generate_stream(..., client=None)` yields text chunks; `is_transient`/`is_permanent` = today's `llm_client._is_transient`/`_is_permanent` logic verbatim; helpers `supports_thinking(model) -> bool`, `pool` property, `next_client_idx() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_gemini.py
from providers.gemini import GeminiBackend


def test_name():
    assert GeminiBackend().name == "gemini"


def test_is_transient_recognizes_503_and_429():
    b = GeminiBackend()
    assert b.is_transient(Exception("503 UNAVAILABLE"))
    assert b.is_transient(Exception("RESOURCE_EXHAUSTED"))
    assert not b.is_transient(Exception("400 INVALID_ARGUMENT"))


def test_is_permanent_only_for_malformed_request():
    b = GeminiBackend()
    assert b.is_permanent(Exception("INVALID_ARGUMENT"))
    assert not b.is_permanent(Exception("401 UNAUTHENTICATED"))  # must fail over keys


def test_supports_thinking_gates_gemma():
    b = GeminiBackend()
    assert b.supports_thinking("gemini-3.5-flash")
    assert not b.supports_thinking("gemma-4-26b-a4b-it")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_providers_gemini.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'providers.gemini'`

- [ ] **Step 3: Create GeminiBackend**

Move the pool + classifier logic out of `llm_client.py` into a class. Copy the bodies of `_is_transient`, `_is_permanent`, `_supports_thinking`, `_with_cache`, and the client-pool helpers verbatim.

```python
# providers/gemini.py
"""Gemini backend: client pool, round-robin, per-key failover, thinking gate.

Extracted verbatim from the original llm_client.py so behavior is unchanged.
generate() returns the raw google-genai response (already the native shape the
agent nodes read) — no shim needed for this backend.
"""

import itertools
import logging
import threading
from typing import Iterator

import config
from google import genai
from providers.base import LLMBackend

logger = logging.getLogger(__name__)


class GeminiBackend(LLMBackend):
    def __init__(self):
        self._pool: list[genai.Client] = []
        self._lock = threading.Lock()
        self._index = itertools.cycle([])

    @property
    def name(self) -> str:
        return "gemini"

    # ── pool ────────────────────────────────────────────────────────────
    def _init_pool(self) -> None:
        if not config.LLM_API_KEY_POOL:
            raise ValueError(
                "Google Gemini API key not configured. "
                "Set LLM_API_KEY (single) or LLM_API_KEYS (comma-separated) in .env."
            )
        self._pool = [genai.Client(api_key=k) for k in config.LLM_API_KEY_POOL]
        self._index = itertools.cycle(range(len(self._pool)))

    def _ensure_pool(self) -> None:
        if not self._pool:
            with self._lock:
                if not self._pool:
                    self._init_pool()

    def next_client_idx(self) -> int:
        with self._lock:
            return next(self._index)

    @property
    def pool(self) -> list:
        self._ensure_pool()
        return self._pool

    # ── classifiers (verbatim from old llm_client) ──────────────────────
    def is_transient(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (429, 503):
            return True
        msg = str(exc)
        return "503" in msg or "429" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg

    def is_permanent(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (400, 404):
            return True
        return "INVALID_ARGUMENT" in str(exc)

    def supports_thinking(self, model: str) -> bool:
        return "gemma" not in model.lower()

    def _with_cache(self, client, model: str, gen_config):
        try:
            sys_inst = getattr(gen_config, "system_instruction", None)
            if not sys_inst:
                return gen_config
            import gemini_cache
            name = gemini_cache.get_or_create(client, model, sys_inst,
                                              getattr(gen_config, "tools", None))
            if not name:
                return gen_config
            return gen_config.model_copy(update={
                "system_instruction": None, "tools": None, "cached_content": name,
            })
        except Exception:
            return gen_config

    def _prep_config(self, client, model, gen_config):
        call_config = self._with_cache(client, model, gen_config)
        if not self.supports_thinking(model) and getattr(call_config, "thinking_config", None) is not None:
            call_config = call_config.model_copy(update={"thinking_config": None})
        return call_config

    # ── single-client calls (dispatch/failover lives in llm_client) ─────
    def generate(self, model: str, contents, gen_config, client=None):
        client = client or self.pool[self.next_client_idx()]
        call_config = self._prep_config(client, model, gen_config)
        return client.models.generate_content(model=model, contents=contents, config=call_config)

    def generate_stream(self, model: str, contents, gen_config, client=None) -> Iterator[str]:
        client = client or self.pool[self.next_client_idx()]
        call_config = self._prep_config(client, model, gen_config)
        emitted = False
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=call_config):
            try:
                if chunk.text:
                    emitted = True
                    yield chunk.text
            except (ValueError, AttributeError) as exc:
                logger.debug("Skipping non-text Gemini stream chunk: %s", exc)
        if not emitted:
            raise RuntimeError("No text generated from Gemini stream")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_providers_gemini.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add providers/gemini.py tests/test_providers_gemini.py
git commit -m "feat(llm): Phase 8 Task 3 — GeminiBackend extraction"
```

---

### Task 4: `providers/openrouter.py` — OpenRouterBackend + translation

**Files:**
- Create: `providers/openrouter.py`
- Modify: `requirements.txt` (add `openai` under Agentic layer)
- Test: `tests/test_providers_openrouter.py`

**Interfaces:**
- Consumes: `providers.base.LLMBackend`, `providers.base.ShimResponse`.
- Produces: `providers.openrouter.OpenRouterBackend` — `.name == "openrouter"`; `generate(...)` returns a `ShimResponse`; `generate_stream(...)` yields text; `is_transient`/`is_permanent` classify `openai` SDK exceptions; module helpers `_to_messages(contents, gen_config)`, `_to_tools(gen_config)`.

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, under the `# Agentic layer` block, add:

```
openai>=1.50.0                 # OpenRouter backend (OpenAI-compatible Chat Completions)
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_providers_openrouter.py
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.genai import types
from providers.openrouter import OpenRouterBackend, _to_messages, _to_tools
from providers.base import ShimResponse


def _cfg(**kw):
    return types.GenerateContentConfig(**kw)


def test_to_messages_prepends_system():
    msgs = _to_messages("hello", _cfg(system_instruction="be terse"))
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_to_tools_wraps_json_schema():
    tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="indicrag_retrieval",
            description="retrieve",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
    ])
    out = _to_tools(_cfg(tools=[tool]))
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "indicrag_retrieval"
    assert out[0]["function"]["parameters"]["required"] == ["query"]


def test_generate_returns_shim_with_text():
    b = OpenRouterBackend()
    fake_msg = SimpleNamespace(content="the answer", tool_calls=None)
    fake_resp = SimpleNamespace(choices=[SimpleNamespace(message=fake_msg)])
    b._client = MagicMock()
    b._client.chat.completions.create.return_value = fake_resp
    resp = b.generate("anthropic/claude-haiku", "q", _cfg(temperature=0.1))
    assert isinstance(resp, ShimResponse)
    assert resp.text == "the answer"


def test_generate_returns_shim_with_function_call():
    import json
    b = OpenRouterBackend()
    tc = SimpleNamespace(function=SimpleNamespace(
        name="indicrag_retrieval", arguments=json.dumps({"query": "x"})))
    fake_msg = SimpleNamespace(content=None, tool_calls=[tc])
    fake_resp = SimpleNamespace(choices=[SimpleNamespace(message=fake_msg)])
    b._client = MagicMock()
    b._client.chat.completions.create.return_value = fake_resp
    resp = b.generate("anthropic/claude-haiku", "q", _cfg())
    part = resp.candidates[0].content.parts[0]
    assert part.function_call.name == "indicrag_retrieval"
    assert dict(part.function_call.args) == {"query": "x"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_providers_openrouter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'providers.openrouter'`

- [ ] **Step 4: Create OpenRouterBackend**

```python
# providers/openrouter.py
"""OpenRouter backend: OpenAI-compatible Chat Completions → Gemini-shaped shim.

Translates a google-genai GenerateContentConfig call into an OpenAI request and
wraps the reply in ShimResponse so agent nodes read it unchanged. SAFETY_SETTINGS
has no OpenRouter equivalent and is silently dropped — a stated behavior gap.
"""

import json
import logging
import threading
from typing import Iterator

import config
from providers.base import LLMBackend, ShimResponse

logger = logging.getLogger(__name__)


def _to_messages(contents, gen_config) -> list[dict]:
    """contents (str) + system_instruction → OpenAI messages."""
    messages = []
    sys_inst = getattr(gen_config, "system_instruction", None)
    if sys_inst:
        messages.append({"role": "system", "content": str(sys_inst)})
    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    else:
        # contents is a list of parts/strings — flatten to text
        messages.append({"role": "user", "content": str(contents)})
    return messages


def _as_plain(params):
    """FunctionDeclaration.parameters may be a dict (as authored) or a Schema
    object. Return a plain JSON-Schema dict either way."""
    if params is None:
        return None
    if isinstance(params, dict):
        return params
    if hasattr(params, "model_dump"):
        return params.model_dump(exclude_none=True)
    return dict(params)


def _to_tools(gen_config):
    """types.Tool(function_declarations) → OpenAI tools[]. Params pass through."""
    tools = getattr(gen_config, "tools", None)
    if not tools:
        return None
    out = []
    for tool in tools:
        for fd in getattr(tool, "function_declarations", []) or []:
            out.append({
                "type": "function",
                "function": {
                    "name": fd.name,
                    "description": getattr(fd, "description", "") or "",
                    "parameters": _as_plain(getattr(fd, "parameters", None)) or {"type": "object", "properties": {}},
                },
            })
    return out or None


class OpenRouterBackend(LLMBackend):
    def __init__(self):
        self._client = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "openrouter"

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    if not config.OPENROUTER_API_KEY:
                        raise ValueError(
                            "OPENROUTER_API_KEY not configured. Set it in .env to use OpenRouter."
                        )
                    from openai import OpenAI
                    self._client = OpenAI(
                        api_key=config.OPENROUTER_API_KEY,
                        base_url=config.OPENROUTER_BASE_URL,
                    )
        return self._client

    def _params(self, model, contents, gen_config, stream: bool) -> dict:
        params = {
            "model": model,
            "messages": _to_messages(contents, gen_config),
            "stream": stream,
        }
        temp = getattr(gen_config, "temperature", None)
        if temp is not None:
            params["temperature"] = temp
        max_tok = getattr(gen_config, "max_output_tokens", None)
        if max_tok is not None:
            params["max_tokens"] = max_tok
        tools = _to_tools(gen_config)
        if tools:
            params["tools"] = tools
        # thinking_config → reasoning is best-effort; dropped for models that
        # don't advertise it (capability gate handles the agent path).
        return params

    def generate(self, model: str, contents, gen_config):
        client = self._get_client()
        resp = client.chat.completions.create(**self._params(model, contents, gen_config, stream=False))
        msg = resp.choices[0].message
        function_calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            function_calls.append((tc.function.name, args))
        return ShimResponse(text=msg.content or "", function_calls=function_calls)

    def generate_stream(self, model: str, contents, gen_config) -> Iterator[str]:
        client = self._get_client()
        emitted = False
        for chunk in client.chat.completions.create(**self._params(model, contents, gen_config, stream=True)):
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                emitted = True
                yield text
        if not emitted:
            raise RuntimeError("No text generated from OpenRouter stream")

    def is_transient(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (429, 500, 502, 503):
            return True
        name = type(exc).__name__
        return name in ("RateLimitError", "APIConnectionError", "InternalServerError", "APITimeoutError")

    def is_permanent(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in (400, 404, 422):
            return True
        return type(exc).__name__ in ("BadRequestError", "NotFoundError")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_providers_openrouter.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add providers/openrouter.py requirements.txt tests/test_providers_openrouter.py
git commit -m "feat(llm): Phase 8 Task 4 — OpenRouterBackend + request/response translation"
```

---

### Task 5: `llm_client.py` dispatcher — routing, (provider,model) circuit key, cross-provider failover

**Files:**
- Modify: `llm_client.py` (full rewrite of dispatch, keeping module-level `generate_with_failover` + `llm_generate_stream` symbols/signatures)
- Test: `tests/test_llm_client_dispatch.py`

**Interfaces:**
- Consumes: `providers.gemini.GeminiBackend`, `providers.openrouter.OpenRouterBackend`, config knobs from Task 1.
- Produces:
  - `llm_client.generate_with_failover(model, contents, gen_config, provider=None)` — unchanged first 3 params; new optional `provider`.
  - `llm_client.llm_generate_stream(prompt, max_tokens=None, system_instruction=None, model=None, provider=None)` — first 3 params unchanged.
  - `llm_client.resolve_provider(model, provider=None) -> str` — `"/"` in model → `"openrouter"`, else `"gemini"`; explicit `provider` wins.
  - `llm_client.get_backend(provider) -> LLMBackend`.
  - `llm_client.model_supports_tools(provider, model) -> bool`.
  - module-level `llm_client._circuit_breaker: dict[tuple[str,str], float]`, `llm_client._circuit_key(provider, model)`, `llm_client._config` (alias to `config`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_client_dispatch.py
import llm_client


def test_resolve_provider_by_model_shape():
    assert llm_client.resolve_provider("gemini-3.5-flash") == "gemini"
    assert llm_client.resolve_provider("anthropic/claude-haiku") == "openrouter"


def test_explicit_provider_overrides_shape():
    assert llm_client.resolve_provider("gemini-3.5-flash", provider="openrouter") == "openrouter"


def test_circuit_key_is_provider_scoped():
    assert llm_client._circuit_key("gemini", "x") != llm_client._circuit_key("openrouter", "x")


def test_generate_with_failover_routes_to_gemini(monkeypatch):
    calls = {}

    class FakeGemini:
        name = "gemini"
        def generate(self, model, contents, gen_config, client=None):
            calls["model"] = model
            return "GEMINI_RESP"
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    monkeypatch.setattr(llm_client, "_backends", {"gemini": FakeGemini()})
    monkeypatch.setattr(llm_client, "get_backend", lambda p: llm_client._backends[p])
    out = llm_client.generate_with_failover("gemini-3.5-flash", "q", object())
    assert out == "GEMINI_RESP"
    assert calls["model"] == "gemini-3.5-flash"


def test_cross_provider_failover(monkeypatch):
    class FailingGemini:
        name = "gemini"
        def generate(self, *a, **k): raise Exception("503 UNAVAILABLE")
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    class OkOpenRouter:
        name = "openrouter"
        def generate(self, model, contents, gen_config): return "OR_RESP"
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    monkeypatch.setattr(llm_client, "_backends",
                        {"gemini": FailingGemini(), "openrouter": OkOpenRouter()})
    monkeypatch.setattr(llm_client, "get_backend", lambda p: llm_client._backends[p])
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_MODEL", "")  # skip same-provider fallback
    llm_client._circuit_breaker.clear()
    out = llm_client.generate_with_failover("gemini-3.5-flash", "q", object())
    assert out == "OR_RESP"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_client_dispatch.py -v`
Expected: FAIL with `AttributeError: module 'llm_client' has no attribute 'resolve_provider'`

- [ ] **Step 3: Rewrite llm_client.py as a dispatcher**

Replace the entire body of `llm_client.py` with:

```python
"""LLM dispatcher: route by provider/model, per-(provider,model) circuit breaker,
same-provider then cross-provider failover.

generate_with_failover and llm_generate_stream keep their names, module location,
and leading signatures — rag.py re-exports them and ~52 tests patch them.
"""

import logging
import time

import config as _config
from providers.base import LLMBackend
from providers.gemini import GeminiBackend
from providers.openrouter import OpenRouterBackend

logger = logging.getLogger(__name__)

_backends: dict[str, LLMBackend] = {}
_circuit_breaker: dict[tuple[str, str], float] = {}
_CIRCUIT_COOLDOWN = 60


def _init_backends() -> None:
    global _backends
    if not _backends:
        _backends = {"gemini": GeminiBackend(), "openrouter": OpenRouterBackend()}


def get_backend(provider: str) -> LLMBackend:
    _init_backends()
    if provider not in _backends:
        raise ValueError(f"Unknown provider: {provider}")
    return _backends[provider]


def resolve_provider(model: str, provider: str | None = None) -> str:
    """Explicit provider wins; else infer from model shape ('/' → openrouter)."""
    if provider:
        return provider
    return "openrouter" if "/" in (model or "") else "gemini"


def _circuit_key(provider: str, model: str) -> tuple[str, str]:
    return (provider, model)


def _fallback_model_for(provider: str) -> str:
    """The provider's default model for cross-provider fallback."""
    if provider == "gemini":
        return _config.LLM_MODEL_NAME
    return _config.LLM_SELECTABLE_MODELS[0] if _config.LLM_SELECTABLE_MODELS else _config.LLM_MODEL_NAME


def model_supports_tools(provider: str, model: str) -> bool:
    """Gemini always supports tools; OpenRouter is checked against the catalog.
    Default True so a catalog outage doesn't over-block."""
    if provider == "gemini":
        return True
    try:
        import routes.models as models_route
        return models_route.model_supports_tools(model)
    except Exception:
        return True


def _attempts(model: str, provider: str) -> list[tuple[str, str]]:
    """Ordered (provider, model) attempts: requested → same-provider fallback →
    cross-provider fallback."""
    attempts = [(provider, model)]
    if provider == "gemini" and _config.LLM_FALLBACK_MODEL and _config.LLM_FALLBACK_MODEL != model:
        attempts.append((provider, _config.LLM_FALLBACK_MODEL))
    fb_provider = _config.LLM_FALLBACK_PROVIDER
    if fb_provider and fb_provider != provider:
        attempts.append((fb_provider, _fallback_model_for(fb_provider)))
    return attempts


def generate_with_failover(model: str, contents, gen_config, provider: str | None = None):
    """Try requested (provider, model), then same-provider then cross-provider
    fallback. Per-(provider,model) circuit breaker skips recently-dead paths."""
    provider = resolve_provider(model, provider)
    last_exc: Exception | None = None
    any_attempted = False

    for prov, mdl in _attempts(model, provider):
        key = _circuit_key(prov, mdl)
        if time.monotonic() < _circuit_breaker.get(key, 0):
            logger.info(f"[failover] {prov}:{mdl} circuit open, skipping")
            continue
        backend = get_backend(prov)
        any_attempted = True
        try:
            result = backend.generate(mdl, contents, gen_config)
            _circuit_breaker.pop(key, None)
            return result
        except Exception as exc:
            last_exc = exc
            if backend.is_permanent(exc):
                raise
            logger.warning(f"[failover] {prov}:{mdl} failed ({exc!s:.120}) — next path")
            _circuit_breaker[key] = time.monotonic() + _CIRCUIT_COOLDOWN
            continue

    if not any_attempted:
        raise RuntimeError("All configured LLM paths are circuit-open; retry after cooldown.")
    raise last_exc  # type: ignore[misc]


def _build_gemini_stream_config(model, max_tokens, system_instruction):
    from google.genai import types
    kwargs = dict(
        temperature=_config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        safety_settings=_config.SAFETY_SETTINGS,
        system_instruction=system_instruction or _config.SYSTEM_PROMPT,
    )
    if get_backend("gemini").supports_thinking(model):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def _build_openrouter_stream_config(max_tokens, system_instruction):
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=_config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        system_instruction=system_instruction or _config.SYSTEM_PROMPT,
    )


def llm_generate_stream(prompt: str, max_tokens: int = None, system_instruction: str = None,
                        model: str = None, provider: str | None = None):
    """Stream chunks with same-provider then cross-provider failover. Failover only
    BEFORE the first chunk — a mid-stream failure re-raises (can't restart output)."""
    if max_tokens is None:
        max_tokens = _config.LLM_MAX_TOKENS
    model = model or _config.LLM_MODEL_NAME
    provider = resolve_provider(model, provider)

    last_exc: Exception | None = None
    any_attempted = False
    for prov, mdl in _attempts(model, provider):
        key = _circuit_key(prov, mdl)
        if time.monotonic() < _circuit_breaker.get(key, 0):
            continue
        backend = get_backend(prov)
        if prov == "gemini":
            gen_config = _build_gemini_stream_config(mdl, max_tokens, system_instruction)
        else:
            gen_config = _build_openrouter_stream_config(max_tokens, system_instruction)
        any_attempted = True
        emitted = False
        try:
            for chunk in backend.generate_stream(mdl, prompt, gen_config):
                emitted = True
                yield chunk
            _circuit_breaker.pop(key, None)
            return
        except Exception as exc:
            last_exc = exc
            if emitted:
                raise  # committed to this stream
            if backend.is_permanent(exc):
                raise
            logger.warning(f"[stream failover] {prov}:{mdl} failed ({exc!s:.120}) — next path")
            _circuit_breaker[key] = time.monotonic() + _CIRCUIT_COOLDOWN
            continue

    if not any_attempted:
        raise RuntimeError("All configured LLM paths are circuit-open; retry after cooldown.")
    raise last_exc  # type: ignore[misc]
```

Note: `_config` is the module-level alias (`import config as _config`) so tests can `monkeypatch.setattr(llm_client._config, ...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_client_dispatch.py -v`
Expected: PASS

- [ ] **Step 5: Run the full agent suite (re-export constraint)**

Run: `pytest tests/test_agent.py tests/test_hyde.py -v`
Expected: PASS — no test modified. If any fail on a signature/symbol mismatch, fix `llm_client.py` (not the tests). Note: tests patch `rag.generate_with_failover`, which is `rag.py:577`'s re-export of this module's function — the symbol name and first-3-params signature are preserved, so patches still bind.

- [ ] **Step 6: Commit**

```bash
git add llm_client.py tests/test_llm_client_dispatch.py
git commit -m "feat(llm): Phase 8 Task 5 — dispatcher, (provider,model) circuit key, cross-provider failover"
```

---

### Task 6: `routes/models.py` — `/models` endpoint + catalog cache

**Files:**
- Create: `routes/models.py`
- Modify: `api_server.py` (register router after line 150)
- Test: `tests/test_models_route.py`

**Interfaces:**
- Consumes: `config.LLM_SELECTABLE_MODELS`, `config.MODELS_CACHE_TTL`, `cache.TTLCache`.
- Produces:
  - `routes.models.router` (`GET /models`).
  - `routes.models.model_supports_tools(model: str) -> bool` (used by `llm_client.model_supports_tools`).
  - `routes.models.list_models() -> list[dict]` — each `{"id", "provider", "tools"}`.
  - `routes.models.validate_model(model: str | None, provider: str | None) -> None` — raises `ValueError` if off-allowlist.
  - module-level `routes.models._catalog_cache` (TTLCache), `routes.models._fetch_openrouter_catalog()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_route.py
import pytest
import routes.models as m


def test_bare_name_is_gemini_no_network(monkeypatch):
    monkeypatch.setattr(m.config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash"])
    def _boom():
        raise AssertionError("should not fetch catalog for bare Gemini name")
    monkeypatch.setattr(m, "_fetch_openrouter_catalog", _boom)
    out = m.list_models()
    assert out == [{"id": "gemini-3.5-flash", "provider": "gemini", "tools": True}]


def test_openrouter_model_enriched_from_catalog(monkeypatch):
    monkeypatch.setattr(m.config, "LLM_SELECTABLE_MODELS", ["anthropic/claude-haiku"])
    monkeypatch.setattr(m, "_fetch_openrouter_catalog",
                        lambda: {"anthropic/claude-haiku": {"supported_parameters": ["tools", "temperature"]}})
    m._catalog_cache.invalidate()
    out = m.list_models()
    assert out == [{"id": "anthropic/claude-haiku", "provider": "openrouter", "tools": True}]


def test_model_without_tools_flagged(monkeypatch):
    monkeypatch.setattr(m.config, "LLM_SELECTABLE_MODELS", ["openai/gpt-5.4-nano"])
    monkeypatch.setattr(m, "_fetch_openrouter_catalog",
                        lambda: {"openai/gpt-5.4-nano": {"supported_parameters": ["temperature"]}})
    m._catalog_cache.invalidate()
    assert m.model_supports_tools("openai/gpt-5.4-nano") is False


def test_validate_model_rejects_off_allowlist(monkeypatch):
    monkeypatch.setattr(m.config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash"])
    with pytest.raises(ValueError):
        m.validate_model("evil/model", None)


def test_validate_model_allows_none():
    m.validate_model(None, None)  # no override — fine
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_route.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'routes.models'`

- [ ] **Step 3: Create the route**

```python
# routes/models.py
"""GET /models — the curated model allowlist enriched with tool-capability.

Bare names → Gemini (always tool-capable, no network). OpenRouter slugs are
enriched from the OpenRouter /models catalog (cached MODELS_CACHE_TTL seconds)
so the UI can grey out tool-incapable models in agent mode.
"""

import logging

from fastapi import APIRouter

import config
from cache import TTLCache

logger = logging.getLogger(__name__)
router = APIRouter()

_catalog_cache = TTLCache(max_size=4, ttl_seconds=config.MODELS_CACHE_TTL)
_CATALOG_KEY = "openrouter_catalog"


def _fetch_openrouter_catalog() -> dict:
    """slug -> {'supported_parameters': [...]}. Network call; best-effort."""
    import httpx
    url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/models"
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return {mod["id"]: mod for mod in data if "id" in mod}


def _catalog() -> dict:
    cached = _catalog_cache.get(_CATALOG_KEY)
    if cached is not None:
        return cached
    try:
        cat = _fetch_openrouter_catalog()
    except Exception as exc:
        logger.warning(f"[/models] OpenRouter catalog fetch failed: {exc!s:.120}")
        cat = {}
    _catalog_cache.put(_CATALOG_KEY, cat)
    return cat


def _is_openrouter(model: str) -> bool:
    return "/" in model


def model_supports_tools(model: str) -> bool:
    if not _is_openrouter(model):
        return True
    entry = _catalog().get(model)
    if not entry:
        return True  # unknown → don't over-block; capability gate is best-effort
    return "tools" in (entry.get("supported_parameters") or [])


def list_models() -> list[dict]:
    out = []
    for mid in config.LLM_SELECTABLE_MODELS:
        if _is_openrouter(mid):
            out.append({"id": mid, "provider": "openrouter", "tools": model_supports_tools(mid)})
        else:
            out.append({"id": mid, "provider": "gemini", "tools": True})
    return out


def validate_model(model: str | None, provider: str | None) -> None:
    """Raise ValueError if a requested model is off the allowlist."""
    if model is None:
        return
    if model not in config.LLM_SELECTABLE_MODELS:
        raise ValueError(f"Model '{model}' is not in LLM_SELECTABLE_MODELS allowlist.")


@router.get("/models", tags=["Models"])
async def get_models():
    return {"models": list_models(), "default": (config.LLM_SELECTABLE_MODELS or [None])[0]}
```

- [ ] **Step 4: Register the router**

In `api_server.py`, after line 150 (`app.include_router(report.router)`), add. Match the existing import style — if routers are imported at the top with `from routes import query, chat, ...`, add `models as models_route` there and register in the mount block:

```python
from routes import models as models_route  # noqa: E402
app.include_router(models_route.router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_models_route.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add routes/models.py api_server.py tests/test_models_route.py
git commit -m "feat(llm): Phase 8 Task 6 — /models endpoint + catalog cache"
```

---

### Task 7: Capability gating on the agent path

**Files:**
- Modify: `agent/nodes/tool_selector.py` (add gate + route the LLM call through it)
- Modify: `agent/state.py` (add `requested_model`, `requested_provider` to `AgentState`)
- Test: `tests/test_capability_gating.py`

**Interfaces:**
- Consumes: `llm_client.model_supports_tools(provider, model)`, `llm_client.resolve_provider(model, provider)`.
- Produces: `agent.nodes.tool_selector._gate_model(state) -> tuple[str, str]` returning the `(provider, model)` to actually use (Gemini default if the requested model can't call tools).

- [ ] **Step 1: Add state fields**

In `agent/state.py`, add to the `AgentState` TypedDict (near the other optional fields):

```python
    requested_model: Optional[str]      # user-selected model for this request (Phase 8)
    requested_provider: Optional[str]   # user-selected provider for this request (Phase 8)
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_capability_gating.py
import agent.nodes.tool_selector as ts


def test_gate_falls_back_to_gemini_for_tool_incapable(monkeypatch):
    monkeypatch.setattr(ts.llm_client, "resolve_provider", lambda m, p=None: "openrouter")
    monkeypatch.setattr(ts.llm_client, "model_supports_tools", lambda prov, m: False)
    state = {"requested_model": "openai/gpt-5.4-nano", "requested_provider": None}
    provider, model = ts._gate_model(state)
    assert provider == "gemini"
    assert model == ts.config.LLM_MODEL_NAME


def test_gate_keeps_tool_capable_model(monkeypatch):
    monkeypatch.setattr(ts.llm_client, "resolve_provider", lambda m, p=None: "openrouter")
    monkeypatch.setattr(ts.llm_client, "model_supports_tools", lambda prov, m: True)
    state = {"requested_model": "anthropic/claude-haiku", "requested_provider": None}
    provider, model = ts._gate_model(state)
    assert provider == "openrouter"
    assert model == "anthropic/claude-haiku"


def test_gate_default_when_no_request():
    state = {}
    provider, model = ts._gate_model(state)
    assert provider == "gemini"
    assert model == ts.config.LLM_MODEL_NAME
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_capability_gating.py -v`
Expected: FAIL with `AttributeError: module 'agent.nodes.tool_selector' has no attribute '_gate_model'`

- [ ] **Step 4: Add the gate + wire it in**

In `agent/nodes/tool_selector.py`, add `import llm_client` at the top, and the gate function:

```python
def _gate_model(state) -> tuple[str, str]:
    """Return the (provider, model) to use for tool selection.

    If the user picked a model that can't call functions, fall back to the
    Gemini default so the agent doesn't silently degrade to the retrieval
    default and *look* like it worked."""
    model = state.get("requested_model") or config.LLM_MODEL_NAME
    provider = state.get("requested_provider")
    provider = llm_client.resolve_provider(model, provider)
    if not llm_client.model_supports_tools(provider, model):
        logger.warning(
            f"[ToolSelector] model {provider}:{model} can't call tools — "
            f"falling back to gemini:{config.LLM_MODEL_NAME}"
        )
        return "gemini", config.LLM_MODEL_NAME
    return provider, model
```

Then in `tool_selector_node`, replace the `resp = rag.generate_with_failover(...)` call (line 134) with:

```python
    gate_provider, gate_model = _gate_model(state)
    try:
        resp = rag.generate_with_failover(
            model=gate_model,
            contents=user_content,
            gen_config=gen_cfg,
            provider=gate_provider,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_capability_gating.py -v`
Expected: PASS

- [ ] **Step 6: Run the agent suite (unchanged behavior when no model requested)**

Run: `pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add agent/nodes/tool_selector.py agent/state.py tests/test_capability_gating.py
git commit -m "feat(llm): Phase 8 Task 7 — capability gating on agent path"
```

---

### Task 8: Non-Gemini JSON parse fallback in structured-output nodes

**Files:**
- Modify: `agent/json_utils.py` (add `extract_json_with_gemini_retry`)
- Modify: `agent/nodes/query_planner.py`
- Modify: `agent/nodes/reflexion_evaluator.py`
- Test: `tests/test_structured_output_fallback.py`

**Interfaces:**
- Consumes: `agent.json_utils.extract_json`.
- Produces: `agent.json_utils.extract_json_with_gemini_retry(raw, provider, gemini_call, prompt, system) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_structured_output_fallback.py
import pytest
import agent.json_utils as ju


def test_extract_json_gemini_retry_used_for_non_gemini():
    out = ju.extract_json_with_gemini_retry(
        raw="not json at all",
        provider="openrouter",
        gemini_call=lambda p, s: '{"sub_queries": ["ok"]}',
        prompt="p", system="s",
    )
    assert out == {"sub_queries": ["ok"]}


def test_extract_json_no_retry_for_gemini():
    with pytest.raises(ValueError):
        ju.extract_json_with_gemini_retry(
            raw="not json", provider="gemini",
            gemini_call=lambda p, s: '{"x": 1}', prompt="p", system="s",
        )


def test_extract_json_passes_through_valid():
    out = ju.extract_json_with_gemini_retry(
        raw='{"a": 1}', provider="openrouter",
        gemini_call=lambda p, s: "{}", prompt="p", system="s",
    )
    assert out == {"a": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_structured_output_fallback.py -v`
Expected: FAIL with `AttributeError: module 'agent.json_utils' has no attribute 'extract_json_with_gemini_retry'`

- [ ] **Step 3: Add the shared helper to `agent/json_utils.py`**

```python
def extract_json_with_gemini_retry(raw: str, provider: str, gemini_call, prompt: str, system: str) -> dict:
    """extract_json(raw); on failure AND provider != gemini, retry once against
    Gemini via gemini_call(prompt, system) then extract_json that. For Gemini,
    behave exactly like extract_json (raise on failure)."""
    try:
        return extract_json(raw)
    except ValueError:
        if provider == "gemini":
            raise
        retry_raw = gemini_call(prompt, system)
        return extract_json(retry_raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_structured_output_fallback.py -v`
Expected: PASS

- [ ] **Step 5: Wire into query_planner**

In `agent/nodes/query_planner.py`, add `import llm_client` at the top and `from agent.json_utils import extract_json, extract_json_with_gemini_retry`. Route the request through the requested model/provider and use the retry helper. Replace lines 93–105 (the `resp = rag.generate_with_failover(...)` call through `parsed = extract_json(raw_resp)`) with:

```python
        _model = state.get("requested_model") or config.LLM_MODEL_NAME
        _provider = state.get("requested_provider")
        _prompt = _DECOMPOSE_PROMPT.format(query=query, max_sq=_MAX_SUB_QUERIES)
        resp = rag.generate_with_failover(
            model=_model,
            contents=_prompt,
            gen_config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1024,
                system_instruction=_DECOMPOSE_SYSTEM,
                thinking_config=types.ThinkingConfig(thinking_budget=config.AGENT_THINKING_BUDGET),
            ),
            provider=_provider,
        )
        raw_resp = resp.text or ""

        active_provider = llm_client.resolve_provider(_model, _provider)

        def _gemini_retry(p, s):
            r = rag.generate_with_failover(
                model=config.LLM_MODEL_NAME, contents=p,
                gen_config=types.GenerateContentConfig(
                    temperature=0, max_output_tokens=1024, system_instruction=s,
                    thinking_config=types.ThinkingConfig(thinking_budget=config.AGENT_THINKING_BUDGET),
                ),
                provider="gemini",
            )
            return r.text or ""

        parsed = extract_json_with_gemini_retry(
            raw_resp, active_provider, _gemini_retry, _prompt, _DECOMPOSE_SYSTEM,
        )
```

(Leave the existing `except Exception` block and its regex fallback below unchanged.)

- [ ] **Step 6: Wire into reflexion_evaluator**

In `agent/nodes/reflexion_evaluator.py`, add `import llm_client` and `from agent.json_utils import extract_json, extract_json_with_gemini_retry`. Replace lines 124–141 (the `resp = rag.generate_with_failover(...)` call through `parsed = extract_json(raw_text)`) with:

```python
    raw_text = ""
    try:
        _model = state.get("requested_model") or config.LLM_MODEL_NAME
        _provider = state.get("requested_provider")
        _completeness_prompt = _COMPLETENESS_PROMPT.format(
            query=state["original_query"],
            source_titles=source_titles,
            answer=_truncate_at_sentence(answer, EVAL_ANSWER_CHARS),
        )
        resp = rag.generate_with_failover(
            model=_model,
            contents=_completeness_prompt,
            gen_config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=config.AGENT_THINKING_BUDGET),
            ),
            provider=_provider,
        )
        raw_text = rag.safe_extract_text(resp)

        active_provider = llm_client.resolve_provider(_model, _provider)

        def _gemini_retry(p, s):
            r = rag.generate_with_failover(
                model=config.LLM_MODEL_NAME, contents=p,
                gen_config=types.GenerateContentConfig(
                    temperature=0, max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=config.AGENT_THINKING_BUDGET),
                ),
                provider="gemini",
            )
            return rag.safe_extract_text(r)

        parsed = extract_json_with_gemini_retry(
            raw_text, active_provider, _gemini_retry, _completeness_prompt, "",
        )
```

(Leave everything below `parsed = ...` — the `completeness_score`/`action` derivation and the `except Exception` block — unchanged.)

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_structured_output_fallback.py tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add agent/json_utils.py agent/nodes/query_planner.py agent/nodes/reflexion_evaluator.py tests/test_structured_output_fallback.py
git commit -m "feat(llm): Phase 8 Task 8 — non-Gemini JSON parse Gemini-retry fallback"
```

---

### Task 9: User selection plumbing (request schemas → agent state)

**Files:**
- Modify: `routes/query.py` (`QueryRequest` fields + validator)
- Modify: `routes/agent.py` (`AgentQueryRequest` fields + validator + seed AgentState)
- Test: `tests/test_model_selection_plumbing.py`

**Interfaces:**
- Consumes: `routes.models.validate_model`.
- Produces: `QueryRequest.model`, `QueryRequest.provider`, `AgentQueryRequest.model`, `AgentQueryRequest.provider` (all `Optional[str]`, validated against the allowlist); agent request seeds `requested_model`/`requested_provider` into the initial `AgentState`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_selection_plumbing.py
import pytest
from pydantic import ValidationError

import config
from routes.agent import AgentQueryRequest
from routes.query import QueryRequest


def test_agent_request_accepts_allowlisted_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash", "anthropic/claude-haiku"])
    req = AgentQueryRequest(question="hi", model="anthropic/claude-haiku")
    assert req.model == "anthropic/claude-haiku"


def test_agent_request_rejects_off_allowlist(monkeypatch):
    monkeypatch.setattr(config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash"])
    with pytest.raises(ValidationError):
        AgentQueryRequest(question="hi", model="evil/model")


def test_query_request_model_optional(monkeypatch):
    monkeypatch.setattr(config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash"])
    req = QueryRequest(question="hi")
    assert req.model is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_selection_plumbing.py -v`
Expected: FAIL — `ValidationError` for unknown field `model`, or the model field doesn't exist yet.

- [ ] **Step 3: Add fields + validator to QueryRequest**

In `routes/query.py`, add to `QueryRequest` (after `paper_ids`):

```python
    model: Optional[str] = Field(None, description="LLM model id from the /models allowlist. Omit for default.")
    provider: Optional[str] = Field(None, description="LLM provider override (gemini|openrouter). Usually inferred from model.")

    @field_validator("model")
    @classmethod
    def validate_model_allowlisted(cls, v):
        from routes.models import validate_model
        validate_model(v, None)
        return v
```

- [ ] **Step 4: Add fields + validator to AgentQueryRequest**

In `routes/agent.py`, add to `AgentQueryRequest` (after `strategy`):

```python
    model: Optional[str] = None
    provider: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model_allowlisted(cls, v):
        from routes.models import validate_model
        validate_model(v, None)
        return v
```

- [ ] **Step 5: Seed the agent state**

In `routes/agent.py`, in `agent_query`, where the initial `AgentState` dict is built (search for `original_query`), add these two keys:

```python
        "requested_model": body.model,
        "requested_provider": body.provider,
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_model_selection_plumbing.py -v`
Expected: PASS

- [ ] **Step 7: Run the agent + query route suites**

Run: `pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add routes/query.py routes/agent.py tests/test_model_selection_plumbing.py
git commit -m "feat(llm): Phase 8 Task 9 — user model/provider selection plumbing"
```

---

### Task 10: UI model dropdown

**Files:**
- Modify: `static/index.html` (add dropdown, fetch `/models`, send `model` in query/agent requests)
- Test: manual (no JS test harness in repo)

**Interfaces:**
- Consumes: `GET /models` → `{models: [{id, provider, tools}], default}`.

- [ ] **Step 1: Add the dropdown element**

Near the query/agent controls in `static/index.html`, add a `<select id="modelSelect">`, then populate it on load:

```html
<select id="modelSelect" title="LLM model"></select>
```

```javascript
async function loadModels() {
  try {
    const r = await fetch('/models');
    const { models, default: def } = await r.json();
    const sel = document.getElementById('modelSelect');
    sel.innerHTML = '';
    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.id + (m.tools ? '' : ' (no tools)');
      opt.dataset.tools = m.tools;
      if (m.id === def) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (e) { console.warn('model list unavailable', e); }
}
loadModels();
```

- [ ] **Step 2: Send the selected model**

In the functions that POST to `/query`, `/query/stream`, and `/agent/query`, add `model: document.getElementById('modelSelect').value` to the JSON body.

- [ ] **Step 3: Grey out tool-incapable models in agent mode**

Where the UI toggles agent mode, iterate `#modelSelect` options and set `opt.disabled = (isAgentMode && opt.dataset.tools === 'false')`.

- [ ] **Step 4: Manual verification**

Run: `python start_server.py --dev`
Open the UI, confirm the dropdown lists the allowlist, defaults to the first entry, and that switching to agent mode disables any `(no tools)` entries. Submit a query and confirm the network request body carries `model`.

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat(ui): Phase 8 Task 10 — model selection dropdown fed by /models"
```

---

### Task 11: Per-model eval hook

**Files:**
- Modify: `docs/Eval/evaluate.py` (add `--model`/`--provider` CLI flags, thread to query calls)
- Test: manual (eval harness is a script)

**Interfaces:**
- Consumes: the query/agent call the harness already makes.

- [ ] **Step 1: Add CLI flags**

In `docs/Eval/evaluate.py`, in the `argparse` setup, add:

```python
    parser.add_argument("--model", default=None, help="LLM model id from the allowlist to eval against.")
    parser.add_argument("--provider", default=None, help="LLM provider override (gemini|openrouter).")
```

- [ ] **Step 2: Thread to the query calls**

Wherever the harness issues a query (search for the request payload / `generate_with_failover` / the HTTP body it POSTs), include `model`/`provider` when set. If it POSTs to the running server, add them to the JSON body; if it calls `rag.generate_with_failover` directly, pass `provider=args.provider` and use `args.model or config.LLM_MODEL_NAME`.

- [ ] **Step 3: Manual verification**

Run: `python docs/Eval/evaluate.py --model anthropic/claude-haiku` (with `OPENROUTER_API_KEY` set)
Expected: the report reflects the chosen model; without the flag, behavior is identical to today.

- [ ] **Step 4: Commit**

```bash
git add docs/Eval/evaluate.py
git commit -m "feat(eval): Phase 8 Task 11 — per-model/provider eval flag"
```

---

### Task 12: Docs + memory + full sweep

**Files:**
- Modify: `implementation_plan.md` (mark Phase 8 shipped, off by default)
- Modify: memory `MEMORY.md` + new `project_v230-phase8-openrouter.md`
- Test: full `pytest tests/`

- [ ] **Step 1: Update the plan status**

In `implementation_plan.md`, annotate Phase 8 as shipped (behind `OPENROUTER_API_KEY`; Gemini remains the default with identical behavior when the key is unset).

- [ ] **Step 2: Write the memory file**

Create `C:\Users\SANJAY\.claude\projects\F--Indicragv2-IndicRAG\memory\project_v230-phase8-openrouter.md` per the memory conventions (type: project; absolute date 2026-07-12), summarizing: `providers/` package, Gemini-shaped shim, `(provider,model)` circuit key, cross-provider failover, `/models` allowlist, capability gate, off unless `OPENROUTER_API_KEY` set. Add the one-line pointer to `MEMORY.md`.

- [ ] **Step 3: Full test sweep**

Run: `pytest tests/ -q`
Expected: PASS (all new + existing suites green).

- [ ] **Step 4: Commit**

```bash
git add implementation_plan.md
git commit -m "docs: Phase 8 OpenRouter provider shipped"
```

---

## Self-Review notes

- **Spec coverage:** Task 1↔config; Tasks 2-4↔abstraction+shim+translation (spec §1,2); Task 5↔failover+circuit key+classifiers (spec §3); Task 6↔/models (spec §6); Task 7↔capability gate (spec §4); Task 8↔JSON fallback (spec §5); Tasks 9-10↔user plumbing+UI (spec §7); Task 11↔eval hook (spec §8). All 8 spec sections covered.
- **Re-export constraint:** enforced by re-running `tests/test_agent.py`+`tests/test_hyde.py` unmodified after Tasks 5, 7, 8, 9.
- **Type consistency:** `resolve_provider`, `model_supports_tools`, `validate_model`, `ShimResponse`, `_gate_model`, `extract_json_with_gemini_retry`, `_circuit_key`, `_config`, `_fetch_openrouter_catalog`, `_catalog_cache` names are consistent across the tasks that consume them.
