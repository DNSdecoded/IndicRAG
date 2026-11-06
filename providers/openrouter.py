"""OpenAI-compatible Chat Completions backend (OpenRouter, NVIDIA NIM, OpenAI, Ollama, vLLM...).

LLMRequest is already in this API's shape, so translation is thin: the system
prompt becomes the first message, chat turns and image parts pass through, and
the reply is normalized to LLMResponse. SAFETY_SETTINGS has no equivalent here
and is not sent (B9).
"""

import collections
import json
import logging
import threading
import time
from typing import Iterator

import config
from providers.base import LLMBackend, LLMRequest, LLMResponse, ToolCall, TRUNCATION_NOTE

logger = logging.getLogger(__name__)

_ROLES = ("user", "assistant")
# OpenAI finish reasons, normalized to LLMResponse's vocabulary.
_FINISH = {"stop": "stop", "length": "length", "tool_calls": "tool_calls",
           "function_call": "tool_calls", "content_filter": "content_filter"}


def _to_messages(request: LLMRequest) -> list[dict]:
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for m in request.messages:
        if m.get("role") not in _ROLES:
            raise ValueError(f"unsupported message role {m.get('role')!r}; use LLMRequest.system for system text")
        messages.append({"role": m["role"], "content": m["content"]})
    return messages


def _usage(resp) -> dict | None:
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    return {"prompt": getattr(u, "prompt_tokens", 0) or 0,
            "completion": getattr(u, "completion_tokens", 0) or 0}


class LocalRateLimitError(RuntimeError):
    """Our own per-provider RPM cap is full; treated like a 429 so failover moves on."""


class _RpmLimiter:
    """Sliding 60s window. Waits up to LLM_RATE_LIMIT_MAX_WAIT_S for a slot, else raises.

    ponytail: per-process; with N uvicorn workers the effective cap is N x rpm.
    Share it via Redis if multi-worker deployments hit the provider's 429s.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._sent = collections.deque()
        self._lock = threading.Lock()

    def acquire(self, name: str) -> None:
        deadline = time.monotonic() + config.LLM_RATE_LIMIT_MAX_WAIT_S
        while True:
            with self._lock:
                now = time.monotonic()
                while self._sent and now - self._sent[0] >= 60:
                    self._sent.popleft()
                if len(self._sent) < self.rpm:
                    self._sent.append(now)
                    return
                wait = 60 - (now - self._sent[0])
            if now + wait > deadline:
                raise LocalRateLimitError(f"{name}: {self.rpm} requests/min cap reached")
            time.sleep(min(wait, 0.5))


class OpenRouterBackend(LLMBackend):
    """OpenAI Chat Completions backend. Defaults to OpenRouter; any other
    OpenAI-compatible provider (NVIDIA NIM, OpenAI, Groq, Ollama, vLLM...) is the
    same class with its own name, base URL and key (see config.provider_settings)."""

    def __init__(self, name: str = "openrouter", base_url: str | None = None,
                 api_key: str | None = None, rpm: int = 0):
        self._limiter = _RpmLimiter(rpm) if rpm > 0 else None
        self._client = None
        self._lock = threading.Lock()
        self._name = name
        self._base_url = base_url
        self._api_key = api_key
        # Current OpenAI models reject `max_tokens` on Chat Completions and require
        # `max_completion_tokens`; other compatible servers expect `max_tokens`.
        self._token_param = "max_completion_tokens" if name == "openai" else "max_tokens"

    @property
    def name(self) -> str:
        return self._name

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    # OpenRouter reads config lazily so its env can change in tests.
                    # provider_settings also reads LLM_OPENROUTER_API_KEY, the
                    # documented form; the pre-registered backend saw only the old name.
                    api_key = self._api_key or (
                        config.OPENROUTER_API_KEY
                        or (config.provider_settings("openrouter") or {}).get("api_key", "")
                        if self._name == "openrouter" else "")
                    base_url = self._base_url or config.OPENROUTER_BASE_URL
                    if not api_key:
                        env = self._name.upper().replace("-", "_")
                        raise ValueError(
                            f"{env}_API_KEY not configured. Set it (or LLM_{env}_API_KEY) "
                            f"in .env to use {self._name}."
                        )
                    from openai import OpenAI
                    # SDK defaults are 600s per request with 2 retries — one stalled
                    # call would outlive the agent budget and 504 the whole run.
                    self._client = OpenAI(
                        api_key=api_key,
                        base_url=base_url,
                        timeout=config.LLM_REQUEST_TIMEOUT_S,
                        max_retries=1,
                    )
        return self._client

    def _params(self, model: str, request: LLMRequest, stream: bool) -> dict:
        params = {"model": model, "messages": _to_messages(request), "stream": stream}
        if stream:
            # Per-request override: the streamed request stays open for the whole
            # generation, so the client's unary timeout would cut long answers off
            # mid-stream. Client default still applies to non-streaming calls.
            params["timeout"] = config.LLM_STREAM_TIMEOUT_S
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params[self._token_param] = request.max_tokens
        if request.tools:
            params["tools"] = request.tools
            params["tool_choice"] = "auto"
        # request.thinking has no portable Chat Completions equivalent; not sent.
        return params

    def generate(self, model: str, request: LLMRequest) -> LLMResponse:
        client = self._get_client()
        if self._limiter:
            self._limiter.acquire(self._name)
        resp = client.chat.completions.create(**self._params(model, request, stream=False))
        # OpenRouter can answer 200 with an error body and `choices: null` (seen live
        # on an overloaded :free model); indexing it raised an opaque TypeError.
        if not getattr(resp, "choices", None):
            raise RuntimeError(f"{self._name} returned no choices for {model}: "
                               f"{getattr(resp, 'error', None)!s:.200}")
        choice = resp.choices[0]
        msg = choice.message
        calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(tc.function.name, args if isinstance(args, dict) else {}))
        raw = getattr(choice, "finish_reason", None) or "stop"
        return LLMResponse(text=msg.content or "", tool_calls=calls,
                           finish_reason=_FINISH.get(raw, raw), usage=_usage(resp))

    def generate_stream(self, model: str, request: LLMRequest) -> Iterator[str]:
        client = self._get_client()
        if self._limiter:
            self._limiter.acquire(self._name)
        emitted = False
        finish = None
        for chunk in client.chat.completions.create(**self._params(model, request, stream=True)):
            if not chunk.choices:
                continue
            finish = getattr(chunk.choices[0], "finish_reason", None) or finish
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                emitted = True
                yield text
        if not emitted:
            raise RuntimeError(f"No text generated from {self._name} stream")
        # "length" is Gemini's MAX_TOKENS: a clean stream end that hides a cut-off
        # answer. Same sentinel so both providers behave alike.
        if finish == "length":
            logger.warning("%s stream hit max_tokens for %s — answer truncated", self._name, model)
            yield TRUNCATION_NOTE

    def is_transient(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (429, 500, 502, 503):
            return True
        name = type(exc).__name__
        return name in ("RateLimitError", "LocalRateLimitError", "APIConnectionError",
                        "InternalServerError", "APITimeoutError")

    def is_permanent(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in (400, 404, 422):
            return True
        return type(exc).__name__ in ("BadRequestError", "NotFoundError")
