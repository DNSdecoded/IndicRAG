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
from providers.base import LLMBackend, ShimResponse, TRUNCATION_NOTE

logger = logging.getLogger(__name__)


def _flatten_contents(contents) -> str:
    """Extract prompt text from google-genai Content/Part objects (or plain
    strings). `str(contents)` would emit a Python repr, not the real text, so
    walk the structure and join the actual `.text` fields."""
    if isinstance(contents, str):
        return contents
    parts: list[str] = []
    for item in contents if isinstance(contents, (list, tuple)) else [contents]:
        if hasattr(item, "parts") and item.parts is not None:
            for part in item.parts:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
        elif getattr(item, "text", None):
            parts.append(item.text)
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts) if parts else str(contents)


def _to_messages(contents, gen_config) -> list[dict]:
    """contents (str | Content/Part objects) + system_instruction → OpenAI messages."""
    messages = []
    sys_inst = getattr(gen_config, "system_instruction", None)
    if sys_inst:
        messages.append({"role": "system", "content": str(sys_inst)})
    messages.append({"role": "user", "content": _flatten_contents(contents)})
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
                    # SDK defaults are 600s per request with 2 retries — one stalled
                    # call would outlive the agent budget and 504 the whole run.
                    self._client = OpenAI(
                        api_key=config.OPENROUTER_API_KEY,
                        base_url=config.OPENROUTER_BASE_URL,
                        timeout=config.LLM_REQUEST_TIMEOUT_S,
                        max_retries=1,
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
        finish = None
        for chunk in client.chat.completions.create(**self._params(model, contents, gen_config, stream=True)):
            if not chunk.choices:
                continue
            finish = getattr(chunk.choices[0], "finish_reason", None) or finish
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                emitted = True
                yield text
        if not emitted:
            raise RuntimeError("No text generated from OpenRouter stream")
        # OpenAI's "length" is Gemini's MAX_TOKENS: a clean stream end that hides
        # a cut-off answer. Same sentinel so both providers behave alike.
        if finish == "length":
            logger.warning("OpenRouter stream hit max_tokens for %s — answer truncated", model)
            yield TRUNCATION_NOTE

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
