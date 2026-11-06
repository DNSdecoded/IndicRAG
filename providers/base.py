"""Provider-neutral LLM request/response and the backend interface (v3 LLM-1).

The internal format is OpenAI Chat Completions: `messages` are chat turns and
`tools` are function schemas. Each backend translates at its own edge; nothing
above `providers/` sees a vendor type (B8: google.genai stays in providers/gemini.py).
"""

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

# Appended to a stream that stopped because it ran out of output tokens. Backends
# end such streams normally, so every layer above sees a clean finish and the text
# is the only channel left to tell the user the answer is incomplete.
TRUNCATION_NOTE = (
    "\n\n*[Answer truncated — output token limit reached. "
    "Ask a narrower question or raise `LLM_MAX_TOKENS`.]*"
)

THINKING_SCOPES = ("standard", "agent", "off")


@dataclass
class LLMRequest:
    """One generation request, in OpenAI chat shape.

    messages: user/assistant turns; `content` is a str or a list of parts
        {"type": "text", "text": ...} / {"type": "image_url", "image_url": {"url": data URL}}.
    tools: OpenAI function tools: [{"type": "function", "function": {name, description, parameters}}].
    thinking: "standard" | "agent" | "off" | None (None = send nothing, model default).
        Only backends with a thinking control read it.
    safety: send the Gemini safety settings (no OpenAI-compatible equivalent, B9).
    """

    messages: list
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list | None = None
    thinking: str | None = None
    safety: bool = False

    def __post_init__(self):
        if self.thinking is not None and self.thinking not in THINKING_SCOPES:
            raise ValueError(f"thinking must be one of {THINKING_SCOPES} or None, got {self.thinking!r}")
        if not self.messages:
            raise ValueError("LLMRequest needs at least one message")


def user_text(text: str, **kw) -> LLMRequest:
    """A single-turn text request: the common case."""
    return LLMRequest(messages=[{"role": "user", "content": text}], **kw)


def image_part(data: bytes, mime_type: str = "image/png") -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64.b64encode(data).decode()}"}}


def decode_data_url(url: str) -> tuple[bytes, str]:
    """data:<mime>;base64,<payload> -> (bytes, mime). Only data URLs are accepted:
    a backend must never fetch a URL a prompt handed it."""
    if not url.startswith("data:") or ";base64," not in url:
        raise ValueError("only base64 data URLs are supported for images")
    head, payload = url[5:].split(";base64,", 1)
    return base64.b64decode(payload), head or "application/octet-stream"


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class LLMResponse:
    """What every backend returns.

    finish_reason: "stop" | "length" | "tool_calls" | "content_filter", or the
        provider's raw value when it maps to none of these.
    usage: {"prompt": int, "completion": int} when the provider reports it.
    """

    text: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict | None = None


class LLMBackend(ABC):
    """One LLM vendor. Dispatch/failover lives in llm_client, not here."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(self, model: str, request: LLMRequest) -> LLMResponse: ...

    @abstractmethod
    def generate_stream(self, model: str, request: LLMRequest) -> Iterator[str]: ...

    @abstractmethod
    def is_transient(self, exc: Exception) -> bool: ...

    @abstractmethod
    def is_permanent(self, exc: Exception) -> bool: ...
