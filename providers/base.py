"""Backend interface + a Gemini-shaped response shim.

Agent nodes read responses as `resp.candidates[0].content.parts[*].function_call`
and `resp.text`. Non-Gemini backends wrap their output in ShimResponse so those
call sites never change.
"""

from abc import ABC, abstractmethod
from typing import Iterator

# Appended to a stream that stopped because it ran out of output tokens. Backends
# end such streams normally, so every layer above sees a clean finish and the text
# is the only channel left to tell the user the answer is incomplete.
TRUNCATION_NOTE = (
    "\n\n*[Answer truncated — output token limit reached. "
    "Ask a narrower question or raise `LLM_MAX_TOKENS`.]*"
)


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
