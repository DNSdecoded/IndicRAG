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


def _chunks(texts, finish_reason=None):
    """OpenAI reports finish_reason on the final choice, alongside an empty delta."""
    out = [SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=t), finish_reason=None)]) for t in texts]
    if finish_reason:
        out.append(SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None), finish_reason=finish_reason)]))
    return out


def _stream_backend(texts, finish_reason=None):
    b = OpenRouterBackend()
    b._client = MagicMock()
    b._client.chat.completions.create.return_value = _chunks(texts, finish_reason)
    return b


def test_stream_appends_truncation_note_on_length():
    from providers.base import TRUNCATION_NOTE
    b = _stream_backend(["half an ans", "wer that stops"], "length")
    out = "".join(b.generate_stream("anthropic/claude-haiku", "q", _cfg()))
    assert out == "half an answer that stops" + TRUNCATION_NOTE


def test_stream_stays_clean_when_model_finishes():
    b = _stream_backend(["all ", "done"], "stop")
    out = "".join(b.generate_stream("anthropic/claude-haiku", "q", _cfg()))
    assert out == "all done"


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
