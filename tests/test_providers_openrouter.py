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
