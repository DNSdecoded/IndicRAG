"""Provider contract suite (v3 LLM-1): one LLMRequest, two backends, one LLMResponse.

Each backend translates the neutral request at its own edge. These tests pin
what each one sends and that both replies normalize to the same LLMResponse,
so a caller never needs to know which provider served it.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import config
from providers import gemini as gem
from providers.base import (LLMBackend, LLMRequest, LLMResponse, ToolCall, decode_data_url,
                            image_part, user_text)
from providers.gemini import GeminiBackend
from providers.openrouter import OpenRouterBackend, _to_messages

TOOLS = [{"type": "function", "function": {
    "name": "indicrag_retrieval", "description": "retrieve",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
PNG = b"\x89PNG\r\n\x1a\nfake"


def _full_request(**kw):
    base = dict(
        messages=[{"role": "user", "content": "first question"},
                  {"role": "assistant", "content": "first answer"},
                  {"role": "user", "content": [image_part(PNG), {"type": "text", "text": "and this figure?"}]}],
        system="be terse", temperature=0.2, max_tokens=300, tools=TOOLS,
    )
    base.update(kw)
    return LLMRequest(**base)


# ── the neutral types ───────────────────────────────────────────────────────

def test_request_rejects_unknown_thinking_scope_and_empty_messages():
    with pytest.raises(ValueError):
        user_text("q", thinking="maximum")
    with pytest.raises(ValueError):
        LLMRequest(messages=[])


def test_images_must_be_data_urls():
    data, mime = decode_data_url(image_part(PNG)["image_url"]["url"])
    assert (data, mime) == (PNG, "image/png")
    with pytest.raises(ValueError):
        decode_data_url("https://example.com/x.png")   # a backend must never fetch a prompt's URL


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        LLMBackend()


# ── OpenAI-compatible: request side ─────────────────────────────────────────

def test_openai_keeps_roles_system_and_image_parts():
    msgs = _to_messages(_full_request())
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == "be terse"
    assert msgs[3]["content"][0]["type"] == "image_url"   # v2.6 flattened roles and dropped images


def test_openai_rejects_system_role_inside_messages():
    with pytest.raises(ValueError):
        _to_messages(LLMRequest(messages=[{"role": "system", "content": "x"}]))


@pytest.mark.parametrize("name,param", [("openrouter", "max_tokens"), ("openai", "max_completion_tokens")])
def test_openai_params(name, param):
    p = OpenRouterBackend(name=name)._params("m", _full_request(), stream=False)
    assert p["temperature"] == 0.2 and p[param] == 300
    assert p["tools"] == TOOLS and p["tool_choice"] == "auto"
    assert "timeout" not in p


def test_openai_stream_gets_its_own_timeout_and_no_tools_without_tools():
    p = OpenRouterBackend()._params("m", user_text("q"), stream=True)
    assert p["timeout"] == config.LLM_STREAM_TIMEOUT_S
    assert "tools" not in p and "temperature" not in p


# ── OpenAI-compatible: response side ────────────────────────────────────────

def _openai_backend(resp):
    b = OpenRouterBackend()
    b._client = MagicMock()
    b._client.chat.completions.create.return_value = resp
    return b


def _openai_resp(content=None, tool_calls=None, finish="stop", usage=(11, 7)):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]))


def test_openai_text_reply():
    r = _openai_backend(_openai_resp("the answer")).generate("m", user_text("q"))
    assert r == LLMResponse(text="the answer", finish_reason="stop", usage={"prompt": 11, "completion": 7})


def test_openai_tool_call_reply_and_bad_json_args():
    calls = [SimpleNamespace(function=SimpleNamespace(name="indicrag_retrieval", arguments=json.dumps({"query": "x"}))),
             SimpleNamespace(function=SimpleNamespace(name="calculate", arguments="{not json"))]
    r = _openai_backend(_openai_resp(tool_calls=calls, finish="tool_calls")).generate("m", user_text("q"))
    assert r.tool_calls == [ToolCall("indicrag_retrieval", {"query": "x"}), ToolCall("calculate", {})]
    assert r.finish_reason == "tool_calls" and r.text == ""


@pytest.mark.parametrize("raw,norm", [("length", "length"), ("content_filter", "content_filter"),
                                      ("function_call", "tool_calls"), ("weird", "weird")])
def test_openai_finish_reasons(raw, norm):
    assert _openai_backend(_openai_resp("x", finish=raw)).generate("m", user_text("q")).finish_reason == norm


def test_openai_no_choices_is_an_error():
    with pytest.raises(RuntimeError):
        _openai_backend(SimpleNamespace(choices=None, error="overloaded")).generate("m", user_text("q"))


# ── Gemini: request side ────────────────────────────────────────────────────

def test_gemini_maps_roles_system_images_and_tools():
    contents, cfg = GeminiBackend().to_genai(_full_request())
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[2].parts[0].inline_data.data == PNG and contents[2].parts[1].text == "and this figure?"
    assert cfg.system_instruction == "be terse" and cfg.temperature == 0.2 and cfg.max_output_tokens == 300
    assert [f.name for f in cfg.tools[0].function_declarations] == ["indicrag_retrieval"]
    assert cfg.tool_config.function_calling_config.mode.name == "AUTO"


def test_gemini_safety_only_when_requested():
    _, cfg = GeminiBackend().to_genai(user_text("q"))
    assert cfg.safety_settings is None
    _, cfg = GeminiBackend().to_genai(user_text("q", safety=True))
    assert len(cfg.safety_settings) == 4
    assert {str(s.threshold).rsplit(".", 1)[-1] for s in cfg.safety_settings} == {config.SAFETY_THRESHOLD}


def test_gemini_thinking_scopes(monkeypatch):
    monkeypatch.setattr(config, "LLM_THINKING_LEVEL", "")
    monkeypatch.setattr(config, "AGENT_THINKING_LEVEL", "")
    monkeypatch.setattr(config, "AGENT_THINKING_BUDGET", 128)
    b = GeminiBackend()
    assert b.to_genai(user_text("q"))[1].thinking_config is None              # None: model default
    assert b.to_genai(user_text("q", thinking="off"))[1].thinking_config.thinking_budget == 0
    assert b.to_genai(user_text("q", thinking="standard"))[1].thinking_config.thinking_budget == 0
    assert b.to_genai(user_text("q", thinking="agent"))[1].thinking_config.thinking_budget == 128


def test_gemini_thinking_level_preferred_when_configured(monkeypatch):
    monkeypatch.setattr(config, "LLM_THINKING_LEVEL", "low")
    tc = GeminiBackend().to_genai(user_text("q", thinking="standard"))[1].thinking_config
    if GeminiBackend._thinking_level("low") is None:
        pytest.skip("installed google-genai has no ThinkingLevel")
    assert tc.thinking_level.name == "LOW" and tc.thinking_budget is None


# ── Gemini: response side ───────────────────────────────────────────────────

def _gem_raw(parts=(), finish="STOP", usage=(5, 3), candidates=True, block=None):
    p = [SimpleNamespace(**x) for x in parts]
    cand = SimpleNamespace(content=SimpleNamespace(parts=p), finish_reason=SimpleNamespace(name=finish))
    return SimpleNamespace(candidates=[cand] if candidates else [],
                           prompt_feedback=SimpleNamespace(block_reason=block),
                           usage_metadata=SimpleNamespace(prompt_token_count=usage[0], candidates_token_count=usage[1]))


def test_gemini_text_reply_skips_thought_parts():
    r = gem._to_response(_gem_raw([{"text": "thinking...", "thought": True}, {"text": "the "}, {"text": "answer"}]))
    assert r == LLMResponse(text="the answer", finish_reason="stop", usage={"prompt": 5, "completion": 3})


def test_gemini_function_call_reply():
    fc = SimpleNamespace(name="indicrag_retrieval", args={"query": "x"})
    r = gem._to_response(_gem_raw([{"function_call": fc, "text": None}]))
    assert r.tool_calls == [ToolCall("indicrag_retrieval", {"query": "x"})] and r.finish_reason == "tool_calls"


@pytest.mark.parametrize("raw,norm", [("MAX_TOKENS", "length"), ("SAFETY", "content_filter"),
                                      ("RECITATION", "content_filter"), ("OTHER", "other")])
def test_gemini_finish_reasons(raw, norm):
    assert gem._to_response(_gem_raw([{"text": "x"}], finish=raw)).finish_reason == norm


def test_gemini_blocked_prompt_has_no_candidates():
    r = gem._to_response(_gem_raw(candidates=False, block="SAFETY"))
    assert r.text == "" and r.finish_reason == "content_filter"


def test_gemini_generate_goes_through_translation():
    client = MagicMock()
    client.models.generate_content.return_value = _gem_raw([{"text": "ok"}])
    r = GeminiBackend().generate("gemini-x", user_text("q", system="s"), client=client)
    assert isinstance(r, LLMResponse) and r.text == "ok"
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["config"].system_instruction == "s" and kwargs["contents"][0].role == "user"


# ── same request, same meaning on both sides ────────────────────────────────

def test_both_backends_carry_the_same_request():
    req = _full_request()
    msgs = _to_messages(req)
    contents, cfg = GeminiBackend().to_genai(req)
    assert msgs[0]["content"] == cfg.system_instruction
    assert len(msgs) - 1 == len(contents)
    oa = OpenRouterBackend()._params("m", req, stream=False)
    assert oa["temperature"] == cfg.temperature and oa["max_tokens"] == cfg.max_output_tokens
    assert [t["function"]["name"] for t in oa["tools"]] == [f.name for f in cfg.tools[0].function_declarations]


def test_both_backends_normalize_a_truncated_reply_identically():
    oa = _openai_backend(_openai_resp("cut", finish="length", usage=(5, 3))).generate("m", user_text("q"))
    gm = gem._to_response(_gem_raw([{"text": "cut"}], finish="MAX_TOKENS"))
    assert oa == gm
