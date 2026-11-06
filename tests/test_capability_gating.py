import pytest

import agent.nodes.tool_selector as ts


@pytest.fixture
def no_utility_model(monkeypatch):
    """Routing follows the answer model when AGENT_UTILITY_MODEL is unset."""
    monkeypatch.setattr(ts.llm_client._config, "AGENT_UTILITY_MODEL", "")


def test_gate_falls_back_to_gemini_for_tool_incapable(monkeypatch, no_utility_model):
    monkeypatch.setattr(ts.llm_client, "resolve_provider", lambda m, p=None: "openrouter")
    monkeypatch.setattr(ts.llm_client, "model_supports_tools", lambda prov, m: False)
    state = {"requested_model": "openai/gpt-5.4-nano", "requested_provider": None}
    provider, model = ts._gate_model(state)
    assert provider == "gemini"
    assert model == ts.config.LLM_MODEL_NAME


def test_gate_keeps_tool_capable_model(monkeypatch, no_utility_model):
    monkeypatch.setattr(ts.llm_client, "resolve_provider", lambda m, p=None: "openrouter")
    monkeypatch.setattr(ts.llm_client, "model_supports_tools", lambda prov, m: True)
    state = {"requested_model": "anthropic/claude-haiku", "requested_provider": None}
    provider, model = ts._gate_model(state)
    assert provider == "openrouter"
    assert model == "anthropic/claude-haiku"


def test_gate_default_when_no_request(no_utility_model):
    state = {}
    provider, model = ts._gate_model(state)
    assert provider == "gemini"
    assert model == ts.config.LLM_MODEL_NAME


def test_utility_model_routes_even_when_user_picked_another(monkeypatch):
    """The user's pick governs the answer; routing uses the fast utility model."""
    monkeypatch.setattr(ts.llm_client._config, "AGENT_UTILITY_MODEL", "gemini-3.5-flash-lite")
    state = {"requested_model": "anthropic/claude-haiku", "requested_provider": None}
    assert ts._gate_model(state) == ("gemini", "gemini-3.5-flash-lite")


def test_selector_uses_the_models_tool_calls(monkeypatch, no_utility_model):
    """The model's ToolCalls become the node's requests, and the request carries the
    schemas. Neither was covered before LLM-1: only the fallback path was tested."""
    from agent.tool_registry import openai_tools
    from providers.base import LLMResponse, ToolCall

    sent = {}

    def fake(model, request, provider=None, **kw):
        sent["request"] = request
        return LLMResponse(tool_calls=[ToolCall("web_search", {"query": "latest 6G news"})],
                           finish_reason="tool_calls")

    monkeypatch.setattr(ts.rag, "generate_with_failover", fake)
    out = ts.tool_selector_node({"original_query": "latest 6G news"})
    names = [c["name"] for c in out["tool_calls_requested"]]
    assert "web_search" in names
    assert {"name": "web_search", "args": {"query": "latest 6G news"}} in out["tool_calls_requested"]
    assert sent["request"].tools == openai_tools() and sent["request"].thinking == "agent"
