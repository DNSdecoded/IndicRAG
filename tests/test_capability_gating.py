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
