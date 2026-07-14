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
