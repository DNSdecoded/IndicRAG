import llm_client


def test_resolve_provider_by_model_shape():
    assert llm_client.resolve_provider("gemini-3.5-flash") == "gemini"
    assert llm_client.resolve_provider("anthropic/claude-haiku") == "openrouter"


def test_explicit_provider_overrides_shape():
    assert llm_client.resolve_provider("gemini-3.5-flash", provider="openrouter") == "openrouter"


def test_circuit_key_is_provider_scoped():
    assert llm_client._circuit_key("gemini", "x") != llm_client._circuit_key("openrouter", "x")


def test_generate_with_failover_routes_to_gemini(monkeypatch):
    calls = {}

    class FakeGemini:
        name = "gemini"
        def generate(self, model, contents, gen_config, client=None):
            calls["model"] = model
            return "GEMINI_RESP"
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    monkeypatch.setattr(llm_client, "_backends", {"gemini": FakeGemini()})
    monkeypatch.setattr(llm_client, "get_backend", lambda p: llm_client._backends[p])
    out = llm_client.generate_with_failover("gemini-3.5-flash", "q", object())
    assert out == "GEMINI_RESP"
    assert calls["model"] == "gemini-3.5-flash"


def test_cross_provider_failover(monkeypatch):
    class FailingGemini:
        name = "gemini"
        def generate(self, *a, **k): raise Exception("503 UNAVAILABLE")
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    class OkOpenRouter:
        name = "openrouter"
        def generate(self, model, contents, gen_config): return "OR_RESP"
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    monkeypatch.setattr(llm_client, "_backends",
                        {"gemini": FailingGemini(), "openrouter": OkOpenRouter()})
    monkeypatch.setattr(llm_client, "get_backend", lambda p: llm_client._backends[p])
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_MODEL", "")  # skip same-provider fallback
    llm_client._circuit_breaker.clear()
    out = llm_client.generate_with_failover("gemini-3.5-flash", "q", object())
    assert out == "OR_RESP"
