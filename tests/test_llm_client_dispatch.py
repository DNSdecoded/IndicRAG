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


def test_openrouter_fallback_model_is_an_openrouter_slug(monkeypatch):
    """Regression: the allowlist is Gemini-first, so LLM_SELECTABLE_MODELS[0] fed
    OpenRouter a bare Gemini name. OpenRouter rewrites that to google/<model>,
    sending the cross-vendor fallback back to the vendor that just failed."""
    monkeypatch.setattr(llm_client._config, "LLM_SELECTABLE_MODELS",
                        ["gemini-3.6-flash", "gemini-3.5-flash", "openai/gpt-oss-20b:free"])
    assert llm_client._fallback_model_for("openrouter") == "openai/gpt-oss-20b:free"
    assert llm_client._fallback_model_for("gemini") == llm_client._config.LLM_MODEL_NAME


def test_openrouter_fallback_falls_back_to_default_when_allowlist_has_no_slug(monkeypatch):
    monkeypatch.setattr(llm_client._config, "LLM_SELECTABLE_MODELS", ["gemini-3.6-flash"])
    assert llm_client._fallback_model_for("openrouter") == llm_client._config.LLM_MODEL_NAME


def test_attempt_chain_ends_on_a_real_openrouter_model(monkeypatch):
    monkeypatch.setattr(llm_client._config, "LLM_SELECTABLE_MODELS",
                        ["gemini-3.6-flash", "openai/gpt-oss-20b:free"])
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_PROVIDER", "openrouter")
    attempts = llm_client._attempts("gemini-3.6-flash", "gemini")
    assert ("openrouter", "openai/gpt-oss-20b:free") in attempts
    assert not any(p == "openrouter" and "/" not in m for p, m in attempts)


# ── deadline-aware failover (A4) ────────────────────────────────────────────

def _fake_backends(monkeypatch, gemini, openrouter=None):
    backends = {"gemini": gemini}
    if openrouter is not None:
        backends["openrouter"] = openrouter
    monkeypatch.setattr(llm_client, "_backends", backends)
    monkeypatch.setattr(llm_client, "get_backend", lambda p: llm_client._backends[p])
    llm_client._circuit_breaker.clear()


def test_deadline_already_passed_starts_no_attempt(monkeypatch):
    """A stalled chain used to walk three attempts at LLM_REQUEST_TIMEOUT_S each,
    running ~180s — past the agent budget it was supposed to fit inside."""
    import time

    import pytest

    class NeverCalled:
        name = "gemini"
        def generate(self, *a, **k):
            raise AssertionError("must not start an attempt past the deadline")
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    _fake_backends(monkeypatch, NeverCalled())

    with pytest.raises(llm_client.DeadlineExceeded):
        llm_client.generate_with_failover("gemini-3.5-flash", "q", object(),
                                          deadline=time.monotonic() - 1)


def test_deadline_stops_the_chain_and_reraises_the_real_error(monkeypatch):
    """When something was tried, the caller must see why it failed — not a
    generic deadline error that hides the provider's own message."""
    import time

    import pytest

    class FailingGemini:
        name = "gemini"
        def generate(self, *a, **k): raise Exception("503 UNAVAILABLE")
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    class SlowToReach:
        name = "openrouter"
        def generate(self, *a, **k):
            raise AssertionError("no budget left for the cross-provider attempt")
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    _fake_backends(monkeypatch, FailingGemini(), SlowToReach())
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm_client._config, "LLM_MIN_ATTEMPT_S", 20.0)

    # A clock the test drives: 25s of budget is enough to start the first
    # attempt, and the 15s it burns leaves too little for the second.
    ticks = iter([0.0, 15.0, 15.0, 15.0])
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: next(ticks, 15.0))

    with pytest.raises(Exception) as excinfo:
        llm_client.generate_with_failover("gemini-3.5-flash", "q", object(), deadline=25.0)
    assert "503 UNAVAILABLE" in str(excinfo.value)


def test_no_deadline_keeps_the_old_behaviour(monkeypatch):
    class FailingGemini:
        name = "gemini"
        def generate(self, *a, **k): raise Exception("503 UNAVAILABLE")
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    class OkOpenRouter:
        name = "openrouter"
        def generate(self, *a, **k): return "OR_RESP"
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    _fake_backends(monkeypatch, FailingGemini(), OkOpenRouter())
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_MODEL", "")

    assert llm_client.generate_with_failover("gemini-3.5-flash", "q", object()) == "OR_RESP"
