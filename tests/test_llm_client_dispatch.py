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


# ── circuit threshold ───────────────────────────────────────────────────────

def test_one_blip_does_not_open_the_circuit_but_three_in_a_row_do(monkeypatch):
    """Tripping on the first 503 took a path out process-wide for a minute; with
    the fallbacks also blipping, every LLM call then failed instantly."""
    calls = []

    class Flaky:
        name = "gemini"
        def generate(self, model, contents, gen_config):
            calls.append(model)
            raise Exception("503 UNAVAILABLE")
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    class Ok:
        name = "openrouter"
        def generate(self, model, contents, gen_config): return "OR_RESP"
        def is_permanent(self, e): return False
        def is_transient(self, e): return True

    _fake_backends(monkeypatch, Flaky(), Ok())
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_MODEL", "")
    key = llm_client._circuit_key("gemini", "gemini-3.5-flash")

    for attempt in range(1, llm_client._CIRCUIT_TRIP_AFTER + 1):
        assert not llm_client._circuit_blocked(key), f"open after {attempt - 1} failure(s)"
        assert llm_client.generate_with_failover("gemini-3.5-flash", "q", object()) == "OR_RESP"
    assert llm_client._circuit_blocked(key)
    assert len(calls) == llm_client._CIRCUIT_TRIP_AFTER


def test_a_success_resets_the_failure_count(monkeypatch):
    llm_client._circuit_failures.clear()
    key = llm_client._circuit_key("gemini", "m")
    for _ in range(llm_client._CIRCUIT_TRIP_AFTER - 1):
        llm_client._circuit_fail(key, Exception("503"))
    llm_client._circuit_clear(key)
    llm_client._circuit_fail(key, Exception("503"))
    assert not llm_client._circuit_blocked(key)


# ── configurable OpenAI-compatible providers + preferred order ─────────────

def _set_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_provider_prefix_routes_to_that_provider_and_strips_it(monkeypatch):
    _set_env(monkeypatch, NVIDIA_API_KEY="k")
    assert llm_client.resolve_provider("nvidia:meta/llama-3.3-70b-instruct") == "nvidia"
    assert llm_client.split_model("nvidia:meta/llama-3.3-70b-instruct") == ("nvidia", "meta/llama-3.3-70b-instruct")
    # An OpenRouter slug that merely contains a colon is not a provider prefix.
    assert llm_client.resolve_provider("nvidia/nemotron-3-super-120b-a12b:free") == "openrouter"
    assert llm_client.resolve_provider("gemini-3.8-flash") == "gemini"


def test_unconfigured_prefix_is_not_treated_as_a_provider(monkeypatch):
    monkeypatch.delenv("FOO_API_KEY", raising=False)
    monkeypatch.delenv("LLM_FOO_API_KEY", raising=False)
    assert llm_client.split_model("foo:bar") == (None, "foo:bar")


def test_preset_and_explicit_base_urls(monkeypatch):
    _set_env(monkeypatch, LLM_NVIDIA_API_KEY="k", LLM_NVIDIA_MODEL="meta/llama-3.3-70b-instruct")
    s = llm_client._config.provider_settings("nvidia")
    assert s["base_url"] == "https://integrate.api.nvidia.com/v1" and s["model"] == "meta/llama-3.3-70b-instruct"
    # No preset: unusable until a base URL is configured.
    _set_env(monkeypatch, LLM_ACME_API_KEY="k")
    assert llm_client._config.provider_settings("acme") is None
    _set_env(monkeypatch, LLM_ACME_BASE_URL="https://llm.acme.example/v1")
    assert llm_client._config.provider_settings("acme")["base_url"] == "https://llm.acme.example/v1"


def test_get_backend_builds_configured_provider_lazily(monkeypatch):
    _set_env(monkeypatch, LLM_GROQ_API_KEY="k")
    llm_client._init_backends()
    monkeypatch.delitem(llm_client._backends, "groq", raising=False)
    b = llm_client.get_backend("groq")
    assert b.name == "groq" and b._base_url == "https://api.groq.com/openai/v1"
    llm_client._backends.pop("groq", None)


def test_preferred_order_drives_the_failover_chain(monkeypatch):
    _set_env(monkeypatch, LLM_NVIDIA_API_KEY="k", LLM_NVIDIA_MODEL="meta/llama-3.3-70b-instruct",
             LLM_OPENAI_API_KEY="k", LLM_OPENAI_MODEL="gpt-x")
    monkeypatch.delenv("LLM_ACME_BASE_URL", raising=False)
    monkeypatch.setattr(llm_client._config, "LLM_PROVIDER_ORDER", ["nvidia", "acme", "openai", "gemini"])
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_MODEL", "")
    monkeypatch.setattr(llm_client._config, "LLM_MODEL_NAME", "gemini-3.8-flash")
    assert llm_client._attempts("gemini-3.8-flash", "gemini") == [
        ("gemini", "gemini-3.8-flash"),
        ("nvidia", "meta/llama-3.3-70b-instruct"),   # acme skipped: not configured
        ("openai", "gpt-x"),
    ]


def test_prefixed_primary_never_sends_its_model_to_gemini(monkeypatch):
    _set_env(monkeypatch, LLM_NVIDIA_API_KEY="k")
    monkeypatch.delenv("LLM_GEMINI_MODEL", raising=False)
    monkeypatch.setattr(llm_client._config, "LLM_PROVIDER_ORDER", ["gemini"])
    monkeypatch.setattr(llm_client._config, "LLM_MODEL_NAME", "nvidia:meta/llama-3.3-70b-instruct")
    monkeypatch.setattr(llm_client._config, "LLM_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    assert llm_client._attempts("nvidia:meta/llama-3.3-70b-instruct", "nvidia") == [
        ("nvidia", "meta/llama-3.3-70b-instruct"),
        ("gemini", "gemini-3.5-flash-lite"),
    ]


def test_openai_uses_max_completion_tokens():
    from providers.openrouter import OpenRouterBackend
    from google.genai import types
    cfg = types.GenerateContentConfig(max_output_tokens=50)
    assert "max_completion_tokens" in OpenRouterBackend("openai", "u", "k")._params("m", "hi", cfg, False)
    assert "max_tokens" in OpenRouterBackend("nvidia", "u", "k")._params("m", "hi", cfg, False)


def test_rpm_limiter_caps_requests_then_raises(monkeypatch):
    from providers import openrouter
    monkeypatch.setattr(openrouter.config, "LLM_RATE_LIMIT_MAX_WAIT_S", 0)
    lim = openrouter._RpmLimiter(2)
    lim.acquire("nvidia")
    lim.acquire("nvidia")
    import pytest
    with pytest.raises(openrouter.LocalRateLimitError):
        lim.acquire("nvidia")
    assert openrouter.OpenRouterBackend("nvidia", "u", "k").is_transient(openrouter.LocalRateLimitError("x"))


def test_nvidia_defaults_to_40_rpm(monkeypatch):
    monkeypatch.setenv("LLM_NVIDIA_API_KEY", "k")
    monkeypatch.delenv("LLM_NVIDIA_RPM", raising=False)
    assert llm_client._config.provider_settings("nvidia")["rpm"] == 40
    monkeypatch.setenv("LLM_NVIDIA_RPM", "10")
    assert llm_client._config.provider_settings("nvidia")["rpm"] == 10


def test_openrouter_fallback_skips_prefixed_slugs(monkeypatch):
    monkeypatch.setenv("LLM_NVIDIA_API_KEY", "k")
    monkeypatch.delenv("LLM_OPENROUTER_MODEL", raising=False)
    monkeypatch.setattr(llm_client._config, "LLM_SELECTABLE_MODELS",
                        ["gemini-3.8-flash", "nvidia:meta/llama-3.3-70b-instruct", "google/gemma-4-31b-it:free"])
    assert llm_client._fallback_model_for("openrouter") == "google/gemma-4-31b-it:free"
