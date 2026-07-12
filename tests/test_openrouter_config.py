import importlib


def test_provider_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_SELECTABLE_MODELS", raising=False)
    import config
    importlib.reload(config)
    assert config.LLM_PROVIDER == "gemini"
    assert config.LLM_FALLBACK_PROVIDER == "openrouter"
    assert config.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert config.LLM_SELECTABLE_MODELS[0] == "gemini-3.5-flash"
    assert config.MODELS_CACHE_TTL == 3600


def test_selectable_models_parsed_from_env(monkeypatch):
    monkeypatch.setenv("LLM_SELECTABLE_MODELS", "gemini-3.5-flash, anthropic/claude-haiku ,openai/gpt-5.4-nano")
    import config
    importlib.reload(config)
    assert config.LLM_SELECTABLE_MODELS == [
        "gemini-3.5-flash", "anthropic/claude-haiku", "openai/gpt-5.4-nano",
    ]
