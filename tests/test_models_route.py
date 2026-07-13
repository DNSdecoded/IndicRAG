import pytest
import routes.models as m


def test_bare_name_is_gemini_no_network(monkeypatch):
    monkeypatch.setattr(m.config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash"])
    def _boom():
        raise AssertionError("should not fetch catalog for bare Gemini name")
    monkeypatch.setattr(m, "_fetch_openrouter_catalog", _boom)
    out = m.list_models()
    assert out == [{"id": "gemini-3.5-flash", "provider": "gemini", "tools": True}]


def test_openrouter_model_enriched_from_catalog(monkeypatch):
    monkeypatch.setattr(m.config, "LLM_SELECTABLE_MODELS", ["anthropic/claude-haiku"])
    monkeypatch.setattr(m, "_fetch_openrouter_catalog",
                        lambda: {"anthropic/claude-haiku": {"supported_parameters": ["tools", "temperature"]}})
    m._catalog_cache.invalidate()
    out = m.list_models()
    assert out == [{"id": "anthropic/claude-haiku", "provider": "openrouter", "tools": True}]


def test_model_without_tools_flagged(monkeypatch):
    monkeypatch.setattr(m.config, "LLM_SELECTABLE_MODELS", ["openai/gpt-5.4-nano"])
    monkeypatch.setattr(m, "_fetch_openrouter_catalog",
                        lambda: {"openai/gpt-5.4-nano": {"supported_parameters": ["temperature"]}})
    m._catalog_cache.invalidate()
    assert m.model_supports_tools("openai/gpt-5.4-nano") is False


def test_validate_model_rejects_off_allowlist(monkeypatch):
    monkeypatch.setattr(m.config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash"])
    with pytest.raises(ValueError):
        m.validate_model("evil/model", None)


def test_validate_model_allows_none():
    m.validate_model(None, None)  # no override — fine
