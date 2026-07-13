import pytest
from pydantic import ValidationError

import config
from routes.agent import AgentQueryRequest
from routes.query import QueryRequest


def test_agent_request_accepts_allowlisted_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash", "anthropic/claude-haiku"])
    req = AgentQueryRequest(question="hi", model="anthropic/claude-haiku")
    assert req.model == "anthropic/claude-haiku"


def test_agent_request_rejects_off_allowlist(monkeypatch):
    monkeypatch.setattr(config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash"])
    with pytest.raises(ValidationError):
        AgentQueryRequest(question="hi", model="evil/model")


def test_query_request_model_optional(monkeypatch):
    monkeypatch.setattr(config, "LLM_SELECTABLE_MODELS", ["gemini-3.5-flash"])
    req = QueryRequest(question="hi")
    assert req.model is None
