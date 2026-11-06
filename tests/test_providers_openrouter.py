# tests/test_providers_openrouter.py — streaming contract of the OpenAI-compatible backend.
# Request/response translation lives in tests/test_provider_contract.py.
from types import SimpleNamespace
from unittest.mock import MagicMock

from providers.base import TRUNCATION_NOTE, user_text
from providers.openrouter import OpenRouterBackend


def _chunks(texts, finish_reason=None):
    """OpenAI reports finish_reason on the final choice, alongside an empty delta."""
    out = [SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=t), finish_reason=None)]) for t in texts]
    if finish_reason:
        out.append(SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None), finish_reason=finish_reason)]))
    return out


def _stream_backend(texts, finish_reason=None):
    b = OpenRouterBackend()
    b._client = MagicMock()
    b._client.chat.completions.create.return_value = _chunks(texts, finish_reason)
    return b


def test_stream_appends_truncation_note_on_length():
    b = _stream_backend(["half an ans", "wer that stops"], "length")
    out = "".join(b.generate_stream("anthropic/claude-haiku", user_text("q")))
    assert out == "half an answer that stops" + TRUNCATION_NOTE


def test_stream_stays_clean_when_model_finishes():
    b = _stream_backend(["all ", "done"], "stop")
    out = "".join(b.generate_stream("anthropic/claude-haiku", user_text("q")))
    assert out == "all done"


def test_stream_with_no_text_is_an_error():
    import pytest
    b = _stream_backend([], "stop")
    with pytest.raises(RuntimeError):
        list(b.generate_stream("m", user_text("q")))
