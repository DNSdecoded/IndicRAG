from types import SimpleNamespace

import pytest

from providers.gemini import GeminiBackend


class _FakeModels:
    """Rejects thinking_budget=0 the way gemini-3.6-flash does (400 INVALID_ARGUMENT)."""

    def __init__(self, stream_chunks=("hi",), finish_reason=None):
        self.calls = []
        self._stream_chunks = stream_chunks
        self.finish_reason = finish_reason

    def _check(self, config):
        tc = getattr(config, "thinking_config", None)
        budget = getattr(tc, "thinking_budget", None) if tc is not None else None
        self.calls.append(budget)
        if budget == 0:
            raise Exception("400 INVALID_ARGUMENT. Request contains an invalid argument.")

    def generate_content(self, model, contents, config):
        self._check(config)
        return SimpleNamespace(text="ok")

    def generate_content_stream(self, model, contents, config):
        self._check(config)
        return iter(_stream(self._stream_chunks, self.finish_reason))


def _stream(chunks, finish_reason=None):
    """Gemini reports finish_reason on the last chunk's candidate, not the chunk."""
    out = []
    for i, c in enumerate(chunks):
        last = i == len(chunks) - 1
        cands = [SimpleNamespace(finish_reason=finish_reason)] if (last and finish_reason) else None
        out.append(SimpleNamespace(text=c, candidates=cands))
    return out


def _backend_with(models):
    b = GeminiBackend()
    b._zero_budget_rejected = set()          # per-test, not the shared class set
    b._pool = [SimpleNamespace(models=models)]
    return b


@pytest.fixture
def zero_budget_config():
    from google.genai import types
    return types.GenerateContentConfig(
        max_output_tokens=16, thinking_config=types.ThinkingConfig(thinking_budget=0)
    )


def test_generate_retries_without_thinking_when_zero_budget_rejected(zero_budget_config):
    models = _FakeModels()
    b = _backend_with(models)
    resp = b.generate("gemini-3.6-flash", "q", zero_budget_config, client=b._pool[0])
    assert resp.text == "ok"
    assert models.calls == [0, None]                       # rejected, then retried without
    assert "gemini-3.6-flash" in b._zero_budget_rejected


def test_second_call_skips_thinking_config_entirely(zero_budget_config):
    models = _FakeModels()
    b = _backend_with(models)
    client = b._pool[0]
    b.generate("gemini-3.6-flash", "q", zero_budget_config, client=client)
    b.generate("gemini-3.6-flash", "q2", zero_budget_config, client=client)
    assert models.calls == [0, None, None]                 # no repeat of the failing call


def test_stream_retries_before_any_token_emitted(zero_budget_config):
    models = _FakeModels(stream_chunks=("a", "b"))
    b = _backend_with(models)
    out = "".join(b.generate_stream("gemini-3.6-flash", "q", zero_budget_config, client=b._pool[0]))
    assert out == "ab"
    assert models.calls == [0, None]


def test_stream_appends_truncation_note_on_max_tokens(zero_budget_config):
    from providers.base import TRUNCATION_NOTE
    models = _FakeModels(stream_chunks=("half an ans", "wer that stops"),
                         finish_reason="FinishReason.MAX_TOKENS")
    b = _backend_with(models)
    out = "".join(b.generate_stream("gemini-3.5-flash", "q", zero_budget_config, client=b._pool[0]))
    assert out == "half an answer that stops" + TRUNCATION_NOTE


def test_stream_stays_clean_when_model_finishes(zero_budget_config):
    models = _FakeModels(stream_chunks=("all ", "done"), finish_reason="FinishReason.STOP")
    b = _backend_with(models)
    out = "".join(b.generate_stream("gemini-3.5-flash", "q", zero_budget_config, client=b._pool[0]))
    assert out == "all done"


def test_other_invalid_argument_errors_still_propagate():
    from google.genai import types

    class _AlwaysBad(_FakeModels):
        def generate_content(self, model, contents, config):
            self.calls.append(getattr(config, "thinking_config", None))
            raise Exception("400 INVALID_ARGUMENT. Bad tool schema.")

    models = _AlwaysBad()
    b = _backend_with(models)
    # No zero thinking budget → nothing to adapt, the error must surface.
    cfg = types.GenerateContentConfig(max_output_tokens=16)
    with pytest.raises(Exception, match="INVALID_ARGUMENT"):
        b.generate("gemini-3.6-flash", "q", cfg, client=b._pool[0])
    assert b._zero_budget_rejected == set()


def test_name():
    assert GeminiBackend().name == "gemini"


def test_is_transient_recognizes_503_and_429():
    b = GeminiBackend()
    assert b.is_transient(Exception("503 UNAVAILABLE"))
    assert b.is_transient(Exception("RESOURCE_EXHAUSTED"))
    assert not b.is_transient(Exception("400 INVALID_ARGUMENT"))


def test_is_permanent_only_for_malformed_request():
    b = GeminiBackend()
    assert b.is_permanent(Exception("INVALID_ARGUMENT"))
    assert not b.is_permanent(Exception("401 UNAUTHENTICATED"))  # must fail over keys


def test_supports_thinking_gates_gemma():
    b = GeminiBackend()
    assert b.supports_thinking("gemini-3.5-flash")
    assert not b.supports_thinking("gemma-4-26b-a4b-it")
