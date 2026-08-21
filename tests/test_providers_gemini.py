from types import SimpleNamespace

import pytest

from providers.gemini import GeminiBackend


class _FakeModels:
    """Rejects thinking_budget=0 the way gemini-3.6-flash does (400 INVALID_ARGUMENT)."""

    def __init__(self, stream_chunks=("hi",), finish_reason=None):
        self.calls = []
        self.levels = []  # thinking_level per call, parallel to calls
        self._stream_chunks = stream_chunks
        self.finish_reason = finish_reason

    def _check(self, config):
        tc = getattr(config, "thinking_config", None)
        budget = getattr(tc, "thinking_budget", None) if tc is not None else None
        self.calls.append(budget)
        self.levels.append(getattr(tc, "thinking_level", None) if tc is not None else None)
        if budget is not None:
            # Gemini 3.x rejects the legacy budget field outright, whatever its value.
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


def test_generate_retries_with_a_level_when_budget_rejected(zero_budget_config):
    """The retry must ask for MINIMAL thinking, not simply drop the field.

    Omitting thinking_config means the model's own default — MEDIUM on
    gemini-3.6-flash — so "thinking off" used to produce medium thinking, billed
    and taken out of the answer's share of max_output_tokens.
    """
    from google.genai import types

    models = _FakeModels()
    b = _backend_with(models)
    resp = b.generate("gemini-3.6-flash", "q", zero_budget_config, client=b._pool[0])
    assert resp.text == "ok"
    assert models.calls == [0, None]                       # budget rejected, then not resent
    assert models.levels == [None, types.ThinkingLevel.MINIMAL]
    assert "gemini-3.6-flash" in b._zero_budget_rejected


def test_second_call_sends_the_level_without_retrying(zero_budget_config):
    from google.genai import types

    models = _FakeModels()
    b = _backend_with(models)
    client = b._pool[0]
    b.generate("gemini-3.6-flash", "q", zero_budget_config, client=client)
    b.generate("gemini-3.6-flash", "q2", zero_budget_config, client=client)
    assert models.calls == [0, None, None]                 # no repeat of the failing call
    assert models.levels[-1] == types.ThinkingLevel.MINIMAL


def test_dynamic_budget_translates_to_no_thinking_config():
    """-1 means "model decides", which is exactly what omitting the field does."""
    from google.genai import types

    models = _FakeModels()
    b = _backend_with(models)
    cfg = types.GenerateContentConfig(
        max_output_tokens=16, thinking_config=types.ThinkingConfig(thinking_budget=-1)
    )
    b.generate("gemini-3.6-flash", "q", cfg, client=b._pool[0])
    assert models.calls == [-1, None]
    assert models.levels == [None, None]


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


# ---------------------------------------------------------------------------
# Thinking LEVEL rejection (gemini-3.7-flash refuses MINIMAL)
# ---------------------------------------------------------------------------
def _level_config(name="MINIMAL"):
    from google.genai import types
    return types.GenerateContentConfig(
        max_output_tokens=32,
        thinking_config=types.ThinkingConfig(
            thinking_level=getattr(types.ThinkingLevel, name)),
    )


class _RejectsLevel:
    """Refuses one named level the way gemini-3.7-flash refuses MINIMAL."""

    def __init__(self, bad="MINIMAL"):
        self.bad = bad
        self.levels_seen = []
        self.models = self

    def generate_content(self, model=None, contents=None, config=None):
        from providers.gemini import GeminiBackend
        lvl = GeminiBackend._current_level_name(config)
        self.levels_seen.append(lvl)
        if lvl == self.bad:
            raise RuntimeError(
                "400 INVALID_ARGUMENT. Thinking level MINIMAL is not supported "
                "for this model. Please retry with other thinking level.")
        return types_ns(text="ok")


def types_ns(**kw):
    return type("R", (), kw)()


def _fresh_backend():
    from providers.gemini import GeminiBackend
    GeminiBackend._level_rejected.clear()
    return GeminiBackend()


def test_unsupported_thinking_level_escalates_and_succeeds():
    b = _fresh_backend()
    client = _RejectsLevel()
    out = b.generate("gemini-3.7-flash", "q", _level_config(), client=client)
    assert out.text == "ok"
    # Retried one level UP, not down: a refused level means the model will not
    # think that little, so less thinking cannot succeed.
    assert client.levels_seen == ["MINIMAL", "LOW"]


def test_the_rejection_is_remembered_so_only_one_call_pays_for_it():
    from providers.gemini import GeminiBackend
    b = _fresh_backend()
    client = _RejectsLevel()
    b.generate("gemini-3.7-flash", "q", _level_config(), client=client)
    b.generate("gemini-3.7-flash", "q2", _level_config(), client=client)
    # Second call skips MINIMAL entirely — three attempts total, not four.
    assert client.levels_seen == ["MINIMAL", "LOW", "LOW"]
    assert ("gemini-3.7-flash", "MINIMAL") in GeminiBackend._level_rejected


def test_rejection_is_learned_per_model_not_globally():
    """3.6-flash accepts MINIMAL; 3.7 refusing it must not change 3.6's behavior."""
    b = _fresh_backend()
    b.generate("gemini-3.7-flash", "q", _level_config(), client=_RejectsLevel())
    ok = _RejectsLevel(bad="NOTHING")
    b.generate("gemini-3.6-flash", "q", _level_config(), client=ok)
    assert ok.levels_seen == ["MINIMAL"]


def test_an_unrelated_400_still_propagates():
    """Narrowness matters: only a thinking-level refusal may be retried, or a
    genuinely bad request gets silently re-sent with different thinking."""
    import pytest
    b = _fresh_backend()

    class _OtherError:
        models = None

        def generate_content(self, **kw):
            raise RuntimeError("400 INVALID_ARGUMENT. Unsupported MIME type.")

    c = _OtherError()
    c.models = c
    with pytest.raises(RuntimeError, match="MIME"):
        b.generate("gemini-3.7-flash", "q", _level_config(), client=c)
