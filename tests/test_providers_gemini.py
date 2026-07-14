from providers.gemini import GeminiBackend


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
