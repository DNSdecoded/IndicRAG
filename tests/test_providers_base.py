from providers.base import ShimResponse, LLMBackend


def test_shim_exposes_text():
    r = ShimResponse(text="hello", function_calls=[])
    assert r.text == "hello"


def test_shim_exposes_function_calls_gemini_shape():
    r = ShimResponse(text="", function_calls=[("indicrag_retrieval", {"query": "x"})])
    parts = r.candidates[0].content.parts
    assert parts[0].function_call.name == "indicrag_retrieval"
    assert dict(parts[0].function_call.args) == {"query": "x"}


def test_shim_text_part_when_no_function_calls():
    r = ShimResponse(text="answer body", function_calls=[])
    parts = r.candidates[0].content.parts
    assert parts[0].text == "answer body"
    assert getattr(parts[0], "function_call", None) is None


def test_llmbackend_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        LLMBackend()
