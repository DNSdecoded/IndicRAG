import pytest
import agent.json_utils as ju


def test_extract_json_gemini_retry_used_for_non_gemini():
    out = ju.extract_json_with_gemini_retry(
        raw="not json at all",
        provider="openrouter",
        gemini_call=lambda p, s: '{"sub_queries": ["ok"]}',
        prompt="p", system="s",
    )
    assert out == {"sub_queries": ["ok"]}


def test_extract_json_no_retry_for_gemini():
    with pytest.raises(ValueError):
        ju.extract_json_with_gemini_retry(
            raw="not json", provider="gemini",
            gemini_call=lambda p, s: '{"x": 1}', prompt="p", system="s",
        )


def test_extract_json_passes_through_valid():
    out = ju.extract_json_with_gemini_retry(
        raw='{"a": 1}', provider="openrouter",
        gemini_call=lambda p, s: "{}", prompt="p", system="s",
    )
    assert out == {"a": 1}
