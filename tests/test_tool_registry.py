"""Tool registry (v3 Phase 1): schema, validation, timeout, trust, disable switch."""

import dataclasses
import time
from unittest.mock import patch

import pytest

import config
from agent import tool_registry as reg
from agent.tool_declarations import FUNCTION_DECLARATIONS


def _with(name, **changes):
    """Patch one registry entry for the duration of a test."""
    return patch.dict(reg.REGISTRY, {name: dataclasses.replace(reg.REGISTRY[name], **changes)})


def test_every_declared_tool_has_executor_trust_and_timeout():
    assert set(reg.REGISTRY) == {d["name"] for d in FUNCTION_DECLARATIONS}
    for t in reg.REGISTRY.values():
        assert t.trust in ("T3", "T4", "T5") and t.timeout_s > 0
    assert reg.REGISTRY["indicrag_retrieval"].trust == "T3"     # owned corpus
    assert reg.REGISTRY["web_search"].trust == "T4"             # external content
    assert reg.REGISTRY["execute_python"].trust == "T5"         # tool output


@pytest.mark.parametrize("args,expected", [
    ({"query": "q", "top_k": "5", "expand_query": "true"}, {"query": "q", "top_k": 5, "expand_query": True}),
    ({"query": "q", "top_k": 7.0}, {"query": "q", "top_k": 7}),
    ({"query": "q", "year_from": None}, {"query": "q"}),
    ({"query": "q", "made_up": 1}, {"query": "q"}),                 # undeclared keys dropped
])
def test_valid_arguments_are_coerced(args, expected):
    assert reg.validate(reg.REGISTRY["indicrag_retrieval"], args) == expected


@pytest.mark.parametrize("args,needle", [
    ({}, "missing required"),
    ({"query": 3}, "query: expected string"),
    ({"query": "q", "top_k": "five"}, "top_k: expected integer"),
    ({"query": "q", "top_k": True}, "top_k: expected integer"),
    ({"query": "q", "expand_query": "maybe"}, "expand_query: expected boolean"),
    ("not an object", "must be an object"),
])
def test_invalid_arguments_are_rejected(args, needle):
    with pytest.raises(ValueError, match=needle):
        reg.validate(reg.REGISTRY["indicrag_retrieval"], args)


def test_unsupported_schema_type_fails_closed():
    tool = dataclasses.replace(reg.REGISTRY["calculate"],
                               parameters={"type": "object", "properties": {"x": {"type": "array"}}})
    with pytest.raises(ValueError, match="unsupported schema type"):
        reg.validate(tool, {"x": [1]})


def test_invalid_call_never_reaches_the_executor():
    called = []
    with _with("calculate", run=lambda a: called.append(a) or {"passages": []}):
        out = reg.run("calculate", {"expression": 12})
    assert called == [] and "invalid arguments for calculate" in out["error"]


def test_unknown_tool():
    assert reg.run("rm_rf", {})["error"] == "Unknown tool: rm_rf"


def test_disabled_tool_is_not_offered_and_is_refused(monkeypatch):
    monkeypatch.setattr(config, "AGENT_DISABLED_TOOLS", {"execute_python"})
    assert "execute_python" not in [t["function"]["name"] for t in reg.openai_tools()]
    assert "disabled" in reg.run("execute_python", {"code": "print(1)"})["error"]


def test_a_hung_tool_is_cut_off_at_its_timeout(monkeypatch):
    monkeypatch.setitem(config.TOOL_TIMEOUTS, "calculate", 0.3)
    with _with("calculate", run=lambda a: time.sleep(5) or {"passages": []}):
        start = time.monotonic()
        out = reg.run("calculate", {"expression": "1+1"})
    assert time.monotonic() - start < 2
    assert "timed out" in out["error"]


def test_passages_carry_the_tools_trust_level_without_overwriting():
    fake = lambda a: {"passages": [{"text": "x"}, {"text": "y", "trust": "T4"}]}  # noqa: E731
    with _with("web_search", run=fake):
        out = reg.run("web_search", {"query": "q"})
    assert [p["trust"] for p in out["passages"]] == ["T4", "T4"]
    with _with("calculate", run=fake):
        out = reg.run("calculate", {"expression": "1"})
    assert [p["trust"] for p in out["passages"]] == ["T5", "T4"]


def test_non_dict_result_is_an_error():
    with _with("calculate", run=lambda a: "42"):
        assert "not a result object" in reg.run("calculate", {"expression": "6*7"})["error"]


def test_timeouts_come_from_config_and_env_override(tmp_path):
    """TOOL_TIMEOUTS="name=seconds" overrides a default; parsed at config import."""
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    shutil.copy(Path(__file__).resolve().parents[1] / "config.py", tmp_path / "config.py")
    env = dict(os.environ, TOOL_TIMEOUTS="web_search=7, calculate=2.5", AGENT_DISABLED_TOOLS="execute_python, ")
    r = subprocess.run([sys.executable, "-c", "import config; print(config.TOOL_TIMEOUTS['web_search'], "
                        "config.TOOL_TIMEOUTS['calculate'], sorted(config.AGENT_DISABLED_TOOLS))"],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.stdout.split("\n")[-2] == "7.0 2.5 ['execute_python']", r.stderr
