"""Model roles (v3 LLM-1, B6): MODEL_ROLE_<ROLE>, legacy keys with a warning, defaults.

config resolves at import time and load_dotenv() walks up from config.py's own
directory, so each case imports a COPY of config.py from an empty temp dir in a
fresh interpreter: the developer's .env cannot leak in.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROLE_ENV = [f"MODEL_ROLE_{r.upper()}" for r in
            ("answer", "utility", "embedding", "reranker", "nli", "translation", "vision")]
LEGACY = ["LLM_MODEL_NAME", "AGENT_UTILITY_MODEL", "EMBEDDING_MODEL_NAME", "NLI_MODEL_NAME"]


def _load(tmp_path, **env):
    shutil.copy(ROOT / "config.py", tmp_path / "config.py")
    clean = {k: v for k, v in os.environ.items() if k not in ROLE_ENV + LEGACY}
    clean.update(env)
    code = ("import json, config; print(json.dumps({'roles': config.MODEL_ROLES, "
            "'deprecated': config.DEPRECATED_SETTINGS, 'llm': config.LLM_MODEL_NAME, "
            "'utility': config.AGENT_UTILITY_MODEL, 'rerank': config.RERANK_MODEL_NAME, "
            "'nllb': [config.TRANSLATION_MODEL_EN_TO_INDIC, config.TRANSLATION_MODEL_INDIC_TO_EN], "
            "'vision': config.VISION_MODEL}))")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=clean, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1]), r.stderr


def test_defaults_cover_all_seven_roles_and_warn_nothing(tmp_path):
    out, err = _load(tmp_path)
    assert set(out["roles"]) == {"answer", "utility", "embedding", "reranker", "nli", "translation", "vision"}
    assert out["roles"]["embedding"] == "BAAI/bge-m3" and out["rerank"] == "BAAI/bge-reranker-v2-m3"
    assert out["deprecated"] == [] and "deprecated" not in err


def test_role_sets_the_model_without_a_warning(tmp_path):
    out, err = _load(tmp_path, MODEL_ROLE_ANSWER="nvidia:meta/llama-3.3-70b-instruct")
    assert out["llm"] == out["roles"]["answer"] == "nvidia:meta/llama-3.3-70b-instruct"
    assert out["vision"] == out["llm"]          # vision follows the answer model unless set
    assert out["deprecated"] == []


def test_legacy_key_still_works_with_a_deprecation_warning(tmp_path):
    out, err = _load(tmp_path, LLM_MODEL_NAME="gemini-legacy")
    assert out["llm"] == "gemini-legacy"
    assert out["deprecated"] == ["LLM_MODEL_NAME"]
    assert "LLM_MODEL_NAME is deprecated; set MODEL_ROLE_ANSWER" in err


def test_role_wins_over_legacy_key(tmp_path):
    out, _ = _load(tmp_path, LLM_MODEL_NAME="old", MODEL_ROLE_ANSWER="new")
    assert out["llm"] == "new" and out["deprecated"] == []


def test_empty_legacy_utility_still_means_follow_the_answer_model(tmp_path):
    out, _ = _load(tmp_path, AGENT_UTILITY_MODEL="", MODEL_ROLE_ANSWER="ans")
    assert out["utility"] == "" and out["roles"]["utility"] == "ans"


@pytest.mark.parametrize("env,key,value", [
    ("MODEL_ROLE_RERANKER", "rerank", "BAAI/bge-reranker-base"),          # was hard-coded (B6)
    ("MODEL_ROLE_VISION", "vision", "openrouter:qwen/qwen-vl"),
])
def test_previously_fixed_models_are_configurable(tmp_path, env, key, value):
    out, _ = _load(tmp_path, **{env: value})
    assert out[key] == value


def test_translation_role_sets_both_directions(tmp_path):
    out, _ = _load(tmp_path, MODEL_ROLE_TRANSLATION="facebook/nllb-200-1.3B")
    assert out["nllb"] == ["facebook/nllb-200-1.3B"] * 2
