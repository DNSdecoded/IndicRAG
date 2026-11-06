"""Failover simulation suite (v3 P0-LLM-2).

Four releases shipped failover fixes found in production. These scenarios script
what each provider returns and assert the attempt sequence, the circuit state
and the deadline behaviour. The fakes subclass the REAL backends and replace
only the network call, so production error classification (is_permanent /
is_transient) and the dispatch loop are tested together, the seam where those
bugs lived.

Chain under test, for the default answer model "g-main":
    gemini:g-main -> gemini:g-lite (same-provider fallback)
    -> nvidia:n-model -> openrouter:o-model (LLM_PROVIDER_ORDER)
"""

import pytest

import config
import llm_client
from providers.base import LLMResponse, user_text
from providers.gemini import GeminiBackend
from providers.openrouter import LocalRateLimitError, OpenRouterBackend

CHAIN = [("gemini", "g-main"), ("gemini", "g-lite"), ("nvidia", "n-model"), ("openrouter", "o-model")]


class StatusError(Exception):
    """An SDK error carrying an HTTP status, as openai/google-genai errors do."""

    def __init__(self, status, msg=""):
        super().__init__(f"{status} {msg}".strip())
        self.status_code = status


class APITimeoutError(Exception):
    """Same class name the openai SDK raises; classification matches on it."""


class Sim:
    """Scripted outcomes per (provider, model), consumed one per attempt."""

    def __init__(self):
        self.script: dict = {}
        self.log: list = []

    def on(self, prov, model, *outcomes):
        self.script.setdefault((prov, model), []).extend(outcomes)
        return self

    def _next(self, prov, model):
        self.log.append((prov, model))
        queue = self.script.get((prov, model)) or ["ok"]
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _fake(base, sim, name):
    class Fake(base):
        @property
        def name(self):
            return name

        def generate(self, model, request, client=None):
            out = sim._next(name, model)
            return LLMResponse(text=f"{name}:{model}", usage={"prompt": 3, "completion": 2}) if out == "ok" else out

        def generate_stream(self, model, request, client=None):
            out = sim._next(name, model)
            if out == "ok":
                yield f"{name}:{model}"
            elif out == "midstream":
                yield "half an answer"
                raise StatusError(503, "UNAVAILABLE")
            elif out == "empty":
                raise RuntimeError(f"No text generated from {name} stream")
            else:
                yield from out
    return Fake() if base is GeminiBackend else Fake(name=name)


@pytest.fixture
def sim(monkeypatch):
    s = Sim()
    backends = {"gemini": _fake(GeminiBackend, s, "gemini"),
                "nvidia": _fake(OpenRouterBackend, s, "nvidia"),
                "openrouter": _fake(OpenRouterBackend, s, "openrouter")}
    monkeypatch.setattr(llm_client, "get_backend", lambda p: backends[p])
    monkeypatch.setattr(config, "LLM_MODEL_NAME", "g-main")
    monkeypatch.setattr(config, "LLM_FALLBACK_MODEL", "g-lite")
    monkeypatch.setattr(config, "LLM_PROVIDER_ORDER", ["gemini", "nvidia", "openrouter"])
    monkeypatch.setattr(config, "LLM_MIN_ATTEMPT_S", 20.0)
    monkeypatch.delenv("LLM_GEMINI_MODEL", raising=False)
    models = {"nvidia": "n-model", "openrouter": "o-model"}
    monkeypatch.setattr(config, "provider_settings",
                        lambda p: {"model": models[p], "base_url": "u", "api_key": "k", "rpm": 0} if p in models else None)
    tokens = []
    monkeypatch.setattr(llm_client.metrics, "record_tokens", lambda prov, mdl, **kw: tokens.append((prov, mdl, kw)))
    s.tokens = tokens
    llm_client._circuit_breaker.clear()
    llm_client._circuit_failures.clear()
    yield s
    llm_client._circuit_breaker.clear()
    llm_client._circuit_failures.clear()


def gen(**kw):
    return llm_client.generate_with_failover("g-main", user_text("q"), **kw)


def stream(**kw):
    return "".join(llm_client.generate_stream_with_failover("g-main", user_text("q"), **kw))


# ── unary: which path answers ───────────────────────────────────────────────

def test_01_healthy_primary_answers_alone(sim):
    assert gen().text == "gemini:g-main" and sim.log == CHAIN[:1]


@pytest.mark.parametrize("exc", [StatusError(503, "UNAVAILABLE"), Exception("429 RESOURCE_EXHAUSTED")],
                         ids=["02_gemini_503", "03_gemini_429"])
def test_gemini_overload_falls_to_the_lighter_gemini(sim, exc):
    sim.on("gemini", "g-main", exc)
    assert gen().text == "gemini:g-lite" and sim.log == CHAIN[:2]


def test_04_both_gemini_paths_down_crosses_provider(sim):
    sim.on("gemini", "g-main", StatusError(503)).on("gemini", "g-lite", StatusError(503))
    assert gen().text == "nvidia:n-model" and sim.log == CHAIN[:3]


def test_05_everything_down_raises_the_last_real_error(sim):
    for prov, mdl in CHAIN:
        sim.on(prov, mdl, StatusError(503, f"{prov} down"))
    with pytest.raises(StatusError, match="openrouter down"):
        gen()
    assert sim.log == CHAIN


def test_06_malformed_request_is_not_retried_anywhere(sim):
    sim.on("gemini", "g-main", Exception("400 INVALID_ARGUMENT. Bad tool schema."))
    with pytest.raises(Exception, match="INVALID_ARGUMENT"):
        gen()
    assert sim.log == CHAIN[:1]


def test_07_retired_model_410_fails_over(sim):
    """A provider retiring the configured model (410 Gone) must not end the request."""
    sim.on("nvidia", "n-model", StatusError(410, "Gone"))
    out = llm_client.generate_with_failover("nvidia:n-model", user_text("q"))
    assert out.text == "gemini:g-main"
    assert sim.log == [("nvidia", "n-model"), ("gemini", "g-main")]


def test_08_unknown_model_404_on_openai_path_surfaces(sim):
    """404 is a configuration error the operator must see, not a blip to route around."""
    sim.on("nvidia", "n-model", StatusError(404, "model not found"))
    with pytest.raises(StatusError):
        llm_client.generate_with_failover("nvidia:n-model", user_text("q"))
    assert sim.log == [("nvidia", "n-model")]


@pytest.mark.parametrize("exc", [APITimeoutError("timed out"), StatusError(500), StatusError(502),
                                 RuntimeError("nvidia returned no choices for n-model: overloaded")],
                         ids=["09_timeout", "10_500", "11_502", "12_empty_choices"])
def test_openai_path_transient_failures_move_on(sim, exc):
    sim.on("gemini", "g-main", StatusError(503)).on("gemini", "g-lite", StatusError(503))
    sim.on("nvidia", "n-model", exc)
    assert gen().text == "openrouter:o-model" and sim.log == CHAIN


# ── circuit breaker ─────────────────────────────────────────────────────────

def test_13_three_consecutive_failures_open_the_circuit(sim):
    sim.on("gemini", "g-main", StatusError(503), StatusError(503), StatusError(503), "ok")
    for _ in range(3):
        gen()
    sim.log.clear()
    assert gen().text == "gemini:g-lite"          # g-main skipped without a call
    assert sim.log == [("gemini", "g-lite")]


def test_14_a_success_resets_the_count(sim):
    sim.on("gemini", "g-main", StatusError(503), StatusError(503), "ok", StatusError(503), StatusError(503), "ok")
    for _ in range(5):
        gen()
    assert ("gemini", "g-main") not in llm_client._circuit_breaker


def test_15_local_rpm_cap_never_opens_the_circuit(sim):
    """Our own rate limiter being full says nothing about the provider's health."""
    sim.on("gemini", "g-main", StatusError(503)).on("gemini", "g-lite", StatusError(503))
    sim.on("nvidia", "n-model", *[LocalRateLimitError("nvidia: 40 requests/min cap reached")] * 4)
    for _ in range(4):
        sim.on("gemini", "g-main", StatusError(503)).on("gemini", "g-lite", StatusError(503))
        assert gen().text == "openrouter:o-model"
    assert ("nvidia", "n-model") not in llm_client._circuit_breaker


def test_16_cooldown_expiry_lets_the_path_back_in(sim):
    llm_client._circuit_breaker[("gemini", "g-main")] = 0.0   # an expired trip
    assert gen().text == "gemini:g-main"


def test_17_all_paths_circuit_open_fails_fast(sim):
    for key in CHAIN:
        llm_client._circuit_breaker[key] = float("inf")
    with pytest.raises(RuntimeError, match="circuit-open"):
        gen()
    assert sim.log == []


# ── deadlines ───────────────────────────────────────────────────────────────

def test_18_no_budget_starts_no_attempt(sim, monkeypatch):
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: 100.0)
    with pytest.raises(llm_client.DeadlineExceeded):
        gen(deadline=110.0)                         # 10 s left < 20 s an attempt needs
    assert sim.log == []


def test_19_running_out_mid_chain_reraises_the_real_error(sim, monkeypatch):
    # 50 s left before the first attempt, 15 s (< the 20 s minimum) after it. Keyed
    # on attempts made, not call count: other code reads the same clock.
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: 135.0 if sim.log else 100.0)
    sim.on("gemini", "g-main", StatusError(503, "the real cause"))
    with pytest.raises(StatusError, match="the real cause"):
        gen(deadline=150.0)
    assert sim.log == CHAIN[:1]


# ── streaming ───────────────────────────────────────────────────────────────

def test_20_stream_fails_over_before_the_first_token(sim):
    sim.on("gemini", "g-main", StatusError(503)).on("gemini", "g-lite", "empty")
    assert stream() == "nvidia:n-model" and sim.log == CHAIN[:3]


def test_21_stream_never_switches_after_a_token(sim):
    sim.on("gemini", "g-main", "midstream")
    got = []
    with pytest.raises(StatusError):
        for piece in llm_client.generate_stream_with_failover("g-main", user_text("q")):
            got.append(piece)
    assert got == ["half an answer"] and sim.log == CHAIN[:1]


def test_22_stream_permanent_error_is_not_retried(sim):
    sim.on("gemini", "g-main", Exception("400 INVALID_ARGUMENT"))
    with pytest.raises(Exception, match="INVALID_ARGUMENT"):
        stream()
    assert sim.log == CHAIN[:1]


# ── what gets recorded ──────────────────────────────────────────────────────

def test_23_usage_is_recorded_for_the_model_that_actually_served(sim):
    sim.on("gemini", "g-main", StatusError(503)).on("gemini", "g-lite", StatusError(503))
    gen()
    assert sim.tokens == [("nvidia", "n-model", {"prompt": 3, "completion": 2})]


def test_24_a_stale_caller_cannot_reach_a_backend(sim):
    with pytest.raises(TypeError):
        llm_client.generate_with_failover("g-main", "q", object())
    assert sim.log == []


def test_scenario_count():
    """The gate requires at least 20 scripted scenarios (P0-LLM-2 acceptance)."""
    import inspect
    import sys
    mod = sys.modules[__name__]
    n = 0
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_") or name == "test_scenario_count":
            continue
        marks = [m for m in getattr(fn, "pytestmark", []) if m.name == "parametrize"]
        n += len(marks[0].args[1]) if marks else 1
    assert n >= 20, n
