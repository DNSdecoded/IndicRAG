"""LLM dispatcher: route by provider/model, per-(provider,model) circuit breaker,
same-provider then cross-provider failover.

generate_with_failover and llm_generate_stream keep their names, module location,
and leading signatures — rag.py re-exports them and ~52 tests patch them.
"""

import itertools
import logging
import os
import threading
import time

import config as _config
import metrics
from providers.base import LLMBackend
from providers.gemini import GeminiBackend
from providers.openrouter import OpenRouterBackend

logger = logging.getLogger(__name__)

_backends: dict[str, LLMBackend] = {}
_backends_lock = threading.Lock()
_circuit_breaker: dict[tuple[str, str], float] = {}
_circuit_failures: dict[tuple[str, str], int] = {}
_circuit_lock = threading.Lock()
_CIRCUIT_COOLDOWN = 60
# Consecutive failures before a path is skipped for _CIRCUIT_COOLDOWN. Tripping
# on the FIRST failure meant one 503 "high demand" blip took a path out for every
# request in the process for a minute; when the fallbacks blipped too, every LLM
# call then failed instantly (measured: an agent run returned "model unavailable"
# after 25.7s). Same threshold as the ChromaDB breaker in vector_store.py.
_CIRCUIT_TRIP_AFTER = 3


def _circuit_blocked(key: tuple[str, str]) -> bool:
    with _circuit_lock:
        return time.monotonic() < _circuit_breaker.get(key, 0)


def _circuit_fail(key: tuple[str, str], exc: Exception) -> None:
    """Count a failed attempt; open the circuit once failures are consecutive enough."""
    metrics.record_failover(key[0], key[1], type(exc).__name__)
    with _circuit_lock:
        _circuit_failures[key] = _circuit_failures.get(key, 0) + 1
        if _circuit_failures[key] < _CIRCUIT_TRIP_AFTER:
            return
        _circuit_failures.pop(key, None)
        _circuit_breaker[key] = time.monotonic() + _CIRCUIT_COOLDOWN
    metrics.record_circuit_trip(f"llm:{key[0]}:{key[1]}")


def _circuit_clear(key: tuple[str, str]) -> None:
    with _circuit_lock:
        _circuit_breaker.pop(key, None)
        _circuit_failures.pop(key, None)


def _record_usage(provider: str, model: str, response) -> None:
    """Export token counts when the provider reports them (Gemini does)."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    metrics.record_tokens(provider, model,
                          prompt=getattr(usage, "prompt_token_count", 0) or 0,
                          completion=getattr(usage, "candidates_token_count", 0) or 0)

# ponytail: legacy back-compat shim. Pool state now lives in GeminiBackend;
# these module globals are unused by the dispatcher and exist only so
# tests/test_agent.py::test_client_pool_idx_stays_in_bounds_under_concurrency
# (which pokes llm_client._client_pool/_client_index directly) keeps passing
# without editing the test. Drop if that test is ever migrated to GeminiBackend.
_client_pool: list = []
_client_index = itertools.cycle([])
_client_lock = threading.Lock()


def _next_client_idx() -> int:
    with _client_lock:
        return next(_client_index)


def _init_backends() -> None:
    global _backends
    if not _backends:
        with _backends_lock:
            if not _backends:
                _backends = {"gemini": GeminiBackend(), "openrouter": OpenRouterBackend()}


def get_backend(provider: str) -> LLMBackend:
    _init_backends()
    if provider not in _backends:
        # Any OpenAI-compatible provider configured in env (LLM_<NAME>_*) is built
        # on first use, so adding one is a .env change, not a code change.
        settings = _config.provider_settings(provider)
        if settings is None:
            raise ValueError(f"Unknown or unconfigured provider: {provider}")
        with _backends_lock:
            if provider not in _backends:
                _backends[provider] = OpenRouterBackend(
                    name=provider, base_url=settings["base_url"], api_key=settings["api_key"],
                    rpm=settings["rpm"])
    return _backends[provider]


def split_model(model: str | None) -> tuple[str | None, str]:
    """'nvidia:meta/llama-3.3-70b-instruct' -> ('nvidia', 'meta/llama-3.3-70b-instruct').

    The prefix counts only when it names a usable provider and has no '/', so an
    OpenRouter slug like 'nvidia/nemotron-3-super:free' is left untouched.
    """
    model = model or ""
    head, sep, rest = model.partition(":")
    if sep and rest and "/" not in head and (
            head == "gemini" or head in _backends or _config.provider_settings(head)):
        return head, rest
    return None, model


def resolve_provider(model: str, provider: str | None = None) -> str:
    """Explicit provider wins; then a 'provider:model' prefix; else infer from the
    model shape ('/' → openrouter, bare name → gemini)."""
    if provider:
        return provider
    prefixed, _ = split_model(model)
    if prefixed:
        return prefixed
    return "openrouter" if "/" in (model or "") else "gemini"


def agent_utility_model(state) -> tuple[str, str | None]:
    """(model, provider) for the agent's planner, tool routing and completeness calls.

    AGENT_UTILITY_MODEL when set; otherwise the model the answer uses (the user's
    pick, else LLM_MODEL_NAME), which was the only behaviour before it existed.
    """
    if _config.AGENT_UTILITY_MODEL:
        return _config.AGENT_UTILITY_MODEL, resolve_provider(_config.AGENT_UTILITY_MODEL)
    return state.get("requested_model") or _config.LLM_MODEL_NAME, state.get("requested_provider")


def _circuit_key(provider: str, model: str) -> tuple[str, str]:
    return (provider, model)


def _fallback_model_for(provider: str) -> str:
    """The provider's default model for cross-provider fallback.

    Must return a model that actually belongs to `provider`. The allowlist is
    Gemini-first, so taking LLM_SELECTABLE_MODELS[0] handed OpenRouter a bare
    Gemini name — OpenRouter silently rewrites that to google/<model>, routing
    the "cross-vendor" fallback straight back to the vendor that just failed
    (and onto a paid route, while the allowlist lists :free slugs).
    """
    if provider == "gemini":
        # LLM_MODEL_NAME may itself be another provider's model ("nvidia:..."),
        # which must never be sent to Gemini.
        explicit = os.getenv("LLM_GEMINI_MODEL", "").strip()
        if explicit:
            return explicit
        prefixed, _ = split_model(_config.LLM_MODEL_NAME)
        return _config.LLM_MODEL_NAME if prefixed in (None, "gemini") else _config.LLM_FALLBACK_MODEL
    configured = (_config.provider_settings(provider) or {}).get("model")
    if configured:
        return configured
    if provider != "openrouter":
        return ""
    for model in _config.LLM_SELECTABLE_MODELS:
        if resolve_provider(model) == "openrouter":   # skips 'nvidia:...' entries
            return model
    return _config.LLM_MODEL_NAME


def model_supports_tools(provider: str, model: str) -> bool:
    """Gemini always supports tools; OpenRouter is checked against the catalog.
    Default True so a catalog outage doesn't over-block."""
    if provider == "gemini":
        return True
    try:
        import routes.models as models_route
        return models_route.model_supports_tools(model)
    except Exception:
        return True


def _attempts(model: str, provider: str) -> list[tuple[str, str]]:
    """Ordered (provider, model) attempts: requested → same-provider fallback →
    cross-provider fallback."""
    _, model = split_model(model)   # the backend gets the bare model id
    attempts = [(provider, model)]
    if provider == "gemini" and _config.LLM_FALLBACK_MODEL and _config.LLM_FALLBACK_MODEL != model:
        attempts.append((provider, _config.LLM_FALLBACK_MODEL))
    if _config.LLM_PROVIDER_ORDER:
        # User-defined preference order replaces the built-in cross-provider chain.
        # Providers that are not configured (no key / no model) are skipped.
        for prov in _config.LLM_PROVIDER_ORDER:
            mdl = _fallback_model_for(prov)
            if prov == provider or not mdl:
                continue
            if prov != "gemini" and _config.provider_settings(prov) is None:
                logger.debug("[failover] provider %s in LLM_PROVIDER_ORDER is not configured", prov)
                continue
            attempts.append((prov, mdl))
        return attempts
    fb_provider = _config.LLM_FALLBACK_PROVIDER
    if fb_provider and fb_provider != provider:
        attempts.append((fb_provider, _fallback_model_for(fb_provider)))
    # Guarantee a gemini backstop. A selected OpenRouter model whose fallback
    # provider is also OpenRouter (fb_provider == provider) would otherwise have
    # no working fallback and fail outright when the free-tier model 429s.
    if _config.LLM_MODEL_NAME and not any(p == "gemini" for p, _ in attempts):
        attempts.append(("gemini", _config.LLM_MODEL_NAME))
    return attempts


class DeadlineExceeded(RuntimeError):
    """No remaining budget for another failover attempt."""


def generate_with_failover(model: str, contents, gen_config, provider: str | None = None,
                           *, deadline: float | None = None):
    """Try requested (provider, model), then same-provider then cross-provider
    fallback. Per-(provider,model) circuit breaker skips recently-dead paths.

    `deadline` is a time.monotonic() timestamp by which the caller needs an
    answer. Without it the chain walks up to three attempts at
    LLM_REQUEST_TIMEOUT_S each, so a fully-stalled chain runs ~180s — past the
    agent's own reflexion budget, which is the case config.py:415-424 describes.
    With it, an attempt that cannot finish before the deadline is not started:
    the caller gets its remaining time back to finalise a draft instead of
    spending it on a request whose answer would arrive too late to use.

    ponytail: this skips attempts, it does not cancel one already in flight —
    the provider SDKs are synchronous and own their own socket timeouts. Bounding
    what we start is the part that changes behaviour; true cancellation needs a
    request-scoped abort the SDKs do not currently expose.
    """
    provider = resolve_provider(model, provider)
    last_exc: Exception | None = None
    any_attempted = False

    for prov, mdl in _attempts(model, provider):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining < _config.LLM_MIN_ATTEMPT_S:
                logger.warning(
                    "[failover] %.1fs left, below the %.1fs an attempt needs — "
                    "stopping the chain instead of starting %s:%s",
                    remaining, _config.LLM_MIN_ATTEMPT_S, prov, mdl)
                if last_exc is not None:
                    raise last_exc
                raise DeadlineExceeded(
                    f"Out of budget before any LLM attempt ({remaining:.1f}s left)")
        key = _circuit_key(prov, mdl)
        if _circuit_blocked(key):
            logger.info(f"[failover] {prov}:{mdl} circuit open, skipping")
            continue
        backend = get_backend(prov)
        any_attempted = True
        try:
            with metrics.stage("llm_generate"):
                result = backend.generate(mdl, contents, gen_config)
            _circuit_clear(key)
            _record_usage(prov, mdl, result)
            return result
        except Exception as exc:
            last_exc = exc
            if backend.is_permanent(exc):
                raise
            logger.warning(f"[failover] {prov}:{mdl} failed ({exc!s:.120}) — next path")
            _circuit_fail(key, exc)
            continue

    if not any_attempted:
        raise RuntimeError("All configured LLM paths are circuit-open; retry after cooldown.")
    raise last_exc  # type: ignore[misc]


def generate_stream_with_failover(model: str, contents, gen_config,
                                  provider: str | None = None, *,
                                  deadline: float | None = None):
    """Stream text chunks, failing over between paths ONLY before the first chunk.

    Once a token has been handed to the caller it has been shown to a user, so a
    silent switch to another model mid-answer would splice two different answers
    together. Before the first chunk nothing is committed and the usual chain
    applies.

    Same deadline semantics as generate_with_failover: an attempt that cannot
    start in time is not started.
    """
    provider = resolve_provider(model, provider)
    last_exc: Exception | None = None
    any_attempted = False

    for prov, mdl in _attempts(model, provider):
        if deadline is not None and (deadline - time.monotonic()) < _config.LLM_MIN_ATTEMPT_S:
            if last_exc is not None:
                raise last_exc
            raise DeadlineExceeded("Out of budget before any streaming attempt")
        key = _circuit_key(prov, mdl)
        if _circuit_blocked(key):
            logger.info(f"[failover] {prov}:{mdl} circuit open, skipping (stream)")
            continue
        backend = get_backend(prov)
        any_attempted = True
        emitted = False
        started = time.monotonic()
        try:
            for chunk in backend.generate_stream(mdl, contents, gen_config):
                if not emitted:
                    metrics.stage_seconds.labels(stage="llm_ttft").observe(time.monotonic() - started)
                emitted = True
                yield chunk
            _circuit_clear(key)
            return
        except Exception as exc:
            if emitted:
                # Half an answer is already on the user's screen; the caller must
                # see the break rather than have a second model continue it.
                logger.error(f"[failover] {prov}:{mdl} broke mid-stream — no failover")
                raise
            last_exc = exc
            if backend.is_permanent(exc):
                raise
            logger.warning(f"[failover] {prov}:{mdl} failed before first token "
                           f"({exc!s:.120}) — next path")
            _circuit_fail(key, exc)
            continue

    if not any_attempted:
        raise RuntimeError("All configured LLM paths are circuit-open; retry after cooldown.")
    raise last_exc  # type: ignore[misc]


def thinking_config_for(scope: str = "standard"):
    """ThinkingConfig for a call scope ("standard" or "agent"), or None to send nothing.

    Prefers the Gemini 3.x thinking_level knob and falls back to the legacy
    thinking_budget when the configured level is empty or the installed SDK has no
    ThinkingLevel enum. Returning None means "omit the field", which lets the model
    apply its own default — MEDIUM on gemini-3.6-flash, so it is a real choice, not
    a neutral one.
    """
    from google.genai import types

    level_name = (_config.AGENT_THINKING_LEVEL if scope == "agent"
                  else _config.LLM_THINKING_LEVEL)
    if level_name:
        level = get_backend("gemini")._thinking_level(level_name)
        if level is not None:
            return types.ThinkingConfig(thinking_level=level)
        logger.warning(
            "Unknown thinking level %r for scope %s — falling back to thinking_budget",
            level_name, scope,
        )
    budget = _config.AGENT_THINKING_BUDGET if scope == "agent" else 0
    return types.ThinkingConfig(thinking_budget=budget)


def _build_gemini_stream_config(model, max_tokens, system_instruction):
    from google.genai import types
    kwargs = dict(
        temperature=_config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        safety_settings=_config.SAFETY_SETTINGS,
        system_instruction=system_instruction or _config.SYSTEM_PROMPT,
    )
    if get_backend("gemini").supports_thinking(model):
        kwargs["thinking_config"] = thinking_config_for("standard")
    return types.GenerateContentConfig(**kwargs)


def _build_openrouter_stream_config(max_tokens, system_instruction):
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=_config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        system_instruction=system_instruction or _config.SYSTEM_PROMPT,
    )


def llm_generate_stream(prompt: str, max_tokens: int = None, system_instruction: str = None,
                        model: str = None, provider: str | None = None):
    """Stream chunks with same-provider then cross-provider failover. Failover only
    BEFORE the first chunk — a mid-stream failure re-raises (can't restart output)."""
    if max_tokens is None:
        max_tokens = _config.LLM_MAX_TOKENS
    model = model or _config.LLM_MODEL_NAME
    provider = resolve_provider(model, provider)

    last_exc: Exception | None = None
    any_attempted = False
    for prov, mdl in _attempts(model, provider):
        key = _circuit_key(prov, mdl)
        if _circuit_blocked(key):
            continue
        backend = get_backend(prov)
        if prov == "gemini":
            gen_config = _build_gemini_stream_config(mdl, max_tokens, system_instruction)
        else:
            gen_config = _build_openrouter_stream_config(max_tokens, system_instruction)
        any_attempted = True
        emitted = False
        chars = 0
        started = time.monotonic()
        try:
            for chunk in backend.generate_stream(mdl, prompt, gen_config):
                if not emitted:
                    metrics.stage_seconds.labels(stage="llm_ttft").observe(time.monotonic() - started)
                emitted = True
                chars += len(chunk)
                yield chunk
            _circuit_clear(key)
            return
        except Exception as exc:
            last_exc = exc
            if emitted:
                # A mid-stream death can't be retried (the client already holds the
                # prefix), so log what tells the causes apart: elapsed near
                # LLM_STREAM_TIMEOUT_S means our own timeout cut it; elapsed well
                # under it means the provider dropped the connection.
                logger.warning(
                    "[stream] %s:%s died after %.0fs and %d chars (limit %ds) — %s: %s",
                    prov, mdl, time.monotonic() - started, chars,
                    _config.LLM_STREAM_TIMEOUT_S, type(exc).__name__, str(exc)[:200],
                )
                raise  # committed to this stream
            if backend.is_permanent(exc):
                raise
            logger.warning(f"[stream failover] {prov}:{mdl} failed ({exc!s:.120}) — next path")
            _circuit_fail(key, exc)
            continue

    if not any_attempted:
        raise RuntimeError("All configured LLM paths are circuit-open; retry after cooldown.")
    raise last_exc  # type: ignore[misc]
