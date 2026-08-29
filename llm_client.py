"""LLM dispatcher: route by provider/model, per-(provider,model) circuit breaker,
same-provider then cross-provider failover.

generate_with_failover and llm_generate_stream keep their names, module location,
and leading signatures — rag.py re-exports them and ~52 tests patch them.
"""

import itertools
import logging
import threading
import time

import config as _config
from providers.base import LLMBackend
from providers.gemini import GeminiBackend
from providers.openrouter import OpenRouterBackend

logger = logging.getLogger(__name__)

_backends: dict[str, LLMBackend] = {}
_backends_lock = threading.Lock()
_circuit_breaker: dict[tuple[str, str], float] = {}
_circuit_lock = threading.Lock()
_CIRCUIT_COOLDOWN = 60


def _circuit_blocked(key: tuple[str, str]) -> bool:
    with _circuit_lock:
        return time.monotonic() < _circuit_breaker.get(key, 0)


def _circuit_trip(key: tuple[str, str]) -> None:
    with _circuit_lock:
        _circuit_breaker[key] = time.monotonic() + _CIRCUIT_COOLDOWN


def _circuit_clear(key: tuple[str, str]) -> None:
    with _circuit_lock:
        _circuit_breaker.pop(key, None)

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
        raise ValueError(f"Unknown provider: {provider}")
    return _backends[provider]


def resolve_provider(model: str, provider: str | None = None) -> str:
    """Explicit provider wins; else infer from model shape ('/' → openrouter)."""
    if provider:
        return provider
    return "openrouter" if "/" in (model or "") else "gemini"


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
        return _config.LLM_MODEL_NAME
    for model in _config.LLM_SELECTABLE_MODELS:
        if "/" in model:                      # slug shape == OpenRouter
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
    attempts = [(provider, model)]
    if provider == "gemini" and _config.LLM_FALLBACK_MODEL and _config.LLM_FALLBACK_MODEL != model:
        attempts.append((provider, _config.LLM_FALLBACK_MODEL))
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
            result = backend.generate(mdl, contents, gen_config)
            _circuit_clear(key)
            return result
        except Exception as exc:
            last_exc = exc
            if backend.is_permanent(exc):
                raise
            logger.warning(f"[failover] {prov}:{mdl} failed ({exc!s:.120}) — next path")
            _circuit_trip(key)
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
        try:
            for chunk in backend.generate_stream(mdl, contents, gen_config):
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
            _circuit_trip(key)
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
            _circuit_trip(key)
            continue

    if not any_attempted:
        raise RuntimeError("All configured LLM paths are circuit-open; retry after cooldown.")
    raise last_exc  # type: ignore[misc]
