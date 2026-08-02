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


def generate_with_failover(model: str, contents, gen_config, provider: str | None = None):
    """Try requested (provider, model), then same-provider then cross-provider
    fallback. Per-(provider,model) circuit breaker skips recently-dead paths."""
    provider = resolve_provider(model, provider)
    last_exc: Exception | None = None
    any_attempted = False

    for prov, mdl in _attempts(model, provider):
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


def _build_gemini_stream_config(model, max_tokens, system_instruction):
    from google.genai import types
    kwargs = dict(
        temperature=_config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        safety_settings=_config.SAFETY_SETTINGS,
        system_instruction=system_instruction or _config.SYSTEM_PROMPT,
    )
    if get_backend("gemini").supports_thinking(model):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
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
