"""
Gemini client pool: round-robin load balancing, failover, circuit breaker.
"""

import itertools
import logging
import threading
import time

import config
from google import genai

logger = logging.getLogger(__name__)

# Client pool — one genai.Client per API key, round-robin load balanced.
# Lazily initialised so the module can be imported without keys (retrieval-only mode).
_client_pool: list[genai.Client] = []
_client_lock = threading.Lock()
_client_index = itertools.cycle([])  # replaced on init

_circuit_breaker: dict[str, float] = {}
_CIRCUIT_COOLDOWN = 60


def _init_client_pool() -> None:
    """Build the client pool from config.LLM_API_KEY_POOL (called under lock)."""
    global _client_pool, _client_index
    if not config.LLM_API_KEY_POOL:
        raise ValueError(
            "Google Gemini API key not configured. "
            "Set LLM_API_KEY (single) or LLM_API_KEYS (comma-separated) in .env."
        )
    _client_pool = [genai.Client(api_key=k) for k in config.LLM_API_KEY_POOL]
    _client_index = itertools.cycle(range(len(_client_pool)))


def _ensure_pool() -> None:
    """Double-checked-locking lazy init, shared by every call site."""
    if not _client_pool:
        with _client_lock:
            if not _client_pool:
                _init_client_pool()


def _next_client_idx() -> int:
    """Advance the round-robin counter under the pool lock (BUG-001/002)."""
    with _client_lock:
        return next(_client_index)


def _get_client() -> genai.Client:
    """Return the next client from the round-robin pool (thread-safe)."""
    _ensure_pool()
    return _client_pool[_next_client_idx()]


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (429, 503):
        return True
    msg = str(exc)
    return "503" in msg or "429" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg


def _is_permanent(exc: Exception) -> bool:
    """Errors where retrying other keys/models is pointless: malformed request only.

    Auth failures (401/403, UNAUTHENTICATED, API key not valid) are NOT permanent:
    in a multi-key pool a single revoked/invalid key must fail over to the next key,
    not abort the whole pool. Everything else (500s, connection resets, read timeouts,
    unclassified) is likewise treated as worth failing over.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (400, 404):
        return True
    msg = str(exc)
    return "INVALID_ARGUMENT" in msg


def _supports_thinking(model: str) -> bool:
    """Gemini models accept thinking_config; Gemma (the fallback) rejects it with a
    permanent 400. Gate the field on the model so failover to Gemma doesn't abort."""
    return "gemma" not in model.lower()


def _with_cache(client, model: str, gen_config):
    """Return gen_config, or a copy that references a cached system prompt.

    When explicit caching is enabled and a cache exists for this client's
    (model, system_instruction, tools) prefix, swap the inline system_instruction
    and tools for `cached_content` (they're mutually exclusive). Otherwise return
    gen_config unchanged. Never raises — caching is best-effort.
    """
    try:
        sys_inst = getattr(gen_config, "system_instruction", None)
        if not sys_inst:
            return gen_config
        import gemini_cache
        name = gemini_cache.get_or_create(client, model, sys_inst,
                                          getattr(gen_config, "tools", None))
        if not name:
            return gen_config
        return gen_config.model_copy(update={
            "system_instruction": None, "tools": None, "cached_content": name,
        })
    except Exception:
        return gen_config


def generate_with_failover(model: str, contents, gen_config):
    """
    Call generate_content rotating through all API keys on 503/429 errors.

    Tries every client in the pool exactly once.  If all keys fail on the
    primary model, retries with LLM_FALLBACK_MODEL before giving up.
    Uses a circuit breaker to skip models that recently failed on all keys.
    """
    _ensure_pool()

    pool = _client_pool
    models_to_try = [model]
    if config.LLM_FALLBACK_MODEL and config.LLM_FALLBACK_MODEL != model:
        models_to_try.append(config.LLM_FALLBACK_MODEL)

    # Round-robin: start from the next key in rotation, not always pool[0]
    start = _next_client_idx()
    ordered_pool = pool[start:] + pool[:start]

    last_exc: Exception | None = None
    any_attempted = False
    for current_model in models_to_try:
        tripped_until = _circuit_breaker.get(current_model, 0)
        if time.monotonic() < tripped_until:
            logger.info(f"[Gemini failover] {current_model} circuit open, skipping")
            continue

        any_attempted = True
        all_failed = True
        for offset, client in enumerate(ordered_pool, 1):
            try:
                call_config = _with_cache(client, current_model, gen_config)
                result = client.models.generate_content(
                    model=current_model, contents=contents, config=call_config
                )
                _circuit_breaker.pop(current_model, None)
                return result
            except Exception as exc:
                last_exc = exc
                if _is_permanent(exc):
                    raise  # bad request / auth — other keys and models won't help
                logger.warning(
                    f"[Gemini failover] {current_model} key #{offset}/{len(pool)} "
                    f"failed ({getattr(exc, 'status_code', '?')}): {exc!s:.120} — trying next"
                )
                continue

        if all_failed:
            _circuit_breaker[current_model] = time.monotonic() + _CIRCUIT_COOLDOWN
            if current_model == model and len(models_to_try) > 1:
                logger.warning(f"[Gemini failover] {model} circuit tripped for {_CIRCUIT_COOLDOWN}s, falling back to {config.LLM_FALLBACK_MODEL}")

    if not any_attempted:
        raise RuntimeError(
            "All configured Gemini models are currently circuit-open; retry after cooldown."
        )
    raise last_exc  # type: ignore[misc]


def _stream_gen_config(model: str, max_tokens: int, system_instruction: str):
    """Build a streaming GenerateContentConfig for a specific model.

    thinking_config is only set for models that support it — Gemma rejects it with
    a permanent 400. Disabling thinking on Gemini keeps it from spending most of
    max_output_tokens on thoughts and truncating the visible answer.
    """
    from google.genai import types

    kwargs = dict(
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        safety_settings=config.SAFETY_SETTINGS,
        system_instruction=system_instruction or config.SYSTEM_PROMPT,
    )
    if _supports_thinking(model):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def llm_generate_stream(prompt: str, max_tokens: int = None, system_instruction: str = None):
    """Generator: stream LLM response chunks with key rotation and model failover.

    Mirrors generate_with_failover for the streaming path: rotates through every API
    key and falls back to LLM_FALLBACK_MODEL on transient errors. Failover only works
    BEFORE the first chunk is emitted — once tokens start flowing we're committed to
    that stream, so a mid-stream failure re-raises rather than restarting the answer.
    """
    if max_tokens is None:
        max_tokens = config.LLM_MAX_TOKENS

    _ensure_pool()
    pool = _client_pool
    models_to_try = [config.LLM_MODEL_NAME]
    if config.LLM_FALLBACK_MODEL and config.LLM_FALLBACK_MODEL != config.LLM_MODEL_NAME:
        models_to_try.append(config.LLM_FALLBACK_MODEL)

    start = _next_client_idx()
    ordered_pool = pool[start:] + pool[:start]

    last_exc: Exception | None = None
    any_attempted = False
    for current_model in models_to_try:
        if time.monotonic() < _circuit_breaker.get(current_model, 0):
            logger.info(f"[Gemini stream failover] {current_model} circuit open, skipping")
            continue

        base_config = _stream_gen_config(current_model, max_tokens, system_instruction)
        for offset, client in enumerate(ordered_pool, 1):
            any_attempted = True
            gen_config = _with_cache(client, current_model, base_config)
            emitted = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=current_model, contents=prompt, config=gen_config
                ):
                    try:
                        if chunk.text:
                            emitted = True
                            yield chunk.text
                    except (ValueError, AttributeError) as exc:
                        logger.debug("Skipping non-text Gemini stream chunk: %s", exc)
                if not emitted:
                    raise RuntimeError("No text generated from Gemini stream")
                _circuit_breaker.pop(current_model, None)
                return
            except Exception as exc:
                last_exc = exc
                if emitted:
                    raise  # committed to this stream — cannot restart the answer
                if _is_permanent(exc):
                    raise  # malformed request — other keys/models won't help
                logger.warning(
                    f"[Gemini stream failover] {current_model} key #{offset}/{len(pool)} "
                    f"failed ({getattr(exc, 'status_code', '?')}): {exc!s:.120} — trying next"
                )
                continue

        _circuit_breaker[current_model] = time.monotonic() + _CIRCUIT_COOLDOWN
        if current_model == config.LLM_MODEL_NAME and len(models_to_try) > 1:
            logger.warning(
                f"[Gemini stream failover] {current_model} exhausted all keys, "
                f"falling back to {config.LLM_FALLBACK_MODEL}"
            )

    if not any_attempted:
        raise RuntimeError("All configured Gemini models are currently circuit-open; retry after cooldown.")
    raise last_exc  # type: ignore[misc]
