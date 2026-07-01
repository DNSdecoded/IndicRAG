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
                result = client.models.generate_content(
                    model=current_model, contents=contents, config=gen_config
                )
                _circuit_breaker.pop(current_model, None)
                return result
            except Exception as exc:
                if _is_transient(exc):
                    logger.warning(
                        f"[Gemini failover] {current_model} key #{offset}/{len(pool)} "
                        f"returned {getattr(exc, 'status_code', '?')}: {exc!s:.120} — trying next"
                    )
                    last_exc = exc
                    continue
                raise

        if all_failed:
            _circuit_breaker[current_model] = time.monotonic() + _CIRCUIT_COOLDOWN
            if current_model == model and len(models_to_try) > 1:
                logger.warning(f"[Gemini failover] {model} circuit tripped for {_CIRCUIT_COOLDOWN}s, falling back to {config.LLM_FALLBACK_MODEL}")

    if not any_attempted:
        raise RuntimeError(
            "All configured Gemini models are currently circuit-open; retry after cooldown."
        )
    raise last_exc  # type: ignore[misc]


def llm_generate_stream(prompt: str, max_tokens: int = None, system_instruction: str = None):
    """Generator: stream LLM response chunks using primary model with key cycling.

    Yields non-empty text chunks as they arrive. No model failover — primary model only.
    ponytail: single-model streaming; add failover if primary is unreliable.
    """
    from google.genai import types

    if max_tokens is None:
        max_tokens = config.LLM_MAX_TOKENS

    _ensure_pool()

    gen_config = types.GenerateContentConfig(
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=max_tokens,
        safety_settings=config.SAFETY_SETTINGS,
        system_instruction=system_instruction or config.SYSTEM_PROMPT,
    )

    client = _get_client()
    emitted = False
    for chunk in client.models.generate_content_stream(
        model=config.LLM_MODEL_NAME, contents=prompt, config=gen_config
    ):
        try:
            if chunk.text:
                emitted = True
                yield chunk.text
        except (ValueError, AttributeError) as exc:
            logger.debug("Skipping non-text Gemini stream chunk: %s", exc)
    if not emitted:
        raise RuntimeError("No text generated from Gemini stream")
