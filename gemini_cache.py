"""Explicit Gemini context caching for stable system-instruction prefixes.

Why per-client: llm_client round-robins a POOL of API keys with failover. An
explicit CachedContent is scoped to the credential/project that created it, so a
cache made on key A is unusable on key B — each client gets its own cache.

Why opt-in: gemini-3.5-flash already does IMPLICIT caching for free (automatic
prefix discount, no storage cost). Explicit caching adds guaranteed reuse but is
billed per token-hour of storage, so it's gated behind config.GEMINI_CACHE_ENABLED.

Fail-open: any create() failure (content below the model's min-token floor,
unsupported model, quota) records a short cooldown and returns None — the caller
then sends the system_instruction inline as usual. Caching never breaks a call.
"""

import hashlib
import logging
import threading
import time

from google.genai import types

import config

logger = logging.getLogger(__name__)

# key -> (cache_name, monotonic_expiry). key = (id(client), model, prefix_hash)
_registry: dict = {}
# key -> monotonic time until which we skip retrying a failed create()
_cooldown: dict = {}
_lock = threading.Lock()
# key -> Lock serializing the network create() for that key. ponytail: never
# evicted, but bounded by distinct (client, model, prefix) keys — a handful.
_inflight: dict = {}

_CREATE_FAIL_COOLDOWN = 300  # s — don't hammer create() when it keeps failing
_EXPIRY_SAFETY = 60          # s — refresh a cache this long before it expires


def _prefix_hash(system_instruction, tools) -> str:
    raw = repr(system_instruction) + "|" + repr(tools)
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:16]


def get_or_create(client, model: str, system_instruction, tools=None):
    """Return a cached_content resource name for this (client, model, prefix), or None.

    None means "don't use caching" — the caller keeps system_instruction/tools inline.
    """
    if not config.GEMINI_CACHE_ENABLED or not system_instruction:
        return None

    key = (id(client), model, _prefix_hash(system_instruction, tools))
    now = time.monotonic()

    # Fast path: registry/cooldown lookups under the global lock only. The slow
    # network create() runs OUTSIDE _lock so a stalled create for one key can't
    # block cache lookups for every other key in the process.
    with _lock:
        cooled = _cooldown.get(key)
        if cooled is not None and now < cooled:
            return None
        entry = _registry.get(key)
        if entry is not None and now < entry[1] - _EXPIRY_SAFETY:
            return entry[0]
        inflight = _inflight.get(key)
        if inflight is None:
            inflight = _inflight[key] = threading.Lock()

    # Serialize create() per key (avoid a stampede of duplicate creates) without
    # holding the global lock.
    with inflight:
        now = time.monotonic()
        with _lock:
            cooled = _cooldown.get(key)
            if cooled is not None and now < cooled:
                return None
            entry = _registry.get(key)
            if entry is not None and now < entry[1] - _EXPIRY_SAFETY:
                return entry[0]

        try:
            cache = client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    tools=tools,
                    ttl=f"{config.GEMINI_CACHE_TTL}s",
                    display_name="indicrag-sysprompt",
                ),
            )
        except Exception as exc:
            # Below min-token floor, unsupported model, quota, etc. — fall back to
            # inline system_instruction and don't retry for a while.
            with _lock:
                _cooldown[key] = time.monotonic() + _CREATE_FAIL_COOLDOWN
            logger.info(f"[GeminiCache] disabled for {model} ({exc!s:.140}); using inline prompt")
            return None

        with _lock:
            _registry[key] = (cache.name, time.monotonic() + config.GEMINI_CACHE_TTL)
            _cooldown.pop(key, None)
        logger.info(f"[GeminiCache] created {cache.name} for {model} (ttl={config.GEMINI_CACHE_TTL}s)")
        return cache.name
