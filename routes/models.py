"""GET /models — the curated model allowlist enriched with tool-capability.

Bare names → Gemini (always tool-capable, no network). OpenRouter slugs are
enriched from the OpenRouter /models catalog (cached MODELS_CACHE_TTL seconds)
so the UI can grey out tool-incapable models in agent mode.
"""

import logging

from fastapi import APIRouter

import config
from cache import TTLCache

logger = logging.getLogger(__name__)
router = APIRouter()

_catalog_cache = TTLCache(max_size=4, ttl_seconds=config.MODELS_CACHE_TTL)
_CATALOG_KEY = "openrouter_catalog"


def _fetch_openrouter_catalog() -> dict:
    """slug -> {'supported_parameters': [...]}. Network call; best-effort."""
    import httpx
    url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/models"
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return {mod["id"]: mod for mod in data if "id" in mod}


def _catalog() -> dict:
    cached = _catalog_cache.get(_CATALOG_KEY)
    if cached is not None:
        return cached
    try:
        cat = _fetch_openrouter_catalog()
    except Exception as exc:
        logger.warning(f"[/models] OpenRouter catalog fetch failed: {exc!s:.120}")
        cat = {}
    _catalog_cache.put(_CATALOG_KEY, cat)
    return cat


def _is_openrouter(model: str) -> bool:
    return "/" in model


def model_supports_tools(model: str) -> bool:
    if not _is_openrouter(model):
        return True
    entry = _catalog().get(model)
    if not entry:
        return True  # unknown → don't over-block; capability gate is best-effort
    return "tools" in (entry.get("supported_parameters") or [])


def list_models() -> list[dict]:
    out = []
    for mid in config.LLM_SELECTABLE_MODELS:
        if _is_openrouter(mid):
            out.append({"id": mid, "provider": "openrouter", "tools": model_supports_tools(mid)})
        else:
            out.append({"id": mid, "provider": "gemini", "tools": True})
    return out


def validate_model(model: str | None, provider: str | None) -> None:
    """Raise ValueError if a requested model is off the allowlist."""
    if model is None:
        return
    if model not in config.LLM_SELECTABLE_MODELS:
        raise ValueError(f"Model '{model}' is not in LLM_SELECTABLE_MODELS allowlist.")


@router.get("/models", tags=["Models"])
def get_models():
    # Sync (not async): list_models() → _fetch_openrouter_catalog() does a blocking
    # httpx.get on cache miss. A sync route runs in FastAPI's threadpool, so it
    # never blocks the event loop; an `async def` would stall all requests for
    # up to the 10s HTTP timeout.
    return {"models": list_models(), "default": (config.LLM_SELECTABLE_MODELS or [None])[0]}
