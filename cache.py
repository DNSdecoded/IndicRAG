"""
Thread-safe TTL LRU cache for IndicRAG.

Used by:
- rag.llm_generate()       — avoid duplicate LLM calls for identical prompts
- rag.retrieve_context()    — avoid re-retrieving identical queries
- agent/tool_executor.py    — avoid duplicate tool calls across reflexion loops
"""

import hashlib
import json
import threading
import time
import logging
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-safe LRU cache with per-entry TTL expiration."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            value, ts = self._store[key]
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (value, time.monotonic())
                return
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)
            self._store[key] = (value, time.monotonic())

    def invalidate(self, key: str = None) -> None:
        with self._lock:
            if key is None:
                self._store.clear()
            elif key in self._store:
                del self._store[key]

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "size": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }


def make_key(*args) -> str:
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Shared cache instances ──────────────────────────────────────────────────
# Sizes and TTLs are configurable via environment variables (see config.py).

import config as _cfg

llm_cache = TTLCache(max_size=_cfg.LLM_CACHE_SIZE, ttl_seconds=_cfg.LLM_CACHE_TTL)

retrieval_cache = TTLCache(max_size=_cfg.RETRIEVAL_CACHE_SIZE, ttl_seconds=_cfg.RETRIEVAL_CACHE_TTL)

tool_cache = TTLCache(max_size=_cfg.TOOL_CACHE_SIZE, ttl_seconds=_cfg.TOOL_CACHE_TTL)
