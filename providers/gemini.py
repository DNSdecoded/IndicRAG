"""Gemini backend: client pool, round-robin, per-key failover, thinking gate.

Extracted verbatim from the original llm_client.py so behavior is unchanged.
generate() returns the raw google-genai response (already the native shape the
agent nodes read) — no shim needed for this backend.
"""

import itertools
import logging
import threading
from typing import Iterator

import config
from google import genai
from providers.base import LLMBackend

logger = logging.getLogger(__name__)


class GeminiBackend(LLMBackend):
    def __init__(self):
        self._pool: list[genai.Client] = []
        self._lock = threading.Lock()
        self._index = itertools.cycle([])

    @property
    def name(self) -> str:
        return "gemini"

    # ── pool ────────────────────────────────────────────────────────────
    def _init_pool(self) -> None:
        if not config.LLM_API_KEY_POOL:
            raise ValueError(
                "Google Gemini API key not configured. "
                "Set LLM_API_KEY (single) or LLM_API_KEYS (comma-separated) in .env."
            )
        self._pool = [genai.Client(api_key=k) for k in config.LLM_API_KEY_POOL]
        self._index = itertools.cycle(range(len(self._pool)))

    def _ensure_pool(self) -> None:
        if not self._pool:
            with self._lock:
                if not self._pool:
                    self._init_pool()

    def next_client_idx(self) -> int:
        self._ensure_pool()
        with self._lock:
            return next(self._index)

    @property
    def pool(self) -> list:
        self._ensure_pool()
        return self._pool

    # ── classifiers (verbatim from old llm_client) ──────────────────────
    def is_transient(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (429, 503):
            return True
        msg = str(exc)
        return "503" in msg or "429" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg

    def is_permanent(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (400, 404):
            return True
        return "INVALID_ARGUMENT" in str(exc)

    def supports_thinking(self, model: str) -> bool:
        return "gemma" not in model.lower()

    def _with_cache(self, client, model: str, gen_config):
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
        except Exception as exc:
            logger.debug("Gemini context caching skipped: %s", exc)
            return gen_config

    def _prep_config(self, client, model, gen_config):
        call_config = self._with_cache(client, model, gen_config)
        if not self.supports_thinking(model) and getattr(call_config, "thinking_config", None) is not None:
            call_config = call_config.model_copy(update={"thinking_config": None})
        return call_config

    # ── single-client calls (dispatch/failover lives in llm_client) ─────
    def generate(self, model: str, contents, gen_config, client=None):
        client = client or self.pool[self.next_client_idx()]
        call_config = self._prep_config(client, model, gen_config)
        return client.models.generate_content(model=model, contents=contents, config=call_config)

    def generate_stream(self, model: str, contents, gen_config, client=None) -> Iterator[str]:
        client = client or self.pool[self.next_client_idx()]
        call_config = self._prep_config(client, model, gen_config)
        emitted = False
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=call_config):
            try:
                if chunk.text:
                    emitted = True
                    yield chunk.text
            except (ValueError, AttributeError) as exc:
                logger.debug("Skipping non-text Gemini stream chunk: %s", exc)
        if not emitted:
            raise RuntimeError("No text generated from Gemini stream")
