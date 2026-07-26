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

    # Models that reject thinking_budget=0 outright (gemini-3.6-flash returns
    # 400 INVALID_ARGUMENT; gemini-3.5-flash accepts it). Learned at runtime and
    # remembered per model, so a new model generation doesn't need a hardcoded
    # list here — omitting thinking_config is the closest thing to "no thinking"
    # those models accept.
    _zero_budget_rejected: set = set()

    @staticmethod
    def _has_zero_thinking_budget(gen_config) -> bool:
        tc = getattr(gen_config, "thinking_config", None)
        return tc is not None and getattr(tc, "thinking_budget", None) == 0

    @staticmethod
    def _is_invalid_argument(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        return status == 400 or "INVALID_ARGUMENT" in str(exc)

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
        drop_thinking = (
            not self.supports_thinking(model)
            or (model in self._zero_budget_rejected and self._has_zero_thinking_budget(call_config))
        )
        if drop_thinking and getattr(call_config, "thinking_config", None) is not None:
            call_config = call_config.model_copy(update={"thinking_config": None})
        return call_config

    def _remember_zero_budget_rejection(self, model: str, gen_config, exc: Exception) -> bool:
        """True if `exc` is this model refusing thinking_budget=0, so the caller
        can retry once without the field. Narrow on purpose: any other 400 is a
        genuine bad request and must keep propagating."""
        if not (self._has_zero_thinking_budget(gen_config) and self._is_invalid_argument(exc)):
            return False
        if model not in self._zero_budget_rejected:
            logger.warning(
                "Model %s rejects thinking_budget=0; retrying without thinking_config "
                "and omitting it for this model from now on.", model,
            )
            self._zero_budget_rejected.add(model)
        return True

    # ── single-client calls (dispatch/failover lives in llm_client) ─────
    def generate(self, model: str, contents, gen_config, client=None):
        client = client or self.pool[self.next_client_idx()]
        call_config = self._prep_config(client, model, gen_config)
        try:
            return client.models.generate_content(model=model, contents=contents, config=call_config)
        except Exception as exc:
            if not self._remember_zero_budget_rejection(model, call_config, exc):
                raise
            retry_config = call_config.model_copy(update={"thinking_config": None})
            return client.models.generate_content(model=model, contents=contents, config=retry_config)

    def generate_stream(self, model: str, contents, gen_config, client=None) -> Iterator[str]:
        client = client or self.pool[self.next_client_idx()]
        call_config = self._prep_config(client, model, gen_config)
        emitted = False

        def _iter(cfg):
            nonlocal emitted
            for chunk in client.models.generate_content_stream(model=model, contents=contents, config=cfg):
                try:
                    if chunk.text:
                        emitted = True
                        yield chunk.text
                except (ValueError, AttributeError) as exc:
                    logger.debug("Skipping non-text Gemini stream chunk: %s", exc)

        try:
            yield from _iter(call_config)
        except Exception as exc:
            # A zero-budget rejection is refused before any token is produced, so
            # retrying the stream here is safe. Once text has been emitted the
            # error is something else and must propagate.
            if emitted or not self._remember_zero_budget_rejection(model, call_config, exc):
                raise
            yield from _iter(call_config.model_copy(update={"thinking_config": None}))

        if not emitted:
            raise RuntimeError("No text generated from Gemini stream")
