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
from google.genai import types as genai_types
from providers.base import LLMBackend, TRUNCATION_NOTE

logger = logging.getLogger(__name__)


class GeminiBackend(LLMBackend):
    def __init__(self):
        self._pool: list[genai.Client] = []
        self._stream_pool: list[genai.Client] = []
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
        # Explicit per-request timeout (google-genai takes milliseconds). The SDK
        # default is generous enough that one stalled call outlives the agent's whole
        # budget and surfaces as "Agent pipeline timed out" instead of failing over.
        #
        # Streaming gets a SEPARATE, much larger budget because this timeout covers
        # the whole stream rather than the gap between chunks — sharing the unary
        # value tore down long answers mid-generation (WinError 10054, truncated text).
        def _client(key, seconds):
            return genai.Client(
                api_key=key,
                http_options=genai_types.HttpOptions(timeout=seconds * 1000),
            )

        self._pool = [_client(k, config.LLM_REQUEST_TIMEOUT_S) for k in config.LLM_API_KEY_POOL]
        self._stream_pool = [_client(k, config.LLM_STREAM_TIMEOUT_S)
                             for k in config.LLM_API_KEY_POOL]
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
    def stream_pool(self) -> list:
        """Clients whose HTTP timeout fits a whole streamed generation."""
        self._ensure_pool()
        return self._stream_pool

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
    # list here.
    # Class-level on purpose (the learning is about the model, not the instance),
    # so concurrent requests must not race on the check-then-add.
    _zero_budget_rejected: set = set()
    _zero_budget_lock = threading.Lock()

    # Budget-to-level translation for models on the newer thinking API. Dropping
    # thinking_config was the old fallback, but "send nothing" means the model's
    # OWN default — MEDIUM on gemini-3.6-flash — so asking for no thinking produced
    # medium thinking, billed and taken out of the answer's token budget.
    _LOW_BUDGET_CEILING = 1024  # above this a positive budget reads as MEDIUM

    @staticmethod
    def _thinking_level(name: str):
        """SDK ThinkingLevel by name, or None if unsupported/unknown.

        Guarded so a pinned older google-genai without the enum keeps working.
        """
        enum = getattr(genai_types, "ThinkingLevel", None)
        if enum is None or not name:
            return None
        return getattr(enum, name.upper(), None)

    @classmethod
    def _level_for_budget(cls, budget):
        """Legacy thinking_budget → the equivalent ThinkingLevel, or None to omit.

        0 is "off", and MINIMAL is the least thinking the new API offers. -1 is
        "model decides", which is exactly what omitting the field already means.
        """
        if budget is None or budget < 0:
            return None
        if budget == 0:
            return cls._thinking_level("MINIMAL")
        return cls._thinking_level("LOW" if budget <= cls._LOW_BUDGET_CEILING else "MEDIUM")

    @staticmethod
    def _has_thinking_budget(gen_config) -> bool:
        """True if the config carries a legacy thinking_budget of any value."""
        tc = getattr(gen_config, "thinking_config", None)
        return tc is not None and getattr(tc, "thinking_budget", None) is not None

    # Not every model supports every level. gemini-3.7-flash rejects MINIMAL
    # outright ("Thinking level MINIMAL is not supported for this model"), while
    # 3.6-flash accepts it — so the configured default cannot be assumed valid for
    # whatever model is selected. Learned per (model, level), exactly like the
    # budget rejections above, so a new model generation needs no hardcoded list.
    _LEVEL_LADDER = ("MINIMAL", "LOW", "MEDIUM", "HIGH")
    _level_rejected: set = set()
    _level_lock = threading.Lock()

    @staticmethod
    def _current_level_name(gen_config):
        """Name of the thinking_level on this config, or None."""
        tc = getattr(gen_config, "thinking_config", None)
        level = getattr(tc, "thinking_level", None) if tc is not None else None
        if level is None:
            return None
        return getattr(level, "name", str(level)).upper()

    def _escalate_level(self, gen_config):
        """Same config at the next level UP the ladder, or thinking dropped.

        Upwards, never downwards: a rejected level means the model will not think
        that little, so the only direction that can succeed is more thinking.
        Running off the end drops thinking_config entirely, which means "model
        decides" — worse than asking, but better than failing the request.
        """
        current = self._current_level_name(gen_config)
        if current is None:
            return None
        try:
            nxt = self._LEVEL_LADDER[self._LEVEL_LADDER.index(current) + 1]
        except (ValueError, IndexError):
            return gen_config.model_copy(update={"thinking_config": None})
        level = self._thinking_level(nxt)
        if level is None:
            return gen_config.model_copy(update={"thinking_config": None})
        return gen_config.model_copy(update={
            "thinking_config": genai_types.ThinkingConfig(thinking_level=level),
        })

    def _skip_rejected_levels(self, model: str, gen_config):
        """Advance past levels this model already refused, before calling out."""
        for _ in self._LEVEL_LADDER:
            name = self._current_level_name(gen_config)
            if name is None:
                return gen_config
            with self._level_lock:
                if (model, name) not in self._level_rejected:
                    return gen_config
            nxt = self._escalate_level(gen_config)
            if nxt is None:
                return gen_config
            gen_config = nxt
        return gen_config

    def _remember_level_rejection(self, model: str, gen_config, exc: Exception) -> bool:
        """True if `exc` is this model refusing the requested thinking_level, so
        the caller can retry one level up. Narrow on purpose, and matched on the
        message: any other 400 is a genuine bad request that must keep
        propagating rather than being retried with different thinking."""
        name = self._current_level_name(gen_config)
        if not name or not self._is_invalid_argument(exc):
            return False
        if "thinking level" not in str(exc).lower():
            return False
        with self._level_lock:
            first_time = (model, name) not in self._level_rejected
            self._level_rejected.add((model, name))
        if first_time:
            logger.warning(
                "Model %s does not support thinking level %s; using the next level up "
                "for this model from now on.", model, name,
            )
        return True

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

    def _translate_thinking(self, call_config):
        """Re-express a thinking_budget this model rejects as a thinking_level.

        Falls back to dropping thinking_config when no level applies (budget -1,
        or an SDK too old to expose the enum) — that means "model decides".
        """
        tc = getattr(call_config, "thinking_config", None)
        if tc is None:
            return call_config
        level = self._level_for_budget(getattr(tc, "thinking_budget", None))
        if level is None:
            return call_config.model_copy(update={"thinking_config": None})
        return call_config.model_copy(update={
            "thinking_config": genai_types.ThinkingConfig(thinking_level=level),
        })

    def _prep_config(self, client, model, gen_config):
        call_config = self._with_cache(client, model, gen_config)
        if not self.supports_thinking(model):
            if getattr(call_config, "thinking_config", None) is not None:
                call_config = call_config.model_copy(update={"thinking_config": None})
            return call_config

        with self._zero_budget_lock:
            known_rejected = model in self._zero_budget_rejected
        if known_rejected and self._has_thinking_budget(call_config):
            call_config = self._translate_thinking(call_config)
        # A level this model already refused would 400 again; step past it here so
        # only the FIRST call per (model, level) pays a round-trip to learn it.
        return self._skip_rejected_levels(model, call_config)

    def _remember_zero_budget_rejection(self, model: str, gen_config, exc: Exception) -> bool:
        """True if `exc` is this model refusing a legacy thinking_budget, so the
        caller can retry once with the translated thinking_level. Narrow on
        purpose: any other 400 is a genuine bad request and must keep
        propagating."""
        if not (self._has_thinking_budget(gen_config) and self._is_invalid_argument(exc)):
            return False
        with self._zero_budget_lock:
            first_time = model not in self._zero_budget_rejected
            self._zero_budget_rejected.add(model)
        if first_time:
            logger.warning(
                "Model %s rejects thinking_budget (superseded by thinking_level on "
                "Gemini 3.x); translating and using a level for this model from now on.",
                model,
            )
        return True

    # ── single-client calls (dispatch/failover lives in llm_client) ─────
    def generate(self, model: str, contents, gen_config, client=None):
        client = client or self.pool[self.next_client_idx()]
        call_config = self._prep_config(client, model, gen_config)
        try:
            return client.models.generate_content(model=model, contents=contents, config=call_config)
        except Exception as exc:
            # Two distinct refusals, each retried once: a legacy budget this model
            # no longer accepts, and a level it does not offer.
            if self._remember_zero_budget_rejection(model, call_config, exc):
                retry_config = self._translate_thinking(call_config)
            elif self._remember_level_rejection(model, call_config, exc):
                retry_config = self._escalate_level(call_config)
            else:
                raise
            if retry_config is None:
                raise
            return client.models.generate_content(
                model=model, contents=contents, config=retry_config,
            )

    def generate_stream(self, model: str, contents, gen_config, client=None) -> Iterator[str]:
        client = client or self.stream_pool[self.next_client_idx()]
        call_config = self._prep_config(client, model, gen_config)
        emitted = False

        def _iter(cfg):
            nonlocal emitted
            finish = None
            for chunk in client.models.generate_content_stream(model=model, contents=contents, config=cfg):
                cands = getattr(chunk, "candidates", None)
                if cands and getattr(cands[0], "finish_reason", None):
                    finish = cands[0].finish_reason
                try:
                    if chunk.text:
                        emitted = True
                        yield chunk.text
                except (ValueError, AttributeError) as exc:
                    logger.debug("Skipping non-text Gemini stream chunk: %s", exc)
            # max_output_tokens caps thinking + answer together, and this model's
            # thinking budget is dynamic, so the cut is unpredictable. The stream
            # still ends normally — say so in the text or nobody ever finds out.
            if emitted and "MAX_TOKENS" in str(finish):
                logger.warning("Gemini stream hit max_output_tokens for %s — answer truncated", model)
                yield TRUNCATION_NOTE

        try:
            yield from _iter(call_config)
        except Exception as exc:
            # A zero-budget rejection is refused before any token is produced, so
            # retrying the stream here is safe. Once text has been emitted the
            # error is something else and must propagate.
            # `emitted` guard: once bytes are out, a retry would duplicate the
            # answer's opening. Same two refusal kinds as the unary path.
            if emitted:
                raise
            if self._remember_zero_budget_rejection(model, call_config, exc):
                retry_config = self._translate_thinking(call_config)
            elif self._remember_level_rejection(model, call_config, exc):
                retry_config = self._escalate_level(call_config)
            else:
                raise
            if retry_config is None:
                raise
            yield from _iter(retry_config)

        if not emitted:
            raise RuntimeError("No text generated from Gemini stream")
