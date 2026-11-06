"""Per-answer research traces (v3 OBS-1).

A trace explains one answer: which stages ran and how long they took, which
chunks it used, which model actually served it after failover, and enough of a
reproducibility stamp (model roles, prompt version, config hash) to re-run it.

Lifecycle: a route calls start() before the pipeline and finish() once the answer
is final. In between, nothing passes a trace around: it lives in a contextvar, so
metrics.stage() spans, llm_client and rag.retrieve_context record into whichever
trace is current, and are no-ops when there is none. Threads started for a request
must copy the context (contextvars.copy_context().run) to keep recording.

Question and answer text are not stored unless TRACE_STORE_TEXT=true; a hash and
length are, so a trace still correlates with query_log and feedback.
"""

import contextvars
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import config

logger = logging.getLogger(__name__)

_current: contextvars.ContextVar = contextvars.ContextVar("indicrag_trace", default=None)
_MAX_SPANS = 500           # one runaway loop must not bloat a row
_PRUNE_EVERY = 200         # finishes between retention sweeps
_finishes = 0


@dataclass
class Trace:
    trace_id: str
    kind: str
    question: str = ""
    owner: str | None = None
    started: float = field(default_factory=time.monotonic)
    spans: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


def start(kind: str, trace_id: str | None = None, question: str = "", owner: str | None = None) -> Trace | None:
    """Begin a trace for the current request (None when TRACE_ENABLED is off)."""
    if not config.TRACE_ENABLED:
        return None
    t = Trace(trace_id or str(uuid.uuid4()), kind, question, owner)
    _current.set(t)
    return t


def current() -> Trace | None:
    return _current.get()


def record_span(stage: str, seconds: float, ok: bool = True) -> None:
    t = _current.get()
    if t is not None and len(t.spans) < _MAX_SPANS:
        t.spans.append({"stage": stage, "ms": round(seconds * 1000, 1), "ok": ok})


def note(**attrs) -> None:
    """Attach facts to the current trace (later values for a key win)."""
    t = _current.get()
    if t is not None:
        t.attrs.update(attrs)


def note_llm(provider: str, model: str, usage: dict | None = None, *, attempt: int = 1,
             stream: bool = False) -> None:
    """The call that actually produced output, after any failover."""
    t = _current.get()
    if t is not None and len(t.llm_calls) < _MAX_SPANS:
        t.llm_calls.append({"provider": provider, "model": model, "attempt": attempt,
                            "stream": stream, "usage": usage or None})


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "ignore")).hexdigest()[:16]


def repro_stamp() -> dict:
    """What a re-run must match: models per role, prompt version, config hash."""
    prompts = "\x1f".join(str(getattr(config, k, "")) for k in
                          ("SYSTEM_PROMPT", "QUERY_PROMPT_TEMPLATE", "AGENT_SYSTEM_PROMPT"))
    knobs = {k: getattr(config, k, None) for k in (
        "USE_HYBRID_SEARCH", "USE_RERANKER", "RERANK_POOL_MULT", "RERANK_POOL_MAX", "MAX_CONTEXT_CHUNKS",
        "DEFAULT_TOP_K", "RRF_K", "USE_HYDE", "USE_COLBERT_RERANK", "LLM_TEMPERATURE", "LLM_MAX_TOKENS")}
    try:
        import vector_store
        chunker = vector_store.CHUNKER_VERSION
    except Exception:
        chunker = None
    return {"models": dict(getattr(config, "MODEL_ROLES", {})), "prompt_version": _digest(prompts)[:12],
            "config_hash": _digest(json.dumps(knobs, sort_keys=True, default=str))[:12],
            "chunker_version": chunker}


def build_record(t: Trace, question: str = "", answer: str = "", **extra) -> dict:
    """The persisted trace body (pure, so tests can check redaction directly)."""
    text = {"question_sha": _digest(question), "question_len": len(question or ""),
            "answer_sha": _digest(answer), "answer_len": len(answer or "")}
    if config.TRACE_STORE_TEXT:
        text.update(question=question, answer=answer)
    served = t.llm_calls[-1] if t.llm_calls else None
    return {"total_ms": round((time.monotonic() - t.started) * 1000, 1), "spans": t.spans,
            "llm_calls": t.llm_calls, "served_by": served, **text, **t.attrs, **extra,
            "repro": repro_stamp()}


def finish(t: Trace | None, answer: str = "", **extra) -> None:
    """Persist a trace. Never raises: a lost trace must not cost the user their answer."""
    global _finishes
    if t is None:
        return
    try:
        import persistence
        now = datetime.now(timezone.utc)
        persistence.save_trace(t.trace_id, t.kind, t.owner, now.isoformat(),
                               build_record(t, t.question, answer, **extra))
        _finishes += 1
        if _finishes % _PRUNE_EVERY == 0:
            persistence.prune_traces((now - timedelta(days=config.TRACE_RETENTION_DAYS)).isoformat())
    except Exception:
        logger.warning("Trace %s could not be saved", t.trace_id, exc_info=True)
        try:
            import metrics
            metrics.stage_errors.labels(stage="trace_write").inc()
        except Exception:
            pass
