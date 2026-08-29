"""Stage-level Prometheus metrics.

The HTTP instrumentator in api_server gives per-route latency and counts, which
answers "is /query slow" but never "why". When p99 doubles, request-level
numbers cannot distinguish a BM25 rebuild after an ingest from a slow LLM
provider from a reranker thrashing on an oversubscribed box — and each of those
has a different fix.

These are the per-stage timings and counters that make them distinguishable.
Everything registers on the default registry, so the existing /metrics endpoint
serves them (behind the same API key) with no extra wiring.

Metrics must never break a request: prometheus_client arrives transitively via
prometheus_fastapi_instrumentator, so if it is ever absent this module degrades
to no-ops instead of taking the pipeline down with it.
"""

from contextlib import contextmanager
import logging
import time

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram
    _ENABLED = True
except Exception:  # pragma: no cover - depends on the deployment's deps
    _ENABLED = False
    logger.warning("prometheus_client unavailable; stage metrics disabled")


class _NoopMetric:
    """Stand-in exposing the subset of the API used here."""

    def labels(self, *a, **k):
        return self

    def observe(self, *a, **k):
        pass

    def inc(self, *a, **k):
        pass

    def set(self, *a, **k):
        pass


if _ENABLED:
    # Buckets span the real spread: a cache hit is sub-millisecond, a cross-encoder
    # pass is hundreds of ms, an LLM call with failover is tens of seconds. The
    # default buckets top out at 10s and would dump every interesting slow case
    # into +Inf, which is precisely the case worth seeing.
    _BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

    stage_seconds = Histogram(
        "indicrag_stage_seconds",
        "Time spent in one pipeline stage",
        ["stage"],
        buckets=_BUCKETS,
    )
    stage_errors = Counter(
        "indicrag_stage_errors_total",
        "Exceptions raised inside a pipeline stage",
        ["stage"],
    )
    cache_events = Counter(
        "indicrag_cache_events_total",
        "Cache hits and misses by cache name",
        ["cache", "event"],
    )
    llm_tokens = Counter(
        "indicrag_llm_tokens_total",
        "Tokens billed, by provider/model and direction",
        ["provider", "model", "direction"],
    )
    llm_failovers = Counter(
        "indicrag_llm_failovers_total",
        "Failover hops taken, labelled by why the previous path was abandoned",
        ["provider", "model", "reason"],
    )
    circuit_trips = Counter(
        "indicrag_circuit_trips_total",
        "Circuit breaker openings by component",
        ["component"],
    )
    reflexion_iterations = Histogram(
        "indicrag_reflexion_iterations",
        "Reflexion loops per agent answer",
        buckets=(0, 1, 2, 3, 4),
    )
    answers = Counter(
        "indicrag_answers_total",
        "Answers produced, by pipeline mode and outcome",
        ["mode", "outcome"],
    )
    corpus_chunks = Gauge(
        "indicrag_corpus_chunks",
        "Chunks currently indexed",
    )
    admission_inflight = Gauge(
        "indicrag_admission_inflight",
        "Requests currently holding an admission slot, by pool",
        ["pool"],
    )
    admission_shed = Counter(
        "indicrag_admission_shed_total",
        "Requests rejected with 429 because the pool was saturated",
        ["pool"],
    )
    cascade_failures = Counter(
        "indicrag_cascade_failures_total",
        "Delete-cascade steps that failed after a retry, leaving a derived view "
        "diverged from the ingest log",
        ["step"],
    )
else:  # pragma: no cover
    stage_seconds = stage_errors = cache_events = _NoopMetric()
    llm_tokens = llm_failovers = circuit_trips = _NoopMetric()
    reflexion_iterations = answers = corpus_chunks = _NoopMetric()
    cascade_failures = admission_inflight = admission_shed = _NoopMetric()


@contextmanager
def stage(name: str):
    """Time a pipeline stage.

    Records the elapsed time whether or not the body raised — a stage that fails
    after 30s is exactly the one worth seeing, and dropping it on the error path
    would make an outage look like reduced latency.
    """
    start = time.perf_counter()
    try:
        yield
    except BaseException:
        stage_errors.labels(stage=name).inc()
        raise
    finally:
        stage_seconds.labels(stage=name).observe(time.perf_counter() - start)


def record_cache(cache_name: str, hit: bool) -> None:
    cache_events.labels(cache=cache_name, event="hit" if hit else "miss").inc()


def record_tokens(provider: str, model: str, prompt: int = 0, completion: int = 0) -> None:
    if prompt:
        llm_tokens.labels(provider=provider or "?", model=model or "?", direction="prompt").inc(prompt)
    if completion:
        llm_tokens.labels(provider=provider or "?", model=model or "?", direction="completion").inc(completion)


def record_failover(provider: str, model: str, reason: str) -> None:
    llm_failovers.labels(provider=provider or "?", model=model or "?", reason=reason).inc()


def record_circuit_trip(component: str) -> None:
    circuit_trips.labels(component=component).inc()


def admission_enter(pool: str) -> None:
    admission_inflight.labels(pool=pool).inc()


def admission_exit(pool: str) -> None:
    admission_inflight.labels(pool=pool).dec()


def record_admission_shed(pool: str) -> None:
    admission_shed.labels(pool=pool).inc()


def record_cascade_failure(step: str) -> None:
    """step: bm25 | ingest_log. A delete that left a derived view behind."""
    cascade_failures.labels(step=step).inc()


def record_answer(mode: str, outcome: str) -> None:
    """outcome: ok | abstained | degraded | no_documents | error."""
    answers.labels(mode=mode, outcome=outcome).inc()
