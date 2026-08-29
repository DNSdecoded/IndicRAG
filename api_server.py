"""
FastAPI server for the Multilingual Scientific RAG system.
Production-ready REST API with authentication, validation, and monitoring.

Route handlers live in routes/*.py; shared auth/state in deps.py.
"""

from contextlib import asynccontextmanager
import asyncio
import logging
import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

import config
from deps import (
    STATIC_DIR,
    limiter,
    verify_api_key,
)
from routes import query, chat, ingest, agent, management, feedback, watch, report, models as models_route
from middleware import RequestIdFilter, RequestIdMiddleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - [%(request_id)s] - %(levelname)s - %(message)s'
)
logging.getLogger().addFilter(RequestIdFilter())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    import embeddings
    import vector_store
    embeddings.load_embedding_model()
    collection = vector_store.get_or_create_collection()

    # Loud at startup rather than silent at query time: comparing vectors from two
    # different embedding spaces is meaningless but never raises, so a model or
    # chunker swap otherwise shows up only as retrieval quality quietly getting
    # worse — which is nearly undebuggable from the outside.
    problem = vector_store.check_index_compatibility(collection)
    if problem:
        logger.warning("INDEX PROVENANCE: %s", problem)

    # Jobs run inside this process, so anything left in-flight by the previous
    # one is dead — but the row still said `running`, and clients polled it
    # forever. Reap by expired lease, so a job another live worker is still
    # heartbeating is left alone.
    from datetime import datetime, timezone
    import persistence
    import deps
    reaped = persistence.reap_stale_jobs(datetime.now(timezone.utc).isoformat())
    if reaped:
        # deps._jobs was hydrated from SQLite at import, before this ran, so the
        # in-memory copies still say `running`. Status is served from memory, so
        # both have to move or the API would keep reporting the stale state.
        with deps._jobs_lock:
            for job in reaped:
                jid = job.get("job_id")
                if jid in deps._jobs:
                    deps._jobs[jid].update(job)
        logger.warning("Reaped %d job(s) abandoned by a previous process: %s",
                       len(reaped), ", ".join(j.get("job_id", "?") for j in reaped))
    if config.USE_RERANKER:
        import rerank
        rerank._load()
    if config.USE_COLBERT_RERANK:
        # Warm here or the FIRST query after start pays the model load inside the
        # retrieval path, while holding the loader lock every concurrent query
        # then queues behind — the same cold-start cliff the reranker warm-up
        # above exists to avoid.
        import colbert_rerank
        colbert_rerank._load()
    if config.USE_HYBRID_SEARCH:
        import bm25_search
        bm25_search.get_or_build_index()

    # Phase 6 Increment 4: background schedule loop (single-worker, in-process).
    watch_task = None
    if config.WATCH_ENABLE:
        import watch_runner
        watch_task = asyncio.create_task(watch_runner.watch_loop())

        def _log_watch_task_result(task: asyncio.Task) -> None:
            # Surface a crashed schedule loop immediately instead of waiting for
            # GC to log "Task exception was never retrieved".
            if not task.cancelled() and task.exception() is not None:
                logger.error("[Watch] schedule loop died unexpectedly",
                             exc_info=task.exception())

        watch_task.add_done_callback(_log_watch_task_result)

    yield

    if watch_task:
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
    import persistence as _persistence
    _persistence.checkpoint()
    logger.info("Shutting down: draining in-flight requests complete.")

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual Scientific RAG API",
    description="Retrieval-Augmented Generation system for multilingual scientific Q&A",
    version=config.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus Monitoring
#
# prometheus_fastapi_instrumentator==7.1.0 walks app.routes expecting flat
# Route/Mount objects. fastapi>=0.130's app.include_router() wraps included
# sub-routers in an internal lazy _IncludedRouter node (no .path attribute),
# which crashes the instrumentator's route-name walker on every request once
# routes are split across routers. Degrade gracefully instead of 500ing:
# fall back to the raw URL path (unmatched-route grouping) when route-name
# resolution can't handle the wrapper node.
import prometheus_fastapi_instrumentator.routing as _pi_routing  # noqa: E402 — must patch before Instrumentator import
_original_get_route_name = _pi_routing.get_route_name


def _safe_get_route_name(request):
    try:
        return _original_get_route_name(request)
    except AttributeError:
        return None


_pi_routing.get_route_name = _safe_get_route_name

from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402 — after routing monkeypatch above
# /metrics requires the same X-API-Key as the rest of the API (no-op when
# API_KEYS is unset) — route latency/count data shouldn't be public.
Instrumentator().instrument(app).expose(
    app,
    include_in_schema=False,
    should_gzip=True,
    dependencies=[Depends(verify_api_key)],
)

# Mount static files directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Phase 3: serve figure/table crops so the UI can render cited figures.
# StaticFiles is path-traversal safe; read-only. Only mounted when the dir exists.
if config.FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(config.FIGURES_DIR)), name="figures")

# CORS configuration — env-driven for deployment flexibility
_cors_origins_env = os.getenv("CORS_ORIGINS")
_cors_origins = (
    [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins_env
    else [
        "http://localhost:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8000",
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)
app.add_middleware(RequestIdMiddleware)

# Mount routers
app.include_router(query.router)
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(agent.router)
app.include_router(management.router)
app.include_router(feedback.router)
app.include_router(watch.router)
app.include_router(report.router)
app.include_router(models_route.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=config.LOG_LEVEL.lower()
    )
