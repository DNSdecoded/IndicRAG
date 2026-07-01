"""
FastAPI server for the Multilingual Scientific RAG system.
Production-ready REST API with authentication, validation, and monitoring.

Route handlers live in routes/*.py; shared auth/state in deps.py.
"""

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config
from deps import (
    STATIC_DIR,
    limiter,
    verify_api_key,
    verify_admin_key,
    _jobs,
    _jobs_lock,
    _update_job,
    _sessions,
    _sessions_lock,
    _get_or_create_session,
    _append_session_messages,
)
from routes import query, chat, ingest, agent, management, feedback
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
    import embeddings, vector_store
    embeddings.load_embedding_model()
    vector_store.get_or_create_collection()
    if config.USE_RERANKER:
        import rerank
        rerank._load()
    if config.USE_HYBRID_SEARCH:
        import bm25_search
        bm25_search.get_or_build_index()
    yield
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
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
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
import prometheus_fastapi_instrumentator.routing as _pi_routing
_original_get_route_name = _pi_routing.get_route_name


def _safe_get_route_name(request):
    try:
        return _original_get_route_name(request)
    except AttributeError:
        return None


_pi_routing.get_route_name = _safe_get_route_name

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# Mount static files directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=config.LOG_LEVEL.lower()
    )
