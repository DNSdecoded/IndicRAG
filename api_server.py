"""
FastAPI server for the Multilingual Scientific RAG system.
Production-ready REST API with authentication, validation, and monitoring.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Security, status, BackgroundTasks, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta, timezone
import os
import uuid
import threading

import rag
import config

# ---------------------------------------------------------------------------
# In-memory job store for background ingestion tasks
# ---------------------------------------------------------------------------
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_last_job_eviction = 0.0


def _update_job(job_id: str, **kwargs):
    """Thread-safe update of a job's fields; evicts completed jobs once per hour."""
    global _last_job_eviction
    with _jobs_lock:
        _jobs[job_id].update(kwargs)
        now = time.monotonic()
        if now - _last_job_eviction >= 3600:
            _last_job_eviction = now
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            for jid in [j for j, v in _jobs.items()
                        if v.get("completed_at") and
                        datetime.fromisoformat(v["completed_at"]) < cutoff]:
                del _jobs[jid]


# ---------------------------------------------------------------------------
# In-memory chat session store
# ---------------------------------------------------------------------------
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
_last_session_eviction = 0.0


def _evict_stale_sessions():
    """Remove sessions older than SESSION_MAX_AGE_HOURS. Must be called under _sessions_lock."""
    global _last_session_eviction
    now = time.monotonic()
    if now - _last_session_eviction < 60:
        return
    _last_session_eviction = now
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.SESSION_MAX_AGE_HOURS)
    for sid in [s for s, v in _sessions.items()
                if datetime.fromisoformat(v["updated_at"]) < cutoff]:
        del _sessions[sid]


def _get_or_create_session(session_id: Optional[str]) -> tuple[str, list]:
    """Return (session_id, messages_list). Creates a new session when id is None."""
    with _sessions_lock:
        _evict_stale_sessions()
        if session_id and session_id in _sessions:
            return session_id, list(_sessions[session_id]["messages"])
        new_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        _sessions[new_id] = {
            "id": new_id,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        return new_id, list(_sessions[new_id]["messages"])


def _append_session_messages(session_id: str, user_text: str, assistant_text: str) -> None:
    with _sessions_lock:
        sess = _sessions[session_id]
        msgs = sess["messages"]
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": assistant_text})
        max_msgs = config.CHAT_HISTORY_MAX_TURNS * 2
        if len(msgs) > max_msgs:
            del msgs[:len(msgs) - max_msgs]
        sess["updated_at"] = datetime.now(timezone.utc).isoformat()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus Monitoring
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# Mount static files directory
STATIC_DIR = Path(__file__).parent / "static"
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

# API Key authentication (optional)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Parse and validate API keys from environment
raw_keys = os.getenv("API_KEYS")
if raw_keys:
    # Filter out empty strings and whitespace-only keys
    VALID_API_KEYS = {k.strip() for k in raw_keys.split(",") if k.strip()}
    # If all keys were empty, disable auth
    if not VALID_API_KEYS:
        VALID_API_KEYS = None
else:
    VALID_API_KEYS = None


_admin_key = os.getenv("ADMIN_API_KEY")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key if authentication is enabled."""
    if VALID_API_KEYS is None:
        return True  # No authentication required

    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Invalid or missing API key",
                "code": "INVALID_API_KEY"
            }
        )
    return True


async def verify_admin_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify admin API key for destructive operations."""
    if _admin_key:
        if not api_key or api_key != _admin_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Admin API key required for destructive operations",
                        "code": "ADMIN_KEY_REQUIRED"}
            )
        return True
    return await verify_api_key(api_key)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., min_length=1, max_length=1000, description="User question in any language")
    strategy: str = Field("A", description="Strategy: 'A' for multilingual LLM, 'B' for English + translation")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    
    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        if v not in ['A', 'B']:
            raise ValueError("Strategy must be 'A' or 'B'")
        return v


class Citation(BaseModel):
    """Citation information."""
    number: str
    title: str
    section: str


class QueryResponse(BaseModel):
    """Response model for question answering."""
    answer: str
    language: str
    language_name: str
    chunks_used: int
    citations: List[Citation]
    processing_time: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    gemini_configured: bool


import re as _re

# Block the genuinely dangerous characters: null bytes, shell metacharacters.
# We rely on the is_absolute() + relative_to() checks for traversal; the regex
# only needs to reject characters that can't appear in safe filenames.
_UNSAFE_CHARS_RE = _re.compile(r'[\x00\|;&`$<>"\'\!\*\?\{\}\[\]\\~]')

class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    pdf_path: str = Field(..., description="Relative path to PDF file inside the papers/ directory")

    @field_validator('pdf_path')
    @classmethod
    def sanitize_pdf_path(cls, v: str) -> str:
        """Reject absolute paths, traversal sequences, and unsafe characters (CWE-22/23/36/73/99)."""
        from pathlib import PurePosixPath, PureWindowsPath
        # Reject empty
        if not v or not v.strip():
            raise ValueError("pdf_path must not be empty.")
        # Reject absolute paths on both POSIX and Windows
        if PurePosixPath(v).is_absolute() or PureWindowsPath(v).is_absolute():
            raise ValueError("pdf_path must be a relative path, not an absolute path.")
        # Reject any path component that is '..' (CWE-22/23)
        parts = PurePosixPath(v.replace('\\', '/')).parts
        if '..' in parts:
            raise ValueError("pdf_path must not contain '..' traversal sequences.")
        # Reject shell-dangerous characters (null bytes, metacharacters)
        if _UNSAFE_CHARS_RE.search(v):
            raise ValueError("pdf_path contains invalid characters.")
        return v.strip()


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    chunks_ingested: int
    paper_id: str
    title: str
    processing_time: float


class BulkIngestResponse(BaseModel):
    """Response model for bulk document ingestion."""
    status: str
    total_files: int
    successful: int
    failed: int
    chunks_ingested: int
    processing_time: float


class IngestJobResponse(BaseModel):
    """Immediate response when a bulk ingestion job is accepted."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Status of a background ingestion job."""
    job_id: str
    status: str          # pending | running | success | partial | failed
    total_files: Optional[int] = None
    successful: Optional[int] = None
    failed: Optional[int] = None
    chunks_ingested: Optional[int] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    submitted_at: str
    completed_at: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for a single chat turn."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message in any language")
    session_id: Optional[str] = Field(None, description="Existing session ID; omit to start a new conversation")
    strategy: str = Field("A", description="Strategy: 'A' for multilingual LLM, 'B' for English + translation")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in ("A", "B"):
            raise ValueError("Strategy must be 'A' or 'B'")
        return v


class ChatResponse(BaseModel):
    """Response model for a single chat turn."""
    session_id: str
    turn_index: int
    answer: str
    language: str
    language_name: str
    chunks_used: int
    citations: List[Citation]
    processing_time: float
    timestamp: str


# Routes

@app.get("/", tags=["General"])
async def root():
    """Serve the web frontend."""
    if STATIC_DIR.exists():
        return FileResponse(str(STATIC_DIR / "index.html"))
    else:
        return {
            "name": "Multilingual Scientific RAG API",
            "version": config.VERSION,
            "description": "Ask scientific questions in any Indian language",
            "endpoints": {
                "docs": "/api/docs",
                "health": "/health",
                "query": "/query",
                "ingest": "/ingest"
            },
            "note": "Frontend not found. Create static/index.html to enable web UI."
        }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=config.VERSION,
        gemini_configured=bool(config.LLM_API_KEY_POOL)
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit("30/minute")
async def query_question(
    request: Request,
    body: QueryRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Answer a question in any language using the RAG system.
    
    Supports 10+ Indian languages plus English.
    """
    start_time = time.time()
    
    try:
        # Log request (truncate question for privacy/brevity)
        question_preview = body.question[:80] + "..." if len(body.question) > 80 else body.question
        logger.info(f"Query received: strategy={body.strategy}, question='{question_preview}'")

        # Enforce top_k bounds even if client sends higher
        top_k = body.top_k
        if top_k is not None:
            top_k = max(1, min(top_k, 20))  # Clamp to [1, 20]

        # Process query (run in thread pool to avoid blocking event loop)
        result = await run_in_threadpool(
            rag.answer_question,
            user_query=body.question,
            strategy=body.strategy,
            top_k=top_k
        )
        
        processing_time = time.time() - start_time
        logger.info(
            f"Query completed: lang={result['language']}, chunks={result['chunks_used']}, "
            f"time={processing_time:.2f}s"
        )
        
        # Convert citations to response model
        citations = [
            Citation(
                number=cite['number'],
                title=cite['title'],
                section=cite['section']
            )
            for cite in result['citations']
        ]
        
        return QueryResponse(
            answer=result['answer'],
            language=result['language'],
            language_name=result['language_name'],
            chunks_used=result['chunks_used'],
            citations=citations,
            processing_time=processing_time,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    except ValueError as e:
        # Handle configuration or validation errors
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(e),
                "code": "VALIDATION_ERROR"
            }
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """
    Send a message in a multi-turn conversation.

    Pass ``session_id`` from a previous response to continue that conversation.
    Omit it (or pass ``null``) to start a fresh session.
    History is kept server-side; only the new ``message`` is required each turn.
    """
    start_time = time.time()

    top_k = body.top_k
    if top_k is not None:
        top_k = max(1, min(top_k, 20))

    session_id, messages = _get_or_create_session(body.session_id)
    turn_index = len(messages) // 2  # number of completed user+assistant pairs

    # Build the full message list the RAG function expects
    full_messages = list(messages) + [{"role": "user", "content": body.message}]

    try:
        result = await run_in_threadpool(
            rag.answer_with_history,
            messages=full_messages,
            strategy=body.strategy,
            top_k=top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": str(e), "code": "VALIDATION_ERROR"})
    except Exception as e:
        logger.error(f"Error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"})

    # Persist both turns in session history
    _append_session_messages(session_id, body.message, result["answer"])

    processing_time = time.time() - start_time
    logger.info(
        f"Chat turn {turn_index + 1} session={session_id[:8]}… "
        f"lang={result['language']} chunks={result['chunks_used']} time={processing_time:.2f}s"
    )

    return ChatResponse(
        session_id=session_id,
        turn_index=turn_index + 1,
        answer=result["answer"],
        language=result["language"],
        language_name=result["language_name"],
        chunks_used=result["chunks_used"],
        citations=[Citation(**c) for c in result["citations"]],
        processing_time=processing_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.delete("/chat/{session_id}", tags=["Chat"])
async def delete_session(
    session_id: str,
    authenticated: bool = Depends(verify_api_key),
):
    """Delete a chat session and its history."""
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found.")
        del _sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.post("/ingest", response_model=IngestResponse, tags=["Management"])
@limiter.limit("5/minute")
async def ingest_document(
    request: Request,
    body: IngestRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Ingest a PDF document into the vector store.

    Requires authentication if API keys are configured.
    """
    start_time = time.time()

    try:
        import ingest as ingest_module
        from pathlib import Path

        # At this point body.pdf_path is already sanitized by the Pydantic validator:
        # - not absolute, no '..' components, safe characters only.
        base_dir = Path(config.PAPERS_DIR).resolve()

        # Build and resolve the candidate path
        candidate = (base_dir / body.pdf_path).resolve()

        # Use Path.relative_to() as the authoritative containment check.
        # This raises ValueError if candidate is not inside base_dir, ensuring
        # the taint chain is severed: safe_pdf_path is reconstructed from
        # base_dir (trusted) + the verified relative portion only.
        try:
            relative_part = candidate.relative_to(base_dir)
        except ValueError:
            logger.warning(f"Path traversal blocked after resolve: {body.pdf_path!r}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid pdf_path: path escapes the papers directory."
            )

        # Reconstruct from trusted base only (breaks taint chain for CodeQL)
        safe_pdf_path = base_dir / relative_part

        # Confirm it exists and is a regular file
        if not safe_pdf_path.exists() or not safe_pdf_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found: {body.pdf_path}"
            )

        logger.info(f"Ingesting document: {safe_pdf_path}")

        # Ingest the PDF (run in thread pool to avoid blocking event loop)
        num_chunks, title = await run_in_threadpool(
            ingest_module.ingest_pdf,
            pdf_path=str(safe_pdf_path),
            paper_id=safe_pdf_path.stem
        )

        try:
            import bm25_search
            bm25_search.invalidate()
            threading.Thread(target=bm25_search.get_or_build_index, daemon=True).start()
        except Exception:
            pass
        try:
            from cache import retrieval_cache, tool_cache
            retrieval_cache.invalidate()
            tool_cache.invalidate()
        except Exception:
            logger.warning("Failed to invalidate caches after ingestion", exc_info=True)

        processing_time = time.time() - start_time

        return IngestResponse(
            status="success",
            chunks_ingested=num_chunks,
            paper_id=safe_pdf_path.stem,
            title=title,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


def _run_bulk_ingest(job_id: str):
    """Background worker: runs ingest_directory and updates the job store."""
    import ingest as ingest_module
    start_time = time.time()
    _update_job(job_id, status="running")
    try:
        stats = ingest_module.ingest_directory(pdf_dir=str(config.PAPERS_DIR))
        try:
            import bm25_search
            bm25_search.invalidate()
            threading.Thread(target=bm25_search.get_or_build_index, daemon=True).start()
        except Exception:
            pass
        try:
            from cache import retrieval_cache, tool_cache
            retrieval_cache.invalidate()
            tool_cache.invalidate()
        except Exception:
            logger.warning("Failed to invalidate caches after bulk ingestion", exc_info=True)
        processing_time = time.time() - start_time
        status_value = "partial" if stats.get("failed", 0) > 0 else "success"
        _update_job(
            job_id,
            status=status_value,
            total_files=stats.get("total_files", 0),
            successful=stats.get("successful", 0),
            failed=stats.get("failed", 0),
            chunks_ingested=stats.get("total_chunks", 0),
            processing_time=processing_time,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"Bulk ingest job {job_id} finished: {status_value}")
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Bulk ingest job {job_id} failed: {e}", exc_info=True)
        _update_job(
            job_id,
            status="failed",
            processing_time=processing_time,
            error=str(e),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


@app.post("/ingest/all", response_model=IngestJobResponse, status_code=202, tags=["Management"])
@limiter.limit("2/minute")
async def ingest_all_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Kick off background ingestion of all PDFs in the papers directory.

    Returns **202 Accepted** with a `job_id` immediately.
    Poll `GET /ingest/status/{job_id}` to check progress and results.
    """
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "total_files": None,
            "successful": None,
            "failed": None,
            "chunks_ingested": None,
            "processing_time": None,
            "error": None,
        }
    background_tasks.add_task(_run_bulk_ingest, job_id)
    logger.info(f"Bulk ingest job {job_id} queued")
    return IngestJobResponse(
        job_id=job_id,
        status="pending",
        message="Ingestion started. Poll /ingest/status/{job_id} for progress."
    )


@app.get("/ingest/status/{job_id}", response_model=JobStatusResponse, tags=["Management"])
async def get_ingest_status(
    job_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Retrieve the status and results of a background ingestion job.

    Possible `status` values: `pending`, `running`, `success`, `partial`, `failed`.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found."
        )
    return JobStatusResponse(**job)


@app.get("/cache/stats", tags=["Management"])
async def cache_stats(authenticated: bool = Depends(verify_api_key)):
    """Get cache hit/miss statistics for LLM, retrieval, and tool caches."""
    from cache import llm_cache, retrieval_cache, tool_cache
    return {
        "llm": llm_cache.stats,
        "retrieval": retrieval_cache.stats,
        "tool": tool_cache.stats,
    }


@app.delete("/cache", tags=["Management"])
async def clear_caches(authenticated: bool = Depends(verify_api_key)):
    """Clear all caches."""
    from cache import llm_cache, retrieval_cache, tool_cache
    llm_cache.invalidate()
    retrieval_cache.invalidate()
    tool_cache.invalidate()
    return {"status": "cleared"}


@app.get("/stats", tags=["Management"])
async def get_stats(authenticated: bool = Depends(verify_api_key)):
    """Get vector store statistics."""
    try:
        import vector_store
        
        collection = vector_store.get_or_create_collection()
        stats = vector_store.get_collection_stats(collection)
        
        return {
            "collection_name": stats['name'],
            "document_count": stats['count'],
            "metadata": stats.get('metadata', {})
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


# ============================================================================
# Search Endpoint (retrieval-only, no LLM generation)
# ============================================================================


class SearchRequest(BaseModel):
    """Request model for retrieval-only search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: Optional[int] = Field(10, ge=1, le=30, description="Number of results")
    source: str = Field("corpus", description="Source: 'corpus' for indexed docs, 'web' for Tavily web search, 'both' for hybrid")

    @field_validator("source")
    @classmethod
    def validate_source(cls, v):
        if v not in ("corpus", "web", "both"):
            raise ValueError("Source must be 'corpus', 'web', or 'both'")
        return v


class SearchResult(BaseModel):
    """A single search result."""
    text: str
    title: str = ""
    source: str = ""
    section: str = ""
    score: float = 0.0


class SearchResponse(BaseModel):
    """Response model for retrieval-only search."""
    query: str
    language: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    timestamp: str


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(
    request: SearchRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """
    Search for relevant passages without LLM generation.

    Returns raw retrieved chunks from the indexed corpus, Tavily web search,
    or both. Useful for debugging retrieval quality or building custom UIs.
    """
    import lang_utils
    start_time = time.time()

    try:
        results = []
        detected_lang = lang_utils.detect_language(request.query) or "en"

        if request.source in ("corpus", "both"):
            context_data = await run_in_threadpool(
                rag.retrieve_context, request.query, request.top_k
            )
            for chunk, meta, dist in zip(
                context_data["chunks"],
                context_data["metadatas"],
                context_data["distances"],
            ):
                results.append(SearchResult(
                    text=chunk,
                    title=meta.get("title", ""),
                    source=meta.get("paper_id", ""),
                    section=meta.get("section", ""),
                    score=round(float(dist), 4),
                ))

        if request.source in ("web", "both"):
            try:
                from agent.tool_executor import execute_web_search
                web_result = await run_in_threadpool(
                    execute_web_search, request.query, min(request.top_k, 10)
                )
                for p in web_result.get("passages", []):
                    results.append(SearchResult(
                        text=p.get("text", ""),
                        title=p.get("title", ""),
                        source=p.get("source", ""),
                        section="web",
                        score=0.0,
                    ))
            except ValueError:
                logger.warning("Web search requested but TAVILY_API_KEY not configured")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")

        processing_time = time.time() - start_time
        logger.info(
            f"Search: query='{request.query[:50]}', source={request.source}, "
            f"results={len(results)}, time={processing_time:.2f}s"
        )

        return SearchResponse(
            query=request.query,
            language=detected_lang,
            results=results,
            total_results=len(results),
            processing_time=processing_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


# ============================================================================
# Document Management Endpoints
# ============================================================================

from fastapi import UploadFile, File
import shutil


class PaperInfo(BaseModel):
    """Information about an uploaded paper."""
    filename: str
    size_bytes: int
    size_mb: float


class UploadResponse(BaseModel):
    """Response model for file upload."""
    status: str
    filename: str
    size_bytes: int
    message: str


class PurgeResponse(BaseModel):
    """Response model for purge operations."""
    status: str
    deleted_count: int
    message: str


@app.post("/upload", response_model=UploadResponse, tags=["Management"])
async def upload_pdf(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Upload a PDF file to the papers directory.
    
    The file will be saved but NOT automatically ingested.
    Use the /ingest endpoint to add it to the vector store.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    # Sanitize filename
    safe_filename = Path(file.filename).name
    destination = config.PAPERS_DIR / safe_filename
    
    try:
        MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

        # Stream into file so a multi-GB upload never fits in RAM
        received = 0
        with open(destination, "wb") as buffer:
            while chunk := await file.read(65536):
                received += len(chunk)
                if received > MAX_UPLOAD_SIZE:
                    buffer.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File too large (max 50MB)"
                    )
                buffer.write(chunk)

        file_size = received
        logger.info(f"Uploaded file: {safe_filename} ({file_size} bytes)")
        
        return UploadResponse(
            status="success",
            filename=safe_filename,
            size_bytes=file_size,
            message=f"File uploaded successfully. Use /ingest to add to vector store."
        )
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


@app.get("/papers", response_model=List[PaperInfo], tags=["Management"])
def list_papers(authenticated: bool = Depends(verify_api_key)):
    """
    List all PDF files in the papers directory.
    """
    try:
        papers = []
        for pdf_file in config.PAPERS_DIR.glob("*.pdf"):
            size = pdf_file.stat().st_size
            papers.append(PaperInfo(
                filename=pdf_file.name,
                size_bytes=size,
                size_mb=round(size / (1024 * 1024), 2)
            ))
        
        return sorted(papers, key=lambda p: p.filename)
    
    except Exception as e:
        logger.error(f"Error listing papers: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


class DeletePaperResponse(BaseModel):
    paper_id: str
    chunks_deleted: int
    status: str


class PatchPaperRequest(BaseModel):
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    tags: Optional[str] = None

    @field_validator("title", "authors", "year", "tags", mode="before")
    @classmethod
    def reject_empty_string(cls, v):
        if v is not None and v == "":
            raise ValueError("field must not be empty string")
        return v


class PatchPaperResponse(BaseModel):
    paper_id: str
    chunks_updated: int
    updates: dict


@app.delete("/papers/{paper_id}", response_model=DeletePaperResponse, tags=["Management"])
async def delete_paper(
    paper_id: str,
    authenticated: bool = Depends(verify_api_key),
):
    """Delete all indexed chunks for a specific paper from the vector store."""
    import vector_store
    chunks_deleted = await run_in_threadpool(vector_store.delete_by_paper_id, paper_id)
    if chunks_deleted == 0:
        raise HTTPException(status_code=404, detail="paper not found or already deleted")
    try:
        import bm25_search
        bm25_search.invalidate()
        threading.Thread(target=bm25_search.get_or_build_index, daemon=True).start()
    except Exception:
        pass
    try:
        from cache import retrieval_cache, tool_cache
        retrieval_cache.invalidate()
        tool_cache.invalidate()
    except Exception:
        logger.warning("Failed to invalidate caches after paper deletion", exc_info=True)
    return DeletePaperResponse(paper_id=paper_id, chunks_deleted=chunks_deleted, status="deleted")


@app.patch("/papers/{paper_id}", response_model=PatchPaperResponse, tags=["Management"])
async def patch_paper(
    paper_id: str,
    request: PatchPaperRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """Update metadata fields on all chunks for a specific paper."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="no valid fields to update")
    import vector_store
    chunks_updated = await run_in_threadpool(vector_store.update_paper_metadata, paper_id, updates)
    if chunks_updated == 0:
        raise HTTPException(status_code=404, detail="paper not found")
    try:
        from cache import retrieval_cache
        retrieval_cache.invalidate()
    except Exception:
        logger.warning("Failed to invalidate retrieval cache after paper update", exc_info=True)
    return PatchPaperResponse(paper_id=paper_id, chunks_updated=chunks_updated, updates=updates)


@app.delete("/purge/papers", response_model=PurgeResponse, tags=["Management"])
def purge_papers(authenticated: bool = Depends(verify_admin_key)):
    """
    Delete all PDF files from the papers directory.
    
    WARNING: This action cannot be undone!
    """
    try:
        pdf_files = list(config.PAPERS_DIR.glob("*.pdf"))
        deleted_count = 0
        
        for pdf_file in pdf_files:
            try:
                pdf_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted paper: {pdf_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {pdf_file.name}: {e}")
        
        return PurgeResponse(
            status="success",
            deleted_count=deleted_count,
            message=f"Deleted {deleted_count} PDF file(s)"
        )
    
    except Exception as e:
        logger.error(f"Error purging papers: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


@app.delete("/purge/database", response_model=PurgeResponse, tags=["Management"])
def purge_database(authenticated: bool = Depends(verify_admin_key)):
    """
    Clear the vector database (delete all indexed chunks).
    
    WARNING: This action cannot be undone!
    """
    try:
        import vector_store
        
        # Get current count before purge
        collection = vector_store.get_or_create_collection()
        previous_count = collection.count()
        
        # Delete and recreate collection
        vector_store.delete_collection(config.COLLECTION_NAME)
        
        # Recreate empty collection
        vector_store.get_or_create_collection()
        
        logger.info(f"Purged database: {previous_count} chunks deleted")
        
        return PurgeResponse(
            status="success",
            deleted_count=previous_count,
            message=f"Deleted {previous_count} chunks from vector database"
        )
    
    except Exception as e:
        logger.error(f"Error purging database: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


# ============================================================================
# Agentic RAG Endpoint
# ============================================================================

from agent.state import AgentState

_agent_graph = None
_agent_graph_lock = threading.Lock()


def _get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        with _agent_graph_lock:
            if _agent_graph is None:
                from agent.graph import build_agent_graph
                _agent_graph = build_agent_graph()
    return _agent_graph


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    strategy: str = Field("A")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        if v not in ("A", "B"):
            raise ValueError("Strategy must be 'A' or 'B'")
        return v


class AgentSource(BaseModel):
    title: str
    source: str = ""
    section: str = ""
    pdf_url: str = ""
    year: str = ""
    authors: str = ""
    citations: int = 0


class AgentQueryResponse(BaseModel):
    answer: str
    session_id: str
    language: str
    reflexion_iterations: int
    tool_calls: List[dict]
    sources: List[AgentSource]
    processing_time: float
    timestamp: str


@app.post("/agent/query", response_model=AgentQueryResponse, tags=["Agent"])
@limiter.limit("10/minute")
async def agent_query(
    request: Request,
    body: AgentQueryRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """
    Answer a question using the agentic IndicRAG pipeline with reflexion loops.
    Supports the same 10+ languages as /query and /chat.
    """
    start_time = time.time()
    session_id, messages = _get_or_create_session(body.session_id)

    initial_state = AgentState(
        original_query=body.question,
        detected_language="",
        query_plan=[],
        tool_calls_requested=[],
        retrieved_contexts=[],
        draft_answer=None,
        final_answer=None,
        reflexion_count=0,
        reflexion_history=[],
        tool_calls_log=[],
        conversation_history=list(messages),
        session_id=session_id,
        strategy=body.strategy,
    )

    import asyncio
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(_get_agent_graph().invoke, initial_state),
            timeout=float(config.AGENT_TIMEOUT),
        )
    except asyncio.TimeoutError:
        logger.warning(f"Agent timed out after {config.AGENT_TIMEOUT}s for query: {body.question[:80]}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"error": "Agent pipeline timed out. Try a simpler query or use Standard RAG mode.", "code": "AGENT_TIMEOUT"},
        )
    except Exception as e:
        err_str = str(e)
        # Detect Gemini 503/429 — surface as 503 so the client knows to retry
        is_llm_unavailable = (
            "503" in err_str or "429" in err_str
            or "UNAVAILABLE" in err_str
            or "RESOURCE_EXHAUSTED" in err_str
            or "high demand" in err_str.lower()
            or "unreachable" in err_str.lower()
        )
        if is_llm_unavailable:
            logger.warning(f"LLM service unavailable for agent query: {err_str[:200]}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "The AI model is temporarily unavailable due to high demand. "
                             "Please try again in a few seconds.",
                    "code": "LLM_UNAVAILABLE",
                },
            )
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Agent pipeline failed. Please try again later.", "code": "AGENT_ERROR"},
        )

    _append_session_messages(session_id, body.question, result["final_answer"])
    processing_time = time.time() - start_time

    logger.info(
        f"Agent query: lang={result['detected_language']} "
        f"reflexion={result['reflexion_count']} time={processing_time:.2f}s"
    )

    # Extract sources: prefer only those actually cited in the final answer
    all_contexts = result.get("retrieved_contexts", [])
    final_answer = result["final_answer"]
    cited_titles: set[str] = set()
    try:
        metas = [{"title": c.get("title", "Unknown"), "section": c.get("section", "body")}
                 for c in all_contexts]
        chunks = [c.get("text", "") for c in all_contexts]
        for cit in rag.extract_citations(final_answer, metas, chunks):
            cited_titles.add(cit["title"].strip())
    except Exception:
        pass  # fall through to dedup-only logic below

    seen_titles: set[str] = set()
    sources = []
    for ctx in all_contexts:
        title = ctx.get("title", "").strip()
        if not title or title in seen_titles or title in ("Unknown", "No results"):
            continue
        # When citation extraction succeeded, skip uncited sources
        if cited_titles and title not in cited_titles:
            continue
        seen_titles.add(title)
        sources.append(AgentSource(
            title=title,
            source=ctx.get("source", ""),
            section=ctx.get("section", ""),
            pdf_url=ctx.get("pdf_url", ""),
            year=ctx.get("year", ""),
            authors=ctx.get("authors", ""),
            citations=ctx.get("citations", 0),
        ))

    return AgentQueryResponse(
        answer=result["final_answer"],
        session_id=session_id,
        language=result.get("detected_language", "en"),
        reflexion_iterations=result.get("reflexion_count", 0),
        tool_calls=result.get("tool_calls_log", []),
        sources=sources,
        processing_time=processing_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=config.LOG_LEVEL.lower()
    )
