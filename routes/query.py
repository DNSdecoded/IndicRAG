"""Routes: /, /health, /query, /query/stream."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

import config
import persistence
import rag
from agent.nodes.finalizer import citation_coverage
from deps import STATIC_DIR, limiter, verify_api_key, current_owner, _jobs, _jobs_lock, _update_job
from sse_utils import sse_stream

logger = logging.getLogger(__name__)
router = APIRouter()


def build_paper_filter(paper_ids: Optional[List[str]]) -> Optional[Dict]:
    """Build a ChromaDB metadata filter that scopes retrieval to specific papers.

    Returns {'paper_id': {'$in': [...]}} for a non-empty list, else None (no
    scoping). This is what enforces the document boundary — without it, retrieval
    spans the whole corpus and "only this paper" can't be honoured.
    """
    ids = [p.strip() for p in (paper_ids or []) if p and p.strip()]
    if not ids:
        return None
    return {"paper_id": {"$in": ids}}


def build_tags_filter(tags: Optional[str]) -> Optional[Dict]:
    """Parse comma-separated tags into a rag.retrieve_context post-filter sentinel, or None.

    Not a ChromaDB where-clause: PATCH /papers stores tags as one unsplit
    string, so a native $in match against split tag names would never equal
    the stored value for any paper with more than one tag. rag.retrieve_context
    (via rag._extract_tags_post_filter) pulls this sentinel back out and
    applies it as a Python-side post-filter instead.
    """
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    if not tag_list:
        return None
    import rag
    return {rag._TAGS_SENTINEL: tag_list}


def combine_filters(*filters: Optional[Dict]) -> Optional[Dict]:
    """AND together any number of ChromaDB filters, dropping the None ones."""
    clauses = [f for f in filters if f]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class QueryRequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., min_length=1, max_length=1000, description="User question in any language")
    strategy: str = Field("A", description="Strategy: 'A' for multilingual LLM, 'B' for English + translation")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    paper_ids: Optional[List[str]] = Field(
        None, description="Restrict retrieval to these paper_ids (PDF filename stems). Omit for whole corpus."
    )
    tags: Optional[str] = Field(None, description="Comma-separated tags to filter retrieval.")
    model: Optional[str] = Field(None, description="LLM model id from the /models allowlist. Omit for default.")
    provider: Optional[str] = Field(None, description="LLM provider override (gemini|openrouter). Usually inferred from model.")

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        if v not in ['A', 'B']:
            raise ValueError("Strategy must be 'A' or 'B'")
        return v

    @field_validator("model")
    @classmethod
    def validate_model_allowlisted(cls, v):
        from routes.models import validate_model
        validate_model(v, None)
        return v


class FigureRef(BaseModel):
    """A figure/table crop a cited paper contributed (Phase 3 multimodal)."""
    page: Optional[int] = None
    chunk_type: str
    url: str


class Citation(BaseModel):
    """Citation information."""
    number: str
    title: str
    section: str
    figures: List[FigureRef] = []


class QueryResponse(BaseModel):
    """Response model for question answering."""
    query_id: str
    answer: str
    language: str
    language_name: str
    chunks_used: int
    citations: List[Citation]
    processing_time: float
    timestamp: str
    confidence: float = 0.0
    evidence: List[dict] = []


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    gemini_configured: bool
    checks: Optional[Dict[str, str]] = None


@router.get("/", tags=["General"])
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


@router.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check(deep: bool = False):
    """Health check endpoint. Add ?deep=true for component-level checks."""
    checks = None
    health_status = "healthy"
    if deep:
        import vector_store
        import embeddings
        import rerank
        checks = {}

        # ChromaDB (critical)
        try:
            await run_in_threadpool(vector_store.get_collection_stats)
            checks["chromadb"] = "ok"
        except Exception:
            checks["chromadb"] = "error"
            health_status = "unhealthy"

        # Embeddings (non-critical — lazy singleton, None means not yet loaded)
        try:
            if getattr(embeddings, '_embedding_model', None) is None:
                checks["embeddings"] = "not_loaded"
            else:
                checks["embeddings"] = "ok"
        except Exception:
            checks["embeddings"] = "error"

        # Reranker
        try:
            if not config.USE_RERANKER:
                checks["reranker"] = "not_configured"
            elif getattr(rerank, '_model', None) is None:
                checks["reranker"] = "not_loaded"
            else:
                checks["reranker"] = "ok"
        except Exception:
            checks["reranker"] = "error"

        if health_status != "unhealthy" and any(v == "error" for v in checks.values()):
            health_status = "degraded"

    return HealthResponse(
        status=health_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=config.VERSION,
        gemini_configured=bool(config.LLM_API_KEY_POOL),
        checks=checks,
    )


@router.post("/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit("30/minute")
async def query_question(
    request: Request,
    body: QueryRequest,
    authenticated: bool = Depends(verify_api_key),
    owner: Optional[str] = Depends(current_owner),
):
    """
    Answer a question in any language using the RAG system.

    Supports 10+ Indian languages plus English.
    """
    import time
    start_time = time.time()

    try:
        logger.info(
            "Query received: strategy=%s, question_len=%d",
            body.strategy,
            len(body.question),
        )

        top_k = body.top_k
        if top_k is not None:
            top_k = max(1, min(top_k, 20))  # Clamp to [1, 20]

        result = await run_in_threadpool(
            rag.answer_question,
            user_query=body.question,
            strategy=body.strategy,
            top_k=top_k,
            filter_dict=combine_filters(build_paper_filter(body.paper_ids), build_tags_filter(body.tags)),
            model=body.model,
            provider=body.provider,
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Query completed: lang={result['language']}, chunks={result['chunks_used']}, "
            f"time={processing_time:.2f}s"
        )

        citations = [
            Citation(
                number=cite['number'],
                title=cite['title'],
                section=cite['section'],
                figures=cite.get('figures', []),
            )
            for cite in result['citations']
        ]

        query_id = str(uuid.uuid4())
        try:
            persistence.log_query(
                query_id=query_id, question=body.question, answer=result['answer'],
                mode=f"standard_{body.strategy}", model=body.model or "default",
                language=result['language'], confidence=result.get('answer_confidence', 0.0),
                coverage=citation_coverage(result['answer']),
                created_at=datetime.now(timezone.utc).isoformat(),
                owner=owner,
            )
        except Exception:
            logger.warning("Failed to log query for feedback correlation", exc_info=True)

        return QueryResponse(
            query_id=query_id,
            answer=result['answer'],
            language=result['language'],
            language_name=result['language_name'],
            chunks_used=result['chunks_used'],
            citations=citations,
            processing_time=processing_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=result.get('answer_confidence', 0.0),
            evidence=result.get('faithfulness', []),
        )

    except ValueError as e:
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


@router.post("/query/stream", tags=["Query"])
@limiter.limit("30/minute")
async def query_stream(
    request: Request,
    body: QueryRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """Stream a RAG answer as Server-Sent Events."""
    top_k = body.top_k
    if top_k is not None:
        top_k = max(1, min(top_k, 20))

    prepared = await run_in_threadpool(rag.prepare_query_for_stream, body.question, body.strategy, top_k,
                                       combine_filters(build_paper_filter(body.paper_ids), build_tags_filter(body.tags)))
    query_id = str(uuid.uuid4())

    if prepared["chunks_used"] == 0:
        async def _no_docs():
            yield f"data: {json.dumps({'type': 'chunk', 'text': prepared['no_docs_msg']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'citations': [], 'language': prepared['detected_lang'], 'query_id': query_id})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_no_docs(), media_type="text/event-stream")

    return StreamingResponse(
        sse_stream(prepared["prompt"], prepared["metadatas"], prepared["detected_lang"],
                   strategy=body.strategy, query_id=query_id,
                   model=body.model, provider=body.provider,
                   visible_chunks=prepared["chunks_used"]),
        media_type="text/event-stream",
    )


class CompareRequest(BaseModel):
    """Request model for a cross-paper comparison table."""
    paper_ids: List[str] = Field(..., min_length=2, max_length=20, description="Paper IDs to compare (2-20)")
    dimensions: List[str] = Field(..., min_length=1, max_length=10, description="Dimensions to compare on (max 10)")
    model: Optional[str] = Field(None, description="LLM model id from the /models allowlist. Omit for default.")

    @field_validator("model")
    @classmethod
    def validate_model_allowlisted(cls, v):
        from routes.models import validate_model
        validate_model(v, None)
        return v


class CompareJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


def _run_compare_job(job_id: str, paper_ids: List[str], dimensions: List[str], model: Optional[str]):
    """Background worker: N papers x M dimensions is N*M LLM calls, so this
    runs as a job rather than blocking the request (same pattern as bulk ingest)."""
    _update_job(job_id, status="running")
    try:
        result = rag.compare_papers(paper_ids, dimensions, model)
        _update_job(
            job_id, status="success",
            dimensions=result["dimensions"], matrix=result["matrix"],
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Compare job {job_id} failed: {e}", exc_info=True)
        _update_job(
            job_id, status="failed", error=str(e),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


@router.post("/compare", response_model=CompareJobResponse, status_code=202, tags=["Comparison"])
@limiter.limit("5/minute")
async def compare_papers_endpoint(
    request: Request,
    body: CompareRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key),
    owner: Optional[str] = Depends(current_owner),
):
    """Generate a papers x dimensions comparison table across the corpus.

    Returns 202 with a job_id immediately; poll GET /compare/status/{job_id}.
    """
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id, "status": "pending",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None, "dimensions": None, "matrix": None, "error": None,
            # Fingerprint of the submitting key: _jobs is global, so without this
            # any authenticated caller who guesses a job_id reads someone else's
            # comparison results. None when auth is off (single-tenant).
            "owner": owner,
        }
    background_tasks.add_task(_run_compare_job, job_id, body.paper_ids, body.dimensions, body.model)
    return CompareJobResponse(
        job_id=job_id, status="pending",
        message="Comparison started. Poll /compare/status/{job_id} for the result.",
    )


@router.get("/compare/status/{job_id}", tags=["Comparison"])
async def get_compare_status(job_id: str, authenticated: bool = Depends(verify_api_key),
                             owner: Optional[str] = Depends(current_owner)):
    with _jobs_lock:
        job = _jobs.get(job_id)
    # 404 rather than 403 on an ownership mismatch — a 403 would confirm the
    # job_id exists.
    if job is None or (job.get("owner") is not None and job["owner"] != owner):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    return job
