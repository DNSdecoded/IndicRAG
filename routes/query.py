"""Routes: /, /health, /query, /query/stream."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

import config
import rag
from deps import STATIC_DIR, limiter, verify_api_key
from sse_utils import sse_stream

logger = logging.getLogger(__name__)
router = APIRouter()


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
    query_id: str
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
        import vector_store, embeddings, rerank
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
    authenticated: bool = Depends(verify_api_key)
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
            top_k=top_k
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
                section=cite['section']
            )
            for cite in result['citations']
        ]

        return QueryResponse(
            query_id=str(uuid.uuid4()),
            answer=result['answer'],
            language=result['language'],
            language_name=result['language_name'],
            chunks_used=result['chunks_used'],
            citations=citations,
            processing_time=processing_time,
            timestamp=datetime.now(timezone.utc).isoformat()
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

    prepared = await run_in_threadpool(rag.prepare_query_for_stream, body.question, body.strategy, top_k)
    query_id = str(uuid.uuid4())

    if prepared["chunks_used"] == 0:
        async def _no_docs():
            yield f"data: {json.dumps({'type': 'chunk', 'text': prepared['no_docs_msg']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'citations': [], 'language': prepared['detected_lang'], 'query_id': query_id})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_no_docs(), media_type="text/event-stream")

    return StreamingResponse(
        sse_stream(prepared["prompt"], prepared["metadatas"], prepared["detected_lang"],
                   strategy=body.strategy, query_id=query_id),
        media_type="text/event-stream",
    )
