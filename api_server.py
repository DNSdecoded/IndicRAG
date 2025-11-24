"""
FastAPI server for the Multilingual Scientific RAG system.
Production-ready REST API with authentication, validation, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from pathlib import Path
import logging
import time
from datetime import datetime
import os

import rag
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual Scientific RAG API",
    description="Retrieval-Augmented Generation system for multilingual scientific Q&A",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Mount static files directory
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    pdf_path: str = Field(..., description="Path to PDF file (relative to papers/ directory)")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    chunks_ingested: int
    paper_id: str
    title: str
    processing_time: float


# Routes

@app.get("/", tags=["General"])
async def root():
    """Serve the web frontend."""
    if STATIC_DIR.exists():
        return FileResponse(str(STATIC_DIR / "index.html"))
    else:
        return {
            "name": "Multilingual Scientific RAG API",
            "version": "1.0.0",
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
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        gemini_configured=bool(config.LLM_API_KEY)
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_question(
    request: QueryRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Answer a question in any language using the RAG system.
    
    Supports 10+ Indian languages plus English.
    """
    start_time = time.time()
    
    try:
        # Log request (truncate question for privacy/brevity)
        question_preview = request.question[:80] + "..." if len(request.question) > 80 else request.question
        logger.info(f"Query received: strategy={request.strategy}, question='{question_preview}'")
        
        # Enforce top_k bounds even if client sends higher
        top_k = request.top_k
        if top_k is not None:
            top_k = max(1, min(top_k, 20))  # Clamp to [1, 20]
        
        # Process query
        result = rag.answer_question(
            user_query=request.question,
            strategy=request.strategy,
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
            timestamp=datetime.utcnow().isoformat()
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
            detail={
                "error": f"Error processing query: {str(e)}",
                "code": "INTERNAL_ERROR"
            }
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Management"])
async def ingest_document(
    request: IngestRequest,
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
        
        # Validate path
        pdf_path = Path(config.PAPERS_DIR) / request.pdf_path
        if not pdf_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found: {request.pdf_path}"
            )
        
        logger.info(f"Ingesting document: {pdf_path}")
        
        # Ingest the PDF
        num_chunks = ingest_module.ingest_pdf(
            pdf_path=str(pdf_path),
            paper_id=pdf_path.stem
        )
        
        processing_time = time.time() - start_time
        
        return IngestResponse(
            status="success",
            chunks_ingested=num_chunks,
            paper_id=pdf_path.stem,
            title=pdf_path.stem,  # Will be extracted in actual implementation
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting document: {str(e)}"
        )


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
            detail=f"Error getting stats: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level=config.LOG_LEVEL.lower()
    )
