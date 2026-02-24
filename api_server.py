"""
FastAPI server for the Multilingual Scientific RAG system.
Production-ready REST API with authentication, validation, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status, BackgroundTasks
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
from datetime import datetime
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


def _update_job(job_id: str, **kwargs):
    """Thread-safe update of a job's fields."""
    with _jobs_lock:
        _jobs[job_id].update(kwargs)

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

# Prometheus Monitoring
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# Mount static files directory
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", 
        "http://localhost:8000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8000"
    ],  # Explicit origins instead of '*' when credentials=True
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
        
        # Process query (run in thread pool to avoid blocking event loop)
        result = await run_in_threadpool(
            rag.answer_question,
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

        # At this point request.pdf_path is already sanitized by the Pydantic validator:
        # - not absolute, no '..' components, safe characters only.
        base_dir = Path(config.PAPERS_DIR).resolve()

        # Build and resolve the candidate path
        candidate = (base_dir / request.pdf_path).resolve()

        # Use Path.relative_to() as the authoritative containment check.
        # This raises ValueError if candidate is not inside base_dir, ensuring
        # the taint chain is severed: safe_pdf_path is reconstructed from
        # base_dir (trusted) + the verified relative portion only.
        try:
            relative_part = candidate.relative_to(base_dir)
        except ValueError:
            logger.warning(f"Path traversal blocked after resolve: {request.pdf_path!r}")
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
                detail=f"PDF file not found: {request.pdf_path}"
            )

        logger.info(f"Ingesting document: {safe_pdf_path}")

        # Ingest the PDF (run in thread pool to avoid blocking event loop)
        num_chunks, title = await run_in_threadpool(
            ingest_module.ingest_pdf,
            pdf_path=str(safe_pdf_path),
            paper_id=safe_pdf_path.stem
        )

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
            detail=f"Error ingesting document: {str(e)}"
        )


def _run_bulk_ingest(job_id: str):
    """Background worker: runs ingest_directory and updates the job store."""
    import ingest as ingest_module
    start_time = time.time()
    _update_job(job_id, status="running")
    try:
        stats = ingest_module.ingest_directory(pdf_dir=str(config.PAPERS_DIR))
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
            completed_at=datetime.utcnow().isoformat(),
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
            completed_at=datetime.utcnow().isoformat(),
        )


@app.post("/ingest/all", response_model=IngestJobResponse, status_code=202, tags=["Management"])
async def ingest_all_documents(
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
            "submitted_at": datetime.utcnow().isoformat(),
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
        
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large (max 50MB)"
            )
            
        # Save uploaded file
        with open(destination, "wb") as buffer:
            buffer.write(content)
        
        file_size = destination.stat().st_size
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
            detail=f"Error uploading file: {str(e)}"
        )


@app.get("/papers", response_model=List[PaperInfo], tags=["Management"])
async def list_papers(authenticated: bool = Depends(verify_api_key)):
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
            detail=f"Error listing papers: {str(e)}"
        )


@app.delete("/purge/papers", response_model=PurgeResponse, tags=["Management"])
async def purge_papers(authenticated: bool = Depends(verify_api_key)):
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
            detail=f"Error purging papers: {str(e)}"
        )


@app.delete("/purge/database", response_model=PurgeResponse, tags=["Management"])
async def purge_database(authenticated: bool = Depends(verify_api_key)):
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
            detail=f"Error purging database: {str(e)}"
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
