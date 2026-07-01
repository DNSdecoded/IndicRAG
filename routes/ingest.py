"""Routes: /ingest, /ingest/all, /ingest/status/{job_id}, /upload."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import logging
import re as _re
import threading
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

import config
from deps import limiter, verify_api_key, _jobs, _jobs_lock, _update_job

logger = logging.getLogger(__name__)
router = APIRouter()

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
        if not v or not v.strip():
            raise ValueError("pdf_path must not be empty.")
        if PurePosixPath(v).is_absolute() or PureWindowsPath(v).is_absolute():
            raise ValueError("pdf_path must be a relative path, not an absolute path.")
        parts = PurePosixPath(v.replace('\\', '/')).parts
        if '..' in parts:
            raise ValueError("pdf_path must not contain '..' traversal sequences.")
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


class UploadResponse(BaseModel):
    """Response model for file upload."""
    status: str
    filename: str
    size_bytes: int
    message: str


@router.post("/ingest", response_model=IngestResponse, tags=["Management"])
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

        base_dir = Path(config.PAPERS_DIR).resolve()
        candidate = (base_dir / body.pdf_path).resolve()

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

        if not safe_pdf_path.exists() or not safe_pdf_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found: {body.pdf_path}"
            )

        logger.info(f"Ingesting document: {safe_pdf_path}")

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


@router.post("/ingest/all", response_model=IngestJobResponse, status_code=202, tags=["Management"])
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


@router.get("/ingest/status/{job_id}", response_model=JobStatusResponse, tags=["Management"])
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


@router.post("/upload", response_model=UploadResponse, tags=["Management"])
async def upload_pdf(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Upload a PDF file to the papers directory.

    The file will be saved but NOT automatically ingested.
    Use the /ingest endpoint to add it to the vector store.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )

    safe_filename = Path(file.filename).name
    destination = config.PAPERS_DIR / safe_filename

    try:
        MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )
