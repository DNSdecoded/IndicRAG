"""Routes: /ingest, /ingest/all, /ingest/status/{job_id}, /upload."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import logging
import re as _re
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import config
from agent.tool_executor import execute_arxiv_search, execute_open_access_search
from cache_refresh import _post_ingest_refresh
from deps import limiter, verify_api_key, _jobs, _jobs_lock, _update_job

logger = logging.getLogger(__name__)
router = APIRouter()

# Block the genuinely dangerous characters: null bytes, shell metacharacters.
# We rely on the is_absolute() + relative_to() checks for traversal; the regex
# only needs to reject characters that can't appear in safe filenames.
_UNSAFE_CHARS_RE = _re.compile(r'[\x00\|;&`$<>"\'\!\*\?\{\}\[\]\\~]')


def _resolve_papers_path(rel_path: str) -> Path:
    """Resolve a papers-relative path safely and confirm the file exists.

    Reconstructs from the trusted base after a relative_to() check so a resolved
    path can't escape the papers directory (breaks taint chain for CodeQL).
    Raises HTTPException(400) on traversal, 404 when the file is missing.
    """
    base_dir = Path(config.PAPERS_DIR).resolve()
    candidate = (base_dir / rel_path).resolve()
    try:
        relative_part = candidate.relative_to(base_dir)
    except ValueError:
        logger.warning(f"Path traversal blocked after resolve: {rel_path!r}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid path: path escapes the papers directory."
        )
    safe_path = base_dir / relative_part
    if not safe_path.exists() or not safe_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found: {rel_path}"
        )
    return safe_path


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
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    progress_message: Optional[str] = None


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

        safe_pdf_path = _resolve_papers_path(body.pdf_path)

        logger.info(f"Ingesting document: {safe_pdf_path}")

        num_chunks, title = await run_in_threadpool(
            ingest_module.ingest_pdf,
            pdf_path=str(safe_pdf_path),
            paper_id=safe_pdf_path.stem
        )

        _post_ingest_refresh()

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


def _make_progress_cb(job_id: str):
    """Return a progress callback that writes live progress into the in-memory job.

    In-memory only (no persistence write) so per-paper updates don't hammer
    SQLite; progress is transient and surfaced via the SSE stream / status poll.
    """
    def _cb(current: int, total: int, message: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["progress_current"] = current
                job["progress_total"] = total
                job["progress_message"] = message
    return _cb


def _run_bulk_ingest(job_id: str):
    """Background worker: runs ingest_directory and updates the job store."""
    import ingest as ingest_module
    start_time = time.time()
    _update_job(job_id, status="running")
    try:
        stats = ingest_module.ingest_directory(
            pdf_dir=str(config.PAPERS_DIR),
            progress_cb=_make_progress_cb(job_id),
        )
        _post_ingest_refresh()
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
            "progress_current": 0,
            "progress_total": None,
            "progress_message": "Queued",
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


@router.get("/ingest/stream/{job_id}", tags=["Management"])
async def stream_ingest_progress(
    job_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Server-Sent Events stream of a bulk ingestion job's live progress.

    Emits one `data:` event per progress change (e.g. "Ingesting foo.pdf (3/50)"),
    then a final event when the job reaches a terminal state, and closes.
    """
    import asyncio
    import json

    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found."
            )

    _FIELDS = ("status", "progress_current", "progress_total",
               "progress_message", "chunks_ingested", "successful", "failed")
    _TERMINAL = {"success", "partial", "failed"}

    async def event_gen():
        last = None
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
                snapshot = {k: job.get(k) for k in _FIELDS} if job else None
            if snapshot is None:
                break
            key = (snapshot["status"], snapshot["progress_current"], snapshot["progress_message"])
            if key != last:
                last = key
                yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["status"] in _TERMINAL:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


class DryRunResponse(BaseModel):
    """Result of a dry-run ingest: what WOULD be ingested, without storing it."""
    title: str
    text_length: int
    num_sections: int
    total_chunks: int
    content_hash: str
    sections: list


@router.post("/ingest/dry-run", response_model=DryRunResponse, tags=["Management"])
@limiter.limit("10/minute")
async def dry_run_document(
    request: Request,
    body: IngestRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Process a PDF (extract text, detect sections, count chunks) WITHOUT embedding
    or storing it. Useful for debugging ingestion quality before committing.
    """
    import ingest as ingest_module

    safe_pdf_path = _resolve_papers_path(body.pdf_path)
    result = await run_in_threadpool(ingest_module.dry_run_pdf, str(safe_pdf_path))
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not process PDF (possibly a scanned/image PDF needing OCR)."
        )
    return DryRunResponse(**result)


class ReindexRequest(BaseModel):
    """Request model for in-place re-ingestion of an existing paper."""
    paper_id: str = Field(..., description="paper_id (PDF filename stem) to re-embed in place")

    @field_validator('paper_id')
    @classmethod
    def sanitize_paper_id(cls, v: str) -> str:
        """paper_id maps to '<paper_id>.pdf' in papers/; reject anything path-like."""
        v = (v or "").strip()
        if not v:
            raise ValueError("paper_id must not be empty.")
        if '/' in v or '\\' in v or '..' in v or _UNSAFE_CHARS_RE.search(v):
            raise ValueError("paper_id contains invalid characters.")
        return v


@router.post("/ingest/reindex", response_model=IngestResponse, tags=["Management"])
@limiter.limit("5/minute")
async def reindex_document(
    request: Request,
    body: ReindexRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Re-embed a single already-ingested paper in place (e.g. after changing chunk
    parameters). Deletes the paper's existing chunks first so re-embedding runs
    even when the source file is unchanged, then re-ingests from papers/.
    """
    start_time = time.time()
    import ingest as ingest_module
    import vector_store

    safe_pdf_path = _resolve_papers_path(f"{body.paper_id}.pdf")

    # Delete first so ingest_pdf's unchanged-file-hash check doesn't skip it.
    await run_in_threadpool(vector_store.delete_by_paper_id, body.paper_id)

    num_chunks, title = await run_in_threadpool(
        ingest_module.ingest_pdf,
        pdf_path=str(safe_pdf_path),
        paper_id=body.paper_id,
    )

    _post_ingest_refresh()

    return IngestResponse(
        status="success",
        chunks_ingested=num_chunks,
        paper_id=body.paper_id,
        title=title,
        processing_time=time.time() - start_time,
    )


class IngestURLRequest(BaseModel):
    """Frictionless ingestion: arXiv ID, DOI, direct PDF URL, or a reading list
    (newline-separated mix of any of the above)."""
    url: Optional[str] = Field(None, description="Direct PDF URL.")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID, e.g. '2301.07041'.")
    doi: Optional[str] = Field(None, description="DOI, resolved via open-access search.")
    reading_list: Optional[str] = Field(
        None, description="Newline-separated arXiv IDs, DOIs, and/or direct URLs."
    )


def _resolve_one(item: str) -> Optional[dict]:
    """Resolve a single arXiv ID, DOI, or URL to a downloadable {url, id, title}."""
    item = item.strip()
    if not item:
        return None
    if item.startswith("http"):
        return {"url": item, "id": item, "title": ""}
    if item.startswith("10.") or "doi.org" in item:
        result = execute_open_access_search(item, max_results=1)
        passages = result.get("passages", [])
        if passages and passages[0].get("pdf_url"):
            return {"url": passages[0]["pdf_url"], "id": item, "title": passages[0].get("title", "")}
        return None
    # Assume arXiv ID
    result = execute_arxiv_search(item, max_results=1)
    passages = result.get("passages", [])
    if passages and passages[0].get("pdf_url"):
        return {"url": passages[0]["pdf_url"], "id": item, "title": passages[0].get("title", "")}
    return None


def _resolve_urls_to_ingest(
    url: str = None, arxiv_id: str = None, doi: str = None, reading_list: str = None,
) -> List[dict]:
    """Resolve any combination of url/arxiv_id/doi/reading_list into a flat,
    order-preserving list of {url, id, title} dicts ready for download_pdf()."""
    resolved: List[dict] = []

    if arxiv_id:
        # arxiv_id is a bare ID, not free text — resolve directly, not via _resolve_one's sniffing.
        result = execute_arxiv_search(arxiv_id.strip(), max_results=1)
        passages = result.get("passages", [])
        if passages and passages[0].get("pdf_url"):
            resolved.append({"url": passages[0]["pdf_url"], "id": arxiv_id.strip(),
                              "title": passages[0].get("title", "")})

    if doi:
        result = execute_open_access_search(doi.strip(), max_results=1)
        passages = result.get("passages", [])
        if passages and passages[0].get("pdf_url"):
            resolved.append({"url": passages[0]["pdf_url"], "id": doi.strip(),
                              "title": passages[0].get("title", "")})

    if url:
        resolved.append({"url": url.strip(), "id": url.strip(), "title": ""})

    if reading_list:
        for line in reading_list.strip().split("\n"):
            item = _resolve_one(line)
            if item:
                resolved.append(item)

    return resolved


def _run_batch_url_ingest(job_id: str, urls_to_ingest: List[dict]):
    """Background worker: download → ingest → refresh caches for each resolved URL.

    Skips any item whose sanitized paper_id already exists in the corpus —
    unlike /upload (409s on filename collision) or /reindex (explicit,
    intentional overwrite), this route has no confirmation step, so silently
    ingesting over an existing paper_id would let a crafted arxiv_id/doi/url
    overwrite another paper's chunks with attacker-supplied content.
    """
    import ingest as ingest_module
    import vector_store

    from download_utils import download_pdf

    start_time = time.time()
    _update_job(job_id, status="running")
    progress_cb = _make_progress_cb(job_id)
    collection = vector_store.get_or_create_collection()
    got = vector_store._chroma_call(collection.get, include=["metadatas"])
    existing_paper_ids = {
        (m or {}).get("paper_id", "") for m in got.get("metadatas", [])
    } - {""}
    successful = failed = chunks_ingested = 0
    for i, item in enumerate(urls_to_ingest):
        progress_cb(i, len(urls_to_ingest), f"Downloading {item['id']}")
        paper_id = _bibtex_safe_id(item["id"])
        if paper_id in existing_paper_ids:
            logger.warning(f"Skipping {item['id']}: paper_id '{paper_id}' already exists")
            failed += 1
            continue
        path = download_pdf(item["url"])
        if not path:
            failed += 1
            continue
        try:
            n_chunks, _ = ingest_module.ingest_pdf(
                path, paper_id=paper_id,
                metadata={"title": item.get("title", "")},
            )
            chunks_ingested += n_chunks
            successful += 1
            existing_paper_ids.add(paper_id)  # a reading_list can carry duplicate ids too
        except Exception as e:
            logger.warning(f"Ingest failed for {item['id']}: {e}")
            failed += 1
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass

    if successful:
        _post_ingest_refresh()

    status_value = "partial" if failed and successful else ("failed" if failed else "success")
    _update_job(
        job_id, status=status_value, total_files=len(urls_to_ingest),
        successful=successful, failed=failed, chunks_ingested=chunks_ingested,
        processing_time=time.time() - start_time,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(f"URL ingest job {job_id} finished: {status_value}")


def _bibtex_safe_id(raw_id: str) -> str:
    """arXiv IDs/DOIs/URLs can contain '/', '.', ':' — none of which are safe as a
    ChromaDB paper_id used in file-adjacent contexts. Keep it short and filesystem-safe."""
    return _re.sub(r"[^A-Za-z0-9_-]", "_", raw_id)[:100] or "paper"


@router.post("/ingest/from-url", response_model=IngestJobResponse, status_code=202, tags=["Ingest"])
@limiter.limit("5/minute")
async def ingest_from_url(
    request: Request,
    body: IngestURLRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key),
):
    """Ingest paper(s) by arXiv ID, DOI, direct PDF URL, or a reading list of any mix."""
    urls_to_ingest = await run_in_threadpool(
        _resolve_urls_to_ingest, body.url, body.arxiv_id, body.doi, body.reading_list
    )
    if not urls_to_ingest:
        raise HTTPException(status_code=400, detail="Could not resolve any URLs from the provided input")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id, "status": "pending",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None, "total_files": len(urls_to_ingest),
            "successful": None, "failed": None, "chunks_ingested": None,
            "processing_time": None, "error": None,
            "progress_current": 0, "progress_total": len(urls_to_ingest),
            "progress_message": "Queued",
        }
    background_tasks.add_task(_run_batch_url_ingest, job_id, urls_to_ingest)
    return IngestJobResponse(
        job_id=job_id, status="pending",
        message=f"Resolved {len(urls_to_ingest)} paper(s). Poll /ingest/status/{{job_id}} for progress.",
    )


@router.post("/upload", response_model=UploadResponse, tags=["Management"])
@limiter.limit("10/minute")
async def upload_pdf(
    request: Request,
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

    if destination.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A file named '{safe_filename}' already exists. "
                   "Delete it first (DELETE /papers/{paper_id}) or rename the upload."
        )

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
            message="File uploaded successfully. Use /ingest to add to vector store."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )
