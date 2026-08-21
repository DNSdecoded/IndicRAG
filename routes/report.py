"""Routes: /report — Phase 7 literature-review report workflow.

POST /report kicks off an async job that decomposes a topic into sections,
synthesizes a cited section per part from the corpus, and stores a Markdown
artifact. Poll GET /report/status/{job_id}; fetch GET /report/{job_id}/download.

Reuses the shared job store (deps._jobs) exactly like /ingest/all, and is gated
by config.REPORT_ENABLE (404 when off), mirroring /watch.
"""

from datetime import datetime, timezone
from typing import Optional
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

import config
import persistence
from deps import limiter, verify_api_key, current_owner, _jobs, _jobs_lock, _update_job

logger = logging.getLogger(__name__)
router = APIRouter()


def _require_enabled():
    if not config.REPORT_ENABLE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Literature-review reports are not enabled")


class ReportRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    language: str = Field("en", min_length=2, max_length=8)


class ReportJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ReportStatusResponse(BaseModel):
    job_id: str
    status: str                       # pending | running | success | failed
    topic: str
    language: str
    sections: Optional[list] = None
    citation_count: Optional[int] = None
    error: Optional[str] = None
    submitted_at: str
    completed_at: Optional[str] = None
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    progress_message: Optional[str] = None


def _make_progress_cb(job_id: str):
    """In-memory progress writes (no SQLite hammer), surfaced via status poll."""
    def _cb(current: int, total: int, message: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["progress_current"] = current
                job["progress_total"] = total
                job["progress_message"] = message
    return _cb


def _run_report_job(job_id: str, topic: str, language: str, owner: Optional[str] = None):
    """Background worker: build the report and store the Markdown on the job.

    `owner` is carried through so the persisted artifact is scoped to the key that
    requested it — GET /reports filters on it.
    """
    import report_runner
    _update_job(job_id, status="running")
    try:
        result = report_runner.run_report(topic, language, progress_cb=_make_progress_cb(job_id))
        completed_at = datetime.now(timezone.utc).isoformat()
        _update_job(
            job_id,
            status="success",
            sections=result["sections"],
            citation_count=result["citation_count"],
            markdown=result["markdown"],
            completed_at=completed_at,
        )
        persistence.save_report(
            report_id=job_id, watch_id="", topic=topic, language=language,
            markdown=result["markdown"], citation_count=result["citation_count"],
            created_at=completed_at, owner=owner,
        )
        logger.info(f"[Report] job {job_id} finished ({len(result['sections'])} sections)")
    except Exception as e:
        logger.error(f"[Report] job {job_id} failed: {e}", exc_info=True)
        _update_job(job_id, status="failed", error=str(e),
                    completed_at=datetime.now(timezone.utc).isoformat())


@router.post("/report", response_model=ReportJobResponse, status_code=202, tags=["Report"])
@limiter.limit("5/minute")
async def create_report(request: Request, body: ReportRequest, background_tasks: BackgroundTasks,
                        authenticated: bool = Depends(verify_api_key),
                        owner: Optional[str] = Depends(current_owner)):
    """Start a literature-review report job. Returns 202 with a job_id to poll.

    Rate-limited: each call decomposes a topic and synthesizes every section, so
    an unbounded loop here drains the shared LLM quota for every other user.
    """
    _require_enabled()
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id, "status": "pending", "owner": owner,
            "topic": body.topic, "language": body.language,
            "sections": None, "citation_count": None, "markdown": None, "error": None,
            "submitted_at": datetime.now(timezone.utc).isoformat(), "completed_at": None,
            "progress_current": 0, "progress_total": None, "progress_message": "Queued",
        }
    background_tasks.add_task(_run_report_job, job_id, body.topic, body.language, owner)
    logger.info(f"[Report] job {job_id} queued topic={body.topic!r}")
    return ReportJobResponse(job_id=job_id, status="pending",
                             message="Report started. Poll /report/status/{job_id}.")


def _get_report_job(job_id: str, owner: Optional[str] = None) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or "topic" not in job:  # 'topic' distinguishes a report job from an ingest job
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report job '{job_id}' not found.")
    # 404 rather than 403 on an ownership mismatch, matching the ingest-job rule in
    # routes/query.py — a 403 would confirm the job id exists.
    if owner is not None and job.get("owner") != owner:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report job '{job_id}' not found.")
    return job


@router.get("/report/status/{job_id}", response_model=ReportStatusResponse, tags=["Report"])
async def get_report_status(job_id: str, authenticated: bool = Depends(verify_api_key),
                            owner: Optional[str] = Depends(current_owner)):
    _require_enabled()
    job = _get_report_job(job_id, owner)
    return ReportStatusResponse(**{k: v for k, v in job.items()
                                   if k not in ("markdown", "owner")})


@router.get("/report/{job_id}/download", tags=["Report"])
async def download_report(job_id: str, authenticated: bool = Depends(verify_api_key),
                          owner: Optional[str] = Depends(current_owner)):
    """Download the finished report as a Markdown attachment."""
    _require_enabled()
    job = _get_report_job(job_id, owner)
    if job.get("status") != "success" or not job.get("markdown"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"Report not ready (status: {job.get('status')}).")
    filename = f"review-{job_id[:8]}.md"
    return Response(
        content=job["markdown"], media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/reports", tags=["Report"])
async def list_persisted_reports(watch_id: str = None, authenticated: bool = Depends(verify_api_key),
                                 owner: Optional[str] = Depends(current_owner)):
    """List durably-stored reports (survives restart), newest first.

    Distinct from GET /report/status/{job_id}: that's the in-memory job store
    for a report still being generated; this is the persisted artifact,
    including watch-owned living reviews that get regenerated in place.

    Scoped to the caller's own reports — a report body is generated from that
    user's topic and corpus and is not shared across keys.
    """
    return persistence.list_reports(watch_id, owner=owner)


@router.get("/reports/{report_id}", tags=["Report"])
async def get_persisted_report(report_id: str, authenticated: bool = Depends(verify_api_key),
                               owner: Optional[str] = Depends(current_owner)):
    report = persistence.get_report(report_id, owner=owner)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report '{report_id}' not found.")
    return report
