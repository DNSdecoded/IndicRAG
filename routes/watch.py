"""Routes: /watch — Phase 6 "watch a topic" registration.

CRUD plus run-one-watch (POST /watch/{id}/run) and the persisted-digest read
(GET /watch/{id}/digest); the background schedule loop lives in watch_runner.
All reuse persistence.save_watch / get_watch here.

Gated by config.WATCH_ENABLE (404 when off), mirroring the /prefs pattern.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

import config
import persistence
from deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


def _owned_or_404(watch_id: str, user_id: str) -> dict:
    """Load a watch and 404 unless it belongs to user_id (404, not 403 — don't
    leak that a watch exists for someone else)."""
    w = persistence.get_watch(watch_id)
    if w is None or (w.get("user_id") or config.DEFAULT_USER_ID) != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watch not found")
    return w

_CADENCES = {"daily": timedelta(days=1), "weekly": timedelta(days=7), "monthly": timedelta(days=30)}


def _require_enabled():
    if not config.WATCH_ENABLE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topic watches are not enabled")


def _next_run_from(now: datetime, cadence: str) -> str:
    """First auto-run one cadence interval out (nothing runs it until the loop lands)."""
    return (now + _CADENCES[cadence]).isoformat()


class WatchCreateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    language: str = Field("en", min_length=2, max_length=8)
    cadence: Optional[str] = None  # defaults to config.WATCH_DEFAULT_CADENCE

    @field_validator("cadence")
    @classmethod
    def validate_cadence(cls, v):
        if v is not None and v not in _CADENCES:
            raise ValueError(f"cadence must be one of {sorted(_CADENCES)}")
        return v


class WatchResponse(BaseModel):
    id: str
    user_id: str
    topic: str
    language: str
    cadence: str
    next_run: Optional[str]
    last_run: Optional[str]
    created_at: str
    seen_count: int
    has_digest: bool


def _to_response(w: dict) -> WatchResponse:
    return WatchResponse(
        id=w["id"], user_id=w["user_id"], topic=w["topic"], language=w["language"],
        cadence=w["cadence"], next_run=w.get("next_run"), last_run=w.get("last_run"),
        created_at=w["created_at"], seen_count=len(w.get("seen_ids", [])),
        has_digest=bool(w.get("latest_digest")),
    )


@router.post("/watch", response_model=WatchResponse, tags=["Watch"])
async def create_watch(body: WatchCreateRequest, user_id: str = Depends(get_current_user)):
    """Register a topic watch. Runs on POST /watch/{id}/run (or the schedule loop)."""
    _require_enabled()
    now = datetime.now(timezone.utc)
    cadence = body.cadence or config.WATCH_DEFAULT_CADENCE
    if cadence not in _CADENCES:  # guard a misconfigured default
        cadence = "weekly"
    watch = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "topic": body.topic,
        "language": body.language,
        "cadence": cadence,
        "seen_ids": [],
        "latest_digest": None,
        "next_run": _next_run_from(now, cadence),
        "last_run": None,
        "created_at": now.isoformat(),
    }
    persistence.save_watch(watch)
    logger.info(f"[Watch] registered {watch['id']} topic={body.topic!r} cadence={cadence}")
    return _to_response(watch)


@router.get("/watch", response_model=list[WatchResponse], tags=["Watch"])
async def list_watches(user_id: str = Depends(get_current_user)):
    """List the caller's own watches (owner derived from the API key)."""
    _require_enabled()
    return [_to_response(w) for w in persistence.list_watches(user_id)]


@router.get("/watch/{watch_id}", response_model=WatchResponse, tags=["Watch"])
async def get_watch(watch_id: str, user_id: str = Depends(get_current_user)):
    _require_enabled()
    return _to_response(_owned_or_404(watch_id, user_id))


@router.post("/watch/{watch_id}/run", tags=["Watch"])
async def run_watch_now(watch_id: str, user_id: str = Depends(get_current_user)):
    """Run a watch immediately: search the topic, ingest new papers, refresh the digest.

    Synchronous — ingest blocks, so it runs in a threadpool to keep the event loop free.
    """
    _require_enabled()
    _owned_or_404(watch_id, user_id)  # ownership gate before doing any work
    import watch_runner  # lazy: keeps arxiv/ingest imports off the request-path cold start
    try:
        return await run_in_threadpool(watch_runner.run_watch, watch_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watch not found")


@router.get("/watch/{watch_id}/digest", tags=["Watch"])
async def get_watch_digest(watch_id: str, user_id: str = Depends(get_current_user)):
    """Return the watch's latest stored digest (persists between runs)."""
    _require_enabled()
    w = _owned_or_404(watch_id, user_id)
    return {"watch_id": watch_id, "digest": w.get("latest_digest"), "last_run": w.get("last_run")}


@router.delete("/watch/{watch_id}", tags=["Watch"])
async def delete_watch(watch_id: str, user_id: str = Depends(get_current_user)):
    _require_enabled()
    _owned_or_404(watch_id, user_id)
    persistence.delete_watch(watch_id)
    return {"status": "deleted", "id": watch_id}
