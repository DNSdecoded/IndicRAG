"""Routes: /watch — Phase 6 "watch a topic" registration.

CRUD only. The run-one-watch endpoint (POST /watch/{id}/run), the digest
endpoint (GET /watch/{id}/digest), and the background schedule loop are added in
later increments; they all reuse persistence.save_watch / get_watch here.

Gated by config.WATCH_ENABLE (404 when off), mirroring the /prefs pattern.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

import config
import persistence
from deps import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

_CADENCES = {"daily": timedelta(days=1), "weekly": timedelta(days=7), "monthly": timedelta(days=30)}


def _require_enabled():
    if not config.WATCH_ENABLE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topic watches are not enabled")


def _next_run_from(now: datetime, cadence: str) -> str:
    """First auto-run one cadence interval out (nothing runs it until the loop lands)."""
    return (now + _CADENCES[cadence]).isoformat()


class WatchCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
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
async def create_watch(body: WatchCreateRequest, authenticated: bool = Depends(verify_api_key)):
    """Register a topic watch. Runs on POST /watch/{id}/run (or the schedule loop)."""
    _require_enabled()
    now = datetime.now(timezone.utc)
    cadence = body.cadence or config.WATCH_DEFAULT_CADENCE
    if cadence not in _CADENCES:  # guard a misconfigured default
        cadence = "weekly"
    watch = {
        "id": str(uuid.uuid4()),
        "user_id": body.user_id,
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
async def list_watches(user_id: Optional[str] = None, authenticated: bool = Depends(verify_api_key)):
    """List all watches, or just one user's when user_id is supplied."""
    _require_enabled()
    return [_to_response(w) for w in persistence.list_watches(user_id)]


@router.get("/watch/{watch_id}", response_model=WatchResponse, tags=["Watch"])
async def get_watch(watch_id: str, authenticated: bool = Depends(verify_api_key)):
    _require_enabled()
    w = persistence.get_watch(watch_id)
    if w is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watch not found")
    return _to_response(w)


@router.delete("/watch/{watch_id}", tags=["Watch"])
async def delete_watch(watch_id: str, authenticated: bool = Depends(verify_api_key)):
    _require_enabled()
    if persistence.get_watch(watch_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watch not found")
    persistence.delete_watch(watch_id)
    return {"status": "deleted", "id": watch_id}
