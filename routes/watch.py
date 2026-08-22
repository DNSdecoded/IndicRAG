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

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

import config
import persistence
from deps import limiter, verify_api_key, current_owner

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
async def create_watch(body: WatchCreateRequest, authenticated: bool = Depends(verify_api_key),
                       owner: Optional[str] = Depends(current_owner)):
    """Register a topic watch. Runs on POST /watch/{id}/run (or the schedule loop).

    `user_id` is a caller-chosen display label only. Authorization is the API-key
    fingerprint stored in `owner`; it used to be `user_id`, which the caller
    supplies and can therefore set to anyone's.
    """
    _require_enabled()
    now = datetime.now(timezone.utc)
    cadence = body.cadence or config.WATCH_DEFAULT_CADENCE
    if cadence not in _CADENCES:  # guard a misconfigured default
        cadence = "weekly"
    watch = {
        "id": str(uuid.uuid4()),
        "user_id": body.user_id,
        "owner": owner,
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


def _owned_watch_or_404(watch_id: str, owner: Optional[str]) -> dict:
    """Fetch a watch the caller owns, or raise 404.

    404 rather than 403 on a mismatch: a 403 would confirm the id exists.
    """
    w = persistence.get_watch(watch_id)
    if not persistence.owns_watch(w, owner):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watch not found")
    return w


@router.get("/watch", response_model=list[WatchResponse], tags=["Watch"])
async def list_watches(user_id: Optional[str] = None, authenticated: bool = Depends(verify_api_key),
                       owner: Optional[str] = Depends(current_owner)):
    """List the caller's watches, optionally narrowed to one of their user_id labels.

    `user_id` filters within what the caller already owns — it is not a way to
    read another key's watches, which is what it used to be.
    """
    _require_enabled()
    return [_to_response(w) for w in persistence.list_watches(user_id, owner=owner)]


@router.get("/watch/{watch_id}", response_model=WatchResponse, tags=["Watch"])
async def get_watch(watch_id: str, authenticated: bool = Depends(verify_api_key),
                    owner: Optional[str] = Depends(current_owner)):
    _require_enabled()
    return _to_response(_owned_watch_or_404(watch_id, owner))


@router.post("/watch/{watch_id}/run", tags=["Watch"])
@limiter.limit("5/minute")
async def run_watch_now(request: Request, watch_id: str,
                        authenticated: bool = Depends(verify_api_key),
                        owner: Optional[str] = Depends(current_owner)):
    """Run a watch immediately: search the topic, ingest new papers, refresh the digest.

    Synchronous — ingest blocks, so it runs in a threadpool to keep the event loop free.
    Rate-limited because each call spends LLM budget (digest generation) and can
    trigger ingestion; unlimited, it drains the shared quota for every other user.
    """
    _require_enabled()
    _owned_watch_or_404(watch_id, owner)
    import watch_runner  # lazy: keeps arxiv/ingest imports off the request-path cold start
    try:
        return await run_in_threadpool(watch_runner.run_watch, watch_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Watch not found")


@router.get("/watch/{watch_id}/digest", tags=["Watch"])
async def get_watch_digest(watch_id: str, authenticated: bool = Depends(verify_api_key),
                           owner: Optional[str] = Depends(current_owner)):
    """Return the watch's latest stored digest (persists between runs)."""
    _require_enabled()
    w = _owned_watch_or_404(watch_id, owner)
    return {"watch_id": watch_id, "digest": w.get("latest_digest"), "last_run": w.get("last_run")}


@router.delete("/watch/{watch_id}", tags=["Watch"])
async def delete_watch(watch_id: str, authenticated: bool = Depends(verify_api_key),
                       owner: Optional[str] = Depends(current_owner)):
    _require_enabled()
    _owned_watch_or_404(watch_id, owner)
    persistence.delete_watch(watch_id)
    return {"status": "deleted", "id": watch_id}
