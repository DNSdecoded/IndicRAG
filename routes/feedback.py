"""Routes: /feedback, /prefs/{user_id}."""

from datetime import datetime, timezone
from typing import Optional
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

import config
import persistence
from deps import verify_api_key, current_owner

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    rating: str
    comment: Optional[str] = Field(None, max_length=2000)

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v):
        if v not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        return v


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(
    body: FeedbackRequest,
    authenticated: bool = Depends(verify_api_key),
    owner: Optional[str] = Depends(current_owner),
):
    """Record thumbs up/down feedback for a previously returned answer."""
    feedback_id = str(uuid.uuid4())
    persistence.save_feedback(
        feedback_id, body.query_id, body.rating, body.comment or "",
        datetime.now(timezone.utc).isoformat(), owner,
    )
    return FeedbackResponse(status="recorded", feedback_id=feedback_id)


@router.get("/feedback", tags=["Feedback"])
async def list_feedback(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    authenticated: bool = Depends(verify_api_key),
    owner: Optional[str] = Depends(current_owner),
):
    """Return the caller's feedback entries joined with their query context, newest first.

    Scoped by API-key fingerprint: the joined `query_log` rows carry the original
    question and answer text, so an unscoped listing handed every user's queries
    to anyone holding any valid key.
    """
    return persistence.get_feedback_with_context(limit, offset, owner=owner)


@router.get("/feedback/stats", tags=["Feedback"])
async def get_feedback_stats(authenticated: bool = Depends(verify_api_key),
                             owner: Optional[str] = Depends(current_owner)):
    """Aggregate feedback totals and per-language approval rates for the caller."""
    return persistence.feedback_stats(owner=owner)


class PrefsRequest(BaseModel):
    """Opaque per-user preference blob; caller defines the shape."""
    prefs: dict


class PrefsResponse(BaseModel):
    user_id: str
    prefs: dict


def _require_enabled():
    if not config.ENABLE_USER_PREFS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User preferences are not enabled")


def _prefs_key(user_id: str, owner: Optional[str]) -> str:
    """Storage key for a user's preferences.

    When auth is on, prefs are stored against the API-key fingerprint, not the
    `{user_id}` in the path — otherwise anyone could read or overwrite anyone
    else's preferences just by typing their id into the URL. The path segment
    stays in the response so existing clients keep working.
    """
    return owner if owner is not None else user_id


@router.get("/prefs/{user_id}", response_model=PrefsResponse, tags=["Preferences"])
async def get_user_prefs(user_id: str, authenticated: bool = Depends(verify_api_key),
                         owner: Optional[str] = Depends(current_owner)):
    _require_enabled()
    return PrefsResponse(user_id=user_id, prefs=persistence.get_prefs(_prefs_key(user_id, owner)))


@router.put("/prefs/{user_id}", response_model=PrefsResponse, tags=["Preferences"])
async def put_user_prefs(user_id: str, body: PrefsRequest, authenticated: bool = Depends(verify_api_key),
                         owner: Optional[str] = Depends(current_owner)):
    _require_enabled()
    persistence.save_prefs(_prefs_key(user_id, owner), body.prefs,
                           datetime.now(timezone.utc).isoformat())
    return PrefsResponse(user_id=user_id, prefs=body.prefs)
