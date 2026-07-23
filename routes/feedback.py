"""Routes: /feedback, /prefs/{user_id}."""

from datetime import datetime, timezone
from typing import Optional
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

import config
import persistence
from deps import verify_api_key, get_current_user

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
    user_id: str = Depends(get_current_user),
):
    """Record thumbs up/down feedback for a previously returned answer."""
    feedback_id = str(uuid.uuid4())
    persistence.save_feedback(
        feedback_id, body.query_id, body.rating, body.comment or "",
        datetime.now(timezone.utc).isoformat(), user_id=user_id,
    )
    return FeedbackResponse(status="recorded", feedback_id=feedback_id)


@router.get("/feedback", tags=["Feedback"])
async def list_feedback(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(get_current_user),
):
    """Return the caller's own feedback entries joined with query context, newest first."""
    return persistence.get_feedback_with_context(limit, offset, user_id=user_id)


@router.get("/feedback/stats", tags=["Feedback"])
async def get_feedback_stats(user_id: str = Depends(get_current_user)):
    """Aggregate the caller's feedback totals and per-language approval rates."""
    return persistence.feedback_stats(user_id=user_id)


class PrefsRequest(BaseModel):
    """Opaque per-user preference blob; caller defines the shape."""
    prefs: dict


class PrefsResponse(BaseModel):
    user_id: str
    prefs: dict


def _require_enabled():
    if not config.ENABLE_USER_PREFS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User preferences are not enabled")


@router.get("/prefs", response_model=PrefsResponse, tags=["Preferences"])
async def get_user_prefs(user_id: str = Depends(get_current_user)):
    """The caller's own preferences (owner derived from the API key)."""
    _require_enabled()
    return PrefsResponse(user_id=user_id, prefs=persistence.get_prefs(user_id))


@router.put("/prefs", response_model=PrefsResponse, tags=["Preferences"])
async def put_user_prefs(body: PrefsRequest, user_id: str = Depends(get_current_user)):
    _require_enabled()
    persistence.save_prefs(user_id, body.prefs, datetime.now(timezone.utc).isoformat())
    return PrefsResponse(user_id=user_id, prefs=body.prefs)
