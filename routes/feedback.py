"""Routes: /feedback, /prefs/{user_id}."""

from datetime import datetime, timezone
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
):
    """Record thumbs up/down feedback for a previously returned answer."""
    feedback_id = str(uuid.uuid4())
    persistence.save_feedback(
        feedback_id, body.query_id, body.rating, body.comment or "",
        datetime.now(timezone.utc).isoformat(),
    )
    return FeedbackResponse(status="recorded", feedback_id=feedback_id)


class PrefsRequest(BaseModel):
    """Opaque per-user preference blob; caller defines the shape."""
    prefs: dict


class PrefsResponse(BaseModel):
    user_id: str
    prefs: dict


def _require_enabled():
    if not config.ENABLE_USER_PREFS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User preferences are not enabled")


@router.get("/prefs/{user_id}", response_model=PrefsResponse, tags=["Preferences"])
async def get_user_prefs(user_id: str, authenticated: bool = Depends(verify_api_key)):
    _require_enabled()
    return PrefsResponse(user_id=user_id, prefs=persistence.get_prefs(user_id))


@router.put("/prefs/{user_id}", response_model=PrefsResponse, tags=["Preferences"])
async def put_user_prefs(user_id: str, body: PrefsRequest, authenticated: bool = Depends(verify_api_key)):
    _require_enabled()
    persistence.save_prefs(user_id, body.prefs, datetime.now(timezone.utc).isoformat())
    return PrefsResponse(user_id=user_id, prefs=body.prefs)
