"""Routes: /login — exchange name+password for the user's API key.

Users are pre-provisioned (manage_users.py); there is no signup endpoint. The
returned api_key is then sent as X-API-Key on every later request, which the
server resolves back to this user (deps.get_current_user).
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

import auth_utils
import persistence
from deps import limiter, _refresh_key_map

logger = logging.getLogger(__name__)
router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=1024)


class LoginResponse(BaseModel):
    username: str
    api_key: str


# Generic message for every failure mode — never reveal whether the username
# exists or the password was the wrong part (no user-enumeration oracle).
_BAD_LOGIN = {"error": "Invalid username or password", "code": "INVALID_CREDENTIALS"}


@router.post("/login", response_model=LoginResponse, tags=["Auth"])
@limiter.limit("10/minute")  # blunt password guessing
async def login(request: Request, body: LoginRequest):
    user = persistence.get_user(body.username)
    if user is None or not auth_utils.verify_password(
        body.password, user["pw_salt"], user["pw_hash"]
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_BAD_LOGIN)
    _refresh_key_map()  # ensure this user's key resolves even if seeded post-startup
    logger.info(f"[Auth] login ok user={body.username!r}")
    return LoginResponse(username=user["username"], api_key=user["api_key"])
