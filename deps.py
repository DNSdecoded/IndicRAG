"""
Shared FastAPI dependencies: auth, rate limiting, in-memory session/job state.

Session/job dicts live here (not persistence.py yet — that's a later phase task)
so routers can share them without importing api_server and creating a cycle.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import os
import threading
import time
import uuid

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

import config
import persistence

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# API key authentication (optional)
# ---------------------------------------------------------------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

raw_keys = os.getenv("API_KEYS")
if raw_keys:
    VALID_API_KEYS = {k.strip() for k in raw_keys.split(",") if k.strip()}
    if not VALID_API_KEYS:
        VALID_API_KEYS = None
else:
    VALID_API_KEYS = None

_admin_key = os.getenv("ADMIN_API_KEY")


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key if authentication is enabled."""
    if VALID_API_KEYS is None:
        return True  # No authentication required

    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Invalid or missing API key",
                "code": "INVALID_API_KEY"
            }
        )
    return True


async def verify_admin_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify admin API key for destructive operations."""
    if _admin_key:
        if not api_key or api_key != _admin_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Admin API key required for destructive operations",
                        "code": "ADMIN_KEY_REQUIRED"}
            )
        return True
    return await verify_api_key(api_key)


# ---------------------------------------------------------------------------
# In-memory job store for background ingestion tasks, write-through to SQLite
# so a restart doesn't lose in-flight/completed job status.
# ---------------------------------------------------------------------------
_jobs: Dict[str, Dict[str, Any]] = persistence.load_jobs()
_jobs_lock = threading.Lock()
_last_job_eviction = 0.0


def _update_job(job_id: str, **kwargs):
    """Thread-safe update of a job's fields; evicts completed jobs once per hour."""
    global _last_job_eviction
    with _jobs_lock:
        _jobs[job_id].update(kwargs)
        persistence.save_job(job_id, _jobs[job_id])
        now = time.monotonic()
        if now - _last_job_eviction >= 3600:
            _last_job_eviction = now
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            for jid in [j for j, v in _jobs.items()
                        if v.get("completed_at") and
                        datetime.fromisoformat(v["completed_at"]) < cutoff]:
                del _jobs[jid]
                persistence.delete_job(jid)


# ---------------------------------------------------------------------------
# In-memory chat session store, write-through to SQLite so sessions survive
# a server restart.
# ---------------------------------------------------------------------------
_sessions: Dict[str, Dict[str, Any]] = persistence.load_sessions()
_sessions_lock = threading.Lock()
_last_session_eviction = 0.0


def _evict_stale_sessions():
    """Remove sessions older than SESSION_MAX_AGE_HOURS. Must be called under _sessions_lock."""
    global _last_session_eviction
    now = time.monotonic()
    if now - _last_session_eviction < 60:
        return
    _last_session_eviction = now
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.SESSION_MAX_AGE_HOURS)
    for sid in [s for s, v in _sessions.items()
                if datetime.fromisoformat(v["updated_at"]) < cutoff]:
        del _sessions[sid]
        persistence.delete_session(sid)


def _get_or_create_session(session_id: Optional[str]) -> tuple[str, list]:
    """Return (session_id, messages_list). Creates a new session when id is None."""
    with _sessions_lock:
        _evict_stale_sessions()
        if session_id and session_id in _sessions:
            return session_id, list(_sessions[session_id]["messages"])
        new_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        _sessions[new_id] = {
            "id": new_id,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        persistence.save_session(new_id, _sessions[new_id])
        return new_id, list(_sessions[new_id]["messages"])


def _append_session_messages(session_id: str, user_text: str, assistant_text: str) -> None:
    with _sessions_lock:
        sess = _sessions[session_id]
        msgs = sess["messages"]
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": assistant_text})
        max_msgs = config.CHAT_HISTORY_MAX_TURNS * 2
        if len(msgs) > max_msgs:
            del msgs[:len(msgs) - max_msgs]
        sess["updated_at"] = datetime.now(timezone.utc).isoformat()
        persistence.save_session(session_id, sess)
