"""
Shared FastAPI dependencies: auth, rate limiting, in-memory session/job state.

Session/job dicts live here (not persistence.py yet — that's a later phase task)
so routers can share them without importing api_server and creating a cycle.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import hmac
import os
import threading
import time
import uuid

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

import config
import persistence

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def _rate_limit_key(request: Request) -> str:
    """Rate-limit per API key when one is presented, else per client IP.

    IP-only limiting lets many callers behind one NAT/proxy exhaust each other's
    quota; keying on the API key gives each key its own bucket. Falls back to IP
    for unauthenticated traffic.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)

# ---------------------------------------------------------------------------
# API key authentication (optional)
# ---------------------------------------------------------------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def _build_key_to_user():
    """Map every valid X-API-Key -> user_id, from seeded users + env API_KEYS.

    Users (manage_users.py) supply {api_key: username}. Env API_KEYS entries are
    bare keys that map to themselves (back-compat with the pre-multi-user setup).
    Returns None when nothing is configured -> auth disabled (single-user mode).
    """
    mapping = {}
    try:
        import persistence
        for api_key, username in persistence.all_key_user_pairs():
            if api_key:
                mapping[api_key] = username
    except Exception:  # persistence not ready (e.g. during early import) — env only
        pass
    raw = os.getenv("API_KEYS")
    if raw:
        for k in (s.strip() for s in raw.split(",")):
            if k:
                mapping.setdefault(k, k)  # bare key: user_id == key
    return mapping or None


KEY_TO_USER = _build_key_to_user()
# Keys-only set kept for the unchanged bool gate verify_api_key / _key_matches.
VALID_API_KEYS = set(KEY_TO_USER) if KEY_TO_USER else None

_admin_key = os.getenv("ADMIN_API_KEY")


def _refresh_key_map():
    """Rebuild the key map so a freshly-seeded user works without a restart."""
    global KEY_TO_USER, VALID_API_KEYS
    KEY_TO_USER = _build_key_to_user()
    VALID_API_KEYS = set(KEY_TO_USER) if KEY_TO_USER else None


def _key_matches(candidate: Optional[str], valid) -> bool:
    """Constant-time membership check to avoid leaking key length/prefix via timing.

    `valid` is a single key (str) or an iterable of keys. Plain `==`/`in` short-circuit
    on the first differing byte, giving a timing side-channel; hmac.compare_digest does not.
    """
    if not candidate:
        return False
    # Compare as bytes: hmac.compare_digest raises TypeError on non-ASCII str, so a
    # header with bytes >127 would 500 instead of cleanly failing auth. UTF-8 encode
    # both sides — a non-ASCII candidate simply won't match an ASCII key.
    cand = candidate.encode("utf-8")
    keys = [valid] if isinstance(valid, str) else valid
    # Compare against every key (no early exit) so total time doesn't depend on which matched.
    return any(hmac.compare_digest(cand, k.encode("utf-8")) for k in keys)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key if authentication is enabled."""
    if VALID_API_KEYS is None:
        return True  # No authentication required

    if not _key_matches(api_key, VALID_API_KEYS):
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
        if not _key_matches(api_key, _admin_key):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Admin API key required for destructive operations",
                        "code": "ADMIN_KEY_REQUIRED"}
            )
        return True
    return await verify_api_key(api_key)


async def get_current_user(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Resolve the caller to a user_id for per-user data ownership.

    Auth disabled (no users seeded and no env API_KEYS) -> everyone is
    config.DEFAULT_USER_ID (single-user back-compat). Otherwise the presented
    key must match a known key (401 on miss); returns its owner username.
    """
    if KEY_TO_USER is None:
        return config.DEFAULT_USER_ID
    user = _lookup_user(api_key)
    if user is None:  # a user seeded after startup won't be in the cached map yet
        _refresh_key_map()
        user = _lookup_user(api_key)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid or missing API key", "code": "INVALID_API_KEY"},
        )
    return user


def _lookup_user(api_key: Optional[str]) -> Optional[str]:
    """Constant-time match of the presented key against the map; owner or None."""
    if not api_key or not KEY_TO_USER:
        return None
    for k, user in KEY_TO_USER.items():
        if hmac.compare_digest(api_key.encode("utf-8"), k.encode("utf-8")):
            return user
    return None


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


def _get_or_create_session(session_id: Optional[str], user_id: str = None) -> tuple[str, list]:
    """Return (session_id, messages_list). Creates a new session when id is None.

    New sessions are stamped with `user_id` (the owner) so chat-history listing
    and reads can be scoped per user. Reopening an existing session does not
    re-stamp — ownership is set once, at creation.
    """
    if user_id is None:
        user_id = config.DEFAULT_USER_ID
    with _sessions_lock:
        _evict_stale_sessions()
        if session_id and session_id in _sessions:
            return session_id, list(_sessions[session_id]["messages"])
        new_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        _sessions[new_id] = {
            "id": new_id,
            "user_id": user_id,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        persistence.save_session(new_id, _sessions[new_id])
        return new_id, list(_sessions[new_id]["messages"])


def _session_owner(session: dict) -> str:
    """Owner of a session; legacy sessions with no owner belong to DEFAULT_USER_ID."""
    return session.get("user_id") or config.DEFAULT_USER_ID


def _append_session_messages(session_id: str, user_text: str, assistant_text: str) -> None:
    with _sessions_lock:
        # The session can be evicted (stale-age sweep) between _get_or_create_session
        # and here during a slow generation — re-materialize it rather than KeyError
        # after the answer has already been computed.
        sess = _sessions.get(session_id)
        if sess is None:
            now = datetime.now(timezone.utc).isoformat()
            sess = {"id": session_id, "messages": [], "created_at": now, "updated_at": now}
            _sessions[session_id] = sess
        msgs = sess["messages"]
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": assistant_text})
        max_msgs = config.CHAT_HISTORY_MAX_TURNS * 2
        if len(msgs) > max_msgs:
            del msgs[:len(msgs) - max_msgs]
        sess["updated_at"] = datetime.now(timezone.utc).isoformat()
        persistence.save_session(session_id, sess)
