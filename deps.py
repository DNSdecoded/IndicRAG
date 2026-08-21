"""
Shared FastAPI dependencies: auth, rate limiting, in-memory session/job state.

Session/job dicts live here (not persistence.py yet — that's a later phase task)
so routers can share them without importing api_server and creating a cycle.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio
import hashlib
import hmac
import os
import threading
import time
import uuid
import weakref

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

raw_keys = os.getenv("API_KEYS")
if raw_keys:
    VALID_API_KEYS = {k.strip() for k in raw_keys.split(",") if k.strip()}
    if not VALID_API_KEYS:
        VALID_API_KEYS = None
else:
    VALID_API_KEYS = None

_admin_key = os.getenv("ADMIN_API_KEY")


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
    """Verify admin API key for destructive operations.

    Fail-closed: this used to fall back to `verify_api_key`, which meant any
    ordinary user key authorized /purge/* in multi-user mode, and that nothing
    at all was required when API_KEYS was unset (ALLOW_UNAUTHENTICATED). A
    dedicated ADMIN_API_KEY is now mandatory for destructive routes.
    """
    if not _admin_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Destructive operations are disabled: set ADMIN_API_KEY to enable them",
                    "code": "ADMIN_KEY_NOT_CONFIGURED"}
        )
    if not _key_matches(api_key, _admin_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Admin API key required for destructive operations",
                    "code": "ADMIN_KEY_REQUIRED"}
        )
    return True


async def current_owner(api_key: str = Security(API_KEY_HEADER)) -> Optional[str]:
    """Non-reversible fingerprint of the caller's API key, used to scope job
    results to the submitter. None when auth is disabled — single-tenant, so
    there is nothing to scope against."""
    if VALID_API_KEYS is None or not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


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


_turn_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()
_turn_locks_guard = threading.Lock()


def session_turn_lock(session_id: str) -> asyncio.Lock:
    """One lock per session, serializing a whole conversational turn.

    A turn is read-history -> generate -> append, and generation takes seconds.
    Two concurrent requests on one session would otherwise both read history at
    length N, both generate without seeing the other, and both append — so each
    answer is blind to the turn running beside it (a lost update). Holding this
    for the full turn makes them queue instead, which is the correct semantics:
    turns in one conversation are not independent.

    WeakValueDictionary so a lock disappears once no turn holds or awaits it;
    the strong reference lives in the caller's `async with`.
    """
    with _turn_locks_guard:
        lock = _turn_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _turn_locks[session_id] = lock
        return lock


def owns_session(session: Optional[dict], owner: Optional[str]) -> bool:
    """True when `owner` may read, append to, or delete `session`.

    Single-tenant (auth disabled, owner None) sees everything. Otherwise the
    fingerprints must match; a session created before ownership existed carries
    no owner and stays invisible to ordinary keys rather than defaulting open.
    """
    if session is None:
        return False
    if owner is None:
        return True
    return session.get("owner") == owner


def _get_or_create_session(session_id: Optional[str], owner: Optional[str] = None) -> tuple[str, list]:
    """Return (session_id, messages_list). Creates a new session when id is None.

    Raises PermissionError when `session_id` names an existing session belonging
    to a different API key — without that check, supplying a guessed id would
    append to, and then read back, someone else's conversation.
    """
    with _sessions_lock:
        _evict_stale_sessions()
        if session_id and session_id in _sessions:
            existing = _sessions[session_id]
            if not owns_session(existing, owner):
                raise PermissionError(session_id)
            return session_id, list(existing["messages"])
        new_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        _sessions[new_id] = {
            "id": new_id,
            "messages": [],
            "owner": owner,
            "created_at": now,
            "updated_at": now,
        }
        persistence.save_session(new_id, _sessions[new_id])
        return new_id, list(_sessions[new_id]["messages"])


def _append_session_messages(session_id: str, user_text: str, assistant_text: str,
                             owner: Optional[str] = None) -> None:
    with _sessions_lock:
        # The session can be evicted (stale-age sweep) between _get_or_create_session
        # and here during a slow generation — re-materialize it rather than KeyError
        # after the answer has already been computed. It has to be re-created with the
        # same owner, or the resurrected session would be ownerless and drop out of
        # its owner's listing.
        sess = _sessions.get(session_id)
        if sess is None:
            now = datetime.now(timezone.utc).isoformat()
            sess = {"id": session_id, "messages": [], "owner": owner,
                    "created_at": now, "updated_at": now}
            _sessions[session_id] = sess
        msgs = sess["messages"]
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": assistant_text})
        max_msgs = config.CHAT_HISTORY_MAX_TURNS * 2
        if len(msgs) > max_msgs:
            del msgs[:len(msgs) - max_msgs]
        sess["updated_at"] = datetime.now(timezone.utc).isoformat()
        persistence.save_session(session_id, sess)
