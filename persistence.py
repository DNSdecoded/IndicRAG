"""
SQLite-backed session/job persistence so restarts don't lose in-flight state.

Stdlib sqlite3, not aiosqlite: every caller in deps.py is already a sync
function invoked via FastAPI's threadpool, so there's no event loop to keep
async for. WAL mode lets reads and the single writer coexist without
blocking each other.
"""

from datetime import datetime, timedelta, timezone
import json
import sqlite3
import threading

import config

_conn = sqlite3.connect(str(config.SESSIONS_DB_PATH), check_same_thread=False)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute(
    "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT, "
    "created_at TEXT, updated_at TEXT)"
)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT, status TEXT, "
    "submitted_at TEXT, completed_at TEXT)"
)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS feedback (id TEXT PRIMARY KEY, query_id TEXT, rating TEXT, "
    "comment TEXT, created_at TEXT)"
)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS user_prefs (user_id TEXT PRIMARY KEY, prefs TEXT, updated_at TEXT)"
)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS watches (id TEXT PRIMARY KEY, user_id TEXT, data TEXT, "
    "next_run TEXT, last_run TEXT, created_at TEXT)"
)
_conn.commit()
_db_lock = threading.Lock()


def load_sessions(max_age_hours: int = None) -> dict:
    """Load sessions, pruning ones older than max_age_hours from disk first."""
    if max_age_hours is None:
        max_age_hours = config.SESSION_MAX_AGE_HOURS
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
    with _db_lock:
        _conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
        _conn.commit()
        rows = _conn.execute("SELECT id, data FROM sessions").fetchall()
    return {sid: json.loads(data) for sid, data in rows}


def save_session(session_id: str, session: dict) -> None:
    with _db_lock:
        _conn.execute(
            "INSERT INTO sessions (id, data, created_at, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
            (session_id, json.dumps(session), session["created_at"], session["updated_at"]),
        )
        _conn.commit()


def delete_session(session_id: str) -> None:
    with _db_lock:
        _conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        _conn.commit()


def load_jobs(max_age_hours: int = 24) -> dict:
    """Load jobs, pruning completed ones older than max_age_hours from disk first.

    Without this, completed jobs accumulate in sessions.db forever and get
    re-hydrated into memory on every restart — the in-memory-only eviction in
    deps.py._update_job only ever cleared the dict, never the SQLite rows.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
    with _db_lock:
        _conn.execute("DELETE FROM jobs WHERE completed_at IS NOT NULL AND completed_at < ?", (cutoff,))
        _conn.commit()
        rows = _conn.execute("SELECT id, data FROM jobs").fetchall()
    return {jid: json.loads(data) for jid, data in rows}


def delete_job(job_id: str) -> None:
    with _db_lock:
        _conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        _conn.commit()


def save_job(job_id: str, job: dict) -> None:
    with _db_lock:
        _conn.execute(
            "INSERT INTO jobs (id, data, status, submitted_at, completed_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data=excluded.data, status=excluded.status, "
            "completed_at=excluded.completed_at",
            (job_id, json.dumps(job), job.get("status"), job.get("submitted_at"), job.get("completed_at")),
        )
        _conn.commit()


def save_feedback(feedback_id: str, query_id: str, rating: str, comment: str, created_at: str) -> None:
    with _db_lock:
        _conn.execute(
            "INSERT INTO feedback (id, query_id, rating, comment, created_at) VALUES (?, ?, ?, ?, ?)",
            (feedback_id, query_id, rating, comment, created_at),
        )
        _conn.commit()


def get_prefs(user_id: str) -> dict:
    with _db_lock:
        row = _conn.execute("SELECT prefs FROM user_prefs WHERE user_id = ?", (user_id,)).fetchone()
    return json.loads(row[0]) if row else {}


def save_prefs(user_id: str, prefs: dict, updated_at: str) -> None:
    with _db_lock:
        _conn.execute(
            "INSERT INTO user_prefs (user_id, prefs, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET prefs=excluded.prefs, updated_at=excluded.updated_at",
            (user_id, json.dumps(prefs), updated_at),
        )
        _conn.commit()


# ---------------------------------------------------------------------------
# Phase 6 — "watch a topic" registrations.
# The full watch dict lives in `data` (json); user_id/next_run/last_run are
# denormalized columns so the scheduler can select due watches without parsing
# every row. next_run is ISO-8601 UTC; a NULL next_run means "never auto-runs".
# ---------------------------------------------------------------------------
def save_watch(watch: dict) -> None:
    """Insert or update a watch. `watch` must carry at least `id`."""
    with _db_lock:
        _conn.execute(
            "INSERT INTO watches (id, user_id, data, next_run, last_run, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET user_id=excluded.user_id, data=excluded.data, "
            "next_run=excluded.next_run, last_run=excluded.last_run",
            (
                watch["id"], watch.get("user_id"), json.dumps(watch),
                watch.get("next_run"), watch.get("last_run"), watch.get("created_at"),
            ),
        )
        _conn.commit()


def get_watch(watch_id: str) -> dict | None:
    with _db_lock:
        row = _conn.execute("SELECT data FROM watches WHERE id = ?", (watch_id,)).fetchone()
    return json.loads(row[0]) if row else None


def list_watches(user_id: str | None = None) -> list[dict]:
    """All watches, or just one user's, newest first."""
    with _db_lock:
        if user_id is None:
            rows = _conn.execute(
                "SELECT data FROM watches ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = _conn.execute(
                "SELECT data FROM watches WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
    return [json.loads(r[0]) for r in rows]


def due_watches(now_iso: str) -> list[dict]:
    """Watches whose next_run has arrived (next_run non-NULL and <= now)."""
    with _db_lock:
        rows = _conn.execute(
            "SELECT data FROM watches WHERE next_run IS NOT NULL AND next_run <= ? "
            "ORDER BY next_run ASC",
            (now_iso,),
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def delete_watch(watch_id: str) -> None:
    with _db_lock:
        _conn.execute("DELETE FROM watches WHERE id = ?", (watch_id,))
        _conn.commit()
