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
_conn.execute(
    "CREATE TABLE IF NOT EXISTS query_log ("
    "query_id TEXT PRIMARY KEY, question TEXT, answer TEXT, mode TEXT, "
    "model TEXT, language TEXT, confidence REAL, coverage REAL, "
    "created_at TEXT)"
)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, watch_id TEXT, topic TEXT, "
    "language TEXT, markdown TEXT, citation_count INTEGER, created_at TEXT)"
)


def _ensure_column(table: str, column: str, decl: str) -> None:
    """Add a column to an existing table if it isn't there yet.

    `CREATE TABLE IF NOT EXISTS` silently keeps an old database's old shape, so a
    column added in a later release never appears on an already-deployed instance
    and the first query referencing it fails with `no such column`. This is the
    whole migration story for now: idempotent, runs at import, no framework.
    Replace it with a versioned runner once a change needs to backfill or
    transform data rather than just append a nullable column.
    """
    cols = {row[1] for row in _conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        _conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# `owner` is the SHA-256 fingerprint of the caller's API key (deps.current_owner).
# NULL means "written before ownership existed, or written while auth was disabled".
_ensure_column("feedback", "owner", "TEXT")
_ensure_column("reports", "owner", "TEXT")
_ensure_column("query_log", "owner", "TEXT")

# Secondary indexes. Every table above declared a PRIMARY KEY and nothing else, so
# each of these access paths was a full scan: due_watches runs on every scheduler
# tick, the feedback join runs on every read, and the startup prunes delete by
# timestamp.
_conn.execute("CREATE INDEX IF NOT EXISTS idx_watches_next_run ON watches(next_run)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_watches_user ON watches(user_id)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_owner ON feedback(owner)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_completed ON jobs(completed_at)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_watch ON reports(watch_id)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_owner ON reports(owner)")

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


def save_feedback(feedback_id: str, query_id: str, rating: str, comment: str, created_at: str,
                  owner: str | None = None) -> None:
    with _db_lock:
        _conn.execute(
            "INSERT INTO feedback (id, query_id, rating, comment, created_at, owner) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (feedback_id, query_id, rating, comment, created_at, owner),
        )
        _conn.commit()


def log_query(query_id: str, question: str, answer: str, mode: str,
              model: str, language: str, confidence: float, coverage: float,
              created_at: str, owner: str | None = None) -> None:
    """Persist a query/answer record so feedback can be correlated with it."""
    with _db_lock:
        _conn.execute(
            "INSERT INTO query_log "
            "(query_id, question, answer, mode, model, language, confidence, coverage, created_at, owner) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(query_id) DO UPDATE SET "
            "answer=excluded.answer, confidence=excluded.confidence, coverage=excluded.coverage",
            (query_id, question, answer, mode, model, language, confidence, coverage, created_at, owner),
        )
        _conn.commit()


_FEEDBACK_CONTEXT_COLUMNS = [
    "id", "query_id", "rating", "comment", "created_at",
    "question", "answer", "mode", "model", "language", "confidence", "coverage",
]


def get_feedback_with_context(limit: int = 50, offset: int = 0,
                              owner: str | None = None) -> list[dict]:
    """Return feedback joined with its query context, newest first.

    `owner` scopes the result to one caller's submissions. None means unscoped —
    only pass None when auth is disabled (single-tenant) or for an admin read;
    an authenticated multi-user route must always pass the caller's fingerprint,
    or every user reads every other user's feedback and query text.
    """
    where = "" if owner is None else "WHERE f.owner = ? "
    params = (limit, offset) if owner is None else (owner, limit, offset)
    with _db_lock:
        rows = _conn.execute(
            "SELECT f.id, f.query_id, f.rating, f.comment, f.created_at, "
            "q.question, q.answer, q.mode, q.model, q.language, q.confidence, q.coverage "
            "FROM feedback f LEFT JOIN query_log q ON f.query_id = q.query_id "
            f"{where}"
            "ORDER BY f.created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [
        dict(zip(_FEEDBACK_CONTEXT_COLUMNS, r))
        for r in rows
    ]


def feedback_stats(owner: str | None = None) -> dict:
    """Aggregate feedback totals and per-language approval rate, optionally per owner."""
    where = "" if owner is None else " WHERE owner = ?"
    join_where = "" if owner is None else " WHERE f.owner = ?"
    args: tuple = () if owner is None else (owner,)
    with _db_lock:
        total = _conn.execute(f"SELECT COUNT(*) FROM feedback{where}", args).fetchone()[0]
        up = _conn.execute(
            f"SELECT COUNT(*) FROM feedback WHERE rating='up'{'' if owner is None else ' AND owner = ?'}",
            args,
        ).fetchone()[0]
        down = _conn.execute(
            f"SELECT COUNT(*) FROM feedback WHERE rating='down'{'' if owner is None else ' AND owner = ?'}",
            args,
        ).fetchone()[0]
        by_lang = _conn.execute(
            "SELECT COALESCE(q.language, 'unknown'), COUNT(*), AVG(CASE WHEN f.rating='up' THEN 1.0 ELSE 0.0 END) "
            "FROM feedback f LEFT JOIN query_log q ON f.query_id = q.query_id "
            f"{join_where} "
            "GROUP BY COALESCE(q.language, 'unknown')",
            args,
        ).fetchall()
    return {
        "total": total, "up": up, "down": down,
        "by_language": {r[0]: {"count": r[1], "approval_rate": round(r[2], 3)} for r in by_lang},
    }


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


def list_watches(user_id: str | None = None, owner: str | None = None) -> list[dict]:
    """All watches, or just one user's, newest first.

    `user_id` is a caller-supplied label and is NOT an authorization boundary —
    anyone can pass anyone's. `owner` is the API-key fingerprint and is the real
    scope; when it is not None, only that key's watches are returned regardless
    of what user_id says.
    """
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
    watches = [json.loads(r[0]) for r in rows]
    if owner is not None:
        watches = [w for w in watches if w.get("owner") == owner]
    return watches


def owns_watch(watch: dict | None, owner: str | None) -> bool:
    """True when `owner` may act on `watch`.

    Single-tenant (auth disabled, owner None) sees everything. Otherwise the
    fingerprints must match; a watch created before ownership existed has no
    owner and is treated as admin-only, i.e. not visible to an ordinary key.
    """
    if watch is None:
        return False
    if owner is None:
        return True
    return watch.get("owner") == owner


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


# ---------------------------------------------------------------------------
# Literature-review reports — a durable artifact, unlike the generic job store
# (deps._jobs) which prunes completed entries after 24h. A watch-owned "living
# review" needs to survive indefinitely and be regenerated in place.
# ---------------------------------------------------------------------------
def save_report(report_id: str, watch_id: str, topic: str, language: str,
                markdown: str, citation_count: int, created_at: str,
                owner: str | None = None) -> None:
    """Insert or update a report. Re-saving the same report_id overwrites in place
    (that's how a watch-owned living review gets regenerated)."""
    with _db_lock:
        _conn.execute(
            "INSERT INTO reports (id, watch_id, topic, language, markdown, citation_count, created_at, owner) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET topic=excluded.topic, language=excluded.language, "
            "markdown=excluded.markdown, citation_count=excluded.citation_count, "
            "created_at=excluded.created_at",
            (report_id, watch_id, topic, language, markdown, citation_count, created_at, owner),
        )
        _conn.commit()


def get_report(report_id: str, owner: str | None = None) -> dict | None:
    """One report, or None. When `owner` is set, a report belonging to another key
    reads as missing — a 404 rather than a 403, so the response can't be used to
    probe which report ids exist."""
    with _db_lock:
        row = _conn.execute(
            "SELECT id, watch_id, topic, language, markdown, citation_count, created_at, owner "
            "FROM reports WHERE id = ?", (report_id,),
        ).fetchone()
    if not row:
        return None
    if owner is not None and row[7] != owner:
        return None
    return {"id": row[0], "watch_id": row[1], "topic": row[2], "language": row[3],
            "markdown": row[4], "citation_count": row[5], "created_at": row[6]}


def list_reports(watch_id: str | None = None, owner: str | None = None) -> list[dict]:
    """Summary rows (no markdown body) — newest first, optionally scoped to one
    watch and/or one owner."""
    clauses, args = [], []
    if watch_id:
        clauses.append("watch_id = ?")
        args.append(watch_id)
    if owner is not None:
        clauses.append("owner = ?")
        args.append(owner)
    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    with _db_lock:
        rows = _conn.execute(
            "SELECT id, watch_id, topic, language, citation_count, created_at "
            f"FROM reports {where}ORDER BY created_at DESC",
            tuple(args),
        ).fetchall()
    return [{"id": r[0], "watch_id": r[1], "topic": r[2], "language": r[3],
             "citation_count": r[4], "created_at": r[5]} for r in rows]
