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
_conn.execute(
    "CREATE TABLE IF NOT EXISTS graph_edges ("
    "source_paper TEXT, target_paper TEXT, edge_type TEXT, "
    "score REAL, metadata TEXT, created_at TEXT, "
    "PRIMARY KEY (source_paper, target_paper, edge_type))"
)
_conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_paper)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_paper)")
_conn.execute(
    "CREATE TABLE IF NOT EXISTS users ("
    "username TEXT PRIMARY KEY, pw_salt TEXT, pw_hash TEXT, "
    "api_key TEXT UNIQUE, created_at TEXT)"
)
_conn.commit()


def _migrate_add_column(table: str, column: str, decl: str) -> None:
    """Idempotently add a column to an existing table (older DBs predate it).

    Existing rows get NULL for the new column; callers treat NULL owner as the
    config.DEFAULT_USER_ID so legacy single-user data stays reachable.
    """
    cols = [r[1] for r in _conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        _conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# Owner columns for per-user isolation (multi-user support). watches already has
# user_id; user_prefs is keyed by it; sessions carry it inside the JSON blob.
for _t in ("reports", "feedback", "query_log"):
    _migrate_add_column(_t, "user_id", "TEXT")
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


def save_feedback(feedback_id: str, query_id: str, rating: str, comment: str,
                  created_at: str, user_id: str = None) -> None:
    with _db_lock:
        _conn.execute(
            "INSERT INTO feedback (id, query_id, rating, comment, created_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (feedback_id, query_id, rating, comment, created_at, user_id),
        )
        _conn.commit()


def log_query(query_id: str, question: str, answer: str, mode: str,
              model: str, language: str, confidence: float, coverage: float,
              created_at: str, user_id: str = None) -> None:
    """Persist a query/answer record so feedback can be correlated with it."""
    with _db_lock:
        _conn.execute(
            "INSERT INTO query_log "
            "(query_id, question, answer, mode, model, language, confidence, coverage, created_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(query_id) DO UPDATE SET "
            "answer=excluded.answer, confidence=excluded.confidence, coverage=excluded.coverage",
            (query_id, question, answer, mode, model, language, confidence, coverage, created_at, user_id),
        )
        _conn.commit()


_FEEDBACK_CONTEXT_COLUMNS = [
    "id", "query_id", "rating", "comment", "created_at",
    "question", "answer", "mode", "model", "language", "confidence", "coverage",
]


def get_feedback_with_context(limit: int = 50, offset: int = 0, user_id: str = None) -> list[dict]:
    """Return feedback joined with its query context, newest first.

    Scoped to `user_id` when given (NULL-owner legacy rows map to DEFAULT_USER_ID).
    """
    where, params = "", []
    if user_id is not None:
        where = "WHERE COALESCE(f.user_id, ?) = ? "
        params = [config.DEFAULT_USER_ID, user_id]
    with _db_lock:
        rows = _conn.execute(
            "SELECT f.id, f.query_id, f.rating, f.comment, f.created_at, "
            "q.question, q.answer, q.mode, q.model, q.language, q.confidence, q.coverage "
            "FROM feedback f LEFT JOIN query_log q ON f.query_id = q.query_id "
            + where +
            "ORDER BY f.created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    return [
        dict(zip(_FEEDBACK_CONTEXT_COLUMNS, r))
        for r in rows
    ]


def feedback_stats(user_id: str = None) -> dict:
    """Aggregate feedback totals and per-language approval rate, optionally per user."""
    where, params = "", []
    if user_id is not None:
        where = "WHERE COALESCE(f.user_id, ?) = ? "
        params = [config.DEFAULT_USER_ID, user_id]
    with _db_lock:
        total = _conn.execute("SELECT COUNT(*) FROM feedback f " + where, params).fetchone()[0]
        up = _conn.execute("SELECT COUNT(*) FROM feedback f " + where + ("AND" if where else "WHERE") + " rating='up'", params).fetchone()[0]
        down = _conn.execute("SELECT COUNT(*) FROM feedback f " + where + ("AND" if where else "WHERE") + " rating='down'", params).fetchone()[0]
        by_lang = _conn.execute(
            "SELECT q.language, COUNT(*), AVG(CASE WHEN f.rating='up' THEN 1.0 ELSE 0.0 END) "
            "FROM feedback f JOIN query_log q ON f.query_id = q.query_id "
            + where +
            "GROUP BY q.language",
            params,
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


# ---------------------------------------------------------------------------
# Literature-review reports — a durable artifact, unlike the generic job store
# (deps._jobs) which prunes completed entries after 24h. A watch-owned "living
# review" needs to survive indefinitely and be regenerated in place.
# ---------------------------------------------------------------------------
def save_report(report_id: str, watch_id: str, topic: str, language: str,
                markdown: str, citation_count: int, created_at: str, user_id: str = None) -> None:
    """Insert or update a report. Re-saving the same report_id overwrites in place
    (that's how a watch-owned living review gets regenerated)."""
    with _db_lock:
        _conn.execute(
            "INSERT INTO reports (id, watch_id, topic, language, markdown, citation_count, created_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET topic=excluded.topic, language=excluded.language, "
            "markdown=excluded.markdown, citation_count=excluded.citation_count",
            (report_id, watch_id, topic, language, markdown, citation_count, created_at, user_id),
        )
        _conn.commit()


def get_report(report_id: str) -> dict | None:
    with _db_lock:
        row = _conn.execute(
            "SELECT id, watch_id, topic, language, markdown, citation_count, created_at, user_id "
            "FROM reports WHERE id = ?", (report_id,),
        ).fetchone()
    if not row:
        return None
    return {"id": row[0], "watch_id": row[1], "topic": row[2], "language": row[3],
            "markdown": row[4], "citation_count": row[5], "created_at": row[6],
            "user_id": row[7] if row[7] is not None else config.DEFAULT_USER_ID}


def list_reports(watch_id: str | None = None, user_id: str | None = None) -> list[dict]:
    """Summary rows (no markdown body) — newest first; scoped to a watch and/or owner.

    NULL-owner legacy rows map to DEFAULT_USER_ID when filtering by user_id.
    """
    clauses, params = [], []
    if watch_id:
        clauses.append("watch_id = ?")
        params.append(watch_id)
    if user_id is not None:
        clauses.append("COALESCE(user_id, ?) = ?")
        params.extend([config.DEFAULT_USER_ID, user_id])
    where = ("WHERE " + " AND ".join(clauses) + " ") if clauses else ""
    with _db_lock:
        rows = _conn.execute(
            "SELECT id, watch_id, topic, language, citation_count, created_at "
            "FROM reports " + where + "ORDER BY created_at DESC",
            params,
        ).fetchall()
    return [{"id": r[0], "watch_id": r[1], "topic": r[2], "language": r[3],
             "citation_count": r[4], "created_at": r[5]} for r in rows]


# ---------------------------------------------------------------------------
# Knowledge/citation graph (Task 3.3). Edges are undirected — endpoints
# normalize to source<=target so (A,B) and (B,A) dedupe to one row. Repeated
# detections of the same pair accumulate score (co-citation frequency /
# contradiction evidence) via the composite-PK upsert, so the table stays
# bounded no matter how many queries run.
# ---------------------------------------------------------------------------
def save_graph_edges(edges: list) -> None:
    """Batch upsert edges: list of (source, target, edge_type, score, metadata_dict).

    One lock + commit for the whole batch (a query co-cites up to ~C(k,2) pairs).
    """
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for source, target, edge_type, score, metadata in edges:
        s, t = sorted((source, target))  # undirected: normalize ordering
        rows.append((s, t, edge_type, score, json.dumps(metadata or {}), now))
    with _db_lock:
        _conn.executemany(
            "INSERT INTO graph_edges (source_paper, target_paper, edge_type, score, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_paper, target_paper, edge_type) DO UPDATE SET "
            "score = score + excluded.score, created_at = excluded.created_at",
            rows,
        )
        _conn.commit()


def save_graph_edge(source: str, target: str, edge_type: str,
                    score: float, metadata: dict = None) -> None:
    save_graph_edges([(source, target, edge_type, score, metadata)])


def get_all_edges() -> list[dict]:
    with _db_lock:
        rows = _conn.execute(
            "SELECT source_paper, target_paper, edge_type, score FROM graph_edges"
        ).fetchall()
    return [{"source": r[0], "target": r[1], "type": r[2], "score": r[3]} for r in rows]


def get_paper_edges(paper_id: str) -> list[dict]:
    with _db_lock:
        rows = _conn.execute(
            "SELECT source_paper, target_paper, edge_type, score, metadata "
            "FROM graph_edges WHERE source_paper = ? OR target_paper = ?",
            (paper_id, paper_id),
        ).fetchall()
    return [{"source": r[0], "target": r[1], "type": r[2], "score": r[3],
             "metadata": json.loads(r[4])} for r in rows]


# ---------------------------------------------------------------------------
# Users (multi-user support). Pre-provisioned via manage_users.py — no signup
# path here. api_key is the per-user X-API-Key; deps.py maps key -> username.
# ---------------------------------------------------------------------------
def save_user(username: str, pw_salt: str, pw_hash: str, api_key: str, created_at: str) -> None:
    with _db_lock:
        _conn.execute(
            "INSERT INTO users (username, pw_salt, pw_hash, api_key, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(username) DO UPDATE SET pw_salt=excluded.pw_salt, "
            "pw_hash=excluded.pw_hash, api_key=excluded.api_key",
            (username, pw_salt, pw_hash, api_key, created_at),
        )
        _conn.commit()


def get_user(username: str) -> dict | None:
    with _db_lock:
        row = _conn.execute(
            "SELECT username, pw_salt, pw_hash, api_key, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row:
        return None
    return {"username": row[0], "pw_salt": row[1], "pw_hash": row[2],
            "api_key": row[3], "created_at": row[4]}


def list_users() -> list[dict]:
    with _db_lock:
        rows = _conn.execute("SELECT username, created_at FROM users ORDER BY username").fetchall()
    return [{"username": r[0], "created_at": r[1]} for r in rows]


def delete_user(username: str) -> None:
    with _db_lock:
        _conn.execute("DELETE FROM users WHERE username = ?", (username,))
        _conn.commit()


def all_key_user_pairs() -> list[tuple]:
    """(api_key, username) for every user — deps.py builds its key->user map from this."""
    with _db_lock:
        rows = _conn.execute("SELECT api_key, username FROM users").fetchall()
    return [(r[0], r[1]) for r in rows]
