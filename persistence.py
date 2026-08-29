"""
SQLite-backed session/job persistence so restarts don't lose in-flight state.

Stdlib sqlite3, not aiosqlite: every caller in deps.py is already a sync
function invoked via FastAPI's threadpool, so there's no event loop to keep
async for. WAL mode lets reads and the single writer coexist without
blocking each other.
"""

from contextlib import contextmanager
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
    # Append-only record of what was ingested, and of the chunks that ingestion
    # produced. This is the system of record; ChromaDB, the BM25 index and the
    # figure store are derived views over it.
    #
    # Without this, the only way to rebuild the vector store was to re-parse every
    # PDF and re-call the VLM captioner — hours of work, non-reproducible, and
    # dependent on source files still being present. That made changing the
    # embedding model or the chunking strategy so expensive it never happened.
    # With it, reindexing is a replay.
    "CREATE TABLE IF NOT EXISTS ingest_log ("
    "event_id TEXT PRIMARY KEY, paper_id TEXT, content_hash TEXT, title TEXT, "
    "source_path TEXT, chunks TEXT, metadatas TEXT, ids TEXT, "
    "embed_model TEXT, chunker_version INTEGER, created_at TEXT)"
)


def _year_of(metadatas_json) -> str:
    """Publication year off a chunk-metadata blob; empty string when absent.

    Every chunk of a paper carries the same year, so the first one that has it
    answers for the paper.
    """
    try:
        for meta in json.loads(metadatas_json or "[]"):
            year = (meta or {}).get("year")
            if year:
                return str(year)
    except (TypeError, ValueError):
        pass
    return ""


def _ensure_column(table: str, column: str, decl: str) -> None:
    """Add a column to an existing table if it isn't there yet.

    `CREATE TABLE IF NOT EXISTS` silently keeps an old database's old shape, so a
    column added in a later release never appears on an already-deployed instance
    and the first query referencing it fails with `no such column`. Idempotent by
    inspection, which is what lets the migrations below run safely on a database
    that already got these columns from the pre-runner version of this file.
    """
    cols = {row[1] for row in _conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        _conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# ---------------------------------------------------------------------------
# Schema migrations
#
# Ordered, named, applied once, recorded in `schema_migrations`. The previous
# scheme — bare _ensure_column() calls at import — could only ever append a
# nullable column: it had no way to express "backfill this from that", and no
# record of what had run, so a migration needing data transformation had nowhere
# to live. Rules: append only, never edit an applied migration (deployed
# databases already ran the old body), and each migration is one transaction.
# ---------------------------------------------------------------------------
def _m0001_owner_and_lease_columns(conn) -> None:
    # `owner` is the SHA-256 fingerprint of the caller's API key (deps.current_owner).
    # NULL means "written before ownership existed, or written while auth was disabled".
    _ensure_column("feedback", "owner", "TEXT")
    _ensure_column("reports", "owner", "TEXT")
    _ensure_column("query_log", "owner", "TEXT")
    # Job leases. An in-flight job is owned by whichever process is heartbeating it;
    # when that process dies the lease simply expires and the row can be reaped.
    _ensure_column("jobs", "lease_until", "TEXT")
    # Which weights produced the vectors. int8 and fp32 output for the SAME model id
    # are not interchangeable, so embed_model alone cannot detect a mixed corpus.
    _ensure_column("ingest_log", "embed_backend", "TEXT")


def _m0002_paper_index(conn) -> None:
    """Materialise one row per paper from the ingest log.

    Dedup on ingest fuzzy-matches the new title against every existing one, and
    the only place titles lived was ChromaDB chunk metadata — so answering "have
    I seen this paper?" meant pulling the metadata of every CHUNK in the corpus
    and grouping it in Python, once per ingested file. This mirror is derived
    state, rebuildable from the log at any time, and it is a papers-sized table
    rather than a chunks-sized scan.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS paper_index ("
        "paper_id TEXT PRIMARY KEY, title TEXT, year TEXT, "
        "chunk_count INTEGER, updated_at TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_index_year ON paper_index(year)")
    rows = conn.execute(
        "SELECT paper_id, title, metadatas, ids, created_at FROM ingest_log"
    ).fetchall()
    for paper_id, title, metadatas_json, ids_json, created_at in rows:
        conn.execute(
            "INSERT INTO paper_index (paper_id, title, year, chunk_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(paper_id) DO UPDATE SET "
            "title=excluded.title, year=excluded.year, "
            "chunk_count=excluded.chunk_count, updated_at=excluded.updated_at",
            (paper_id, title, _year_of(metadatas_json),
             len(json.loads(ids_json or "[]")), created_at),
        )


def _m0003_metadata_cache(conn) -> None:
    """Cache arXiv title lookups so a re-ingest is not a re-crawl.

    Enrichment is one network round trip per paper behind a 1s politeness delay,
    paid again on every bulk run over the same directory even when the answer
    could not have changed. Misses are cached too — "arXiv does not have this
    paper" is exactly the answer that would otherwise be re-fetched forever.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metadata_cache ("
        "title_key TEXT PRIMARY KEY, authors TEXT, year TEXT, doi TEXT, "
        "found INTEGER, fetched_at TEXT)"
    )


_MIGRATIONS = [
    ("0001_owner_and_lease_columns", _m0001_owner_and_lease_columns),
    ("0002_paper_index", _m0002_paper_index),
    ("0003_metadata_cache", _m0003_metadata_cache),
]


def _run_migrations() -> None:
    _conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "name TEXT PRIMARY KEY, applied_at TEXT)"
    )
    applied = {row[0] for row in _conn.execute("SELECT name FROM schema_migrations")}
    for name, fn in _MIGRATIONS:
        if name in applied:
            continue
        try:
            fn(_conn)
            _conn.execute(
                "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
                (name, datetime.now(timezone.utc).isoformat()),
            )
            _conn.commit()
        except Exception:
            # Leave it unrecorded and re-raise: a half-applied migration that
            # marked itself done would be invisible on the next start, and the
            # failure belongs at startup, not at the first query that needs the
            # column.
            _conn.rollback()
            raise


_run_migrations()


# ---------------------------------------------------------------------------
# Reads and batched writes
#
# WAL lets one writer and many readers coexist, but every access here went
# through one connection behind one global lock, so readers queued behind each
# other and behind the writer for no reason. Reads now use a per-thread
# read-only connection and take no lock at all; WAL gives each one a consistent
# snapshot. Writes still go through _conn under _db_lock — single-writer is the
# correct topology here, and that lock is what keeps it single.
# ---------------------------------------------------------------------------
_local = threading.local()


def _read_conn():
    """A per-thread read-only connection, opened on first use."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(
            f"file:{config.SESSIONS_DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        _local.conn = conn
    return conn


_batch_depth = 0


@contextmanager
def batch_writes():
    """Commit once at the end of the block instead of once per write.

    Bulk paths wrote one row per commit, and every commit is an fsync: a 50-paper
    ingest paid 50 durability barriers to record work the user thinks of as one
    operation. Nested use is refcounted, and a failure inside the block rolls the
    whole batch back rather than leaving half of it recorded.

    ponytail: the depth counter is process-global, so a concurrent write from
    another thread joins whichever batch is open. Single-writer by design; make
    it thread-local if writes ever fan out.
    """
    global _batch_depth
    with _db_lock:
        _batch_depth += 1
    try:
        yield
    except Exception:
        with _db_lock:
            _batch_depth -= 1
            if _batch_depth == 0:
                _conn.rollback()
        raise
    with _db_lock:
        _batch_depth -= 1
        if _batch_depth == 0:
            _conn.commit()


def _maybe_commit() -> None:
    """Commit unless a batch is open. Call while holding _db_lock."""
    if _batch_depth == 0:
        _conn.commit()

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
_conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_paper ON ingest_log(paper_id)")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_hash ON ingest_log(content_hash)")

_conn.commit()
_db_lock = threading.Lock()


def snapshot_to(dest_path) -> dict:
    """Write a consistent copy of this database to `dest_path`, online.

    sqlite3's own backup API, not a file copy: copying sessions.db while the
    process is writing yields a torn file, and copying it without its -wal loses
    the most recent writes entirely. This runs page-by-page against a live
    connection and is safe with the server up.
    """
    import sqlite3 as _sqlite3

    with _db_lock:
        dest = _sqlite3.connect(str(dest_path))
        try:
            _conn.backup(dest)
        finally:
            dest.close()
        papers = _conn.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0]
    return {"path": str(dest_path), "papers": papers}


def restore_from(src_path) -> dict:
    """Replace this database's contents with the snapshot at `src_path`.

    Also the backup API, in the other direction: it overwrites every page inside
    one transaction, so a failure leaves the original intact rather than half a
    database. The vector store is NOT touched — it is a derived view, and the
    caller replays the restored log into it (reindex.py).
    """
    import sqlite3 as _sqlite3

    with _db_lock:
        src = _sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        try:
            # Sanity-check before overwriting: an arbitrary SQLite file that has
            # no ingest_log would silently wipe the system of record.
            tables = {r[0] for r in src.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "ingest_log" not in tables:
                raise ValueError(f"{src_path} is not an IndicRAG database "
                                 "(no ingest_log table)")
            src.backup(_conn)
            papers = _conn.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0]
        finally:
            src.close()
    # A restored database can predate migrations this build expects.
    _run_migrations()
    return {"path": str(src_path), "papers": papers}


def checkpoint() -> None:
    """Fold the WAL back into the main database file. Best-effort.

    WAL mode defers that fold, so an unclean stop leaves the last writes only in
    sessions.db-wal. SQLite recovers them on next open, but only if the -wal file
    travels with the .db — which a naive backup, container image, or volume copy
    does not guarantee. TRUNCATE, not PASSIVE: PASSIVE gives up when a reader
    holds the file, and shutdown is exactly when we want it to finish.
    """
    try:
        with _db_lock:
            _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        import logging
        logging.getLogger(__name__).warning("WAL checkpoint on shutdown failed", exc_info=True)


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


def save_job(job_id: str, job: dict, lease_until: str = None) -> None:
    """Persist a job. `lease_until` renews the caller's claim on an in-flight job.

    Every update to a running job is also a heartbeat: it pushes the lease
    forward, which is how a still-alive worker distinguishes itself from a dead
    one. See reap_stale_jobs.
    """
    with _db_lock:
        _conn.execute(
            "INSERT INTO jobs (id, data, status, submitted_at, completed_at, lease_until) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data=excluded.data, status=excluded.status, "
            "completed_at=excluded.completed_at, lease_until=excluded.lease_until",
            (job_id, json.dumps(job), job.get("status"), job.get("submitted_at"),
             job.get("completed_at"), lease_until),
        )
        _conn.commit()


def reap_stale_jobs(now_iso: str) -> list[dict]:
    """Fail jobs left in-flight by a process that died. Returns what was reaped.

    Ingest and report jobs run inside the API process, so a restart, crash or
    OOM mid-job left the row saying `running` forever — nothing ever moved it,
    and a client polling /ingest/status or /report/status waited on a job that
    no longer existed anywhere.

    Reaping is by expired lease rather than by "not mine": a live worker renews
    its lease on every progress update, so an expired lease is real evidence the
    owner is gone. A job with no lease at all predates this and is also reaped —
    it cannot be running, because running jobs heartbeat.
    """
    with _db_lock:
        rows = _conn.execute(
            "SELECT id, data FROM jobs WHERE status IN ('pending', 'running') "
            "AND (lease_until IS NULL OR lease_until < ?)",
            (now_iso,),
        ).fetchall()
        reaped = []
        for jid, data in rows:
            job = json.loads(data)
            job["status"] = "failed"
            job["error"] = ("Abandoned: the process running this job exited before it "
                            "finished. Resubmit to try again.")
            job["completed_at"] = now_iso
            _conn.execute(
                "UPDATE jobs SET data = ?, status = 'failed', completed_at = ?, "
                "lease_until = NULL WHERE id = ?",
                (json.dumps(job), now_iso, jid),
            )
            reaped.append(job)
        if reaped:
            _conn.commit()
    return reaped


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
    rows = _read_conn().execute(
        "SELECT data FROM watches WHERE next_run IS NOT NULL AND next_run <= ? "
        "ORDER BY next_run ASC",
        (now_iso,),
    ).fetchall()
    return [json.loads(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# Ingest log — the system of record the search indexes are derived from.
# ---------------------------------------------------------------------------
def record_ingest(event_id: str, paper_id: str, content_hash: str, title: str,
                  source_path: str, chunks: list, metadatas: list, ids: list,
                  embed_model: str, chunker_version: int, created_at: str,
                  embed_backend: str = None) -> None:
    """Record one ingestion, including the chunks it produced.

    Re-ingesting a paper replaces its row rather than appending a second one:
    the log describes the CURRENT contents of the indexes, and keeping
    superseded versions would make a replay reinstate deleted chunks.
    """
    with _db_lock:
        _conn.execute(
            "INSERT INTO ingest_log (event_id, paper_id, content_hash, title, source_path, "
            "chunks, metadatas, ids, embed_model, chunker_version, created_at, embed_backend) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(event_id) DO UPDATE SET content_hash=excluded.content_hash, "
            "title=excluded.title, source_path=excluded.source_path, chunks=excluded.chunks, "
            "metadatas=excluded.metadatas, ids=excluded.ids, embed_model=excluded.embed_model, "
            "chunker_version=excluded.chunker_version, created_at=excluded.created_at, "
            "embed_backend=excluded.embed_backend",
            (event_id, paper_id, content_hash, title, source_path,
             json.dumps(chunks), json.dumps(metadatas), json.dumps(ids),
             embed_model, chunker_version, created_at, embed_backend),
        )
        # Same transaction as the log write: the mirror is only trustworthy if it
        # cannot lag the row it mirrors.
        _conn.execute(
            "INSERT INTO paper_index (paper_id, title, year, chunk_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(paper_id) DO UPDATE SET "
            "title=excluded.title, year=excluded.year, "
            "chunk_count=excluded.chunk_count, updated_at=excluded.updated_at",
            (paper_id, title, _year_of(json.dumps(metadatas)), len(ids), created_at),
        )
        _maybe_commit()


def get_ingest_events(paper_id: str = None) -> list[dict]:
    """Every recorded ingestion, oldest first, or just one paper's.

    Oldest first so a replay reproduces the original ingest order — chunk ids
    and citation numbering follow insertion order.
    """
    conn = _read_conn()
    if paper_id is None:
        rows = conn.execute(
            "SELECT event_id, paper_id, content_hash, title, source_path, chunks, "
            "metadatas, ids, embed_model, chunker_version, created_at, embed_backend "
            "FROM ingest_log ORDER BY created_at ASC").fetchall()
    else:
        rows = conn.execute(
            "SELECT event_id, paper_id, content_hash, title, source_path, chunks, "
            "metadatas, ids, embed_model, chunker_version, created_at, embed_backend "
            "FROM ingest_log WHERE paper_id = ? ORDER BY created_at ASC",
            (paper_id,)).fetchall()
    return [{
        "event_id": r[0], "paper_id": r[1], "content_hash": r[2], "title": r[3],
        "source_path": r[4], "chunks": json.loads(r[5]), "metadatas": json.loads(r[6]),
        "ids": json.loads(r[7]), "embed_model": r[8], "chunker_version": r[9],
        "created_at": r[10], "embed_backend": r[11],
    } for r in rows]


def get_metadata_cache(title_key: str):
    """Cached arXiv lookup: a metadata dict, {} for a cached miss, or None if unseen.

    The three-way return matters — {} and None are different answers. {} means
    "asked, arXiv had nothing", which must not trigger another lookup.
    """
    row = _read_conn().execute(
        "SELECT authors, year, doi, found FROM metadata_cache WHERE title_key = ?",
        (title_key,),
    ).fetchone()
    if row is None:
        return None
    if not row[3]:
        return {}
    return {"authors": row[0] or "", "year": row[1] or "", "doi": row[2] or ""}


def put_metadata_cache(title_key: str, data, fetched_at: str) -> None:
    """Record a lookup result. `data` falsy means a cached miss."""
    data = data or {}
    with _db_lock:
        _conn.execute(
            "INSERT INTO metadata_cache (title_key, authors, year, doi, found, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(title_key) DO UPDATE SET "
            "authors=excluded.authors, year=excluded.year, doi=excluded.doi, "
            "found=excluded.found, fetched_at=excluded.fetched_at",
            (title_key, data.get("authors", ""), data.get("year", ""),
             data.get("doi", ""), 1 if data else 0, fetched_at),
        )
        _conn.commit()


def list_papers() -> list[dict]:
    """One row per ingested paper: paper_id, title, year, chunk_count.

    The dedup path's view of the corpus. Papers-sized, so a caller can afford to
    fuzzy-match every title without touching the vector store.
    """
    rows = _read_conn().execute(
        "SELECT paper_id, title, year, chunk_count FROM paper_index"
    ).fetchall()
    return [{"paper_id": r[0], "title": r[1] or "", "year": r[2] or "",
             "chunk_count": r[3] or 0} for r in rows]


def clear_ingest_log() -> int:
    """Delete every ingest event. Returns how many rows went.

    For `purge.py --db` only: the log describes the current contents of the
    indexes, so wiping the indexes without wiping the log leaves a system of
    record that would replay a corpus which no longer exists.
    """
    with _db_lock:
        cur = _conn.execute("DELETE FROM ingest_log")
        _conn.execute("DELETE FROM paper_index")
        _conn.commit()
        return cur.rowcount


def ingest_event_count() -> int:
    return _read_conn().execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0]


def delete_ingest_events(paper_id: str) -> int:
    """Drop a paper from the log. Returns rows removed.

    Must accompany deleting the paper from the indexes — otherwise a later
    replay would resurrect it, which is the failure mode a system of record is
    supposed to prevent.
    """
    with _db_lock:
        cur = _conn.execute("DELETE FROM ingest_log WHERE paper_id = ?", (paper_id,))
        _conn.execute("DELETE FROM paper_index WHERE paper_id = ?", (paper_id,))
        _conn.commit()
        return cur.rowcount


def claim_watch(watch_id: str, expected_next_run: str, lease_until: str) -> bool:
    """Atomically claim a due watch. True if this caller won the claim.

    The scheduler runs in-process, so two workers (or two replicas) both see the
    same watch as due and both run it — duplicate arXiv fetches, duplicate
    ingests, duplicate LLM spend on the digest. This is a compare-and-set on the
    row: the UPDATE only matches while next_run is still what the claimer read,
    so exactly one caller can move it and the losers see rowcount 0.

    `lease_until` parks next_run far enough ahead that a claimer which crashes
    mid-run doesn't wedge the watch forever — the lease simply expires and the
    watch becomes due again. run_watch rewrites next_run properly on success.
    """
    with _db_lock:
        # The JSON copy moves with the column. due_watches reads the row but
        # claim_watch compares the column, so leaving data.next_run behind after a
        # failed run means every later claim compares the stale value against the
        # lease and fails — the watch never runs again.
        cur = _conn.execute(
            "UPDATE watches SET next_run = ?, data = json_set(data, '$.next_run', ?) "
            "WHERE id = ? AND next_run = ?",
            (lease_until, lease_until, watch_id, expected_next_run),
        )
        _conn.commit()
        return cur.rowcount == 1


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
            "created_at=excluded.created_at, owner=excluded.owner",
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
