"""Regression tests for the cross-user data exposure fixed in this batch.

Before the fix, any valid API key could read every other user's data:
  * GET /chat and GET /chat/{id} listed and returned all sessions;
  * GET /watch took `user_id` as a query parameter, so omitting it returned
    every user's watches and supplying someone else's returned theirs;
  * feedback and report listings had no owner column at all.

The rule these tests pin down: `owner` is the API-key fingerprint, it is never
taken from client input, and a record belonging to another key is indistinguishable
from one that does not exist (404, not 403 — a 403 confirms the id is real).
"""

import uuid

import pytest

import deps
import persistence

OWNER_A = "owner-a-fingerprint"
OWNER_B = "owner-b-fingerprint"


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------
def test_owns_session_matches_only_the_same_fingerprint():
    sess = {"id": "s1", "owner": OWNER_A}
    assert deps.owns_session(sess, OWNER_A)
    assert not deps.owns_session(sess, OWNER_B)


def test_owns_session_rejects_missing_session():
    assert not deps.owns_session(None, OWNER_A)


def test_legacy_ownerless_session_is_not_readable_by_a_user_key():
    """Rows written before ownership existed must fail closed, not default open."""
    legacy = {"id": "old", "messages": []}
    assert not deps.owns_session(legacy, OWNER_A)


def test_single_tenant_mode_sees_everything():
    """Auth disabled => current_owner is None => no scoping to apply."""
    assert deps.owns_session({"id": "s", "owner": None}, None)
    assert deps.owns_session({"id": "s", "owner": OWNER_A}, None)


def test_new_session_records_its_owner():
    sid, _ = deps._get_or_create_session(None, OWNER_A)
    try:
        assert deps._sessions[sid]["owner"] == OWNER_A
    finally:
        deps._sessions.pop(sid, None)
        persistence.delete_session(sid)


def test_guessing_another_users_session_id_is_refused():
    """The core of the bug: continuing a chat by id must not cross keys."""
    sid, _ = deps._get_or_create_session(None, OWNER_A)
    try:
        with pytest.raises(PermissionError):
            deps._get_or_create_session(sid, OWNER_B)
    finally:
        deps._sessions.pop(sid, None)
        persistence.delete_session(sid)


def test_owner_survives_eviction_resurrect_path():
    """_append_session_messages re-creates an evicted session; if it dropped the
    owner, the session would vanish from its owner's listing mid-conversation."""
    sid = str(uuid.uuid4())
    try:
        deps._append_session_messages(sid, "q", "a", OWNER_A)
        assert deps._sessions[sid]["owner"] == OWNER_A
    finally:
        deps._sessions.pop(sid, None)
        persistence.delete_session(sid)


# --------------------------------------------------------------------------
# Watches
# --------------------------------------------------------------------------
def _make_watch(owner, user_id="label"):
    w = {
        "id": str(uuid.uuid4()), "user_id": user_id, "owner": owner,
        "topic": "antennas", "language": "en", "cadence": "weekly",
        "seen_ids": [], "latest_digest": None,
        "next_run": None, "last_run": None, "created_at": "2026-01-01T00:00:00+00:00",
    }
    persistence.save_watch(w)
    return w


def test_list_watches_is_scoped_by_owner_not_by_user_id():
    a = _make_watch(OWNER_A)
    b = _make_watch(OWNER_B)
    try:
        ids = {w["id"] for w in persistence.list_watches(owner=OWNER_A)}
        assert a["id"] in ids
        assert b["id"] not in ids
    finally:
        persistence.delete_watch(a["id"])
        persistence.delete_watch(b["id"])


def test_supplying_another_users_user_id_label_reveals_nothing():
    """user_id is a caller-chosen label; it must not act as an access key."""
    victim = _make_watch(OWNER_B, user_id="victim-label")
    try:
        assert persistence.list_watches("victim-label", owner=OWNER_A) == []
    finally:
        persistence.delete_watch(victim["id"])


def test_owns_watch_rules():
    w = {"id": "w", "owner": OWNER_A}
    assert persistence.owns_watch(w, OWNER_A)
    assert not persistence.owns_watch(w, OWNER_B)
    assert not persistence.owns_watch(None, OWNER_A)
    assert persistence.owns_watch(w, None)                   # single-tenant
    assert not persistence.owns_watch({"id": "w"}, OWNER_A)  # legacy, fails closed


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def test_report_of_another_owner_reads_as_missing():
    rid = str(uuid.uuid4())
    persistence.save_report(rid, "", "topic", "en", "# body", 3,
                            "2026-01-01T00:00:00+00:00", owner=OWNER_A)
    try:
        assert persistence.get_report(rid, owner=OWNER_A) is not None
        assert persistence.get_report(rid, owner=OWNER_B) is None   # 404, not 403
        assert persistence.get_report(rid) is not None              # admin/single-tenant
    finally:
        persistence._conn.execute("DELETE FROM reports WHERE id = ?", (rid,))
        persistence._conn.commit()


def test_list_reports_filters_by_owner():
    mine, theirs = str(uuid.uuid4()), str(uuid.uuid4())
    persistence.save_report(mine, "", "mine", "en", "x", 1,
                            "2026-01-01T00:00:00+00:00", owner=OWNER_A)
    persistence.save_report(theirs, "", "theirs", "en", "y", 1,
                            "2026-01-01T00:00:00+00:00", owner=OWNER_B)
    try:
        ids = {r["id"] for r in persistence.list_reports(owner=OWNER_A)}
        assert mine in ids and theirs not in ids
    finally:
        persistence._conn.execute("DELETE FROM reports WHERE id IN (?, ?)", (mine, theirs))
        persistence._conn.commit()


# --------------------------------------------------------------------------
# Feedback  (the join exposes the original question and answer text)
# --------------------------------------------------------------------------
def test_feedback_listing_is_scoped_to_its_submitter():
    qid_a, qid_b = str(uuid.uuid4()), str(uuid.uuid4())
    fid_a, fid_b = str(uuid.uuid4()), str(uuid.uuid4())
    ts = "2026-01-01T00:00:00+00:00"
    persistence.log_query(qid_a, "secret question A", "answer A", "standard_A",
                          "m", "en", 0.9, 0.9, ts, owner=OWNER_A)
    persistence.log_query(qid_b, "secret question B", "answer B", "standard_A",
                          "m", "en", 0.9, 0.9, ts, owner=OWNER_B)
    persistence.save_feedback(fid_a, qid_a, "up", "", ts, owner=OWNER_A)
    persistence.save_feedback(fid_b, qid_b, "down", "", ts, owner=OWNER_B)
    try:
        rows = persistence.get_feedback_with_context(owner=OWNER_A)
        questions = {r["question"] for r in rows}
        assert "secret question A" in questions
        assert "secret question B" not in questions
    finally:
        persistence._conn.execute("DELETE FROM feedback WHERE id IN (?, ?)", (fid_a, fid_b))
        persistence._conn.execute("DELETE FROM query_log WHERE query_id IN (?, ?)", (qid_a, qid_b))
        persistence._conn.commit()


def test_feedback_stats_are_scoped_too():
    qid, fid = str(uuid.uuid4()), str(uuid.uuid4())
    ts = "2026-01-01T00:00:00+00:00"
    persistence.save_feedback(fid, qid, "up", "", ts, owner=OWNER_A)
    try:
        assert persistence.feedback_stats(owner=OWNER_A)["total"] >= 1
        assert persistence.feedback_stats(owner=OWNER_B)["total"] == 0
    finally:
        persistence._conn.execute("DELETE FROM feedback WHERE id = ?", (fid,))
        persistence._conn.commit()


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------
def test_owner_columns_exist_and_ensure_column_is_idempotent():
    for table in ("feedback", "reports", "query_log"):
        cols = {r[1] for r in persistence._conn.execute(f"PRAGMA table_info({table})")}
        assert "owner" in cols, f"{table} missing owner column"
    # Running it again must not raise "duplicate column name".
    persistence._ensure_column("feedback", "owner", "TEXT")
