"""Unit tests for Phase 6 watch persistence (persistence.save_watch etc.).

Runs against the throwaway SQLite DB set up in conftest.py. Each test clears the
watches table first so ordering between tests can't leak state.
"""

import pytest

import persistence


@pytest.fixture(autouse=True)
def _clear_watches():
    with persistence._db_lock:
        persistence._conn.execute("DELETE FROM watches")
        persistence._conn.commit()
    yield


def _watch(wid, user="u1", next_run=None, created="2026-07-11T09:00:00+00:00", **extra):
    base = {
        "id": wid, "user_id": user, "topic": "terahertz antennas", "language": "en",
        "cadence": "weekly", "seen_ids": [], "latest_digest": None,
        "next_run": next_run, "last_run": None, "created_at": created,
    }
    base.update(extra)
    return base


def test_save_and_get_round_trips_full_dict():
    w = _watch("w1", seen_ids=["arxiv:1"], latest_digest="hello")
    persistence.save_watch(w)
    assert persistence.get_watch("w1") == w


def test_get_missing_returns_none():
    assert persistence.get_watch("does-not-exist") is None


def test_list_watches_all_and_by_user_newest_first():
    persistence.save_watch(_watch("w1", user="u1", created="2026-07-11T09:00:00+00:00"))
    persistence.save_watch(_watch("w2", user="u1", created="2026-07-11T10:00:00+00:00"))
    persistence.save_watch(_watch("w3", user="u2", created="2026-07-11T11:00:00+00:00"))

    assert [w["id"] for w in persistence.list_watches()] == ["w3", "w2", "w1"]
    assert {w["id"] for w in persistence.list_watches("u1")} == {"w1", "w2"}
    assert [w["id"] for w in persistence.list_watches("u2")] == ["w3"]


def test_due_watches_respects_next_run_and_null():
    persistence.save_watch(_watch("due", next_run="2026-07-11T00:00:00+00:00"))
    persistence.save_watch(_watch("future", next_run="2099-01-01T00:00:00+00:00"))
    persistence.save_watch(_watch("manual", next_run=None))  # never auto-runs

    now = "2026-07-11T12:00:00+00:00"
    assert [w["id"] for w in persistence.due_watches(now)] == ["due"]


def test_save_watch_upserts_not_duplicates():
    persistence.save_watch(_watch("w1", next_run="2026-07-11T00:00:00+00:00"))
    updated = _watch("w1", seen_ids=["arxiv:99"],
                     next_run="2099-01-01T00:00:00+00:00",
                     last_run="2026-07-11T12:00:00+00:00")
    persistence.save_watch(updated)

    assert len(persistence.list_watches()) == 1
    assert persistence.get_watch("w1")["seen_ids"] == ["arxiv:99"]
    # advancing next_run into the future removes it from the due set
    assert persistence.due_watches("2026-07-11T12:00:00+00:00") == []


def test_delete_watch():
    persistence.save_watch(_watch("w1"))
    persistence.save_watch(_watch("w2"))
    persistence.delete_watch("w1")
    assert persistence.get_watch("w1") is None
    assert {w["id"] for w in persistence.list_watches()} == {"w2"}
