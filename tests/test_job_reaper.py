"""Tests for durable job leases and the stale-job reaper.

Ingest and report jobs run inside the API process, so a restart, crash or OOM
mid-job left the row saying `running` forever. Nothing ever moved it, and a
client polling /ingest/status or /report/status waited on a job that no longer
existed in any process.

Jobs now carry a lease that every progress update renews, and startup reaps the
ones whose lease expired — an expired lease being evidence the owner is gone, as
opposed to merely slow.
"""

import uuid
from datetime import datetime, timedelta, timezone

import deps
import persistence

PAST = "2020-01-01T00:00:00+00:00"
FUTURE = "2099-01-01T00:00:00+00:00"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(job_id, status, lease_until, **extra):
    job = {"job_id": job_id, "status": status, "submitted_at": PAST,
           "completed_at": None, "error": None, **extra}
    persistence.save_job(job_id, job, lease_until=lease_until)
    return job


def _row(job_id):
    return persistence._conn.execute(
        "SELECT status, lease_until FROM jobs WHERE id = ?", (job_id,)).fetchone()


def _cleanup(*ids):
    for jid in ids:
        persistence.delete_job(jid)


def test_expired_lease_is_reaped_and_reported():
    jid = str(uuid.uuid4())
    _save(jid, "running", PAST)
    try:
        reaped = persistence.reap_stale_jobs(_now())
        assert jid in {j["job_id"] for j in reaped}
        status, lease = _row(jid)
        assert status == "failed"
        assert lease is None
        job = [j for j in reaped if j["job_id"] == jid][0]
        # The error has to say what happened and what to do — a bare "failed"
        # tells the user nothing about whether retrying is sensible.
        assert "Abandoned" in job["error"] and "Resubmit" in job["error"]
        assert job["completed_at"] is not None
    finally:
        _cleanup(jid)


def test_a_live_worker_keeps_its_job():
    """The point of leasing: a job someone is still heartbeating must survive a
    reap, or one worker starting up would kill another's in-flight work."""
    jid = str(uuid.uuid4())
    _save(jid, "running", FUTURE)
    try:
        reaped = persistence.reap_stale_jobs(_now())
        assert jid not in {j["job_id"] for j in reaped}
        assert _row(jid)[0] == "running"
    finally:
        _cleanup(jid)


def test_finished_jobs_are_never_reaped():
    done, failed = str(uuid.uuid4()), str(uuid.uuid4())
    _save(done, "success", None)
    _save(failed, "failed", None)
    try:
        reaped = {j["job_id"] for j in persistence.reap_stale_jobs(_now())}
        assert done not in reaped and failed not in reaped
        assert _row(done)[0] == "success"
    finally:
        _cleanup(done, failed)


def test_pending_jobs_are_reaped_too():
    """A job queued but never started is just as abandoned as a running one."""
    jid = str(uuid.uuid4())
    _save(jid, "pending", PAST)
    try:
        assert jid in {j["job_id"] for j in persistence.reap_stale_jobs(_now())}
    finally:
        _cleanup(jid)


def test_legacy_jobs_without_a_lease_are_reaped():
    """Rows written before leasing existed cannot be running — running jobs
    heartbeat — so a NULL lease is not a reason to leave one in-flight."""
    jid = str(uuid.uuid4())
    _save(jid, "running", None)
    try:
        assert jid in {j["job_id"] for j in persistence.reap_stale_jobs(_now())}
    finally:
        _cleanup(jid)


def test_reaping_is_idempotent():
    jid = str(uuid.uuid4())
    _save(jid, "running", PAST)
    try:
        assert len(persistence.reap_stale_jobs(_now())) >= 1
        second = {j["job_id"] for j in persistence.reap_stale_jobs(_now())}
        assert jid not in second  # already failed, nothing left to reap
    finally:
        _cleanup(jid)


def test_update_job_renews_the_lease_while_running():
    jid = str(uuid.uuid4())
    with deps._jobs_lock:
        deps._jobs[jid] = {"job_id": jid, "status": "pending", "submitted_at": PAST,
                           "completed_at": None}
    try:
        deps._update_job(jid, status="running", progress_message="step 1")
        status, lease = _row(jid)
        assert status == "running"
        assert lease is not None and lease > _now()  # pushed into the future
    finally:
        with deps._jobs_lock:
            deps._jobs.pop(jid, None)
        _cleanup(jid)


def test_finishing_a_job_clears_its_lease():
    """A finished job holds no claim — leaving a lease on it would assert
    ongoing work that isn't happening."""
    jid = str(uuid.uuid4())
    with deps._jobs_lock:
        deps._jobs[jid] = {"job_id": jid, "status": "running", "submitted_at": PAST,
                           "completed_at": None}
    try:
        deps._update_job(jid, status="success", completed_at=_now())
        assert _row(jid) == ("success", None)
    finally:
        with deps._jobs_lock:
            deps._jobs.pop(jid, None)
        _cleanup(jid)


def test_lease_horizon_matches_config():
    import config
    lease = datetime.fromisoformat(deps.job_lease())
    expected = datetime.now(timezone.utc) + timedelta(seconds=config.JOB_LEASE_SECONDS)
    assert abs((lease - expected).total_seconds()) < 5
