"""Unit tests for Phase 6 Increment 3 — watch_runner.run_watch.

Mocks the three external seams (arxiv search, PDF download+ingest, LLM digest)
so the test is offline and deterministic. Asserts dedup, seen_ids growth, digest
storage, and next_run advancement.
"""

import asyncio

import pytest

import persistence
import watch_runner


@pytest.fixture(autouse=True)
def _clear_watches():
    with persistence._db_lock:
        persistence._conn.execute("DELETE FROM watches")
        persistence._conn.commit()
    yield


def _save(wid, **extra):
    w = {
        "id": wid, "user_id": "u1", "topic": "terahertz antennas", "language": "en",
        "cadence": "weekly", "seen_ids": [], "latest_digest": None,
        "next_run": None, "last_run": None, "created_at": "2026-07-11T09:00:00+00:00",
    }
    w.update(extra)
    persistence.save_watch(w)
    return w


def _hit(arxiv_id, pdf=True):
    return {
        "text": f"Abstract of {arxiv_id}", "title": f"Paper {arxiv_id}",
        "source": f"http://arxiv.org/abs/{arxiv_id}", "section": "arxiv",
        "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}" if pdf else None,
        "arxiv_id": arxiv_id,
    }


def _patch(monkeypatch, hits, ingest=lambda *a, **k: (5, "Ingested Title"), digest="DIGEST"):
    monkeypatch.setattr(watch_runner, "execute_arxiv_search", lambda *a, **k: {"passages": hits})
    monkeypatch.setattr(watch_runner, "_download_pdf", lambda url: "/tmp/fake.pdf")
    monkeypatch.setattr(watch_runner, "ingest_pdf", ingest)
    monkeypatch.setattr(watch_runner, "_summarize", lambda *a, **k: digest)


def test_ingests_new_papers_and_stores_cited_digest(monkeypatch):
    _save("w1")
    _patch(monkeypatch, [_hit("2401.001"), _hit("2401.002")])

    res = watch_runner.run_watch("w1")

    assert res["new_count"] == 2
    assert res["digest"] == "DIGEST"
    w = persistence.get_watch("w1")
    assert set(w["seen_ids"]) == {"2401.001", "2401.002"}
    assert w["latest_digest"] == "DIGEST"
    assert w["last_run"] is not None
    assert w["next_run"] > w["created_at"]  # advanced one cadence out


def test_dedups_already_seen_ids(monkeypatch):
    _save("w1", seen_ids=["2401.001"])
    _patch(monkeypatch, [_hit("2401.001"), _hit("2401.002")])

    res = watch_runner.run_watch("w1")

    assert res["new_count"] == 1  # only the unseen one
    assert set(persistence.get_watch("w1")["seen_ids"]) == {"2401.001", "2401.002"}


def test_corpus_duplicate_marks_seen_but_not_new(monkeypatch):
    _save("w1")
    # ingest_pdf returns 0 chunks → duplicate/unchanged in corpus
    _patch(monkeypatch, [_hit("2401.001")], ingest=lambda *a, **k: (0, "Dup"))

    res = watch_runner.run_watch("w1")

    assert res["new_count"] == 0                      # nothing genuinely new
    assert persistence.get_watch("w1")["seen_ids"] == ["2401.001"]  # but marked seen


def test_abstract_only_when_no_pdf(monkeypatch):
    _save("w1")
    _patch(monkeypatch, [_hit("2401.003", pdf=False)])

    res = watch_runner.run_watch("w1")

    assert res["new_count"] == 1  # abstract feeds the digest
    assert persistence.get_watch("w1")["seen_ids"] == ["2401.003"]


def test_downloaded_pdf_is_kept_in_papers_dir(monkeypatch, tmp_path):
    """The library lists PAPERS_DIR and derives paper_id from the file stem, so a
    watch that deleted its download left indexed-but-invisible orphan chunks."""
    import config

    papers = tmp_path / "papers"
    monkeypatch.setattr(config, "PAPERS_DIR", papers)

    src = tmp_path / "download.pdf"
    src.write_bytes(b"%PDF-1.4 fake")

    seen = {}

    def _ingest(path, paper_id=None, metadata=None):
        seen["path"], seen["paper_id"] = path, paper_id
        return 5, "Ingested Title"

    _save("w1")
    _patch(monkeypatch, [_hit("2401.001")], ingest=_ingest)
    monkeypatch.setattr(watch_runner, "_download_pdf", lambda url: str(src))

    watch_runner.run_watch("w1")

    kept = papers / "2401_001.pdf"
    assert kept.exists(), "PDF must survive in PAPERS_DIR or the paper never lists"
    assert kept.stem == seen["paper_id"], "stem drives /ingest/health's paper_id lookup"
    assert seen["path"] == str(kept)     # ingested from the kept copy, not the temp file
    assert not src.exists()              # temp moved, not left behind


def test_failed_save_still_ingests_from_temp(monkeypatch, tmp_path):
    """A save failure must degrade to the old behavior, not drop the paper."""
    import config

    monkeypatch.setattr(config, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(watch_runner.shutil, "move",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    seen = {}

    def _ingest(path, paper_id=None, metadata=None):
        seen["path"] = path
        return 5, "Ingested Title"

    _save("w1")
    _patch(monkeypatch, [_hit("2401.001")], ingest=_ingest)
    monkeypatch.setattr(watch_runner, "_download_pdf", lambda url: "/tmp/fake.pdf")

    res = watch_runner.run_watch("w1")

    assert res["new_count"] == 1
    assert seen["path"] == "/tmp/fake.pdf"


def test_missing_watch_raises_keyerror(monkeypatch):
    _patch(monkeypatch, [])
    with pytest.raises(KeyError):
        watch_runner.run_watch("nope")


def test_no_new_papers_keeps_prior_digest(monkeypatch):
    _save("w1", seen_ids=["2401.001"], latest_digest="OLD")
    _patch(monkeypatch, [_hit("2401.001")])  # only an already-seen paper

    res = watch_runner.run_watch("w1")

    assert res["new_count"] == 0
    assert persistence.get_watch("w1")["latest_digest"] == "OLD"  # unchanged


def test_pdf_ingest_triggers_cache_refresh(monkeypatch):
    _save("w1")
    _patch(monkeypatch, [_hit("2401.001")])
    calls = []
    monkeypatch.setattr(watch_runner, "_post_ingest_refresh", lambda: calls.append(1))

    watch_runner.run_watch("w1")

    assert calls == [1]  # BM25/cache must be invalidated so the paper is searchable


def test_corpus_duplicate_does_not_trigger_cache_refresh(monkeypatch):
    _save("w1")
    # ingest_pdf returns 0 chunks → duplicate/unchanged in corpus, nothing indexed
    _patch(monkeypatch, [_hit("2401.001")], ingest=lambda *a, **k: (0, "Dup"))
    calls = []
    monkeypatch.setattr(watch_runner, "_post_ingest_refresh", lambda: calls.append(1))

    watch_runner.run_watch("w1")

    assert calls == []  # duplicate content, nothing new to make searchable


def test_abstract_only_does_not_trigger_cache_refresh(monkeypatch):
    _save("w1")
    _patch(monkeypatch, [_hit("2401.003", pdf=False)])
    calls = []
    monkeypatch.setattr(watch_runner, "_post_ingest_refresh", lambda: calls.append(1))

    watch_runner.run_watch("w1")

    assert calls == []  # nothing was actually indexed, no refresh needed


def test_owned_report_regenerated_when_new_papers_ingested(monkeypatch):
    """A watch with report_id set owns a 'living review': every time it
    actually indexes a new paper, the report regenerates in place under
    the same report_id (no confirmation button a user would forget to click)."""
    import persistence as persistence_module

    _save("w1", report_id="rep-1")
    _patch(monkeypatch, [_hit("2401.001")])  # has a pdf_url -> real ingest, indexed=True

    saved = []
    monkeypatch.setattr(
        "report_runner.run_report",
        lambda topic, language: {"markdown": "regenerated [1]", "citation_count": 1,
                                  "sections": ["s"], "topic": topic, "language": language},
    )
    monkeypatch.setattr(persistence_module, "save_report",
                        lambda **kw: saved.append(kw))

    watch_runner.run_watch("w1")

    assert len(saved) == 1
    assert saved[0]["report_id"] == "rep-1"
    assert saved[0]["markdown"] == "regenerated [1]"


def test_owned_report_not_regenerated_when_nothing_indexed(monkeypatch):
    import persistence as persistence_module

    _save("w1", report_id="rep-1")
    _patch(monkeypatch, [_hit("2401.003", pdf=False)])  # abstract-only, indexed=False

    saved = []
    monkeypatch.setattr("report_runner.run_report", lambda *a, **k: {"markdown": "x", "citation_count": 0})
    monkeypatch.setattr(persistence_module, "save_report", lambda **kw: saved.append(kw))

    watch_runner.run_watch("w1")

    assert saved == []


def test_watch_without_report_id_never_calls_run_report(monkeypatch):
    _save("w1")  # no report_id
    _patch(monkeypatch, [_hit("2401.001")])

    with_call = []
    monkeypatch.setattr("report_runner.run_report", lambda *a, **k: with_call.append(1) or {"markdown": "", "citation_count": 0})

    watch_runner.run_watch("w1")

    assert with_call == []


def test_run_due_watches_runs_each_due_and_survives_failures(monkeypatch):
    ran = []
    monkeypatch.setattr(watch_runner.persistence, "due_watches",
                        lambda now: [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    # Claiming is now part of the sweep; this caller wins every claim.
    monkeypatch.setattr(watch_runner.persistence, "claim_watch",
                        lambda wid, expected, lease: True)

    def _fake_run(wid):
        if wid == "b":
            raise RuntimeError("boom")  # one failure must not abort the sweep
        ran.append(wid)

    monkeypatch.setattr(watch_runner, "run_watch", _fake_run)

    count = asyncio.run(watch_runner.run_due_watches())

    assert count == 3            # all due watches attempted
    assert ran == ["a", "c"]     # b failed but a and c still ran


def test_run_due_watches_skips_watches_claimed_by_another_worker(monkeypatch):
    """The schedule loop runs in-process, so every worker sees the same watch as
    due. Only the one that wins the claim may run it — otherwise each replica
    re-fetches arXiv, re-ingests, and pays for its own digest."""
    ran = []
    monkeypatch.setattr(watch_runner.persistence, "due_watches",
                        lambda now: [{"id": "a"}, {"id": "b"}])
    # 'a' was taken by another worker between our read and our claim.
    monkeypatch.setattr(watch_runner.persistence, "claim_watch",
                        lambda wid, expected, lease: wid != "a")
    monkeypatch.setattr(watch_runner, "run_watch", lambda wid: ran.append(wid))

    count = asyncio.run(watch_runner.run_due_watches())

    assert ran == ["b"]
    assert count == 1  # counts what was claimed and run, not what looked due


def _seed_watch(persistence, wid, due_at):
    persistence.save_watch({
        "id": wid, "user_id": "u", "owner": None, "topic": "t", "language": "en",
        "cadence": "weekly", "seen_ids": [], "latest_digest": None,
        "next_run": due_at, "last_run": None, "created_at": due_at,
    })


def test_claim_watch_is_won_by_exactly_one_caller():
    """The compare-and-set itself: two workers racing on one row, one winner."""
    import uuid
    import persistence

    wid = str(uuid.uuid4())
    due_at = "2026-01-01T00:00:00+00:00"
    _seed_watch(persistence, wid, due_at)
    try:
        lease = "2026-01-01T01:00:00+00:00"
        first = persistence.claim_watch(wid, due_at, lease)
        second = persistence.claim_watch(wid, due_at, lease)
        assert first is True
        assert second is False   # next_run already moved; the loser matches no row
    finally:
        persistence.delete_watch(wid)


def test_claim_parks_next_run_so_a_dead_claimer_cannot_wedge_the_watch():
    import uuid
    import persistence

    wid = str(uuid.uuid4())
    due_at = "2026-01-01T00:00:00+00:00"
    lease = "2026-01-01T01:00:00+00:00"
    _seed_watch(persistence, wid, due_at)
    try:
        assert persistence.claim_watch(wid, due_at, lease) is True
        # Not due while the lease holds...
        assert wid not in {w["id"] for w in persistence.due_watches(due_at)}
        # ...but due again once it expires, so a crashed claimer costs one delayed
        # run rather than a permanently stuck watch.
        assert wid in {w["id"] for w in persistence.due_watches("2026-01-01T02:00:00+00:00")}
    finally:
        persistence.delete_watch(wid)


def test_a_failed_run_can_be_reclaimed_after_the_lease_expires():
    """The retry path: nothing rewrites next_run after a failed run, so the row
    the poller reads must already carry the lease — otherwise the second claim
    compares a stale next_run and the watch is wedged for good."""
    import uuid
    import persistence

    wid = str(uuid.uuid4())
    due_at = "2026-01-01T00:00:00+00:00"
    lease = "2026-01-01T01:00:00+00:00"
    _seed_watch(persistence, wid, due_at)
    try:
        assert persistence.claim_watch(wid, due_at, lease) is True
        # run fails: next_run is never rewritten, the lease just expires
        again = [w for w in persistence.due_watches("2026-01-01T02:00:00+00:00")
                 if w["id"] == wid]
        assert again and again[0]["next_run"] == lease
        assert persistence.claim_watch(
            wid, again[0]["next_run"], "2026-01-01T03:00:00+00:00") is True
    finally:
        persistence.delete_watch(wid)
