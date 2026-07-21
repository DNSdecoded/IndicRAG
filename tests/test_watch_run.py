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


def test_run_due_watches_runs_each_due_and_survives_failures(monkeypatch):
    ran = []
    monkeypatch.setattr(watch_runner.persistence, "due_watches",
                        lambda now: [{"id": "a"}, {"id": "b"}, {"id": "c"}])

    def _fake_run(wid):
        if wid == "b":
            raise RuntimeError("boom")  # one failure must not abort the sweep
        ran.append(wid)

    monkeypatch.setattr(watch_runner, "run_watch", _fake_run)

    count = asyncio.run(watch_runner.run_due_watches())

    assert count == 3            # all due watches attempted
    assert ran == ["a", "c"]     # b failed but a and c still ran
