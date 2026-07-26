"""Concurrency regression tests for embeddings.embed_query in-flight dedup.

embed_query coalesces concurrent identical queries so the CPU-bound encoder runs
once. These tests pin that contract, including the owner-failure path where a
waiter has to recover without deregistering another thread's in-flight entry.
"""

import threading
import time
from unittest.mock import patch

import numpy as np

import embeddings


def _clear_state():
    with embeddings._query_cache_lock:
        embeddings._query_cache.clear()
    with embeddings._in_flight_lock:
        embeddings._in_flight.clear()


def test_concurrent_identical_queries_embed_once():
    _clear_state()
    calls = []
    start = threading.Event()

    def fake_embed_texts(texts, **kwargs):
        calls.append(texts[0])
        start.wait(timeout=5)          # hold the owner so the others pile up
        return np.ones((1, 4), dtype=np.float32)

    results = []
    with patch.object(embeddings, "embed_texts", side_effect=fake_embed_texts):
        threads = [threading.Thread(target=lambda: results.append(embeddings.embed_query("same query")))
                   for _ in range(8)]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=10)

    assert len(calls) == 1, f"expected one encode, got {len(calls)}"
    assert len(results) == 8
    assert all(np.array_equal(r, results[0]) for r in results)
    _clear_state()


def test_waiter_recovers_when_owner_fails():
    """Owner raises → waiters must still return a value, and the in-flight entry
    must not be left dangling (which would block or duplicate later callers)."""
    _clear_state()
    attempts = []
    lock = threading.Lock()

    # Without this handshake the owner can fail before any other thread reaches
    # the in-flight event, so the test would pass without ever exercising the
    # recovery path. Count arrivals on the event and hold the owner until two
    # waiters are actually parked on it.
    waiters_parked = threading.Event()
    waiter_count = [0]

    class CountingEvent(threading.Event):
        def wait(self, timeout=None):
            with lock:
                waiter_count[0] += 1
                if waiter_count[0] >= 2:
                    waiters_parked.set()
            return super().wait(timeout)

    def flaky_embed_texts(texts, **kwargs):
        with lock:
            attempts.append(1)
            first = len(attempts) == 1
        if first:
            assert waiters_parked.wait(timeout=10), "waiters never reached the in-flight event"
            raise RuntimeError("owner encode failed")
        return np.full((1, 4), 7.0, dtype=np.float32)

    errors, values = [], []

    def call():
        try:
            values.append(embeddings.embed_query("flaky query"))
        except Exception as e:            # the owner surfaces its own failure
            errors.append(e)

    with patch.object(embeddings, "embed_texts", side_effect=flaky_embed_texts), \
            patch.object(threading, "Event", CountingEvent):
        owner = threading.Thread(target=call)
        owner.start()
        # Wait until the owner has registered, so the other two are guaranteed
        # to take the waiter branch rather than racing to own the compute.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with embeddings._in_flight_lock:
                if "flaky query" in embeddings._in_flight:
                    break
            time.sleep(0.005)
        else:
            raise AssertionError("owner never registered an in-flight entry")

        waiters = [threading.Thread(target=call) for _ in range(2)]
        for t in waiters:
            t.start()
        for t in [owner, *waiters]:
            t.join(timeout=15)

    assert waiter_count[0] >= 2, "recovery path was not exercised"
    assert len(errors) <= 1, "only the owning thread may propagate the failure"
    assert values, "waiters must recover with a computed embedding"
    with embeddings._in_flight_lock:
        assert "flaky query" not in embeddings._in_flight, "in-flight entry leaked"
    _clear_state()
