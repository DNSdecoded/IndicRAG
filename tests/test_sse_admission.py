import asyncio
import threading

import sse_utils


def test_cancelled_slot_wait_does_not_leak_a_slot(monkeypatch):
    """A client that disconnects while waiting for a producer slot must not
    leave a waiter behind that later takes a slot nothing will ever release."""
    slots = threading.Semaphore(0)
    monkeypatch.setattr(sse_utils, "_producer_slots", slots)

    async def _scenario():
        waiter = asyncio.create_task(sse_utils._acquire_producer_slot(5.0, poll_s=0.01))
        await asyncio.sleep(0.05)
        waiter.cancel()                 # the client disconnected mid-wait
        try:
            await waiter
        except asyncio.CancelledError:
            pass
        slots.release()                 # a running producer finishes
        await asyncio.sleep(0.05)       # give any orphaned waiter a chance to grab it

    asyncio.run(_scenario())
    assert slots.acquire(blocking=False), "the freed slot was taken by a cancelled waiter"


def test_slot_wait_times_out_without_holding_a_slot(monkeypatch):
    monkeypatch.setattr(sse_utils, "_producer_slots", threading.Semaphore(0))
    assert asyncio.run(sse_utils._acquire_producer_slot(0.03, poll_s=0.01)) is False
