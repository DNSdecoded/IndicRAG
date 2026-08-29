"""SSE streaming path: the done event must carry a citation-corrected answer.

Chunks go out as the model produces them, so they carry its raw [N] markers.
An answer citing papers 1 and 4 of 4 streams as "[1] ... [4]" while the panel
holds two entries, and a marker past the prompt's truncation point has no source
behind it at all. The done event therefore carries the compacted answer, and the
client re-renders from that.
"""

import asyncio
import json
from unittest.mock import patch

import sse_utils


def _drive(**kwargs):
    """Run sse_stream to completion, returning the parsed events."""
    async def _collect():
        out = []
        async for raw in sse_utils.sse_stream(**kwargs):
            payload = raw[len("data: "):].strip()
            if payload != "[DONE]":
                out.append(json.loads(payload))
        return out

    return asyncio.run(_collect())


_METADATAS = [
    {"title": "Paper A", "section": "intro"},
    {"title": "Paper B", "section": "body"},
    {"title": "Paper C", "section": "results"},
]


def test_done_event_carries_compacted_answer():
    answer = "reward [1] and bandwidth [3], plus invented [9]"

    with patch("llm_client.llm_generate_stream", return_value=iter([answer])):
        events = _drive(prompt="p", metadatas=_METADATAS, language="en",
                        query_id="q1", visible_chunks=3)

    chunks = [e for e in events if e["type"] == "chunk"]
    done = next(e for e in events if e["type"] == "done")

    # Streaming itself is untouched — the raw text still goes out live.
    assert "".join(c["text"] for c in chunks) == answer

    # ...but the done event carries the corrected answer: [3] renumbered to [2]
    # (Paper B was never cited) and the dangling [9] dropped.
    assert done["answer"] == "reward [1] and bandwidth [2], plus invented"
    assert [c["title"] for c in done["citations"]] == ["Paper A", "Paper C"]
    assert [c["number"] for c in done["citations"]] == ["1", "2"]


def test_partial_answer_keeps_its_citations_after_a_stream_error():
    """A stream that dies partway must still emit done.

    Observed with a dropped upstream connection (WinError 10054): the user kept
    the partial answer on screen but every citation vanished with it, because the
    error path returned before the done event.
    """
    def _die_midway():
        yield "grounded [1] and more"
        raise RuntimeError("[WinError 10054] connection forcibly closed")

    with patch("llm_client.llm_generate_stream", return_value=_die_midway()):
        events = _drive(prompt="p", metadatas=_METADATAS, language="en",
                        query_id="q3", visible_chunks=3)

    assert any(e["type"] == "error" for e in events), "the failure must still surface"
    done = next(e for e in events if e["type"] == "done")
    assert done["answer"].startswith("grounded [1] and more")
    assert [c["title"] for c in done["citations"]] == ["Paper A"]
    # ...and the answer must admit it is incomplete: with the error toast gone,
    # a mid-sentence answer arriving with citations reads as a finished one.
    assert done["answer"].endswith(sse_utils.INTERRUPTED_NOTE)


def test_empty_stream_error_stops_without_a_done_event():
    """Nothing salvageable — no answer text, so there is nothing to attribute."""
    def _die_immediately():
        raise RuntimeError("upstream refused")
        yield  # pragma: no cover - makes this a generator

    with patch("llm_client.llm_generate_stream", return_value=_die_immediately()):
        events = _drive(prompt="p", metadatas=_METADATAS, language="en",
                        query_id="q4", visible_chunks=3)

    assert [e["type"] for e in events] == ["error"]


def test_marker_past_visible_chunks_resolves_to_nothing():
    """A number invented past the prompt's truncation point must not resolve to a
    real paper — Paper C never reached the prompt when only 2 chunks were used."""
    with patch("llm_client.llm_generate_stream",
               return_value=iter(["grounded [1] invented [3]"])):
        events = _drive(prompt="p", metadatas=_METADATAS, language="en",
                        query_id="q2", visible_chunks=2)

    done = next(e for e in events if e["type"] == "done")
    assert [c["title"] for c in done["citations"]] == ["Paper A"]
    assert done["answer"] == "grounded [1] invented"


# ── backpressure (A5) ───────────────────────────────────────────────────────

def test_slow_client_never_stalls_the_producer(monkeypatch):
    """A slow reader used to block the producer thread for up to 30s per chunk,
    pinning a provider connection open for minutes. Chunks are droppable — the
    done event carries the whole compacted answer — so the generation runs on."""
    import asyncio

    import sse_utils

    produced = []

    def _many_chunks(prompt, max_tokens, model=None, provider=None):
        for i in range(500):          # far more than the queue holds
            produced.append(i)
            yield f"chunk-{i} "

    monkeypatch.setattr(sse_utils.llm_client, "llm_generate_stream", _many_chunks)
    monkeypatch.setattr(sse_utils.rag, "compact_citations",
                        lambda text, metas, visible_chunks=None: (text, []))

    async def _drain_slowly():
        events = []
        agen = sse_utils.sse_stream("p", [], "en")
        async for event in agen:
            events.append(event)
            # Let the producer run ahead while this consumer dawdles.
            await asyncio.sleep(0)
        return events

    events = asyncio.run(_drain_slowly())

    assert len(produced) == 500, "the producer must finish regardless of the reader"
    done = [e for e in events if '"type": "done"' in e or "'type': 'done'" in e]
    assert done, "a done event must still arrive"


def test_streaming_capacity_is_bounded(monkeypatch):
    """Every producer holds a provider connection; unbounded threads mean a burst
    of slow clients can pin the upstream API."""
    import asyncio
    import threading

    import sse_utils

    monkeypatch.setattr(sse_utils, "_producer_slots", threading.Semaphore(0))
    monkeypatch.setattr(sse_utils.config, "ADMISSION_WAIT_S", 0.01)

    async def _collect():
        return [e async for e in sse_utils.sse_stream("p", [], "en")]

    events = asyncio.run(_collect())

    assert any("streaming capacity" in e for e in events)
    assert events[-1] == "data: [DONE]\n\n"
