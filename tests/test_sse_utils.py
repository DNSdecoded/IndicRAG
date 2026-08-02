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
