"""Shared SSE bridge for streaming LLM output, used by routes/query.py and routes/chat.py."""

import asyncio
import json
import logging
import threading

from fastapi.concurrency import run_in_threadpool

import config
import lang_utils
import llm_client
import rag
import translation

logger = logging.getLogger(__name__)

# Bounds how many generations can be producing chunks at once. Acquired before the
# producer thread starts and released when it finishes, so a burst of streaming
# clients queues here rather than as an unbounded pile of threads each holding a
# provider connection open.
_producer_slots = threading.Semaphore(config.SSE_MAX_PRODUCERS)

# Distinct from providers.base.TRUNCATION_NOTE, which means "hit the token limit".
# This one means the connection died mid-generation, so the answer stops wherever
# the last chunk landed — usually mid-sentence.
INTERRUPTED_NOTE = (
    "\n\n*[Answer incomplete — the connection to the model dropped mid-response. "
    "The sources below cover only what was generated.]*"
)


async def sse_stream(prompt: str, metadatas: list, language: str, strategy: str = "A",
                      max_tokens: int = None, query_id: str = None,
                      model: str = None, provider: str = None,
                      visible_chunks: int = None, degraded: str = None):
    """Async SSE generator: bridges sync llm_generate_stream via asyncio.Queue.

    Strategy B + Indic target language: buffers all chunks, translates the full
    English answer, then emits a single translated chunk before the done event.

    model/provider select the LLM (allowlist-validated upstream); omit for default.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=128)
    loop = asyncio.get_running_loop()  # fix: get_event_loop() deprecated in async contexts (Python 3.10+)
    stop_event = threading.Event()

    dropped = 0
    # Producer-owned, and the ONLY complete copy of the answer. The consumer sees
    # whatever survived the queue; anything dropped for a slow reader is still
    # text the model generated, and the done event has to carry it.
    produced: list[str] = []
    produced_lock = threading.Lock()

    def _enqueue_terminal(item):
        """Block until the queue has space. Only for error/done, which must land."""
        fut = asyncio.run_coroutine_threadsafe(q.put(item), loop)
        fut.result(timeout=30)

    def _offer_chunk(item):
        """Hand over a chunk without ever blocking the producer.

        A slow client used to stall generation itself: the producer waited up to
        30s per chunk on a full queue, holding a provider connection open for
        minutes because the reader was not draining. Chunks are the one event
        that can be sacrificed — the done event carries the complete, compacted
        answer and the client re-renders from it — so a full queue drops its
        oldest chunk rather than stopping the generation that fills it.
        """
        nonlocal dropped
        with produced_lock:
            produced.append(item[1])

        def _put():
            nonlocal dropped
            if q.full():
                try:
                    q.get_nowait()
                    dropped += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                dropped += 1

        loop.call_soon_threadsafe(_put)

    def _run():
        try:
            for chunk in llm_client.llm_generate_stream(prompt, max_tokens, model=model, provider=provider):
                if stop_event.is_set():
                    break
                _offer_chunk(("chunk", chunk))
        except Exception as exc:
            try:
                _enqueue_terminal(("error", str(exc)))
            except Exception:
                logger.warning("Could not deliver the stream error event", exc_info=True)
        finally:
            # Nested, because delivery can raise on its own (a 30s timeout, or a
            # loop already closed by a disconnect). A slot leaked here is
            # permanent: SSE_MAX_PRODUCERS shrinks by one for the process.
            try:
                _enqueue_terminal(("done", None))
            except Exception:
                logger.warning("Could not deliver the stream done event", exc_info=True)
            finally:
                _producer_slots.release()

    # Off-loop: a blocking semaphore wait here would stall every other request
    # on this worker for up to ADMISSION_WAIT_S, which is the opposite of what
    # bounding producers is for.
    got_slot = await asyncio.to_thread(
        _producer_slots.acquire, True, config.ADMISSION_WAIT_S)
    if not got_slot:
        # Same shape as any other stream error, so the client renders it
        # instead of hanging on a connection that will never produce.
        payload = json.dumps({'type': 'error',
                              'message': 'Server is at streaming capacity. Retry shortly.'})
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"
        return

    threading.Thread(target=_run, daemon=True).start()

    # ponytail: buffer when translation needed, stream otherwise
    needs_translation = strategy == "B" and language != "en" and lang_utils.is_indic_language(language)
    full_answer: list[str] = []
    interrupted = False  # stream died partway, but there is text worth keeping
    try:
        while True:
            kind, data = await q.get()
            if kind == "chunk":
                full_answer.append(data)
                if not needs_translation:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': data})}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': data})}\n\n"
                # Don't discard what already streamed. A stream that dies partway
                # (dropped connection, provider hiccup) used to return here, so the
                # user kept the partial answer on screen but lost every citation
                # with it. Fall through to the done event when there is text left
                # to attribute; only a completely empty answer stops here.
                if not full_answer:
                    yield "data: [DONE]\n\n"
                    return
                # Say so in the answer itself. A stream cut off mid-sentence
                # otherwise reads as a complete answer once the error toast is
                # gone — and it arrives with citations, which makes it look
                # more finished than it is.
                interrupted = True
                break
            else:  # done
                break

        if dropped:
            # The final answer is unaffected — the done event below carries the
            # whole compacted text — but the live typing effect skipped ahead.
            logger.info("SSE backpressure: %d chunk event(s) dropped for a slow client",
                        dropped)

        # produced[], not full_answer[]: the queue is a delivery channel and may
        # have dropped chunks for a slow reader, but the answer that gets
        # compacted, cited, logged and stored must be the whole generation.
        with produced_lock:
            assembled = "".join(produced) if produced else "".join(full_answer)
        # Compact BEFORE translating, so the translated answer inherits the dense
        # numbering (same order rag.answer_question uses). Chunks already streamed
        # carry the raw markers — an answer citing papers 1 and 4 of 4 renders
        # "[1] … [4]" against a two-entry panel, and a marker past visible_chunks
        # has no source at all — so the done event carries the corrected answer and
        # the client re-renders from it. visible_chunks keeps a marker invented past
        # the prompt's truncation point from resolving to a paper never shown.
        compacted, citations = rag.compact_citations(
            assembled, metadatas, visible_chunks=visible_chunks)
        if interrupted:
            compacted += INTERRUPTED_NOTE

        if needs_translation and compacted:
            try:
                compacted = await run_in_threadpool(
                    translation.translate_from_english, compacted, language)
            except Exception:
                pass  # fall back to English
            yield f"data: {json.dumps({'type': 'chunk', 'text': compacted})}\n\n"

        # `degraded` rides on the done event so a streaming client learns the answer
        # came from a reduced pipeline — it has no other way to find out.
        yield f"data: {json.dumps({'type': 'done', 'answer': compacted, 'citations': citations, 'language': language, 'query_id': query_id, 'degraded': degraded})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        stop_event.set()
