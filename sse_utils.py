"""Shared SSE bridge for streaming LLM output, used by routes/query.py and routes/chat.py."""

import asyncio
import json
import threading

from fastapi.concurrency import run_in_threadpool

import lang_utils
import llm_client
import rag
import translation

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

    def _enqueue(item):
        """Block until queue has space, ensuring terminal events are never dropped."""
        fut = asyncio.run_coroutine_threadsafe(q.put(item), loop)
        fut.result(timeout=30)

    def _run():
        try:
            for chunk in llm_client.llm_generate_stream(prompt, max_tokens, model=model, provider=provider):
                if stop_event.is_set():
                    break
                _enqueue(("chunk", chunk))
        except Exception as exc:
            _enqueue(("error", str(exc)))
        finally:
            _enqueue(("done", None))

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

        assembled = "".join(full_answer)
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
