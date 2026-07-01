"""Shared SSE bridge for streaming LLM output, used by routes/query.py and routes/chat.py."""

import asyncio
import json
import threading

from fastapi.concurrency import run_in_threadpool

import lang_utils
import llm_client
import rag
import translation


async def sse_stream(prompt: str, metadatas: list, language: str, strategy: str = "A",
                      max_tokens: int = None, query_id: str = None):
    """Async SSE generator: bridges sync llm_generate_stream via asyncio.Queue.

    Strategy B + Indic target language: buffers all chunks, translates the full
    English answer, then emits a single translated chunk before the done event.
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
            for chunk in llm_client.llm_generate_stream(prompt, max_tokens):
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
    try:
        while True:
            kind, data = await q.get()
            if kind == "chunk":
                full_answer.append(data)
                if not needs_translation:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': data})}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': data})}\n\n"
                yield "data: [DONE]\n\n"
                return
            else:  # done
                break

        assembled = "".join(full_answer)
        if needs_translation and assembled:
            try:
                translated = await run_in_threadpool(translation.translate_from_english, assembled, language)
                yield f"data: {json.dumps({'type': 'chunk', 'text': translated})}\n\n"
            except Exception:
                yield f"data: {json.dumps({'type': 'chunk', 'text': assembled})}\n\n"  # fall back to English

        citations = rag.extract_citations(assembled, metadatas)
        yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'language': language, 'query_id': query_id})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        stop_event.set()
