"""Route: /agent/query, /agent/stream."""

from datetime import datetime, timezone
from typing import List, Optional
import asyncio
import json
import logging
import threading
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import config
import persistence
import rag
from agent.state import AgentState
from agent.nodes.finalizer import citation_coverage
from deps import (
    limiter, verify_api_key, current_owner, admit_agent,
    _get_or_create_session, _append_session_messages, session_turn_lock,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_agent_graph = None
_agent_graph_lock = threading.Lock()


def _get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        with _agent_graph_lock:
            if _agent_graph is None:
                from agent.graph import build_agent_graph
                _agent_graph = build_agent_graph()
    return _agent_graph


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    strategy: str = Field("A")
    model: Optional[str] = None
    provider: Optional[str] = None

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v):
        if v not in ("A", "B"):
            raise ValueError("Strategy must be 'A' or 'B'")
        return v

    @field_validator("model")
    @classmethod
    def validate_model_allowlisted(cls, v):
        from routes.models import validate_model
        validate_model(v, None)
        return v


class AgentSource(BaseModel):
    number: int = 0  # per-paper citation number matching [Cite:N] in the answer
    title: str
    source: str = ""
    section: str = ""
    pdf_url: str = ""
    year: str = ""
    authors: str = ""
    citations: int = 0


class AgentQueryResponse(BaseModel):
    answer: str
    session_id: str
    language: str
    reflexion_iterations: int
    tool_calls: List[dict]
    sources: List[AgentSource]
    processing_time: float
    timestamp: str
    answer_confidence: Optional[float] = None
    abstained: bool = False
    query_id: str = ""


# What each graph node is actually doing, in the user's terms. Node names are an
# implementation detail; "tool_executor" means nothing to someone waiting for an
# answer. A node with no label here simply produces no progress line, so adding a
# node to the graph cannot leak a raw internal name into the UI.
_NODE_LABELS = {
    "query_planner": "Planning the query",
    "tool_selector": "Choosing tools",
    "tool_executor": "Searching the corpus",
    "answer_generator": "Writing the answer",
    "reflexion_evaluator": "Checking the answer against sources",
    "finalizer": "Finishing up",
}


def _sources_preview(contexts: list, limit: int = 8) -> list:
    """Distinct papers from retrieved contexts, for the mid-run evidence preview.

    Deliberately not the final citation list: numbering is only decided after the
    answer exists and its markers are compacted. This is "what was retrieved",
    shown early so the wait has visible content, and the done event replaces it
    with the authoritative cited set.
    """
    seen, out = set(), []
    for ctx in contexts or []:
        title = (ctx.get("title") or "").strip()
        if not title or title in seen or title in ("Unknown", "No results"):
            continue
        seen.add(title)
        out.append({
            "number": len(out) + 1,
            "title": title,
            "section": ctx.get("section", ""),
            "year": ctx.get("year", ""),
            "authors": ctx.get("authors", ""),
        })
        if len(out) >= limit:
            break
    return out


@router.post("/agent/query", response_model=AgentQueryResponse, tags=["Agent"])
@limiter.limit("10/minute")
async def agent_query(
    request: Request,
    body: AgentQueryRequest,
    authenticated: bool = Depends(verify_api_key),
    owner: Optional[str] = Depends(current_owner),
    _slot: None = Depends(admit_agent),
):
    """
    Answer a question using the agentic IndicRAG pipeline with reflexion loops.
    Supports the same 10+ languages as /query and /chat.
    """
    start_time = time.time()
    # One turn at a time per session: agent and chat traffic share the same
    # history, so a concurrent turn must not generate from a history that
    # omits the turn running beside it.
    async with session_turn_lock(body.session_id):
        try:
            session_id, messages = _get_or_create_session(body.session_id, owner)
        except PermissionError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Session '{body.session_id}' not found.")

        initial_state = AgentState(
            original_query=body.question,
            detected_language="",
            query_plan=[],
            tool_calls_requested=[],
            retrieved_contexts=[],
            draft_answer=None,
            final_answer=None,
            reflexion_count=0,
            reflexion_history=[],
            tool_calls_log=[],
            conversation_history=list(messages),
            session_id=session_id,
            strategy=body.strategy,
            start_time=time.monotonic(),
            requested_model=body.model,
            requested_provider=body.provider,
        )

        try:
            result = await asyncio.wait_for(
                run_in_threadpool(_get_agent_graph().invoke, initial_state),
                timeout=float(config.AGENT_TIMEOUT),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent timed out after %ds: strategy=%s, question_len=%d",
                config.AGENT_TIMEOUT, body.strategy, len(body.question),
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"error": "Agent pipeline timed out. Try a simpler query or use Standard RAG mode.", "code": "AGENT_TIMEOUT"},
            )
        except Exception as e:
            err_str = str(e)
            is_llm_unavailable = (
                "503" in err_str or "429" in err_str
                or "UNAVAILABLE" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "high demand" in err_str.lower()
                or "unreachable" in err_str.lower()
            )
            if is_llm_unavailable:
                logger.warning(f"LLM service unavailable for agent query: {err_str[:200]}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "The AI model is temporarily unavailable due to high demand. "
                                 "Please try again in a few seconds.",
                        "code": "LLM_UNAVAILABLE",
                    },
                )
            logger.error(f"Agent error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Agent pipeline failed. Please try again later.", "code": "AGENT_ERROR"},
            )

        all_contexts = result.get("retrieved_contexts", [])
        final_answer = result["final_answer"]
        cited_titles: set[str] = set()
        title_to_num: dict[str, int] = {}
        cits: list = []
        try:
            metas = [{"title": c.get("title", "Unknown"), "section": c.get("section", "body")}
                     for c in all_contexts]
            # The context numbers every retrieved paper, but the panel below keeps
            # only the cited ones — so an answer citing papers 1 and 4 of 4 read
            # "[1] … [4]" beside a two-entry panel. compact_citations renumbers the
            # answer's markers and the citations together to a dense 1..M.
            final_answer, cits = rag.compact_citations(
                final_answer, metas, visible_chunks=result.get("context_chunks_used"))
            for cit in cits:
                title = cit["title"].strip()
                cited_titles.add(title)
                title_to_num[title] = int(cit["number"])
        except Exception:
            pass  # fall through to dedup-only logic below

        _append_session_messages(session_id, body.question, final_answer, owner, cits)
    processing_time = time.time() - start_time

    logger.info(
        f"Agent query: lang={result['detected_language']} "
        f"reflexion={result['reflexion_count']} time={processing_time:.2f}s"
    )

    seen_titles: set[str] = set()
    sources = []
    for ctx in all_contexts:
        title = ctx.get("title", "").strip()
        if not title or title in seen_titles or title in ("Unknown", "No results"):
            continue
        if cited_titles and title not in cited_titles:
            continue
        seen_titles.add(title)
        sources.append(AgentSource(
            number=title_to_num.get(title, len(sources) + 1),
            title=title,
            source=ctx.get("source", ""),
            section=ctx.get("section", ""),
            pdf_url=ctx.get("pdf_url", ""),
            year=ctx.get("year", ""),
            authors=ctx.get("authors", ""),
            citations=ctx.get("citations", 0),
        ))
    # Order the panel by citation number so [1],[2],... read in sequence.
    sources.sort(key=lambda s: s.number)

    query_id = str(uuid.uuid4())
    try:
        persistence.log_query(
            query_id=query_id, question=body.question, answer=final_answer,
            mode=f"agent_{body.strategy}", model=body.model or "default",
            language=result.get("detected_language", "en"),
            confidence=result.get("answer_confidence") or 0.0,
            coverage=citation_coverage(final_answer),
            created_at=datetime.now(timezone.utc).isoformat(),
            owner=owner,
        )
    except Exception:
        logger.warning("Failed to log query for feedback correlation", exc_info=True)

    return AgentQueryResponse(
        answer=final_answer,
        session_id=session_id,
        language=result.get("detected_language", "en"),
        reflexion_iterations=result.get("reflexion_count", 0),
        tool_calls=result.get("tool_calls_log", []),
        sources=sources,
        processing_time=processing_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
        answer_confidence=result.get("answer_confidence"),
        abstained=result.get("abstained", False),
        query_id=query_id,
    )


@router.post("/agent/stream", tags=["Agent"])
@limiter.limit("10/minute")
async def agent_stream(
    request: Request,
    body: AgentQueryRequest,
    authenticated: bool = Depends(verify_api_key),
    owner: Optional[str] = Depends(current_owner),
    _slot: None = Depends(admit_agent),
):
    """Stream agentic query results as Server-Sent Events.

    Runs the full agent pipeline, then streams the final answer in chunks
    along with metadata events for sources, tool calls, and timing.
    """
    start_time = time.time()
    # Same turn lock as the non-streaming path, acquired manually: the turn ends
    # when the generator appends the answer, so the generator's `finally` is what
    # releases it and a client disconnect mid-stream cannot strand it.
    lock = session_turn_lock(body.session_id)
    await lock.acquire()
    try:
        session_id, messages = _get_or_create_session(body.session_id, owner)
    except PermissionError:
        lock.release()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Session '{body.session_id}' not found.")
    except BaseException:
        lock.release()
        raise

    initial_state = AgentState(
        original_query=body.question,
        detected_language="",
        query_plan=[],
        tool_calls_requested=[],
        retrieved_contexts=[],
        draft_answer=None,
        final_answer=None,
        reflexion_count=0,
        reflexion_history=[],
        tool_calls_log=[],
        conversation_history=list(messages),
        session_id=session_id,
        strategy=body.strategy,
        start_time=time.monotonic(),
        requested_model=body.model,
        requested_provider=body.provider,
    )

    async def _run_and_stream():
        stop = threading.Event()   # read by the graph worker below
        try:
            # --- Phase 1: run the agent pipeline, reporting progress as it goes ---
            #
            # This used to be a single invoke() behind run_in_threadpool: one
            # "Running agent pipeline…" line, then total silence for as long as the
            # run took (measured: 145s, of which ~85s showed the client nothing but a
            # caret). A wait that reports nothing is indistinguishable from a hang.
            #
            # graph.stream(stream_mode="updates") yields {node: delta} as each node
            # finishes, which is real progress rather than a spinner. AgentState is a
            # plain TypedDict with no reducers, so replaying the deltas with
            # dict.update() reproduces exactly what invoke() would have returned.
            q: asyncio.Queue = asyncio.Queue(maxsize=64)
            loop = asyncio.get_running_loop()

            def _enqueue(item):
                # Block until there is room, so a terminal event is never dropped.
                asyncio.run_coroutine_threadsafe(q.put(item), loop).result(timeout=30)

            def _emit_token(text: str) -> None:
                """Called from the graph worker for every token of the first draft.

                Non-blocking by design: a token is the one event worth dropping —
                the done event carries the complete, citation-corrected answer and
                the client re-renders from it — so a slow reader must never stall
                the generation that feeds it.
                """
                if stop.is_set():
                    raise RuntimeError("client gone")
                try:
                    loop.call_soon_threadsafe(q.put_nowait, ("token", text, None))
                except RuntimeError:
                    raise RuntimeError("client gone")

            initial_state["token_sink"] = _emit_token

            def _run_graph():
                merged = dict(initial_state)
                try:
                    for update in _get_agent_graph().stream(initial_state, stream_mode="updates"):
                        if stop.is_set():
                            return
                        for node, delta in (update or {}).items():
                            if isinstance(delta, dict):
                                merged.update(delta)
                            _enqueue(("node", node, dict(merged)))
                    _enqueue(("done", None, merged))
                except Exception as exc:  # mirrors the old except-Exception path
                    _enqueue(("error", str(exc)[:200], None))

            threading.Thread(target=_run_graph, daemon=True, name="agent-graph").start()

            result = None
            sources_sent = False
            streamed_tokens = False
            deadline = time.monotonic() + float(config.AGENT_TIMEOUT)
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    kind, payload, state = await asyncio.wait_for(q.get(), timeout=remaining)

                    if kind == "error":
                        yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    if kind == "token":
                        # Raw [N] markers as the model wrote them: numbering cannot
                        # be compacted until the answer is complete, so the done
                        # event ships the corrected text and the client re-renders.
                        # Same contract /query/stream already uses.
                        yield f"data: {json.dumps({'type': 'chunk', 'text': payload})}\n\n"
                        streamed_tokens = True
                        continue
                    if kind == "done":
                        result = state
                        break

                    label = _NODE_LABELS.get(payload)
                    if label:
                        yield f"data: {json.dumps({'type': 'progress', 'node': payload, 'text': label})}\n\n"

                    # Retrieval finishes long before the answer is written, so send the
                    # sources as soon as they exist and let the evidence rail fill during
                    # generation instead of staying blank for the whole wait.
                    if not sources_sent and state and state.get("retrieved_contexts"):
                        preview = _sources_preview(state["retrieved_contexts"])
                        if preview:
                            sources_sent = True
                            yield f"data: {json.dumps({'type': 'sources_preview', 'sources': preview})}\n\n"
            except asyncio.TimeoutError:
                err = json.dumps({"type": "error", "message": "Agent pipeline timed out."})
                yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"
                return

            processing_time = time.time() - start_time

            # --- Phase 2: stream tool calls as thinking events ---
            for tc in result.get("tool_calls_log", []):
                tool_msg = json.dumps({
                    "type": "thinking",
                    "text": f"Tool: {tc.get('tool', '?')} ({tc.get('latency_ms', 0):.0f}ms)",
                })
                yield f"data: {tool_msg}\n\n"

            # --- Phase 3: resolve citations BEFORE streaming ---
            # The markers have to be compacted before the first chunk goes out, or
            # the client renders [1],[4] against a two-entry panel.
            final_answer = result["final_answer"] or ""
            all_contexts = result.get("retrieved_contexts", [])
            cited_titles: set = set()
            title_to_num: dict = {}
            cits: list = []
            try:
                metas = [{"title": c.get("title", "Unknown"), "section": c.get("section", "body")}
                         for c in all_contexts]
                final_answer, cits = rag.compact_citations(
                    final_answer, metas, visible_chunks=result.get("context_chunks_used"))
                for cit in cits:
                    title = cit["title"].strip()
                    cited_titles.add(title)
                    title_to_num[title] = int(cit["number"])
            except Exception:
                pass

            _append_session_messages(session_id, body.question, final_answer, owner, cits)

            # --- Phase 3b: fallback for answers that never streamed ---
            # A regenerated draft, a non-streaming provider path, or a client that
            # connected after generation started all land here. When tokens DID
            # stream, re-sending the text would duplicate the whole answer.
            if not streamed_tokens:
                chunk_size = 80  # characters per SSE chunk
                for i in range(0, len(final_answer), chunk_size):
                    chunk = final_answer[i:i + chunk_size]
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

            # --- Phase 4: build sources list ---
            seen_titles: set = set()
            sources = []
            for ctx in all_contexts:
                title = ctx.get("title", "").strip()
                if not title or title in seen_titles or title in ("Unknown", "No results"):
                    continue
                if cited_titles and title not in cited_titles:
                    continue
                seen_titles.add(title)
                sources.append({
                    "number": title_to_num.get(title, len(sources) + 1),
                    "title": title,
                    "source": ctx.get("source", ""),
                    "section": ctx.get("section", ""),
                    "pdf_url": ctx.get("pdf_url", ""),
                    "year": ctx.get("year", ""),
                    "authors": ctx.get("authors", ""),
                })
            sources.sort(key=lambda s: s["number"])

            # --- Phase 5: done event with full metadata ---
            query_id = str(uuid.uuid4())
            try:
                persistence.log_query(
                    query_id=query_id, question=body.question, answer=final_answer,
                    mode=f"agent_{body.strategy}", model=body.model or "default",
                    language=result.get("detected_language", "en"),
                    confidence=result.get("answer_confidence") or 0.0,
                    coverage=citation_coverage(final_answer),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    owner=owner,
                )
            except Exception:
                pass

            done_payload = {
                "type": "done",
                # The citation-corrected text. Tokens went out with the model's own
                # [N] numbering, which can have gaps; the client renders this in
                # their place once the answer is complete.
                "answer": final_answer,
                "citations": sources,
                "language": result.get("detected_language", "en"),
                "session_id": session_id,
                "query_id": query_id,
                "processing_time": processing_time,
                "model": body.model or "default",
                "reflexion_iterations": result.get("reflexion_count", 0),
                "tool_calls": result.get("tool_calls_log", []),
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # The graph worker outlives this generator otherwise — on a timeout
            # or a client disconnect it would keep burning LLM calls with nobody
            # left to read them.
            stop.set()
            lock.release()

    return StreamingResponse(_run_and_stream(), media_type="text/event-stream")
