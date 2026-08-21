"""Routes: /chat, /chat/stream, /chat/{session_id}."""

from datetime import datetime, timezone
from typing import List, Optional
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import rag
from deps import (
    limiter, verify_api_key, current_owner, owns_session, session_turn_lock,
    _get_or_create_session, _append_session_messages,
)
from routes.query import Citation, build_paper_filter, build_tags_filter, combine_filters
from sse_utils import sse_stream

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for a single chat turn."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message in any language")
    session_id: Optional[str] = Field(None, description="Existing session ID; omit to start a new conversation")
    strategy: str = Field("A", description="Strategy: 'A' for multilingual LLM, 'B' for English + translation")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    paper_ids: Optional[List[str]] = Field(
        None, description="Restrict retrieval to these paper_ids (PDF filename stems). Omit for whole corpus."
    )
    tags: Optional[str] = Field(None, description="Comma-separated tags to filter retrieval.")
    model: Optional[str] = Field(None, description="LLM model id from the /models allowlist. Omit for default.")
    provider: Optional[str] = Field(None, description="LLM provider override (gemini|openrouter). Usually inferred from model.")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in ("A", "B"):
            raise ValueError("Strategy must be 'A' or 'B'")
        return v

    @field_validator("model")
    @classmethod
    def validate_model_allowlisted(cls, v):
        from routes.models import validate_model
        validate_model(v, None)
        return v


class ChatResponse(BaseModel):
    """Response model for a single chat turn."""
    query_id: str
    session_id: str
    turn_index: int
    answer: str
    language: str
    language_name: str
    chunks_used: int
    citations: List[Citation]
    processing_time: float
    timestamp: str
    # 'sparse_only' when the dense retrieval leg was unavailable — see QueryResponse.
    degraded: Optional[str] = None


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    authenticated: bool = Depends(verify_api_key),
    owner: Optional[str] = Depends(current_owner),
):
    """
    Send a message in a multi-turn conversation.

    Pass ``session_id`` from a previous response to continue that conversation.
    Omit it (or pass ``null``) to start a fresh session.
    History is kept server-side; only the new ``message`` is required each turn.
    """
    start_time = time.time()

    top_k = body.top_k
    if top_k is not None:
        top_k = max(1, min(top_k, 20))

    # The lock spans read-history -> generate -> append: concurrent turns on one
    # session must not each answer from a history that omits the other's turn.
    async with session_turn_lock(body.session_id or "new"):
        try:
            session_id, messages = _get_or_create_session(body.session_id, owner)
        except PermissionError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Session '{body.session_id}' not found.")
        turn_index = len(messages) // 2  # number of completed user+assistant pairs

        full_messages = list(messages) + [{"role": "user", "content": body.message}]

        try:
            result = await run_in_threadpool(
                rag.answer_with_history,
                messages=full_messages,
                strategy=body.strategy,
                top_k=top_k,
                filter_dict=combine_filters(build_paper_filter(body.paper_ids), build_tags_filter(body.tags)),
                model=body.model,
                provider=body.provider,
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": str(e), "code": "VALIDATION_ERROR"})
        except Exception as e:
            logger.error(f"Error in /chat: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"})

        _append_session_messages(session_id, body.message, result["answer"], owner,
                                 result.get("citations"))

    processing_time = time.time() - start_time
    logger.info(
        f"Chat turn {turn_index + 1} session={session_id[:8]}… "
        f"lang={result['language']} chunks={result['chunks_used']} time={processing_time:.2f}s"
    )

    return ChatResponse(
        query_id=str(uuid.uuid4()),
        session_id=session_id,
        turn_index=turn_index + 1,
        answer=result["answer"],
        language=result["language"],
        language_name=result["language_name"],
        chunks_used=result["chunks_used"],
        citations=[Citation(**c) for c in result["citations"]],
        processing_time=processing_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
        degraded=result.get("degraded"),
    )


@router.post("/chat/stream", tags=["Chat"])
@limiter.limit("30/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    authenticated: bool = Depends(verify_api_key),
    owner: Optional[str] = Depends(current_owner),
):
    """Stream a multi-turn chat answer as Server-Sent Events."""
    top_k = body.top_k
    if top_k is not None:
        top_k = max(1, min(top_k, 20))

    # Same turn lock as POST /chat, but it has to be acquired manually: the turn
    # is not finished when this function returns — it ends when the generator
    # below appends the answer. The generator's `finally` is what releases it,
    # so a client disconnect mid-stream can't strand the lock.
    lock = session_turn_lock(body.session_id or "new")
    await lock.acquire()
    try:
        try:
            session_id, messages = _get_or_create_session(body.session_id, owner)
        except PermissionError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Session '{body.session_id}' not found.")
        full_messages = list(messages) + [{"role": "user", "content": body.message}]

        prepared = await run_in_threadpool(rag.prepare_chat_for_stream, full_messages, body.strategy, top_k,
                                           combine_filters(build_paper_filter(body.paper_ids), build_tags_filter(body.tags)))
        query_id = str(uuid.uuid4())

        if prepared["chunks_used"] == 0:
            async def _no_docs():
                yield f"data: {json.dumps({'type': 'chunk', 'text': prepared['no_docs_msg']})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'citations': [], 'language': prepared['detected_lang'], 'session_id': session_id, 'query_id': query_id})}\n\n"
                yield "data: [DONE]\n\n"
            # The turn is already complete on this branch (nothing streams from the
            # model), so release here rather than handing the lock to a generator
            # that has no answer left to append.
            _append_session_messages(session_id, body.message, prepared["no_docs_msg"], owner)
            lock.release()
            return StreamingResponse(_no_docs(), media_type="text/event-stream")
    except BaseException:
        lock.release()
        raise

    async def _stream_and_save():
        full_answer: list[str] = []
        final_answer: str | None = None  # compacted answer from the done event
        final_cites: list = []           # its citations, stored with the turn
        hit_error = False
        try:
            async for event in sse_stream(prepared["prompt"], prepared["metadatas"], prepared["detected_lang"],
                                           strategy=body.strategy, query_id=query_id,
                                           model=body.model, provider=body.provider,
                                           visible_chunks=prepared["chunks_used"],
                                           degraded=prepared.get("degraded")):
                if event.startswith('data: {"type": "error"'):
                    hit_error = True
                if event.startswith('data: {"type": "done"'):
                    payload = json.loads(event[6:])
                    payload["session_id"] = session_id
                    final_answer = payload.get("answer")
                    # The done event already carries the resolved citations; keep
                    # them so reopening this turn from history can redraw sources.
                    final_cites = payload.get("citations") or []
                    yield f"data: {json.dumps(payload)}\n\n"
                else:
                    if event.startswith('data: {"type": "chunk"'):
                        try:
                            full_answer.append(json.loads(event[6:])["text"])
                        except Exception:
                            pass
                    yield event
            if not hit_error:
                # Persist the compacted answer, not the raw streamed chunks — otherwise
                # the follow-up turns inherit gapped and dangling [N] markers.
                _append_session_messages(
                    session_id, body.message, final_answer or "".join(full_answer), owner,
                    final_cites)
        finally:
            # Runs on normal completion, on an error, and on GeneratorExit when the
            # client disconnects mid-stream — the turn is over in all three cases.
            lock.release()

    return StreamingResponse(_stream_and_save(), media_type="text/event-stream")


class ChatSessionSummary(BaseModel):
    """One row in the chat-history list."""
    session_id: str
    preview: str
    turns: int
    created_at: str
    updated_at: str


class ChatHistoryResponse(BaseModel):
    """Full message history for a single session."""
    session_id: str
    messages: List[dict]
    created_at: str
    updated_at: str


@router.get("/chat", response_model=List[ChatSessionSummary], tags=["Chat"])
async def list_sessions(authenticated: bool = Depends(verify_api_key),
                        owner: Optional[str] = Depends(current_owner)):
    """List the caller's saved chat sessions, most-recent first.

    Scoped by API-key fingerprint. This listing used to be global, so any valid
    key could read every other user's conversation previews.
    """
    from deps import _sessions, _sessions_lock, _evict_stale_sessions
    summaries: List[ChatSessionSummary] = []
    with _sessions_lock:
        _evict_stale_sessions()
        for sid, s in _sessions.items():
            if not owns_session(s, owner):
                continue
            msgs = s.get("messages", [])
            if not msgs:  # skip empty sessions (created but never used)
                continue
            first_user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            summaries.append(ChatSessionSummary(
                session_id=sid,
                preview=first_user[:120],
                turns=len(msgs) // 2,
                created_at=s.get("created_at", ""),
                updated_at=s.get("updated_at", ""),
            ))
    summaries.sort(key=lambda x: x.updated_at, reverse=True)
    return summaries


@router.get("/chat/{session_id}", response_model=ChatHistoryResponse, tags=["Chat"])
async def get_session_history(session_id: str, authenticated: bool = Depends(verify_api_key),
                              owner: Optional[str] = Depends(current_owner)):
    """Return a session's full message history so the UI can reopen the conversation.

    Another key's session reads as 404, not 403 — a 403 would confirm the id exists.
    """
    from deps import _sessions, _sessions_lock
    with _sessions_lock:
        s = _sessions.get(session_id)
        if not owns_session(s, owner):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found.")
        return ChatHistoryResponse(
            session_id=session_id,
            messages=list(s.get("messages", [])),
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
        )


@router.delete("/chat/{session_id}", tags=["Chat"])
async def delete_session(
    session_id: str,
    authenticated: bool = Depends(verify_api_key),
    owner: Optional[str] = Depends(current_owner),
):
    """Delete a chat session and its history. Only the owning key may delete."""
    import persistence
    from deps import _sessions, _sessions_lock
    with _sessions_lock:
        if not owns_session(_sessions.get(session_id), owner):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found.")
        del _sessions[session_id]
        persistence.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
