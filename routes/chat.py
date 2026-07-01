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
from deps import limiter, verify_api_key, _get_or_create_session, _append_session_messages
from routes.query import Citation
from sse_utils import sse_stream

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for a single chat turn."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message in any language")
    session_id: Optional[str] = Field(None, description="Existing session ID; omit to start a new conversation")
    strategy: str = Field("A", description="Strategy: 'A' for multilingual LLM, 'B' for English + translation")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in ("A", "B"):
            raise ValueError("Strategy must be 'A' or 'B'")
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


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    authenticated: bool = Depends(verify_api_key),
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

    session_id, messages = _get_or_create_session(body.session_id)
    turn_index = len(messages) // 2  # number of completed user+assistant pairs

    full_messages = list(messages) + [{"role": "user", "content": body.message}]

    try:
        result = await run_in_threadpool(
            rag.answer_with_history,
            messages=full_messages,
            strategy=body.strategy,
            top_k=top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": str(e), "code": "VALIDATION_ERROR"})
    except Exception as e:
        logger.error(f"Error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"})

    _append_session_messages(session_id, body.message, result["answer"])

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
    )


@router.post("/chat/stream", tags=["Chat"])
@limiter.limit("30/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """Stream a multi-turn chat answer as Server-Sent Events."""
    top_k = body.top_k
    if top_k is not None:
        top_k = max(1, min(top_k, 20))

    session_id, messages = _get_or_create_session(body.session_id)
    full_messages = list(messages) + [{"role": "user", "content": body.message}]

    prepared = await run_in_threadpool(rag.prepare_chat_for_stream, full_messages, body.strategy, top_k)
    query_id = str(uuid.uuid4())

    if prepared["chunks_used"] == 0:
        async def _no_docs():
            yield f"data: {json.dumps({'type': 'chunk', 'text': prepared['no_docs_msg']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'citations': [], 'language': prepared['detected_lang'], 'session_id': session_id, 'query_id': query_id})}\n\n"
            yield "data: [DONE]\n\n"
        _append_session_messages(session_id, body.message, prepared["no_docs_msg"])
        return StreamingResponse(_no_docs(), media_type="text/event-stream")

    async def _stream_and_save():
        full_answer: list[str] = []
        hit_error = False
        async for event in sse_stream(prepared["prompt"], prepared["metadatas"], prepared["detected_lang"],
                                       strategy=body.strategy, query_id=query_id):
            if event.startswith('data: {"type": "error"'):
                hit_error = True
            if event.startswith('data: {"type": "done"'):
                payload = json.loads(event[6:])
                payload["session_id"] = session_id
                yield f"data: {json.dumps(payload)}\n\n"
            else:
                if event.startswith('data: {"type": "chunk"'):
                    try:
                        full_answer.append(json.loads(event[6:])["text"])
                    except Exception:
                        pass
                yield event
        if not hit_error:
            _append_session_messages(session_id, body.message, "".join(full_answer))

    return StreamingResponse(_stream_and_save(), media_type="text/event-stream")


@router.delete("/chat/{session_id}", tags=["Chat"])
async def delete_session(
    session_id: str,
    authenticated: bool = Depends(verify_api_key),
):
    """Delete a chat session and its history."""
    import persistence
    from deps import _sessions, _sessions_lock
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found.")
        del _sessions[session_id]
        persistence.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
