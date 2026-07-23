"""Route: /agent/query."""

from datetime import datetime, timezone
from typing import List, Optional
import asyncio
import logging
import threading
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

import config
import persistence
import rag
from agent.state import AgentState
from agent.nodes.finalizer import citation_coverage
from deps import limiter, get_current_user, _get_or_create_session, _append_session_messages

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


@router.post("/agent/query", response_model=AgentQueryResponse, tags=["Agent"])
@limiter.limit("10/minute")
async def agent_query(
    request: Request,
    body: AgentQueryRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Answer a question using the agentic IndicRAG pipeline with reflexion loops.
    Supports the same 10+ languages as /query and /chat.
    """
    start_time = time.time()
    session_id, messages = _get_or_create_session(body.session_id, user_id)

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
        user_id=user_id,
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

    _append_session_messages(session_id, body.question, result["final_answer"])
    processing_time = time.time() - start_time

    logger.info(
        f"Agent query: lang={result['detected_language']} "
        f"reflexion={result['reflexion_count']} time={processing_time:.2f}s"
    )

    all_contexts = result.get("retrieved_contexts", [])
    final_answer = result["final_answer"]
    cited_titles: set[str] = set()
    title_to_num: dict[str, int] = {}
    try:
        metas = [{"title": c.get("title", "Unknown"), "section": c.get("section", "body")}
                 for c in all_contexts]
        # Same per-paper numbering the LLM saw in the context, so the source
        # panel's [N] matches [Cite:N] in the answer text.
        for num, meta in rag.citation_number_map(metas).items():
            title_to_num[(meta.get("title") or "Unknown").strip()] = num
        for cit in rag.extract_citations(final_answer, metas):
            cited_titles.add(cit["title"].strip())
    except Exception:
        pass  # fall through to dedup-only logic below

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
            query_id=query_id, question=body.question, answer=result["final_answer"],
            mode=f"agent_{body.strategy}", model=body.model or "default",
            language=result.get("detected_language", "en"),
            confidence=result.get("answer_confidence") or 0.0,
            coverage=citation_coverage(result["final_answer"]),
            created_at=datetime.now(timezone.utc).isoformat(), user_id=user_id,
        )
    except Exception:
        logger.warning("Failed to log query for feedback correlation", exc_info=True)

    return AgentQueryResponse(
        answer=result["final_answer"],
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
