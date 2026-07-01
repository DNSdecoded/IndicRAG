"""Routes: /search, /papers, /papers/{id}, /purge/*, /stats, /cache/*."""

from datetime import datetime, timezone
from typing import List, Optional
import logging
import re
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

import config
import lang_utils
import rag
from deps import verify_api_key, verify_admin_key

logger = logging.getLogger(__name__)
router = APIRouter()


class SearchRequest(BaseModel):
    """Request model for retrieval-only search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: Optional[int] = Field(10, ge=1, le=30, description="Number of results")
    source: str = Field("corpus", description="Source: 'corpus' for indexed docs, 'web' for Tavily web search, 'both' for hybrid")

    @field_validator("source")
    @classmethod
    def validate_source(cls, v):
        if v not in ("corpus", "web", "both"):
            raise ValueError("Source must be 'corpus', 'web', or 'both'")
        return v


class SearchResult(BaseModel):
    """A single search result."""
    text: str
    title: str = ""
    source: str = ""
    section: str = ""
    score: float = 0.0


class SearchResponse(BaseModel):
    """Response model for retrieval-only search."""
    query: str
    language: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    timestamp: str


class PaperInfo(BaseModel):
    """Information about an uploaded paper."""
    filename: str
    size_bytes: int
    size_mb: float


class PurgeResponse(BaseModel):
    """Response model for purge operations."""
    status: str
    deleted_count: int
    message: str


class DeletePaperResponse(BaseModel):
    paper_id: str
    chunks_deleted: int
    status: str


class PatchPaperRequest(BaseModel):
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    tags: Optional[str] = None

    @field_validator("title", "authors", "year", "tags", mode="before")
    @classmethod
    def reject_empty_string(cls, v):
        if v is not None and v == "":
            raise ValueError("field must not be empty string")
        return v


class PatchPaperResponse(BaseModel):
    paper_id: str
    chunks_updated: int
    updates: dict


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(
    request: SearchRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """
    Search for relevant passages without LLM generation.

    Returns raw retrieved chunks from the indexed corpus, Tavily web search,
    or both. Useful for debugging retrieval quality or building custom UIs.
    """
    start_time = time.time()

    try:
        results = []
        detected_lang = lang_utils.detect_language(request.query) or "en"

        if request.source in ("corpus", "both"):
            context_data = await run_in_threadpool(
                rag.retrieve_context, request.query, request.top_k
            )
            for chunk, meta, dist in zip(
                context_data["chunks"],
                context_data["metadatas"],
                context_data["distances"],
            ):
                results.append(SearchResult(
                    text=chunk,
                    title=meta.get("title", ""),
                    source=meta.get("paper_id", ""),
                    section=meta.get("section", ""),
                    score=round(float(dist), 4),
                ))

        if request.source in ("web", "both"):
            try:
                from agent.tool_executor import execute_web_search
                web_result = await run_in_threadpool(
                    execute_web_search, request.query, min(request.top_k, 10)
                )
                for p in web_result.get("passages", []):
                    results.append(SearchResult(
                        text=p.get("text", ""),
                        title=p.get("title", ""),
                        source=p.get("source", ""),
                        section="web",
                        score=0.0,
                    ))
            except ValueError:
                logger.warning("Web search requested but TAVILY_API_KEY not configured")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")

        processing_time = time.time() - start_time
        logger.info(
            f"Search: query='{request.query[:50]}', source={request.source}, "
            f"results={len(results)}, time={processing_time:.2f}s"
        )

        return SearchResponse(
            query=request.query,
            language=detected_lang,
            results=results,
            total_results=len(results),
            processing_time=processing_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


def _bibtex_escape(value: str) -> str:
    return value.replace("{", "").replace("}", "")


def _bibtex_key(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", paper_id) or "paper"


@router.get("/search/export", response_class=PlainTextResponse, tags=["Search"])
async def export_search_results(
    query: str = Query(..., min_length=1, max_length=1000),
    format: str = Query("bibtex"),
    top_k: int = Query(10, ge=1, le=30),
    authenticated: bool = Depends(verify_api_key),
):
    """Export retrieved passages' paper metadata as BibTeX entries."""
    if format != "bibtex":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only format=bibtex is supported")

    context_data = await run_in_threadpool(rag.retrieve_context, query, top_k)

    seen_papers: dict = {}
    for meta in context_data["metadatas"]:
        paper_id = meta.get("paper_id", "")
        if not paper_id or paper_id in seen_papers:
            continue
        seen_papers[paper_id] = meta

    entries = []
    for paper_id, meta in seen_papers.items():
        title = _bibtex_escape(meta.get("title", "Unknown"))
        year = _bibtex_escape(str(meta.get("year", "")))
        authors = _bibtex_escape(meta.get("authors", ""))
        fields = [f"  title = {{{title}}}"]
        if authors:
            fields.append(f"  author = {{{authors}}}")
        if year:
            fields.append(f"  year = {{{year}}}")
        entries.append(f"@article{{{_bibtex_key(paper_id)},\n" + ",\n".join(fields) + "\n}")

    return "\n\n".join(entries)


@router.get("/papers", response_model=List[PaperInfo], tags=["Management"])
def list_papers(authenticated: bool = Depends(verify_api_key)):
    """List all PDF files in the papers directory."""
    try:
        papers = []
        for pdf_file in config.PAPERS_DIR.glob("*.pdf"):
            size = pdf_file.stat().st_size
            papers.append(PaperInfo(
                filename=pdf_file.name,
                size_bytes=size,
                size_mb=round(size / (1024 * 1024), 2)
            ))

        return sorted(papers, key=lambda p: p.filename)

    except Exception as e:
        logger.error(f"Error listing papers: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


@router.delete("/papers/{paper_id}", response_model=DeletePaperResponse, tags=["Management"])
async def delete_paper(
    paper_id: str,
    authenticated: bool = Depends(verify_api_key),
):
    """Delete all indexed chunks for a specific paper from the vector store."""
    import vector_store
    chunks_deleted = await run_in_threadpool(vector_store.delete_by_paper_id, paper_id)
    if chunks_deleted == 0:
        raise HTTPException(status_code=404, detail="paper not found or already deleted")
    try:
        import bm25_search
        bm25_search.invalidate()
        threading.Thread(target=bm25_search.get_or_build_index, daemon=True).start()
    except Exception:
        pass
    try:
        from cache import retrieval_cache, tool_cache
        retrieval_cache.invalidate()
        tool_cache.invalidate()
    except Exception:
        logger.warning("Failed to invalidate caches after paper deletion", exc_info=True)
    return DeletePaperResponse(paper_id=paper_id, chunks_deleted=chunks_deleted, status="deleted")


@router.patch("/papers/{paper_id}", response_model=PatchPaperResponse, tags=["Management"])
async def patch_paper(
    paper_id: str,
    request: PatchPaperRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """Update metadata fields on all chunks for a specific paper."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="no valid fields to update")
    import vector_store
    chunks_updated = await run_in_threadpool(vector_store.update_paper_metadata, paper_id, updates)
    if chunks_updated == 0:
        raise HTTPException(status_code=404, detail="paper not found")
    try:
        from cache import retrieval_cache
        retrieval_cache.invalidate()
    except Exception:
        logger.warning("Failed to invalidate retrieval cache after paper update", exc_info=True)
    return PatchPaperResponse(paper_id=paper_id, chunks_updated=chunks_updated, updates=updates)


@router.delete("/purge/papers", response_model=PurgeResponse, tags=["Management"])
def purge_papers(authenticated: bool = Depends(verify_admin_key)):
    """
    Delete all PDF files from the papers directory.

    WARNING: This action cannot be undone!
    """
    try:
        pdf_files = list(config.PAPERS_DIR.glob("*.pdf"))
        deleted_count = 0

        for pdf_file in pdf_files:
            try:
                pdf_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted paper: {pdf_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {pdf_file.name}: {e}")

        return PurgeResponse(
            status="success",
            deleted_count=deleted_count,
            message=f"Deleted {deleted_count} PDF file(s)"
        )

    except Exception as e:
        logger.error(f"Error purging papers: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


@router.delete("/purge/database", response_model=PurgeResponse, tags=["Management"])
def purge_database(authenticated: bool = Depends(verify_admin_key)):
    """
    Clear the vector database (delete all indexed chunks).

    WARNING: This action cannot be undone!
    """
    try:
        import vector_store

        collection = vector_store.get_or_create_collection()
        previous_count = collection.count()

        vector_store.delete_collection(config.COLLECTION_NAME)
        vector_store.get_or_create_collection()

        logger.info(f"Purged database: {previous_count} chunks deleted")

        return PurgeResponse(
            status="success",
            deleted_count=previous_count,
            message=f"Deleted {previous_count} chunks from vector database"
        )

    except Exception as e:
        logger.error(f"Error purging database: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )


@router.get("/cache/stats", tags=["Management"])
async def cache_stats(authenticated: bool = Depends(verify_api_key)):
    """Get cache hit/miss statistics for LLM, retrieval, and tool caches."""
    from cache import llm_cache, retrieval_cache, tool_cache
    return {
        "llm": llm_cache.stats,
        "retrieval": retrieval_cache.stats,
        "tool": tool_cache.stats,
    }


@router.delete("/cache", tags=["Management"])
async def clear_caches(authenticated: bool = Depends(verify_api_key)):
    """Clear all caches."""
    from cache import llm_cache, retrieval_cache, tool_cache
    llm_cache.invalidate()
    retrieval_cache.invalidate()
    tool_cache.invalidate()
    return {"status": "cleared"}


@router.get("/stats", tags=["Management"])
async def get_stats(authenticated: bool = Depends(verify_api_key)):
    """Get vector store statistics."""
    try:
        import vector_store

        collection = vector_store.get_or_create_collection()
        stats = vector_store.get_collection_stats(collection)

        return {
            "collection_name": stats['name'],
            "document_count": stats['count'],
            "metadata": stats.get('metadata', {})
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error. Please try again.", "code": "INTERNAL_ERROR"}
        )
