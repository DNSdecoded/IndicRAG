"""Shared post-ingest cache/index invalidation, used by both the ingest routes
and the watch runner so newly indexed papers are searchable everywhere.
"""

import logging

logger = logging.getLogger(__name__)


def _post_ingest_refresh():
    """Invalidate retrieval/tool/LLM caches after an ingest or delete.

    The BM25 index is NOT touched here: vector_store.add_documents folds new
    chunks into it and delete_by_paper_id / delete_stale_chunks remove old ones,
    since every chunk write and delete routes through those functions. Dropping
    the index here (as this function used to, because no caller handed it the
    new chunk ids) made the first query after every ingest — including every
    scheduled watch digest — rebuild it from ChromaDB under the build lock.
    """
    try:
        from cache import llm_cache, retrieval_cache, tool_cache
        retrieval_cache.invalidate()
        tool_cache.invalidate()
        # llm_cache too: it is keyed on the prompt, and a prompt embeds the
        # context that was retrieved when it was built. After an ingest or a
        # delete that context is stale, so a cached answer can keep citing a
        # paper that no longer exists for the whole TTL.
        llm_cache.invalidate()
    except Exception:
        logger.warning("Failed to invalidate caches after ingestion", exc_info=True)
