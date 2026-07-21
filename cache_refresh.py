"""Shared post-ingest cache/index invalidation, used by both the ingest routes
and the watch runner so newly indexed papers are searchable everywhere.
"""

import logging
import threading

logger = logging.getLogger(__name__)


def _post_ingest_refresh():
    """Rebuild BM25 index (async) and invalidate retrieval/tool caches after an ingest."""
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
        logger.warning("Failed to invalidate caches after ingestion", exc_info=True)
