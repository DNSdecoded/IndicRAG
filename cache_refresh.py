"""Shared post-ingest cache/index invalidation, used by both the ingest routes
and the watch runner so newly indexed papers are searchable everywhere.
"""

import logging
import threading

logger = logging.getLogger(__name__)


def _post_ingest_refresh(new_ids=None, new_texts=None):
    """Refresh the BM25 index and invalidate retrieval/tool caches after an ingest.

    Pass the newly indexed chunks to fold them in incrementally. Without them this
    falls back to dropping the whole index, which makes the FIRST query after any
    ingest — including every scheduled watch digest — pay a full rebuild from
    ChromaDB while holding the build lock, with every concurrent query queued
    behind it. That is a recurring p99 cliff that grows with corpus size.
    """
    try:
        import bm25_search
        if new_ids and bm25_search.add_to_index(list(new_ids), list(new_texts or [])):
            logger.debug("BM25 index updated incrementally (+%d chunks)", len(new_ids))
            # Re-persist off-thread: the in-memory index just moved ahead of the
            # saved copy, and a stale file only costs a rebuild on next start.
            threading.Thread(target=bm25_search.save_index, daemon=True).start()
        else:
            # Nothing handed over, or no index built yet: drop it and warm a new one
            # off-thread so the next request doesn't eat the rebuild.
            bm25_search.invalidate()
            threading.Thread(target=bm25_search.get_or_build_index, daemon=True).start()
    except Exception:
        logger.warning("Failed to refresh BM25 index after ingestion", exc_info=True)
    try:
        from cache import retrieval_cache, tool_cache
        retrieval_cache.invalidate()
        tool_cache.invalidate()
    except Exception:
        logger.warning("Failed to invalidate caches after ingestion", exc_info=True)
