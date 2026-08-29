"""Auto-fetch authors/year/DOI from arXiv by fuzzy title match, at ingest time."""

import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

_TITLE_MATCH_THRESHOLD = 0.85


def _cache_key(title: str) -> str:
    """Normalised lookup key: the same paper under different whitespace or case
    is the same query, and caching it twice would re-crawl arXiv for nothing."""
    return re.sub(r"\s+", " ", title or "").strip().lower()


def _lookup_arxiv(title: str) -> Optional[dict]:
    """One arXiv round trip. Returns metadata, or None for no confident match."""
    import arxiv

    client = arxiv.Client(page_size=1, delay_seconds=1, num_retries=1)
    search = arxiv.Search(query=f'ti:"{title}"', max_results=1)
    result = next(client.results(search), None)
    if result is None:
        return None

    similarity = SequenceMatcher(None, title.strip().lower(), result.title.strip().lower()).ratio()
    if similarity < _TITLE_MATCH_THRESHOLD:
        logger.debug(f"arXiv match too weak ({similarity:.2f}) for title: {title[:80]}")
        return None

    return {
        "authors": ", ".join(a.name for a in result.authors),
        "year": str(result.published.year),
        "doi": result.doi or "",
    }


def enrich_from_arxiv(title: str) -> Optional[dict]:
    """Look up `title` on arXiv; return metadata dict if a confident match is found, else None.

    Answers are cached in SQLite, misses included: without that, every re-ingest
    of a directory re-crawls arXiv for papers it has already asked about, at one
    network round trip plus a 1s politeness delay each.

    Never raises — enrichment is best-effort and must not block ingestion
    (offline environments, arXiv downtime, or no match are all normal).
    """
    if not title or not title.strip():
        return None

    key = _cache_key(title)
    try:
        import persistence
        cached = persistence.get_metadata_cache(key)
        if cached is not None:
            # {} is a cached miss — a real answer, not an absence of one.
            return cached or None
    except Exception:
        logger.debug("metadata cache unavailable; querying arXiv directly", exc_info=True)

    try:
        found = _lookup_arxiv(title)
    except Exception as e:
        # Do NOT cache a failure: an offline run would otherwise poison the cache
        # with permanent misses for papers arXiv actually has.
        logger.debug(f"arXiv enrichment skipped for '{title[:80]}': {e}")
        return None

    try:
        import persistence
        persistence.put_metadata_cache(key, found, datetime.now(timezone.utc).isoformat())
    except Exception:
        logger.debug("could not cache arXiv lookup", exc_info=True)
    return found


def enrich_many(titles: List[str], workers: int = None) -> Dict[str, Optional[dict]]:
    """Enrich several titles concurrently. Returns {title: metadata or None}.

    Enrichment is network-bound and was run one title at a time inside the bulk
    ingest loop, so a 50-paper run spent ~50s waiting on arXiv before any of it
    could be embedded. A handful of workers is the whole fix; the cap stays small
    because arXiv is a shared public service, not a resource to saturate.
    """
    import concurrent.futures

    unique = list(dict.fromkeys(t for t in titles if t and t.strip()))
    if not unique:
        return {}

    max_workers = max(1, min(workers or config.ENRICH_WORKERS, len(unique)))
    results: Dict[str, Optional[dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(enrich_from_arxiv, t): t for t in unique}
        for future in concurrent.futures.as_completed(futures):
            title = futures[future]
            try:
                results[title] = future.result()
            except Exception:
                # enrich_from_arxiv already swallows its own errors; this is the
                # belt-and-braces case, and one bad title must not fail an ingest.
                logger.debug("enrichment failed for '%s'", title[:80], exc_info=True)
                results[title] = None
    return results
