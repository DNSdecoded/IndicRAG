"""Auto-fetch authors/year/DOI from arXiv by fuzzy title match, at ingest time."""

import logging
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

_TITLE_MATCH_THRESHOLD = 0.85


def enrich_from_arxiv(title: str) -> Optional[dict]:
    """Look up `title` on arXiv; return metadata dict if a confident match is found, else None.

    Never raises — enrichment is best-effort and must not block ingestion
    (offline environments, arXiv downtime, or no match are all normal).
    """
    if not title or not title.strip():
        return None
    try:
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
    except Exception as e:
        logger.debug(f"arXiv enrichment skipped for '{title[:80]}': {e}")
        return None
