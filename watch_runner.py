"""Phase 6 Increment 3 — run one watch: search → dedup → ingest → cited digest.

`run_watch(watch_id)` is a plain blocking function so the route stays a thin
`run_in_threadpool` wrapper and tests can mock the three external seams:
`execute_arxiv_search`, `ingest_pdf`, and `rag.llm_generate`.
"""

from datetime import datetime, timedelta, timezone
import asyncio
import logging
import os
import tempfile
import urllib.request

import config
import persistence
import rag
from agent.tool_executor import execute_arxiv_search, execute_open_access_search
from cache_refresh import _post_ingest_refresh
from ingest import ingest_pdf

logger = logging.getLogger(__name__)

_CADENCES = {"daily": timedelta(days=1), "weekly": timedelta(days=7), "monthly": timedelta(days=30)}
_DIGEST_MAX_TOKENS = 1200
_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB cap — guards against OOM on hostile URLs


def _download_pdf(url: str) -> str | None:
    """Fetch a PDF to a temp file; return its path, or None on failure.

    Rejects non-HTTP(S) URLs (blocks file://, ftp://, gopher:// SSRF vectors)
    and streams with a hard size cap so a huge response can't OOM the process.
    """
    from urllib.parse import urlparse

    if urlparse(url).scheme not in ("http", "https"):
        logger.warning(f"[Watch] Rejected non-HTTP(S) URL: {url}")
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IndicRAG/2.0"})
        fd, path = tempfile.mkstemp(suffix=".pdf")
        with urllib.request.urlopen(req, timeout=30) as resp, os.fdopen(fd, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_PDF_BYTES:
                    logger.warning(f"[Watch] PDF too large (>{_MAX_PDF_BYTES} bytes), aborting: {url}")
                    f.close()
                    os.unlink(path)
                    return None
                f.write(chunk)
        return path
    except Exception as e:
        logger.warning(f"[Watch] PDF download failed {url}: {e}")
        return None


def _summarize(topic: str, papers: list[dict], language: str) -> str:
    """Cited digest of the newly ingested papers via the shared LLM."""
    listing = "\n\n".join(f"[{p['arxiv_id']}] {p['title']}\n{p['text']}" for p in papers)
    prompt = (
        f"You are compiling a research digest on the topic: {topic!r}.\n"
        f"Below are newly published papers. Write a concise digest in {language} "
        f"summarizing what is new. Cite each paper inline by its id in square "
        f"brackets, e.g. [2401.12345]. Use only the papers provided.\n\n"
        f"{listing}"
    )
    return rag.llm_generate(prompt, max_tokens=_DIGEST_MAX_TOKENS)


def run_watch(watch_id: str) -> dict:
    """Search the watch topic, ingest genuinely-new papers, store a cited digest.

    Returns ``{watch_id, new_count, digest, seen_count}``. Raises ``KeyError`` if
    the watch does not exist (the route maps that to a 404).
    """
    w = persistence.get_watch(watch_id)
    if w is None:
        raise KeyError(watch_id)

    topic = w["topic"]
    language = w.get("language", "en")
    seen = set(w.get("seen_ids", []))
    max_results = config.WATCH_MAX_RESULTS

    passages = execute_arxiv_search(topic, max_results=max_results, sort_by="submitted_date").get("passages", [])
    if not passages:  # arXiv empty or timed out → open-access fallback
        passages = execute_open_access_search(topic, max_results=max_results).get("passages", [])

    ingested: list[dict] = []
    new_ids: list[str] = []
    indexed = False  # True once at least one paper was actually added to the vector store
    for p in passages:
        arxiv_id = p.get("arxiv_id")
        if not arxiv_id or arxiv_id in seen:
            continue
        new_ids.append(arxiv_id)  # mark seen even if it dupes the corpus, so we stop re-fetching it

        pdf_url = p.get("pdf_url")
        if pdf_url:
            path = _download_pdf(pdf_url)
            if path:
                try:
                    n_chunks, title = ingest_pdf(
                        path, paper_id=arxiv_id,
                        metadata={"title": p.get("title", ""), "source": p.get("source", "")},
                    )
                finally:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                if n_chunks > 0:  # 0 = duplicate/unchanged in corpus → seen, but not "new"
                    ingested.append({"arxiv_id": arxiv_id, "title": title or p.get("title", ""), "text": p.get("text", "")})
                    indexed = True
                continue
        # ponytail: no PDF (or download failed) → abstract feeds the digest only;
        # indexing an abstract-only chunk is deferred to a later increment.
        ingested.append({"arxiv_id": arxiv_id, "title": p.get("title", ""), "text": p.get("text", "")})

    if indexed:
        # Invalidate caches so newly ingested papers are searchable
        _post_ingest_refresh()

    digest = _summarize(topic, ingested, language) if ingested else w.get("latest_digest")

    now = datetime.now(timezone.utc)
    delta = _CADENCES.get(w.get("cadence", "weekly"), _CADENCES["weekly"])
    w["seen_ids"] = list(seen) + new_ids
    w["latest_digest"] = digest
    w["last_run"] = now.isoformat()
    w["next_run"] = (now + delta).isoformat()
    persistence.save_watch(w)

    logger.info(f"[Watch] ran {watch_id}: {len(ingested)} new, seen now {len(w['seen_ids'])}")
    return {"watch_id": watch_id, "new_count": len(ingested), "digest": digest, "seen_count": len(w["seen_ids"])}


async def run_due_watches() -> int:
    """Run every watch whose next_run has arrived. Returns how many were run.

    run_watch blocks (network + ingest), so each runs in a worker thread to keep
    the event loop free. One watch failing does not abort the sweep.
    """
    now = datetime.now(timezone.utc).isoformat()
    due = persistence.due_watches(now)
    for w in due:
        try:
            await asyncio.to_thread(run_watch, w["id"])
        except Exception as e:
            logger.error(f"[Watch] scheduled run failed for {w['id']}: {e}")
    return len(due)


async def watch_loop() -> None:
    """Poll for due watches every WATCH_POLL_INTERVAL seconds until cancelled.

    Started from the FastAPI lifespan only when WATCH_ENABLE is set.
    """
    logger.info(f"[Watch] schedule loop started (interval {config.WATCH_POLL_INTERVAL}s)")
    try:
        while True:
            await asyncio.sleep(config.WATCH_POLL_INTERVAL)
            await run_due_watches()
    except asyncio.CancelledError:
        logger.info("[Watch] schedule loop stopped")
        raise
