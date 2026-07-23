"""Phase 6 Increment 3 — run one watch: search → dedup → ingest → cited digest.

`run_watch(watch_id)` is a plain blocking function so the route stays a thin
`run_in_threadpool` wrapper and tests can mock the three external seams:
`execute_arxiv_search`, `ingest_pdf`, and `rag.llm_generate`.
"""

from datetime import datetime, timedelta, timezone
import asyncio
import logging
import os

import config
import persistence
import rag
from agent.tool_executor import execute_arxiv_search, execute_open_access_search
from cache_refresh import _post_ingest_refresh
from download_utils import download_pdf as _download_pdf
from ingest import ingest_pdf

logger = logging.getLogger(__name__)

_CADENCES = {"daily": timedelta(days=1), "weekly": timedelta(days=7), "monthly": timedelta(days=30)}
_DIGEST_MAX_TOKENS = 1200


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

        # A watch that owns a "living review" regenerates it in place whenever
        # it actually indexes something new — no button a user could forget to click.
        if w.get("report_id"):
            try:
                import report_runner
                new_report = report_runner.run_report(topic, language)
                persistence.save_report(
                    report_id=w["report_id"], watch_id=watch_id,
                    topic=topic, language=language,
                    markdown=new_report["markdown"],
                    citation_count=new_report["citation_count"],
                    created_at=datetime.now(timezone.utc).isoformat(),
                    user_id=w.get("user_id"),
                )
            except Exception as e:
                logger.error(f"[Watch] living-review regeneration failed for {watch_id}: {e}")

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
