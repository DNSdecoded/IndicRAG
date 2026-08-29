"""Reconcile the ingest log against the search indexes derived from it.

The ingest log is the system of record; ChromaDB and the BM25 index are derived
views. Nothing verified that they still agree. Every divergence class below has
already been observed in this repo — a partially-failed delete cascade, a
rolled-back upsert, a BM25 index that outlived the rows it indexes — and each
one is invisible from the outside: retrieval simply cites a paper that is gone,
or silently stops returning one that is not.

Run it as a command (`python check_db.py`, exit 1 on divergence) or through
`GET /reconcile`, which caches the result for `/quality` to surface.

ponytail: one full metadata scan of the collection per run, which is why this is
a maintenance command and not a per-request check. If the corpus outgrows that,
the D2 SQL mirror gives per-paper counts without the scan.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Last reconcile result, so /quality can report integrity without paying for a
# full scan on every call. Populated by reconcile(); None until one has run.
_last_result: Optional[dict] = None


def _chroma_ids_by_paper(collection) -> dict[str, set]:
    """paper_id -> set of chunk ids currently in the collection."""
    import vector_store

    got = vector_store._chroma_call(collection.get, include=['metadatas'])
    by_paper: dict[str, set] = {}
    for cid, meta in zip(got.get('ids', []), got.get('metadatas', [])):
        pid = (meta or {}).get('paper_id', '')
        if pid:
            by_paper.setdefault(pid, set()).add(cid)
    return by_paper


def _bm25_live_ids(collection) -> Optional[set]:
    """Ids the lexical index would currently serve, or None if there is no index.

    None is not "empty": with hybrid search off, or before the first build, there
    is nothing to compare against and reporting every paper as missing from BM25
    would be pure noise.
    """
    try:
        import config
        if not config.USE_HYBRID_SEARCH:
            return None
        import bm25_search
        idx = bm25_search.get_or_build_index(collection)
        if idx is None:
            return None
        with idx._lock:
            return {doc_id for i, doc_id in enumerate(idx.doc_ids) if i not in idx._deleted}
    except Exception:
        logger.warning("BM25 index unavailable for reconciliation", exc_info=True)
        return None


def reconcile(collection=None) -> dict:
    """Compare the ingest log against ChromaDB and BM25, per paper.

    Divergence kinds:
      missing_from_chroma  the log claims chunks ChromaDB does not hold — retrieval
                           silently lost content, and only a replay restores it
      extra_in_chroma      ChromaDB holds chunks for a logged paper that the log
                           does not list — a replay would drop them
      not_in_log           ChromaDB holds a paper the log has never heard of — a
                           rebuild deletes it; an upsert rollback that failed
      missing_from_bm25    indexed for dense retrieval but invisible to lexical
      stale_in_bm25        the lexical index still serves deleted chunks, so
                           deleted papers go on being cited
    """
    import persistence
    import vector_store

    if collection is None:
        collection = vector_store.get_or_create_collection()

    log_ids_by_paper: dict[str, set] = {}
    for event in persistence.get_ingest_events():
        pid = event.get("paper_id")
        if pid:
            log_ids_by_paper.setdefault(pid, set()).update(event.get("ids") or [])

    chroma_by_paper = _chroma_ids_by_paper(collection)
    bm25_ids = _bm25_live_ids(collection)

    divergences = []

    def _add(paper_id, kind, ids):
        divergences.append({
            "paper_id": paper_id,
            "kind": kind,
            "count": len(ids),
            # A bounded sample: a paper whose whole index vanished would otherwise
            # emit thousands of ids into a health endpoint's response body.
            "sample_ids": sorted(ids)[:5],
        })

    for pid, logged in log_ids_by_paper.items():
        in_chroma = chroma_by_paper.get(pid, set())
        if logged - in_chroma:
            _add(pid, "missing_from_chroma", logged - in_chroma)
        if in_chroma - logged:
            _add(pid, "extra_in_chroma", in_chroma - logged)
        if bm25_ids is not None:
            # Compare against what ChromaDB actually holds, not against the log:
            # ids already reported as missing_from_chroma would otherwise be
            # counted a second time as a BM25 problem they are not.
            if in_chroma - bm25_ids:
                _add(pid, "missing_from_bm25", in_chroma - bm25_ids)

    for pid, in_chroma in chroma_by_paper.items():
        if pid not in log_ids_by_paper:
            _add(pid, "not_in_log", in_chroma)

    if bm25_ids is not None:
        all_chroma_ids = set().union(*chroma_by_paper.values()) if chroma_by_paper else set()
        stale = bm25_ids - all_chroma_ids
        if stale:
            _add(None, "stale_in_bm25", stale)

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "papers_in_log": len(log_ids_by_paper),
        "papers_in_chroma": len(chroma_by_paper),
        "chunks_in_log": sum(len(v) for v in log_ids_by_paper.values()),
        "chunks_in_chroma": sum(len(v) for v in chroma_by_paper.values()),
        "chunks_in_bm25": None if bm25_ids is None else len(bm25_ids),
        "consistent": not divergences,
        "divergences": divergences,
    }

    global _last_result
    _last_result = result
    if divergences:
        logger.warning("Reconciliation found %d divergence(s) across %d paper(s)",
                       len(divergences), len({d["paper_id"] for d in divergences}))
    return result


def last_result() -> dict:
    """The most recent reconcile() result, for endpoints that must not pay for a scan."""
    return _last_result or {"status": "not_run"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = reconcile()
    print(f"papers: log={report['papers_in_log']} chroma={report['papers_in_chroma']}")
    print(f"chunks: log={report['chunks_in_log']} chroma={report['chunks_in_chroma']} "
          f"bm25={report['chunks_in_bm25']}")
    if report["consistent"]:
        print("consistent: log and derived indexes agree")
        return 0
    print(f"DIVERGENT: {len(report['divergences'])} finding(s)")
    for d in report["divergences"]:
        print(f"  {d['kind']:<20} paper={d['paper_id']} chunks={d['count']} "
              f"sample={d['sample_ids']}")
    print("\nFix: reindex.py replays the log into a fresh index; "
          "reindex.py --backfill-log adopts chunks the log never recorded.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
