#!/usr/bin/env python3
"""Rebuild the search indexes by replaying the ingest log.

The vector store, the BM25 index and the figure store are derived views. The
ingest log (see persistence.record_ingest) is the system of record: it holds the
chunk text and metadata each ingestion produced.

Before it existed, rebuilding meant re-parsing every PDF and re-calling the VLM
captioner — hours of work, not reproducible, and impossible once a source file
had moved. That made changing the embedding model or the chunking strategy
expensive enough that it never happened, which is how a corpus ends up
permanently married to whatever model first indexed it.

Replay re-embeds recorded chunks. It does NOT re-chunk: chunk boundaries are
what the log recorded. So this is the right tool for an EMBEDDING model change
and the wrong one for a CHUNKER change — a chunker change needs a real
re-ingest from the PDFs, and --check says so when that is the case.

Usage
-----
    python reindex.py --check          # report drift, change nothing
    python reindex.py --dry-run        # show what a rebuild would do
    python reindex.py --yes            # rebuild into the live collection (destructive)
    python reindex.py --into staging   # rebuild into a different collection first
"""

import argparse
import logging

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("reindex")


def backfill_log_from_collection(collection_name: str) -> int:
    """Reconstruct ingest-log rows from chunks already in the collection.

    For corpora indexed before the log existed, or indexed by a run that wrote the
    chunks but died before recording them — which is exactly what a bulk upsert
    timing out after a successful write produced. Everything the log stores
    (chunk text, metadata, ids, provenance) is already on the chunks themselves,
    so this recovers replayability without re-embedding.

    Not a substitute for logging at ingest time: content_hash and source_path are
    not recoverable from the collection and are left empty.
    """
    from datetime import datetime, timezone
    import persistence
    import vector_store

    collection = vector_store.get_or_create_collection(collection_name)
    got = vector_store._chroma_call(collection.get, include=["documents", "metadatas"],
                                    timeout=config.CHROMA_WRITE_TIMEOUT_S)
    ids, docs, metas = got.get("ids", []), got.get("documents", []), got.get("metadatas", [])
    if not ids:
        logger.error("Collection '%s' is empty — nothing to backfill.", collection_name)
        return 0

    papers = {}
    for cid, doc, meta in zip(ids, docs, metas):
        pid = (meta or {}).get("paper_id")
        if not pid:
            continue
        papers.setdefault(pid, {"ids": [], "chunks": [], "metadatas": []})
        papers[pid]["ids"].append(cid)
        papers[pid]["chunks"].append(doc)
        papers[pid]["metadatas"].append(meta)

    now = datetime.now(timezone.utc).isoformat()
    for pid, p in papers.items():
        first = p["metadatas"][0]
        persistence.record_ingest(
            event_id=pid, paper_id=pid, content_hash="", title=first.get("title", ""),
            source_path="", chunks=p["chunks"], metadatas=p["metadatas"], ids=p["ids"],
            embed_model=first.get("embed_model") or "",
            chunker_version=first.get("chunker_version") or 0,
            created_at=now, embed_backend=first.get("embed_backend"),
        )
        logger.info("  backfilled %-28s %4d chunks", pid, len(p["chunks"]))
    logger.info("Backfilled %d papers / %d chunks into the ingest log.",
                len(papers), sum(len(p["chunks"]) for p in papers.values()))
    return len(papers)


def _drift_report(events: list) -> list:
    """Differences between what the log recorded and what is configured now."""
    problems = []
    models = {e["embed_model"] for e in events if e["embed_model"]}
    chunkers = {e["chunker_version"] for e in events if e["chunker_version"] is not None}

    if len(models) > 1:
        problems.append(f"corpus was indexed with MIXED embedding models: {sorted(models)} "
                        "— vectors from different spaces are not comparable")
    if models and config.EMBEDDING_MODEL_NAME not in models:
        problems.append(f"configured embedding model {config.EMBEDDING_MODEL_NAME!r} differs "
                        f"from the indexed one(s) {sorted(models)} — a replay will re-embed")
    if len(chunkers) > 1:
        problems.append(f"corpus was chunked by MIXED chunker versions: {sorted(chunkers)}")

    import vector_store
    backends = {e.get("embed_backend") for e in events if e.get("embed_backend")}
    if len(backends) > 1:
        problems.append(f"corpus was embedded by MIXED backends: {sorted(backends)} — "
                        "int8 and fp32 output for the same model are not comparable")
    current_backend = vector_store._embed_backend()
    if backends and current_backend not in ("unloaded", "unknown") and current_backend not in backends:
        problems.append(f"configured embedding backend {current_backend!r} differs from the "
                        f"indexed one(s) {sorted(backends)} — a replay will re-embed")

    if chunkers and vector_store.CHUNKER_VERSION not in chunkers:
        problems.append(
            f"configured chunker version {vector_store.CHUNKER_VERSION} differs from the "
            f"recorded one(s) {sorted(chunkers)}. Replay CANNOT fix this — it reuses the "
            "recorded chunk boundaries. Re-ingest from the PDFs instead.")
    return problems


def reindex(collection_name: str, dry_run: bool, batch_size: int,
            confirm: bool = False) -> int:
    import bm25_search
    import embeddings
    import persistence
    import vector_store

    events = persistence.get_ingest_events()
    if not events:
        logger.error("The ingest log is empty — nothing to replay. Papers ingested "
                     "before the log existed are not replayable; re-ingest them from "
                     "the PDFs in %s.", config.PAPERS_DIR)
        return 2

    total_chunks = sum(len(e["chunks"]) for e in events)
    logger.info("Replaying %d papers / %d chunks into '%s'",
                len(events), total_chunks, collection_name)

    for problem in _drift_report(events):
        logger.warning("DRIFT: %s", problem)

    if dry_run:
        for e in events:
            logger.info("  would replay %-28s %4d chunks  (%s)",
                        e["paper_id"], len(e["chunks"]), e["embed_model"] or "unknown model")
        logger.info("Dry run — nothing written.")
        return 0

    # reset=True wipes whatever is there, and the default target is the live
    # collection — so destroying production must be typed out, not defaulted into.
    if collection_name == config.COLLECTION_NAME and not confirm:
        logger.error("Refusing to reset the LIVE collection '%s' without --yes. "
                     "Build into a staging collection with --into NAME, or pass "
                     "--yes if you really mean to replace it.", collection_name)
        return 2

    # Rebuild into a fresh collection. reset=True because a replay must produce
    # exactly the logged contents: merging into an existing collection would keep
    # chunks the log no longer accounts for, which is the drift this exists to end.
    collection = vector_store.get_or_create_collection(collection_name, reset=True)

    done = 0
    for e in events:
        chunks, metas, ids = e["chunks"], e["metadatas"], e["ids"]
        for start in range(0, len(chunks), batch_size):
            sl = slice(start, start + batch_size)
            vecs = embeddings.embed_passages(chunks[sl])
            vector_store.add_documents(
                texts=chunks[sl], embeddings=vecs,
                metadatas=metas[sl], ids=ids[sl], collection=collection)
        done += len(chunks)
        logger.info("  %-28s %4d chunks  (%d/%d)", e["paper_id"], len(chunks), done, total_chunks)

    # The lexical index is derived too — leaving the old one would keep serving
    # chunks from the collection that was just replaced.
    bm25_search.invalidate()
    logger.info("Rebuilt %d chunks into '%s'. BM25 will rebuild on the next query.",
                done, collection_name)

    if collection_name != config.COLLECTION_NAME:
        logger.info("Built into '%s', not the live collection '%s'. Point "
                    "COLLECTION_NAME at it once you have verified retrieval quality.",
                    collection_name, config.COLLECTION_NAME)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", default=None,
                    help="collection to build into (default: the configured one)")
    ap.add_argument("--dry-run", action="store_true", help="report what would happen")
    ap.add_argument("--check", action="store_true",
                    help="report drift between the log and the current config, then exit")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--yes", action="store_true",
                    help="confirm resetting the live collection")
    ap.add_argument("--backfill-log", action="store_true",
                    help="rebuild the ingest log from chunks already in the collection")
    args = ap.parse_args()

    import persistence

    if args.backfill_log:
        return 0 if backfill_log_from_collection(args.into or config.COLLECTION_NAME) else 2

    if args.check:
        events = persistence.get_ingest_events()
        if not events:
            logger.warning("Ingest log is empty — nothing recorded yet.")
            return 1
        problems = _drift_report(events)
        logger.info("%d papers / %d chunks recorded in the ingest log",
                    len(events), sum(len(e["chunks"]) for e in events))
        for p in problems:
            logger.warning("DRIFT: %s", p)
        if not problems:
            logger.info("No drift: the log agrees with the current configuration.")
        return 1 if problems else 0

    return reindex(args.into or config.COLLECTION_NAME, args.dry_run, args.batch_size,
                   confirm=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
