"""
Document ingestion pipeline for scientific papers.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
import os
from datetime import datetime, timezone
from tqdm import tqdm
import hashlib
import concurrent.futures
import threading
import pdf_utils
import embeddings
import vector_store
import config

logger = logging.getLogger(__name__)

def calculate_sha256(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_hash(text: str) -> str:
    """SHA-256 of the leading cleaned text, for content-based dedup — catches the
    same paper re-uploaded under a different filename/title."""
    return hashlib.sha256(text[:2000].encode("utf-8", "ignore")).hexdigest()


def _build_paper_chunks(
    paper_id: str,
    title: str,
    sections: List[tuple],
    metadata: Dict[str, Any],
    collection,
    seen_hashes: Optional[set] = None,
    figures: Optional[List[dict]] = None,
) -> Optional[Dict[str, Any]]:
    """Run dedup checks and build chunk/metadata/id lists for one paper.

    Does NOT embed or write to the store — embedding is batched by the caller so
    BGE-M3 sees one large batch instead of one call per paper.

    Returns {'paper_id', 'chunks', 'metadatas', 'ids', 'needs_deletion'}, or
    None when the paper should be skipped (unchanged, or a duplicate).

    seen_hashes: optional set of content hashes already prepared in this batch,
    so two identical papers in one bulk run don't both get ingested.
    """
    sections = list(sections)

    # Unchanged paper → skip. Changed paper → delete old chunks after embedding.
    existing = collection.get(where={'paper_id': paper_id}, limit=1, include=['metadatas'])
    needs_deletion = False
    if existing and existing.get('ids'):
        existing_metadata = existing['metadatas'][0]
        if 'file_hash' in metadata and existing_metadata.get('file_hash') == metadata['file_hash']:
            logger.info(f'Paper {paper_id} already indexed and unchanged, skipping')
            return None
        logger.info(f'Paper {paper_id} has changed or hash missing. Will delete old chunks after embedding.')
        needs_deletion = True

    content_hash = metadata.get('content_hash')

    # Content-based dedup: same body text re-uploaded under a different paper_id.
    if config.DEDUP_PAPERS and not needs_deletion and content_hash:
        if seen_hashes is not None and content_hash in seen_hashes:
            logger.info(f"Paper '{title[:80]}' duplicates another paper in this batch (content hash), skipping")
            return None
        existing_c = collection.get(where={'content_hash': content_hash}, limit=1, include=['metadatas'])
        if existing_c and existing_c.get('ids'):
            dup_pid = existing_c['metadatas'][0].get('paper_id')
            if dup_pid and dup_pid != paper_id:
                logger.info(f"Paper '{title[:80]}' duplicates existing paper_id={dup_pid} (content hash), skipping")
                return None

    # Cross-ingestion dedup: same paper re-uploaded under a different filename/paper_id
    if config.DEDUP_PAPERS and not needs_deletion:
        dup_id = vector_store.find_similar_paper(
            title, year=metadata.get('year'),
            threshold=config.DEDUP_TITLE_THRESHOLD, collection=collection,
        )
        if dup_id and dup_id != paper_id:
            logger.info(f"Paper '{title[:80]}' looks like a duplicate of existing paper_id={dup_id}, skipping")
            return None

    all_chunks: List[str] = []
    all_metadata: List[dict] = []
    all_ids: List[str] = []
    chunk_counter = 0

    def _build_chunks(sections_iter, skip_refs: bool):
        """Process sections into chunk lists, optionally skipping references."""
        nonlocal chunk_counter
        for section_name, section_text in sections_iter:
            if skip_refs and section_name.lower() in ['references', 'bibliography']:
                logger.debug(f"Skipping '{section_name}' for paper {paper_id}")
                continue
            # Skip very short sections
            if len(section_text) < config.MIN_CHUNK_SIZE:
                continue
            # Per-section chunk size: dense sections smaller, narrative larger.
            # canonical_section maps verbose/structural headers (e.g. "stage 3:
            # reinforcement learning") to a sizing bucket so method equations stay whole.
            max_chars = config.SECTION_CHUNK_SIZES.get(
                pdf_utils.canonical_section(section_name), config.CHUNK_SIZE)
            chunks = pdf_utils.simple_chunk(section_text, max_chars=max_chars)
            for chunk in chunks:
                safe_section = section_name.replace(' ', '_').lower()
                all_chunks.append(chunk)
                all_metadata.append({
                    "paper_id": paper_id,
                    "title": title,
                    "section": section_name,
                    "chunk_index": chunk_counter,
                    **metadata
                })
                all_ids.append(f"{paper_id}_{safe_section}_{chunk_counter}")
                chunk_counter += 1

    # First pass: normal behaviour — skip references/bibliography
    _build_chunks(sections, skip_refs=True)

    # Fallback: multi-chapter books often have every chapter's body mis-labeled
    # as 'references' by the section extractor. If nothing survived the filter,
    # retry including those sections so no content is silently lost.
    if not all_chunks:
        logger.warning(
            f"No chunks after reference-filter for '{paper_id}'. "
            "Retrying without section-name filter (likely a multi-chapter book)."
        )
        chunk_counter = 0
        _build_chunks(sections, skip_refs=False)

    # Phase 3: append figure/table chunks. Captioning (VLM, network) runs here —
    # after every dedup check above has passed — so a duplicate/unchanged paper is
    # never captioned. Regions were extracted CPU-side upstream (worker / ingest_pdf).
    if config.ENABLE_MULTIMODAL_INGEST and figures:
        import figure_captioner
        try:
            captions = figure_captioner.caption_regions(figures, paper_id)
        except Exception as cap_err:
            logger.error(
                f"Figure captioning failed for {paper_id}, continuing without "
                f"figure chunks: {cap_err}"
            )
            captions = []
        for fig in captions:
            all_chunks.append(fig["text"])
            all_metadata.append({
                "paper_id": paper_id,
                "title": title,
                "section": fig["chunk_type"],
                "chunk_index": chunk_counter,
                "chunk_type": fig["chunk_type"],
                "page": fig["page"],
                "crop_path": fig["crop_path"],
                **metadata,
            })
            all_ids.append(f"{paper_id}_{fig['chunk_type']}_{chunk_counter}")
            chunk_counter += 1

    if not all_chunks:
        logger.warning(f"No chunks created for paper {paper_id}")
        return None

    if seen_hashes is not None and content_hash:
        seen_hashes.add(content_hash)

    return {
        'paper_id': paper_id,
        'title': title,
        'content_hash': content_hash or '',
        'chunks': all_chunks,
        'metadatas': all_metadata,
        'ids': all_ids,
        'needs_deletion': needs_deletion,
    }


def _record_ingest(prepared: Dict[str, Any], source_path: str = None) -> None:
    """Write one paper's ingest-log row.

    Shared by both ingest paths on purpose. ingest_paper() and ingest_directory()
    each do their own embed-and-add, so putting the log write in only one of them
    (as the first version did) silently left the bulk path — the one /ingest/all
    uses — unlogged, and reindex.py would report an empty log after a perfectly
    successful ingest.

    Best-effort: a logging failure costs replayability for this paper, not the
    ingest that already succeeded.
    """
    try:
        import persistence
        persistence.record_ingest(
            event_id=prepared['paper_id'],   # one row per paper: the log describes
                                             # current index contents, not history
            paper_id=prepared['paper_id'],
            content_hash=prepared.get('content_hash') or '',
            title=prepared.get('title') or '',
            source_path=source_path or prepared.get('source_path') or '',
            chunks=prepared['chunks'],
            metadatas=prepared['metadatas'],
            ids=prepared['ids'],
            embed_model=config.EMBEDDING_MODEL_NAME,
            chunker_version=vector_store.CHUNKER_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            embed_backend=vector_store._embed_backend(),
        )
    except Exception:
        logger.warning("Could not record ingest log for %s — this paper will not be "
                       "replayable by reindex.py", prepared.get('paper_id'), exc_info=True)


def ingest_paper(
    paper_id: str,
    title: str,
    sections: List[tuple],
    metadata: Optional[Dict[str, Any]] = None,
    collection=None,
    figures: Optional[List[dict]] = None,
    source_path: Optional[str] = None,
) -> int:
    """
    Ingest a single paper into the vector store.

    Args:
        paper_id: Unique identifier for the paper (e.g., "arxiv:2101.00001")
        title: Paper title
        sections: List of (section_name, section_text) tuples
        metadata: Additional metadata (e.g., year, authors, domain)
        collection: ChromaDB collection (uses default if None)

    Returns:
        Number of chunks ingested
    """
    if collection is None:
        collection = vector_store.get_or_create_collection()

    prepared = _build_paper_chunks(paper_id, title, sections, metadata or {}, collection, figures=figures)
    if prepared is None:
        return 0

    logger.info(f"Embedding {len(prepared['chunks'])} chunks from '{title}'...")
    chunk_embeddings = embeddings.embed_passages(prepared['chunks'])

    if prepared['needs_deletion']:
        try:
            vector_store.delete_by_paper_id(paper_id, collection)
        except Exception as del_err:
            logger.error(f"Failed to delete old chunks for paper {paper_id}: {del_err}")

    vector_store.add_documents(
        texts=prepared['chunks'],
        embeddings=chunk_embeddings,
        metadatas=prepared['metadatas'],
        ids=prepared['ids'],
        collection=collection
    )

    # Record what was indexed so the indexes can be rebuilt without re-parsing
    # the PDF. Written AFTER the indexes, so the log never claims chunks that
    # failed to land.
    _record_ingest(prepared, source_path)

    return len(prepared['chunks'])


def dry_run_pdf(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Process a PDF (extract, section, count chunks) WITHOUT embedding or storing.

    Returns per-section chunk stats for debugging ingestion quality before
    committing, or None if the PDF can't be processed (scanned/image PDF).
    """
    result = pdf_utils.process_pdf(pdf_path)
    if result is None:
        return None

    section_stats = []
    total_chunks = 0
    for section_name, section_text in result['sections']:
        if len(section_text) < config.MIN_CHUNK_SIZE:
            n = 0
        else:
            max_chars = config.SECTION_CHUNK_SIZES.get(
                pdf_utils.canonical_section(section_name), config.CHUNK_SIZE)
            n = len(pdf_utils.simple_chunk(section_text, max_chars=max_chars))
        total_chunks += n
        section_stats.append({"section": section_name, "chars": len(section_text), "chunks": n})

    return {
        "title": result['title'],
        "text_length": len(result['text']),
        "num_sections": len(result['sections']),
        "total_chunks": total_chunks,
        "sections": section_stats,
        "content_hash": _content_hash(result['text']),
    }


def ingest_pdf(
    pdf_path: str,
    paper_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    collection=None
) -> Tuple[int, str]:
    """
    Ingest a single PDF file into the vector store.
    
    Args:
        pdf_path: Path to PDF file
        paper_id: Unique identifier (uses filename if None)
        metadata: Additional metadata
        collection: ChromaDB collection (uses default if None)
        
    Returns:
        Tuple of (Number of chunks ingested, title extraction result)
    """
    # Generate paper_id from filename if not provided
    if paper_id is None:
        paper_id = Path(pdf_path).stem
    
    # Process PDF
    logger.info(f"\nProcessing: {pdf_path}")
    
    if metadata is None:
        metadata = {}
    metadata['file_hash'] = calculate_sha256(pdf_path)
    
    result = pdf_utils.process_pdf(pdf_path)

    if result is None:
        logger.error(f"Failed to process PDF: {pdf_path}")
        return 0, ""

    metadata['content_hash'] = _content_hash(result['text'])

    if config.ENRICH_METADATA:
        import metadata_enrich
        enriched = metadata_enrich.enrich_from_arxiv(result['title'])
        if enriched:
            # Don't overwrite metadata the caller explicitly provided
            metadata = {**enriched, **metadata}

    # Phase 3: extract figure/table regions (CPU) before ingest; captioning is
    # deferred to _build_paper_chunks so it only runs if the paper survives dedup.
    figures = None
    if config.ENABLE_MULTIMODAL_INGEST:
        import figure_captioner
        try:
            figures = figure_captioner.extract_regions(pdf_path, paper_id)
        except Exception as ext_err:
            logger.warning(
                f"Figure extraction failed for {paper_id}, continuing without "
                f"figures: {ext_err}"
            )
            figures = None

    # Ingest the paper
    num_chunks = ingest_paper(
        paper_id=paper_id,
        title=result['title'],
        sections=result['sections'],
        metadata=metadata,
        collection=collection,
        figures=figures,
        source_path=str(pdf_path),
    )
    
    logger.info(f"Ingested {num_chunks} chunks from '{result['title']}'")
    return num_chunks, result['title']


def _extract_worker(path: str, metadata: dict = None) -> tuple:
    """Worker function for parallel PDF extraction."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)

    m = dict(metadata) if metadata else {}
    m['file_hash'] = h.hexdigest()

    paper_id = Path(path).stem
    res = pdf_utils.process_pdf(path)

    # CPU-only work here (runs in a ProcessPoolExecutor worker). arXiv metadata
    # enrichment is a network call — kept in the parent loop so we don't fan out
    # N concurrent arXiv requests from the pool.
    if res is not None:
        m['content_hash'] = _content_hash(res['text'])
        # Phase 3: extract figure/table regions here (CPU-only, safe in the pool
        # worker). Captioning is a network call and stays in the parent loop.
        if config.ENABLE_MULTIMODAL_INGEST:
            import figure_captioner
            try:
                res['figures'] = figure_captioner.extract_regions(path, paper_id)
            except Exception as ext_err:
                logger.warning(
                    f"Figure extraction failed for {paper_id}, continuing without "
                    f"figures: {ext_err}"
                )
                res['figures'] = None

    return path, paper_id, res, m


def ingest_directory(
    pdf_dir: str,
    pattern: str = "*.pdf",
    metadata_fn=None,
    collection=None,
    reset: bool = False,
    progress_cb=None,
) -> Dict[str, int]:
    """
    Ingest all PDFs from a directory.

    Args:
        pdf_dir: Directory containing PDF files
        pattern: Glob pattern for PDF files (default: "*.pdf")
        metadata_fn: Optional function that takes pdf_path and returns metadata dict
        collection: ChromaDB collection (uses default if None)
        reset: If True, reset collection before ingesting
        progress_cb: Optional callable(done:int, total:int, message:str) invoked
            per paper during extraction and once before the batch-embed step, for
            live progress reporting (e.g. an SSE stream).

    Returns:
        Dictionary with ingestion statistics
    """
    pdf_dir = Path(pdf_dir)
    
    if not pdf_dir.exists():
        raise ValueError(f"Directory does not exist: {pdf_dir}")
    
    # Get or create collection
    if collection is None:
        collection = vector_store.get_or_create_collection(reset=reset)
    
    # Find all PDFs
    pdf_files = list(pdf_dir.glob(pattern))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir} matching pattern '{pattern}'")
        return {"total_files": 0, "successful": 0, "skipped": 0,
                "failed": 0, "total_chunks": 0, "failed_files": []}
    
    logger.info(f"Found {len(pdf_files)} PDF files to ingest")
    logger.info("=" * 60)
    
    stats = {
        "total_files": len(pdf_files),
        "successful": 0,   # papers that produced chunks to ingest
        "skipped": 0,      # unchanged / duplicate papers (no error, no new chunks)
        "failed": 0,
        "total_chunks": 0,
        "failed_files": []
    }
    
    # Process each PDF using parallel extraction
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Evaluate metadata_fn in the parent process so the result (a plain dict)
        # can be safely pickled and passed to worker processes.
        def _get_metadata(p):
            if metadata_fn is None:
                return {}
            try:
                result = metadata_fn(p)
                return result if isinstance(result, dict) else {}
            except Exception as meta_err:
                logger.warning(f"metadata_fn failed for {p}, using empty metadata: {meta_err}")
                return {}

        # Bound in-flight tasks to avoid IPC queue bloat (PDF parse >> embedding speed)
        max_workers = os.cpu_count() or 4
        sem = threading.Semaphore(max_workers * 2)

        future_to_pdf = {}
        for p in pdf_files:
            sem.acquire()
            f = executor.submit(_extract_worker, str(p), _get_metadata(p))
            f.add_done_callback(lambda _: sem.release())
            future_to_pdf[f] = str(p)

        prepared_papers: List[Dict[str, Any]] = []
        seen_hashes: set = set()
        total = len(pdf_files)
        done = 0

        for future in tqdm(concurrent.futures.as_completed(future_to_pdf), total=total, desc="Extracting PDFs"):
            pdf_path = future_to_pdf[future]
            done += 1
            try:
                path, paper_id, result, metadata = future.result()

                if result is None:
                    stats["failed"] += 1
                    stats["failed_files"].append(path)
                    continue

                # arXiv enrichment (network) in the parent, not the CPU workers.
                if config.ENRICH_METADATA:
                    import metadata_enrich
                    enriched = metadata_enrich.enrich_from_arxiv(result['title'])
                    if enriched:
                        # Caller/worker-computed metadata (file_hash, content_hash)
                        # wins over enriched fields.
                        metadata = {**enriched, **metadata}

                # Build chunks + dedup now; embedding is batched below.
                prepared = _build_paper_chunks(
                    paper_id, result['title'], result['sections'],
                    metadata, collection, seen_hashes,
                    figures=result.get('figures'),
                )
                if prepared is not None:
                    stats["successful"] += 1
                    prepared['source_path'] = str(path)
                    prepared_papers.append(prepared)
                else:
                    # Unchanged or deduplicated — processed fine, just nothing to ingest.
                    stats["skipped"] += 1

                if progress_cb:
                    progress_cb(done, total, f"Ingesting {Path(path).name} ({done}/{total})")

            except Exception as e:
                logger.error(f"\nError processing {pdf_path}: {e}")
                stats["failed"] += 1
                stats["failed_files"].append(str(pdf_path))

    # Batch-embed every chunk across all papers in one pass — BGE-M3 is far more
    # efficient on one large batch than on one embed call per paper.
    if prepared_papers:
        all_chunks = [c for p in prepared_papers for c in p['chunks']]
        all_metadata = [m for p in prepared_papers for m in p['metadatas']]
        all_ids = [i for p in prepared_papers for i in p['ids']]

        if progress_cb:
            progress_cb(total, total, f"Embedding {len(all_chunks)} chunks from {len(prepared_papers)} papers...")
        logger.info(f"Batch-embedding {len(all_chunks)} chunks from {len(prepared_papers)} papers...")

        # Embed BEFORE deleting old chunks: if embedding fails, a changed paper
        # keeps its existing chunks rather than being left empty. Matches the
        # embed → delete → add ordering in ingest_paper().
        all_embeddings = embeddings.embed_passages(all_chunks)

        for p in prepared_papers:
            if p['needs_deletion']:
                try:
                    vector_store.delete_by_paper_id(p['paper_id'], collection)
                except Exception as del_err:
                    logger.error(f"Failed to delete old chunks for {p['paper_id']}: {del_err}")

        # ponytail: one array of shape (n_chunks, 1024) in memory — fine at corpus
        # scale; embed_passages already mini-batches internally. Chunk the add if
        # a single corpus ever exceeds tens of thousands of chunks.
        vector_store.add_documents(
            texts=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadata,
            ids=all_ids,
            collection=collection,
        )
        stats["total_chunks"] = len(all_chunks)

        # Same log write as the single-paper path, after the indexes are written.
        for p in prepared_papers:
            _record_ingest(p)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Successful: {stats['successful']}")
    logger.info(f"  Skipped (unchanged/duplicate): {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    
    if stats['failed_files']:
        logger.info("\nFailed files:")
        for f in stats['failed_files']:
            logger.info(f"  - {f}")
    
    # Final collection stats
    collection_stats = vector_store.get_collection_stats(collection)
    logger.info(f"\nCollection '{collection_stats['name']}' now contains {collection_stats['count']} documents")
    
    return stats


if __name__ == "__main__":
    import sys
   
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Ensure directories exist
    try:
        config.ensure_directories()
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        # Ingest from command line argument
        path = sys.argv[1]
        
        if Path(path).is_file():
            #  Single PDF
            logger.info("Ingesting single PDF...")
            ingest_pdf(path)
        elif Path(path).is_dir():
            # Directory of PDFs
            logger.info("Ingesting directory of PDFs...")
            ingest_directory(path)
        else:
            logger.error(f"Invalid path: {path}")
    else:
        # Default: ingest from papers directory
        papers_dir = config.PAPERS_DIR
        
        if not any(papers_dir.glob("*.pdf")):
            logger.warning(f"No PDFs found in {papers_dir}")
            logger.info(f"Please add PDF files to {papers_dir} and run again.")
            logger.info("\nUsage: python ingest.py [pdf_file_or_directory]")
        else:
            logger.info(f"Ingesting PDFs from {papers_dir}...")
            ingest_directory(str(papers_dir))
