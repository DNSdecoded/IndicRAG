"""
Document ingestion pipeline for scientific papers.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
from tqdm import tqdm
import hashlib
import concurrent.futures
import pdf_utils
import embeddings
import vector_store
import config

logger = logging.getLogger(__name__)

def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ingest_paper(
    paper_id: str,
    title: str,
    sections: List[tuple],
    metadata: Optional[Dict[str, Any]] = None,
    collection=None
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
        
    if metadata is None:
        metadata = {}
        
    # Before embedding, check if paper already exists
    existing = collection.get(where={'paper_id': paper_id}, limit=1, include=['metadatas'])
    if existing and existing.get('ids'):
        existing_metadata = existing['metadatas'][0]
        if 'file_hash' in metadata and existing_metadata.get('file_hash') == metadata['file_hash']:
            logger.info(f'Paper {paper_id} already indexed and unchanged, skipping')
            return 0
        else:
            logger.info(f'Paper {paper_id} has changed or hash missing. Reindexing...')
            try:
                vector_store.delete_by_paper_id(paper_id, collection)
            except Exception as del_err:
                logger.error(f"Failed to delete existing chunks for paper {paper_id}: {del_err}")
                raise RuntimeError(
                    f"Aborting re-index of '{paper_id}' to prevent duplicate chunks: {del_err}"
                ) from del_err
    
    all_chunks = []
    all_metadata = []
    all_ids = []
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
            # Chunk the section
            chunks = pdf_utils.simple_chunk(section_text)
            for chunk in chunks:
                safe_section = section_name.replace(' ', '_').lower()
                chunk_id = f"{paper_id}_{safe_section}_{chunk_counter}"
                chunk_metadata = {
                    "paper_id": paper_id,
                    "title": title,
                    "section": section_name,
                    "chunk_index": chunk_counter,
                    **metadata
                }
                all_chunks.append(chunk)
                all_metadata.append(chunk_metadata)
                all_ids.append(chunk_id)
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

    if not all_chunks:
        logger.warning(f"No chunks created for paper {paper_id}")
        return 0
    
    # Embed all chunks
    logger.info(f"Embedding {len(all_chunks)} chunks from '{title}'...")
    chunk_embeddings = embeddings.embed_passages(all_chunks)
    
    # Add to vector store
    vector_store.add_documents(
        texts=all_chunks,
        embeddings=chunk_embeddings,
        metadatas=all_metadata,
        ids=all_ids,
        collection=collection
    )
    
    return len(all_chunks)


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
    metadata['file_hash'] = calculate_md5(pdf_path)
    
    result = pdf_utils.process_pdf(pdf_path)
    
    if result is None:
        logger.error(f"Failed to process PDF: {pdf_path}")
        return 0, ""
    
    # Ingest the paper
    num_chunks = ingest_paper(
        paper_id=paper_id,
        title=result['title'],
        sections=result['sections'],
        metadata=metadata,
        collection=collection
    )
    
    logger.info(f"Ingested {num_chunks} chunks from '{result['title']}'")
    return num_chunks, result['title']


def _extract_worker(path: str, metadata: dict = None) -> tuple:
    """Worker function for parallel PDF extraction."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    m = dict(metadata) if metadata else {}
    m['file_hash'] = hash_md5.hexdigest()
    
    paper_id = Path(path).stem
    res = pdf_utils.process_pdf(path)
    return path, paper_id, res, m


def ingest_directory(
    pdf_dir: str,
    pattern: str = "*.pdf",
    metadata_fn=None,
    collection=None,
    reset: bool = False
) -> Dict[str, int]:
    """
    Ingest all PDFs from a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        pattern: Glob pattern for PDF files (default: "*.pdf")
        metadata_fn: Optional function that takes pdf_path and returns metadata dict
        collection: ChromaDB collection (uses default if None)
        reset: If True, reset collection before ingesting
        
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
        return {"total_files": 0, "successful": 0, "failed": 0, "total_chunks": 0}
    
    logger.info(f"Found {len(pdf_files)} PDF files to ingest")
    logger.info("=" * 60)
    
    stats = {
        "total_files": len(pdf_files),
        "successful": 0,
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

        future_to_pdf = {
            executor.submit(_extract_worker, str(p), _get_metadata(p)): str(p)
            for p in pdf_files
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_pdf), total=len(pdf_files), desc="Ingesting PDFs"):
            pdf_path = future_to_pdf[future]
            try:
                path, paper_id, result, metadata = future.result()
                
                if result is None:
                    stats["failed"] += 1
                    stats["failed_files"].append(path)
                    continue
                    
                num_chunks = ingest_paper(
                    paper_id=paper_id,
                    title=result['title'],
                    sections=result['sections'],
                    metadata=metadata,
                    collection=collection
                )
                
                stats["successful"] += 1
                stats["total_chunks"] += num_chunks
                
            except Exception as e:
                logger.error(f"\nError processing {pdf_path}: {e}")
                stats["failed"] += 1
                stats["failed_files"].append(str(pdf_path))
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Successful: {stats['successful']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    
    if stats['failed_files']:
        logger.info(f"\nFailed files:")
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
            logger.info(f"\nUsage: python ingest.py [pdf_file_or_directory]")
        else:
            logger.info(f"Ingesting PDFs from {papers_dir}...")
            ingest_directory(str(papers_dir))
