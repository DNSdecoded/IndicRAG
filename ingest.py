"""
Document ingestion pipeline for scientific papers.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
from tqdm import tqdm
import pdf_utils
import embeddings
import vector_store
import config

logger = logging.getLogger(__name__)


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
        
    # Before embedding, check if paper already exists
    existing = collection.get(where={'paper_id': paper_id}, limit=1)
    if existing and existing.get('ids'):
        logger.info(f'Paper {paper_id} already indexed, skipping')
        return 0
    
    if metadata is None:
        metadata = {}
    
    all_chunks = []
    all_metadata = []
    all_ids = []
    chunk_counter = 0
    
    # Process each section
    for section_name, section_text in sections:
        # Skip very short sections
        if len(section_text) < config.MIN_CHUNK_SIZE:
            continue
        
        # Chunk the section
        chunks = pdf_utils.simple_chunk(section_text)
        
        # Create metadata for each chunk
        for chunk in chunks:
            safe_section = section_name.replace(' ', '_').lower()
            chunk_id = f"{paper_id}_{safe_section}_{chunk_counter}"
            
            chunk_metadata = {
                "paper_id": paper_id,
                "title": title,
                "section": section_name,
                "chunk_index": chunk_counter,
                **metadata  # Add any additional metadata
            }
            
            all_chunks.append(chunk)
            all_metadata.append(chunk_metadata)
            all_ids.append(chunk_id)
            chunk_counter += 1
    
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
    
    # Process each PDF
    for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
        try:
            # Get metadata if function provided
            metadata = metadata_fn(str(pdf_path)) if metadata_fn else None
            
            # Ingest PDF
            num_chunks, title = ingest_pdf(
                pdf_path=str(pdf_path),
                metadata=metadata,
                collection=collection
            )
            
            if num_chunks > 0:
                stats["successful"] += 1
                stats["total_chunks"] += num_chunks
            else:
                stats["failed"] += 1
                stats["failed_files"].append(str(pdf_path))
        
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
