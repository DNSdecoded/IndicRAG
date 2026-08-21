"""
Vector store operations using ChromaDB.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import numpy as np
import logging
import threading
import time
import concurrent.futures
import config

logger = logging.getLogger(__name__)


# A timed-out ChromaDB call keeps running (a running future can't be cancelled),
# so a small pool would be permanently saturated after a few hangs — every later
# call would then time out waiting for a free worker. A large pool means a hung
# call leaks at most one thread instead of poisoning the shared timeout facility.
# ponytail: 32 workers; if hangs ever pile up, fix the hang, don't grow the pool.
_chroma_executor = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="chroma-timeout")


# Circuit breaker, same shape as the per-(provider, model) one in llm_client.py.
# Without it an unhealthy ChromaDB (disk full, corrupt segment, hung compaction)
# makes every request pay the full timeout before failing, and each one parks a
# worker in the pool above that cannot be cancelled — the pool drains in seconds
# and the failure spreads to requests that would never have touched Chroma.
_CIRCUIT_COOLDOWN = 60.0
_CIRCUIT_TRIP_AFTER = 3  # consecutive timeouts before the circuit opens
_circuit_until = 0.0
_circuit_failures = 0
_circuit_lock = threading.Lock()


class ChromaUnavailable(RuntimeError):
    """Raised instead of waiting on a ChromaDB already known to be timing out."""


def chroma_circuit_open() -> bool:
    """True while the breaker is open — callers can degrade without trying."""
    with _circuit_lock:
        return time.monotonic() < _circuit_until


def _chroma_record_failure() -> None:
    global _circuit_until, _circuit_failures
    with _circuit_lock:
        _circuit_failures += 1
        if _circuit_failures >= _CIRCUIT_TRIP_AFTER:
            _circuit_until = time.monotonic() + _CIRCUIT_COOLDOWN
            logger.error("ChromaDB circuit OPEN for %.0fs after %d consecutive timeouts",
                         _CIRCUIT_COOLDOWN, _circuit_failures)


def _chroma_record_success() -> None:
    global _circuit_failures
    if _circuit_failures:
        with _circuit_lock:
            _circuit_failures = 0


def _chroma_call(fn, *args, timeout: float = 5.0, **kwargs) -> Any:
    """Run a ChromaDB call with a timeout. Raises TimeoutError if it hangs.

    Note: cancel() cannot stop an already-running call; on timeout the worker
    thread keeps running until the underlying call returns. The large pool above
    bounds the damage of that leak, and the breaker bounds how often we are
    willing to create one.
    """
    if chroma_circuit_open():
        raise ChromaUnavailable("ChromaDB circuit is open; skipping call")
    fut = _chroma_executor.submit(fn, *args, **kwargs)
    try:
        result = fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError as err:
        fut.cancel()
        _chroma_record_failure()
        raise TimeoutError(f"ChromaDB operation timed out after {timeout}s") from err
    _chroma_record_success()
    return result


# Global client cache
_chroma_client = None
_lock = threading.Lock()


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Get or create ChromaDB client with persistence (thread-safe).
    """
    global _chroma_client

    if _chroma_client is not None:
        return _chroma_client

    with _lock:
        if _chroma_client is not None:
            return _chroma_client

        logger.info(f"Initializing ChromaDB at: {config.CHROMA_DB_DIR}")

        _chroma_client = chromadb.PersistentClient(
            path=str(config.CHROMA_DB_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    return _chroma_client


def get_or_create_collection(
    collection_name: str = None,
    reset: bool = False
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection.
    
    Args:
        collection_name: Name of the collection (default from config)
        reset: If True, delete existing collection and create new one
        
    Returns:
        ChromaDB collection instance
    """
    if collection_name is None:
        collection_name = config.COLLECTION_NAME
    
    client = get_chroma_client()
    
    # Reset if requested
    if reset:
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass  # Collection doesn't exist
    
    # Get or create collection
    # ef_construction/max_neighbors only take effect when the collection is first
    # created; get_or_create ignores them for an existing collection. ef_search is
    # applied per query and can be retuned by re-creating with a new value.
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Multilingual scientific papers"},
        configuration={"hnsw": {
            "space": config.DISTANCE_METRIC,
            "ef_construction": config.HNSW_EF_CONSTRUCTION,
            "ef_search": config.HNSW_EF_SEARCH,
            "max_neighbors": config.HNSW_M,
        }}
    )
    
    logger.info(f"Collection '{collection_name}' ready. Current size: {collection.count()}")
    
    return collection


def add_documents(
    texts: List[str],
    embeddings: np.ndarray,
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    collection: chromadb.Collection = None
) -> None:
    """
    Add documents to the vector store.
    
    Args:
        texts: List of text chunks
        embeddings: Numpy array of embeddings, shape (n_docs, embedding_dim)
        metadatas: List of metadata dictionaries for each document
        ids: List of unique IDs for each document
        collection: ChromaDB collection (uses default if None)
    """
    if collection is None:
        collection = get_or_create_collection()
    
    # Convert embeddings to list of lists
    embeddings_list = embeddings.tolist()
    
    logger.info(f"Adding {len(ids)} chunks. Unique IDs: {len(set(ids))}")
    if len(ids) != len(set(ids)):
        from collections import Counter
        duplicates = {k: v for k, v in Counter(ids).items() if v > 1}
        logger.error(f"Duplicate IDs detected before upsert: {duplicates}")
    
    # Add to collection (upsert replaces existing, inserts new)
    _chroma_call(collection.upsert,
        documents=texts,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids
    )
    
    logger.info(f"Added {len(texts)} documents. Total in collection: {collection.count()}")


def search(
    query_embedding: np.ndarray,
    top_k: int = None,
    filter_dict: Optional[Dict[str, Any]] = None,
    collection: chromadb.Collection = None
) -> Dict[str, List]:
    """
    Search for similar documents using vector similarity.
    
    Args:
        query_embedding: Query embedding, shape (embedding_dim,)
        top_k: Number of results to return (default from config)
        filter_dict: Optional metadata filter (e.g., {"year": 2023})
        collection: ChromaDB collection (uses default if None)
        
    Returns:
        Dictionary with keys:
            - 'ids': List of document IDs
            - 'documents': List of document texts
            - 'metadatas': List of metadata dicts
            - 'distances': List of distances (lower is more similar)
    """
    if collection is None:
        collection = get_or_create_collection()
    
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
    
    # Convert embedding to list
    query_embedding_list = query_embedding.tolist()
    
    # Search
    results = _chroma_call(collection.query,
        query_embeddings=[query_embedding_list],
        n_results=top_k,
        where=filter_dict,
        include=["documents", "metadatas", "distances"]
    )
    
    # Flatten results (query returns list of lists)
    return {
        'ids': results['ids'][0],
        'documents': results['documents'][0],
        'metadatas': results['metadatas'][0],
        'distances': results['distances'][0]
    }


def delete_collection(collection_name: str = None) -> None:
    """
    Delete a collection from ChromaDB.
    
    Args:
        collection_name: Name of collection to delete (default from config)
    """
    if collection_name is None:
        collection_name = config.COLLECTION_NAME
    
    client = get_chroma_client()
    
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")


def get_collection_stats(collection: chromadb.Collection = None) -> Dict[str, Any]:
    """
    Get statistics about a collection.
    
    Args:
        collection: ChromaDB collection (uses default if None)
        
    Returns:
        Dictionary with collection statistics
    """
    if collection is None:
        collection = get_or_create_collection()
    
    count = _chroma_call(collection.count)
    
    # Get a sample to inspect metadata
    sample = _chroma_call(collection.peek, limit=1)
    
    stats = {
        'name': collection.name,
        'count': count,
        'metadata': collection.metadata,
    }
    
    if sample['metadatas']:
        stats['sample_metadata'] = sample['metadatas'][0]
    
    return stats

def get_paper_chunk_counts(collection: chromadb.Collection = None) -> Dict[str, int]:
    """Chunk count per paper_id, for ingestion health / re-ingest tooling."""
    if collection is None:
        collection = get_or_create_collection()
    got = _chroma_call(collection.get, include=['metadatas'])
    counts: Dict[str, int] = {}
    for meta in got.get('metadatas', []):
        pid = (meta or {}).get('paper_id', '')
        if pid:
            counts[pid] = counts.get(pid, 0) + 1
    return counts


def delete_by_paper_id(paper_id: str, collection: chromadb.Collection = None) -> int:
    """
    Delete all chunks for a specific paper.

    The BM25 index is updated here rather than in the route, because every
    deletion path routes through this function — doing it per caller meant the
    lexical index kept serving chunks of deleted papers until something happened
    to trigger a full rebuild, so deleted papers went on being cited.
    """
    if collection is None:
        collection = get_or_create_collection()
    ids = _chroma_call(collection.get, where={'paper_id': paper_id}, include=[])['ids']
    _chroma_call(collection.delete, where={'paper_id': paper_id})
    if ids:
        try:
            import bm25_search
            bm25_search.remove_from_index(ids, getattr(collection, "name", None))
        except Exception:
            # A stale lexical index is a quality bug, not a correctness one for the
            # delete itself — never fail the deletion over it.
            logger.warning("BM25 index not updated after deleting %s", paper_id, exc_info=True)
    return len(ids)


def update_paper_metadata(paper_id: str, updates: dict, collection: chromadb.Collection = None) -> int:
    """Update metadata fields on all chunks for a paper. Returns chunk count updated."""
    if collection is None:
        collection = get_or_create_collection()
    result = _chroma_call(collection.get, where={'paper_id': paper_id}, include=['metadatas'])
    ids = result.get('ids', [])
    if not ids:
        return 0
    new_metadatas = [{**m, **updates} for m in result['metadatas']]
    _chroma_call(collection.update, ids=ids, metadatas=new_metadatas)
    return len(ids)


def find_similar_paper(
    title: str,
    year: str = None,
    threshold: float = 0.9,
    collection: chromadb.Collection = None,
) -> Optional[str]:
    """Return an existing paper_id whose title is a near-duplicate of `title`, or None.

    Cross-ingestion dedup for re-uploads under a different filename. Uses
    difflib.SequenceMatcher (stdlib) rather than a fuzzy-matching dependency —
    good enough for near-identical title comparison at this corpus scale.
    """
    if collection is None:
        collection = get_or_create_collection()
    result = _chroma_call(collection.get, include=['metadatas'])
    seen: Dict[str, dict] = {}
    for meta in result.get('metadatas', []):
        pid = meta.get('paper_id')
        if pid and pid not in seen:
            seen[pid] = meta

    from difflib import SequenceMatcher
    norm_title = title.strip().lower()
    best_pid, best_ratio = None, 0.0
    for pid, meta in seen.items():
        if year and meta.get('year') and str(meta['year']) != str(year):
            continue
        ratio = SequenceMatcher(None, norm_title, str(meta.get('title', '')).strip().lower()).ratio()
        if ratio > best_ratio:
            best_pid, best_ratio = pid, ratio

    return best_pid if best_ratio >= threshold else None

if __name__ == "__main__":
    # Test vector store functionality
    print("Testing ChromaDB Vector Store")
    print("=" * 60)
    
    # Create test collection
    print("\n1. Creating test collection...")
    collection = get_or_create_collection("test_collection", reset=True)
    
    # Add test documents
    print("\n2. Adding test documents...")
    test_docs = [
        "Diabetes is a metabolic disease.",
        "Treatment includes insulin therapy.",
        "Machine learning can predict disease outcomes.",
    ]
    
    # Create dummy embeddings (in real use, these come from embedding model)
    test_embeddings = np.random.randn(len(test_docs), config.EMBEDDING_DIMENSION)
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    
    test_metadata = [
        {"paper_id": "paper1", "title": "Diabetes Research", "section": "introduction"},
        {"paper_id": "paper1", "title": "Diabetes Research", "section": "methods"},
        {"paper_id": "paper2", "title": "ML in Medicine", "section": "results"},
    ]
    
    test_ids = ["doc1", "doc2", "doc3"]
    
    add_documents(test_docs, test_embeddings, test_metadata, test_ids, collection)
    
    # Test search
    print("\n3. Testing search...")
    query_emb = np.random.randn(config.EMBEDDING_DIMENSION)
    query_emb = query_emb / np.linalg.norm(query_emb)
    
    results = search(query_emb, top_k=2, collection=collection)
    
    print(f"\nTop {len(results['documents'])} results:")
    for i, (doc, metadata, dist) in enumerate(zip(
        results['documents'],
        results['metadatas'],
        results['distances']
    )):
        print(f"\n{i+1}. Distance: {dist:.4f}")
        print(f"   Text: {doc}")
        print(f"   Metadata: {metadata}")
    
    # Get stats
    print("\n4. Collection statistics...")
    stats = get_collection_stats(collection)
    print(f"Name: {stats['name']}")
    print(f"Count: {stats['count']}")
    print(f"Sample metadata: {stats.get('sample_metadata', {})}")
    
    # Cleanup
    print("\n5. Cleaning up test collection...")
    delete_collection("test_collection")
    print("Done!")
