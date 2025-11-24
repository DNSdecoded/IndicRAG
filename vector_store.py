"""
Vector store operations using ChromaDB.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import numpy as np
import logging
import config

logger = logging.getLogger(__name__)

# Global client cache
_chroma_client = None

# Global collection cache
# NOTE: _collection stores the LAST USED collection. Test code should always use
# the collection object returned by get_or_create_collection(), not this global.
_collection = None


def get_chroma_client() -> chromadb.Client:
    """
    Get or create ChromaDB client with persistence.
    
    Returns:
        ChromaDB client instance
    """
    global _chroma_client
    
    if _chroma_client is not None:
        return _chroma_client
    
    logger.info(f"Initializing ChromaDB at: {config.CHROMA_DB_DIR}")
    
    # Create persistent client
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
    global _collection
    
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
    _collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": config.DISTANCE_METRIC,
            "description": "Multilingual scientific papers"
        }
    )
    
    logger.info(f"Collection '{collection_name}' ready. Current size: {_collection.count()}")
    
    return _collection


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
    
    # Add to collection
    collection.add(
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
    results = collection.query(
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
    
    count = collection.count()
    
    # Get a sample to inspect metadata
    sample = collection.peek(limit=1)
    
    stats = {
        'name': collection.name,
        'count': count,
        'metadata': collection.metadata,
    }
    
    if sample['metadatas']:
        stats['sample_metadata'] = sample['metadatas'][0]
    
    return stats


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
