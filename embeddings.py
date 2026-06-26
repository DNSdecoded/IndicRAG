"""
Multilingual embedding model management.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import logging
import threading
import config
import torch

logger = logging.getLogger(__name__)


# Global model cache
_embedding_model = None
_lock = threading.Lock()


def load_embedding_model(model_name: str = None) -> SentenceTransformer:
    """
    Load the multilingual embedding model with caching (thread-safe).
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    with _lock:
        if _embedding_model is not None:
            return _embedding_model

        if model_name is None:
            model_name = config.EMBEDDING_MODEL_NAME

        logger.info(f"Loading embedding model: {model_name}")
        logger.info("This may take a few minutes on first run...")

        cache_dir = str(config.MODELS_CACHE_DIR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device=device
        )

        logger.info(f"Model loaded on device: {device}")
        logger.info(f"Embedding dimension: {model.get_embedding_dimension()}")
        _embedding_model = model

    return _embedding_model


def embed_texts(
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True,
    is_query: bool = False
) -> np.ndarray:
    """
    Embed a list of texts using the multilingual model.
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar
        is_query: If True, add query prefix for E5 models
        
    Returns:
        Numpy array of embeddings, shape (len(texts), embedding_dim)
    """
    model = load_embedding_model()
    
    # Add E5 prefix if using E5 model (check for 'e5' in model name)
    if "e5" in config.EMBEDDING_MODEL_NAME.lower():
        prefix = config.E5_QUERY_PREFIX if is_query else config.E5_PASSAGE_PREFIX
        texts = [prefix + text for text in texts]
    
    # Encode
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )
    
    return embeddings


_query_cache: dict = {}
_QUERY_CACHE_MAX = 128
_query_cache_lock = threading.Lock()


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query text (with LRU cache).
    """
    key = query.strip().lower()
    with _query_cache_lock:
        if key in _query_cache:
            return _query_cache[key]
    result = embed_texts([query], batch_size=1, show_progress=False, is_query=True)[0]
    with _query_cache_lock:
        if len(_query_cache) >= _QUERY_CACHE_MAX:
            _query_cache.pop(next(iter(_query_cache)))
        _query_cache[key] = result
    return result


def embed_passages(passages: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of passage texts (documents/chunks).
    
    Args:
        passages: List of passage texts to embed
        batch_size: Batch size for encoding
        
    Returns:
        Numpy array of embeddings, shape (len(passages), embedding_dim)
    """
    return embed_texts(passages, batch_size=batch_size, show_progress=True, is_query=False)


def compute_similarity(query_embedding: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple passages.
    
    Args:
        query_embedding: Query embedding, shape (embedding_dim,)
        passage_embeddings: Passage embeddings, shape (n_passages, embedding_dim)
        
    Returns:
        Similarity scores, shape (n_passages,)
    """
    # Cosine similarity (embeddings are already normalized)
    similarities = np.dot(passage_embeddings, query_embedding)
    return similarities


if __name__ == "__main__":
    # Test embedding functionality
    print("Testing Multilingual Embedding Model")
    print("=" * 60)
    
    # Test texts in different languages
    test_texts = [
        "What is the treatment for diabetes?",  # English
        "मधुमेह का इलाज क्या है?",  # Hindi
        "நீரிழிவு நோய்க்கான சிகிச்சை என்ன?",  # Tamil
        "డయాబెటిస్ చికిత్స ఏమిటి?",  # Telugu
    ]
    
    print("\n1. Loading model...")
    model = load_embedding_model()
    
    print("\n2. Embedding test texts...")
    embeddings = embed_passages(test_texts)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Expected shape: ({len(test_texts)}, {config.EMBEDDING_DIMENSION})")
    
    print("\n3. Testing cross-lingual similarity...")
    query = "diabetes treatment"
    query_emb = embed_query(query)
    
    similarities = compute_similarity(query_emb, embeddings)
    
    print(f"\nQuery: '{query}'")
    print("\nSimilarities:")
    for text, sim in zip(test_texts, similarities):
        print(f"  {sim:.4f} - {text}")
    
    print("\n4. Testing multilingual queries...")
    queries = [
        "diabetes treatment",  # English
        "मधुमेह उपचार",  # Hindi
    ]
    
    for q in queries:
        q_emb = embed_query(q)
        sims = compute_similarity(q_emb, embeddings)
        print(f"\nQuery: '{q}'")
        print(f"Top match: {test_texts[np.argmax(sims)]} (similarity: {np.max(sims):.4f})")
