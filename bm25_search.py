"""BM25 lexical search index for hybrid retrieval (dense + sparse)."""
import math
import regex
import threading
import logging
from collections import Counter
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_indices: dict[str, "BM25Index"] = {}
_lock = threading.Lock()


class BM25Index:
    """Lightweight BM25 index that lives alongside ChromaDB's dense vectors."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.df: Counter = Counter()
        self.n_docs: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return regex.findall(r'\w+', text.lower())

    def build(self, ids: List[str], texts: List[str]):
        self.doc_ids = list(ids)
        self.doc_freqs = []
        self.doc_lens = []
        self.df = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            self.doc_lens.append(len(tokens))
            for term in freq:
                self.df[term] += 1

        self.n_docs = len(texts)
        self.avg_dl = sum(self.doc_lens) / self.n_docs if self.n_docs else 1.0

    def search(self, query: str, top_k: int = 30) -> Tuple[List[str], List[float]]:
        if self.n_docs == 0:
            return [], []

        query_terms = self._tokenize(query)
        scores = []

        for i in range(self.n_docs):
            score = 0.0
            dl = self.doc_lens[i]
            for term in query_terms:
                if term not in self.doc_freqs[i]:
                    continue
                tf = self.doc_freqs[i][term]
                df = self.df.get(term, 0)
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                score += idf * numerator / denominator
            scores.append(score)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.doc_ids[i] for i in ranked], [scores[i] for i in ranked]


def rrf(dense_ids: List[str], sparse_ids: List[str], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion of two ranked lists."""
    scores: Dict[str, float] = {}
    for rank, _id in enumerate(dense_ids, 1):
        scores[_id] = scores.get(_id, 0) + 1 / (k + rank)
    for rank, _id in enumerate(sparse_ids, 1):
        scores[_id] = scores.get(_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def get_or_build_index(collection=None) -> Optional[BM25Index]:
    """Return (and lazily build) the BM25 index for the given collection."""
    global _indices
    if collection is None:
        import vector_store
        collection = vector_store.get_or_create_collection()
    coll_name = getattr(collection, "name", "default")

    if coll_name in _indices:
        return _indices[coll_name]

    with _lock:
        if coll_name in _indices:
            return _indices[coll_name]

        count = collection.count()
        if count == 0:
            return None

        logger.info(f"Building BM25 index for '{coll_name}' from {count} documents...")
        all_docs = collection.get(include=["documents"])
        idx = BM25Index()
        idx.build(all_docs["ids"], all_docs["documents"])
        _indices[coll_name] = idx
        logger.info(f"BM25 index built for '{coll_name}'")

    return _indices[coll_name]


def invalidate():
    """Clear all cached indices (call after ingestion)."""
    global _indices
    with _lock:
        _indices = {}
