"""BM25 lexical search index for hybrid retrieval (dense + sparse).

Inverted index: `term -> [(doc_idx, tf), ...]`. Scoring touches only the postings
of the query's own terms, so cost tracks how often those terms occur rather than
how big the corpus is.

The earlier version scored by looping over EVERY document for EVERY query
(`for i in range(self.n_docs)`) and kept a `Counter` per chunk resident to do it.
At ~500k chunks that is multiple GB of Counters and a per-query scan in the
hundreds of milliseconds — sparse retrieval, not the cross-encoder, becomes the
dominant term in p99.

Updates are incremental (`add_documents`, `remove_documents`), so an ingest no
longer throws the whole index away and leaves the next query paying for a full
rebuild.
"""
import gzip
import json
import math
import os
import regex
import threading
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# Bump when the on-disk shape changes; an older file is then ignored, not
# misread — a silently misinterpreted index is worse than a rebuild.
_CACHE_FORMAT = 1

_indices: dict[str, "BM25Index"] = {}
_lock = threading.Lock()


class BM25Index:
    """Lightweight BM25 inverted index that lives alongside ChromaDB's dense vectors."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_lens: List[int] = []
        # term -> [(doc_idx, term_frequency), ...]
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # Removed documents are tombstoned rather than compacted out: reclaiming a
        # slot would renumber every later doc_idx and invalidate every posting.
        self._deleted: set[int] = set()
        self._total_len = 0
        self._lock = threading.Lock()

    @property
    def n_docs(self) -> int:
        """Live document count (tombstones excluded)."""
        return len(self.doc_ids) - len(self._deleted)

    @property
    def avg_dl(self) -> float:
        n = self.n_docs
        return (self._total_len / n) if n else 1.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return regex.findall(r'[\p{L}\p{M}\p{N}]+', text.lower())

    # -- construction -------------------------------------------------------
    def build(self, ids: List[str], texts: List[str]):
        """Build from scratch, discarding any existing content."""
        with self._lock:
            self.doc_ids = []
            self.doc_lens = []
            self.postings = defaultdict(list)
            self._deleted = set()
            self._total_len = 0
            self._add_unlocked(ids, texts)

    def add_documents(self, ids: List[str], texts: List[str]) -> None:
        """Append documents without rebuilding.

        Re-adding an existing id tombstones the old copy first, so an updated
        chunk is not scored twice.
        """
        with self._lock:
            self._add_unlocked(ids, texts)

    def _add_unlocked(self, ids: List[str], texts: List[str]) -> None:
        existing = {d: i for i, d in enumerate(self.doc_ids) if i not in self._deleted}
        for doc_id, text in zip(ids, texts):
            prior = existing.get(doc_id)
            if prior is not None:
                self._tombstone(prior)
            tokens = self._tokenize(text)
            idx = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.doc_lens.append(len(tokens))
            self._total_len += len(tokens)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((idx, tf))
            existing[doc_id] = idx

    def remove_documents(self, ids) -> int:
        """Tombstone documents by id. Returns how many were removed.

        Deleting a paper used to leave its chunks in the index until something
        triggered a full rebuild, so deleted papers kept being retrieved and cited.
        """
        wanted = set(ids)
        removed = 0
        with self._lock:
            for i, doc_id in enumerate(self.doc_ids):
                if doc_id in wanted and i not in self._deleted:
                    self._tombstone(i)
                    removed += 1
        return removed

    def _tombstone(self, idx: int) -> None:
        """Mark a slot dead. Postings stay put and are skipped at query time —
        rewriting every posting list on each delete would cost far more than the
        occasional skipped entry."""
        self._deleted.add(idx)
        self._total_len -= self.doc_lens[idx]

    # -- query --------------------------------------------------------------
    def search(self, query: str, top_k: int = 30) -> Tuple[List[str], List[float]]:
        n_docs = self.n_docs
        if n_docs == 0:
            return [], []

        query_terms = self._tokenize(query)
        if not query_terms:
            return [], []

        avg_dl = self.avg_dl
        scores: Dict[int, float] = defaultdict(float)

        # Each query term is looked at once, and only the documents that actually
        # contain it are touched — this is the whole point of the inverted index.
        for term in set(query_terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            df = sum(1 for idx, _ in postings if idx not in self._deleted)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for idx, tf in postings:
                if idx in self._deleted:
                    continue
                dl = self.doc_lens[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom

        if not scores:
            return [], []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self.doc_ids[i] for i, _ in ranked], [s for _, s in ranked]


def rrf(dense_ids: List[str], sparse_ids: List[str], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion of two ranked lists."""
    scores: Dict[str, float] = {}
    for rank, _id in enumerate(dense_ids, 1):
        scores[_id] = scores.get(_id, 0) + 1 / (k + rank)
    for rank, _id in enumerate(sparse_ids, 1):
        scores[_id] = scores.get(_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _cache_path(coll_name: str):
    # .json.gz, because that is what it is — naming a JSON file .pkl invites the
    # next reader to reach for pickle.load().
    safe = regex.sub(r"[^\w.-]", "_", coll_name)
    return config.BM25_CACHE_DIR / f"bm25_{safe}.json.gz"


def _load_cached(coll_name: str, expected_docs: int) -> Optional["BM25Index"]:
    """Load a persisted index, or None if it is missing, unreadable, or stale.

    Staleness is judged on live document count: the corpus changing under a saved
    index is exactly when it must not be trusted. A corrupt or version-mismatched
    file is not an error — it just means rebuild.
    """
    path = _cache_path(coll_name)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != _CACHE_FORMAT:
            return None
        idx = BM25Index(k1=payload["k1"], b=payload["b"])
        idx.doc_ids = payload["doc_ids"]
        idx.doc_lens = payload["doc_lens"]
        idx.postings = defaultdict(list, {t: [tuple(p) for p in ps]
                                          for t, ps in payload["postings"].items()})
        idx._deleted = set(payload["deleted"])
        idx._total_len = payload["total_len"]
        if idx.n_docs != expected_docs:
            logger.info("BM25 cache for '%s' is stale (%d docs on disk vs %d live); rebuilding",
                        coll_name, idx.n_docs, expected_docs)
            return None
        logger.info("BM25 index for '%s' loaded from cache (%d docs)", coll_name, idx.n_docs)
        return idx
    except Exception:
        logger.warning("BM25 cache unreadable for '%s'; rebuilding", coll_name, exc_info=True)
        return None


def save_index(coll_name: str = None) -> bool:
    """Persist an in-memory index so the next process start skips the rebuild.

    Gzipped JSON rather than pickle: this file is read back and trusted at
    startup, and unpickling attacker-controlled bytes is arbitrary code
    execution. JSON can only ever produce data. Gzip keeps it compact — the
    postings dominate the size.

    Best-effort: a failure to save costs a rebuild later, never a request now.
    Writes to a temp file and replaces atomically, so a crash mid-write cannot
    leave a half-written index for the next start to load.
    """
    with _lock:
        if coll_name is None:
            items = list(_indices.items())
        else:
            items = [(coll_name, _indices[coll_name])] if coll_name in _indices else []
    if not items:
        return False
    ok = True
    for name, idx in items:
        path = _cache_path(name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "format": _CACHE_FORMAT,
                "k1": idx.k1,
                "b": idx.b,
                "doc_ids": idx.doc_ids,
                "doc_lens": idx.doc_lens,
                "postings": {t: [list(p) for p in ps] for t, ps in idx.postings.items()},
                "deleted": sorted(idx._deleted),
                "total_len": idx._total_len,
            }
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"))
            os.replace(tmp, path)
            logger.info("BM25 index for '%s' saved to %s (%d docs)", name, path, idx.n_docs)
        except Exception:
            logger.warning("Could not persist BM25 index for '%s'", name, exc_info=True)
            ok = False
    return ok


def get_or_build_index(collection=None) -> Optional[BM25Index]:
    """Return (and lazily build) the BM25 index for the given collection.

    Tries the on-disk cache first: rebuilding reads every document out of
    ChromaDB, so a restart otherwise pays a full corpus scan on the first query
    (and the lifespan warm-up pays it on every deploy).
    """
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

        if config.BM25_PERSIST:
            cached = _load_cached(coll_name, count)
            if cached is not None:
                _indices[coll_name] = cached
                return cached

        logger.info(f"Building BM25 index for '{coll_name}' from {count} documents...")
        all_docs = collection.get(include=["documents"])
        idx = BM25Index()
        idx.build(all_docs["ids"], all_docs["documents"])
        _indices[coll_name] = idx
        logger.info(f"BM25 index built for '{coll_name}'")

    if config.BM25_PERSIST:
        save_index(coll_name)
    return _indices[coll_name]


def add_to_index(ids: List[str], texts: List[str], collection_name: str = None) -> bool:
    """Fold newly ingested chunks into a live index.

    Returns False when no index is built yet — the caller should then leave it
    alone and let the next query build it, rather than forcing a rebuild.
    """
    if not ids:
        return True
    with _lock:
        if collection_name is not None:
            targets = [_indices[collection_name]] if collection_name in _indices else []
        else:
            targets = list(_indices.values())
    if not targets:
        return False
    for idx in targets:
        idx.add_documents(ids, texts)
    return True


def remove_from_index(ids, collection_name: str = None) -> bool:
    """Drop deleted chunks from a live index. Returns False if none is built."""
    with _lock:
        if collection_name is not None:
            targets = [_indices[collection_name]] if collection_name in _indices else []
        else:
            targets = list(_indices.values())
    if not targets:
        return False
    for idx in targets:
        idx.remove_documents(ids)
    return True


def invalidate():
    """Clear all cached indices (full rebuild on next query).

    Prefer add_to_index / remove_from_index: this drops everything, and the next
    query pays a full rebuild from ChromaDB while holding the build lock.
    """
    global _indices
    with _lock:
        _indices = {}
