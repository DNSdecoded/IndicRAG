"""ColBERT multi-vector MaxSim reranking, query-time only.

ponytail: no persistent per-chunk ColBERT index synced with ChromaDB (the
plan doc's colbert_index.py). It reranks the already-small hybrid-fusion
candidate set (tens of chunks, not the full corpus), so re-encoding token
vectors per query is cheap and avoids a second on-disk store that would need
its own ingest/delete/dedup sync logic to stay consistent with ChromaDB.

Requires the `FlagEmbedding` package (BGEM3FlagModel) for token-level
`colbert_vecs` — the `sentence_transformers` wrapper used elsewhere in this
codebase (embeddings.py) only exposes the pooled dense embedding.
"""

import logging
import threading
from typing import List, Tuple

import numpy as np
import torch

import config

logger = logging.getLogger(__name__)
_model = None
_lock = threading.Lock()


def _load():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from FlagEmbedding import BGEM3FlagModel
                use_fp16 = torch.cuda.is_available()
                logger.info(f"Loading ColBERT model {config.EMBEDDING_MODEL_NAME} (fp16={use_fp16})")
                _model = BGEM3FlagModel(
                    config.EMBEDDING_MODEL_NAME,
                    use_fp16=use_fp16,
                    cache_dir=str(config.MODELS_CACHE_DIR),
                )
    return _model


def _maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """Sum over query tokens of the max cosine sim to any doc token."""
    sims = query_vecs @ doc_vecs.T  # (n_query_tokens, n_doc_tokens)
    return float(sims.max(axis=1).sum())


def rerank(
    query: str,
    docs: List[str],
    metadatas: List[dict],
    dense_similarities: List[float],
    top_k: int,
    weight: float = 0.5,
) -> Tuple[List[str], List[dict], List[float]]:
    """Fuse dense cosine similarity with ColBERT MaxSim score (weighted), return reordered top_k.

    ColBERT token vectors are stored as float16 (config knob is fp16 on GPU;
    numpy arrays here stay float32 for the matmul, converted back down only
    matters for a persistent index, which this module intentionally has none of).
    """
    if not docs:
        return [], [], []

    model = _load()
    query_out = model.encode([query], return_dense=False, return_sparse=False, return_colbert_vecs=True)
    query_vecs = np.asarray(query_out["colbert_vecs"][0], dtype=np.float32)

    doc_out = model.encode(docs, return_dense=False, return_sparse=False, return_colbert_vecs=True)
    colbert_scores = [_maxsim(query_vecs, np.asarray(v, dtype=np.float32)) for v in doc_out["colbert_vecs"]]

    # Normalize ColBERT scores (unbounded sum-of-maxsims) to roughly [0, 1]
    # so the weighted fusion with dense cosine similarity is meaningful.
    max_cs = max(colbert_scores) or 1.0
    norm_colbert = [s / max_cs for s in colbert_scores]

    fused = [weight * d + (1 - weight) * c for d, c in zip(dense_similarities, norm_colbert)]
    order = np.argsort(fused)[::-1][:top_k]
    return (
        [docs[i] for i in order],
        [metadatas[i] for i in order],
        [float(fused[i]) for i in order],
    )
