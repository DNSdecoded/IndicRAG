"""Cross-encoder reranking with a lazy, thread-safe singleton."""
import threading
import logging
import torch
from typing import List, Tuple
from sentence_transformers import CrossEncoder
import config

logger = logging.getLogger(__name__)
_model = None
_lock = threading.Lock()


def _load():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                # int8 ONNX is ~3x faster than fp32 on CPU (35s -> 11s / 15 pairs).
                # Falls back to fp32 torch if the ONNX stack/export is unavailable.
                try:
                    import onnx_ce
                    _model = onnx_ce.load(config.RERANK_MODEL_NAME, "rerank")
                    logger.info("Loading reranker (int8 ONNX)")
                except Exception as e:
                    logger.warning(f"Reranker ONNX load failed ({e}); using fp32 torch")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Loading reranker {config.RERANK_MODEL_NAME} on {device}")
                    _model = CrossEncoder(config.RERANK_MODEL_NAME, device=device,
                                          cache_folder=str(config.MODELS_CACHE_DIR))
    return _model


def rerank(query: str, docs: List[str], metadatas: List[dict],
           top_k: int) -> Tuple[List[str], List[dict], List[float]]:
    if not docs:
        return [], [], []
    model = _load()
    scores = model.predict([(query, d) for d in docs], convert_to_numpy=True)
    order = scores.argsort()[::-1][:top_k]
    return ([docs[i] for i in order],
            [metadatas[i] for i in order],
            [float(scores[i]) for i in order])
