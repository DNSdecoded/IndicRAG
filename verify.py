"""Claim-level faithfulness check via cross-encoder NLI."""
import re
import threading
import torch
from typing import List
from sentence_transformers import CrossEncoder
import config

_model = None
_lock = threading.Lock()


def _load():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device,
                                      cache_folder=str(config.MODELS_CACHE_DIR))
    return _model


def check_claims(answer: str, chunks: List[str]) -> List[dict]:
    """Return per-sentence support scores against the cited chunk."""
    model = _load()
    results = []
    for sent in re.split(r'(?<=[.!?])\s+', answer):
        cited = [int(n) - 1 for n in re.findall(r'\[(\d+)\]', sent)]
        cited = [i for i in cited if 0 <= i < len(chunks)]
        if not cited:
            continue
        score = max(model.predict([(chunks[i], sent)])[0] for i in cited)
        results.append({"claim": sent, "support": float(score),
                        "grounded": score >= config.FAITHFULNESS_THRESHOLD})
    return results
