"""Claim-level faithfulness check via cross-encoder NLI."""
import re
import threading
import numpy as np
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
                # ponytail: nli-deberta-v3-base needed here — bge-reranker scores relevance,
                # not entailment; wrong distribution for faithfulness thresholding (BUG-003)
                _model = CrossEncoder("cross-encoder/nli-deberta-v3-base", device=device,
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
        pairs = [(chunks[i], sent) for i in cited]
        raw = np.atleast_2d(model.predict(pairs))  # (n, 3): contradiction, entailment, neutral
        # softmax → probabilities; entailment is label index 1 for nli-deberta-v3-base
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        score = float(probs[:, 1].max())
        results.append({"claim": sent, "support": score,
                        "grounded": score >= config.FAITHFULNESS_THRESHOLD})
    return results
