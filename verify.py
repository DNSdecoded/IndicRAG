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


_CITE_ONLY_RE = re.compile(r'^(\[(?:Cite:\s*\d+|NOT FOUND[^\]]*)\]\s*)+$')


def check_claims(answer: str, chunks: List[str]) -> List[dict]:
    """Return per-sentence support scores against the cited chunk."""
    model = _load()
    raw_sentences = re.split(r'(?<=[.!?।॥])\s+', answer)

    # Merge citation-only fragments into the previous sentence — the LLM often
    # places [Cite:N] right after the sentence-ending period, so the naive
    # split above orphans the marker into its own "sentence" with no claim
    # text. Scoring a bare "[Cite:N]" against a chunk can't entail anything,
    # so this was silently forcing faithfulness toward 0 on nearly every answer.
    sentences = []
    for frag in raw_sentences:
        if sentences and _CITE_ONLY_RE.match(frag.strip()):
            sentences[-1] = sentences[-1] + ' ' + frag
        else:
            sentences.append(frag)

    results = []
    for sent in sentences:
        cited = [int(n) - 1 for n in re.findall(r'\[Cite:\s*(\d+)\]', sent)]
        cited = [i for i in cited if 0 <= i < len(chunks)]
        if not cited:
            continue
        # Strip citation/not-found markers before scoring — leaving literal
        # "[Cite:1]" text in the NLI hypothesis is out-of-distribution input
        # that collapses entailment probability toward 0 regardless of how
        # well the chunk actually supports the claim (verified: 0.998 -> 0.009
        # entailment on an identical claim/chunk pair, marker text alone
        # flips the model's dominant class to "neutral"). This was silently
        # forcing faithfulness scores toward 0 on essentially every answer,
        # since every cited claim carries a [Cite:N] marker by construction.
        clean_sent = re.sub(r'\[(?:Cite:\s*\d+|NOT FOUND[^\]]*)\]', '', sent).strip()
        if not clean_sent:
            continue
        pairs = [(chunks[i], clean_sent) for i in cited]
        raw = np.atleast_2d(model.predict(pairs))  # (n, 3): contradiction, entailment, neutral
        # softmax → probabilities; entailment is label index 1 for nli-deberta-v3-base
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        score = float(probs[:, 1].max())
        results.append({"claim": sent, "support": score,
                        "grounded": score >= config.FAITHFULNESS_THRESHOLD})
    return results
