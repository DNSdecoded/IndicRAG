"""Claim-level faithfulness check via cross-encoder NLI."""
import logging
import re
import threading
import numpy as np
import torch
from typing import List
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
                # int8 ONNX is ~11x faster than fp32 on CPU (10s -> 0.9s/pair), which
                # is the difference between faithfulness fitting the agent timeout or
                # not. Falls back to fp32 torch if the ONNX stack/export is unavailable.
                try:
                    import onnx_ce
                    _model = onnx_ce.load(config.NLI_MODEL_NAME, "nli")
                    logger.info("[verify] using int8 ONNX NLI model")
                except Exception as e:
                    logger.warning(f"[verify] ONNX load failed ({e}); using fp32 torch")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    # An NLI model (not bge-reranker) is required — bge-reranker scores
                    # relevance, not entailment; wrong distribution for faithfulness
                    # thresholding (BUG-003). Default is now MULTILINGUAL so Indic-language
                    # claims are actually verified, not scored by an English-only model.
                    _model = CrossEncoder(config.NLI_MODEL_NAME, device=device,
                                          cache_folder=str(config.MODELS_CACHE_DIR))
    return _model


_CITE_ONLY_RE = re.compile(r'^(\[(?:\d+|NOT FOUND[^\]]*)\]\s*)+$')


def _paper_chunk_map(chunks: List[str], metadatas):
    """Map each per-paper citation number → the list of chunk texts for that paper.

    Citations are numbered per DISTINCT paper (first-seen title order), matching
    rag.citation_number_map / format_context — several chunks of one paper share
    one [N]. Indexing chunks[N-1] directly is wrong whenever a paper contributes
    more than one chunk (BUG: faithfulness scored claims against the wrong chunk).

    When metadatas is None (or absent), fall back to treating each chunk as its
    own paper, i.e. [N] → chunks[N-1] — the historical behaviour, still correct
    for callers where every retrieved chunk is a distinct document.
    """
    if not metadatas:
        return {i + 1: [c] for i, c in enumerate(chunks)}

    num_to_chunks: dict = {}
    title_to_num: dict = {}
    for chunk, meta in zip(chunks, metadatas):
        title = ((meta or {}).get('title') or 'Unknown').strip() or 'Unknown'
        num = title_to_num.get(title)
        if num is None:
            num = len(title_to_num) + 1
            title_to_num[title] = num
        num_to_chunks.setdefault(num, []).append(chunk)
    return num_to_chunks


def check_claims(answer: str, chunks: List[str], metadatas=None) -> List[dict]:
    """Return per-sentence support scores against the cited paper's chunk(s).

    metadatas (one dict per chunk, aligned with `chunks`) lets a [N] marker
    resolve to ALL chunks of the Nth distinct paper — see _paper_chunk_map.
    """
    model = _load()
    num_to_chunks = _paper_chunk_map(chunks, metadatas)
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
        cited_nums = {int(n) for n in re.findall(r'\[(\d+)\]', sent)}
        cited_chunks = [c for n in cited_nums for c in num_to_chunks.get(n, [])]
        if not cited_chunks:
            continue
        # Strip citation/not-found markers before scoring — leaving literal
        # "[Cite:1]" text in the NLI hypothesis is out-of-distribution input
        # that collapses entailment probability toward 0 regardless of how
        # well the chunk actually supports the claim (verified: 0.998 -> 0.009
        # entailment on an identical claim/chunk pair, marker text alone
        # flips the model's dominant class to "neutral"). This was silently
        # forcing faithfulness scores toward 0 on essentially every answer,
        # since every cited claim carries a [Cite:N] marker by construction.
        clean_sent = re.sub(r'\[(?:\d+|NOT FOUND[^\]]*)\]', '', sent).strip()
        if not clean_sent:
            continue
        pairs = [(chunk, clean_sent) for chunk in cited_chunks]
        raw = np.atleast_2d(model.predict(pairs))  # (n, num_labels) NLI logits
        # softmax → probabilities; entailment column depends on the model
        # (config.NLI_ENTAILMENT_INDEX), since label order differs across NLI models.
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        score = float(probs[:, config.NLI_ENTAILMENT_INDEX].max())
        results.append({"claim": sent, "support": score,
                        "grounded": score >= config.FAITHFULNESS_THRESHOLD})
    return results
