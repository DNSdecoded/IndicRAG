"""
Phase 5 — contradiction / consensus detection.

Reuse the faithfulness NLI cross-encoder (``verify._load`` /
``config.NLI_MODEL_NAME``) to find pairs of retrieved passages that *contradict*
each other. When sources disagree, the answer generator is told to present both
positions with citations instead of silently picking one.

Gated by ``config.CONTRADICTION_DETECT_ENABLE`` in the caller. Cost is bounded:
only the top ``_MAX_ITEMS`` passages are compared, O(n^2) NLI passes over a small
capped set. Threshold: ``config.CONTRADICTION_NLI_THRESHOLD``.
"""

import itertools
import logging
from typing import Any, Dict, List, Optional

import numpy as np

import config
import verify  # reuse the already-loaded NLI model

logger = logging.getLogger(__name__)

_MAX_ITEMS = 8       # cap the passage set → at most C(8,2)*2 = 56 NLI passes
_NLI_MAXLEN = 600    # truncate passages; NLI models cap at ~512 tokens anyway


def _contradiction_probs(pairs: List[tuple]) -> np.ndarray:
    """Contradiction probability for each (premise, hypothesis) pair."""
    model = verify._load()
    raw = np.atleast_2d(model.predict(pairs))  # (n, num_labels) NLI logits
    e = np.exp(raw - raw.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs[:, config.NLI_CONTRADICTION_INDEX]


def find_contradictions(
    chunks: List[str],
    metadatas: Optional[List[Dict]] = None,
    threshold: Optional[float] = None,
    max_items: int = _MAX_ITEMS,
) -> List[Dict[str, Any]]:
    """Find contradicting passage pairs among the top retrieved chunks.

    NLI is directional, so each pair is scored both ways and the max
    contradiction probability is kept. Returns one entry per contradicting pair
    (deduped by title pair), highest score first:

        [{'a_title', 'b_title', 'score'}, ...]

    Passages from the same paper are skipped — a paper contradicting itself is
    almost always a chunking artifact, not a real disagreement.
    """
    if threshold is None:
        threshold = config.CONTRADICTION_NLI_THRESHOLD

    items = [(c or "")[:_NLI_MAXLEN] for c in chunks[:max_items]]
    metas = (metadatas or [])[:max_items]

    def _title(i: int) -> str:
        m = metas[i] if i < len(metas) else None
        return ((m or {}).get("title") or "Unknown").strip() or "Unknown"

    found: Dict[tuple, Dict[str, Any]] = {}
    for i, j in itertools.combinations(range(len(items)), 2):
        ti, tj = _title(i), _title(j)
        if ti == tj:
            continue  # same paper — skip self-contradiction from chunking
        if not items[i].strip() or not items[j].strip():
            continue
        score = float(_contradiction_probs(
            [(items[i], items[j]), (items[j], items[i])]
        ).max())
        if score < threshold:
            continue
        key = tuple(sorted((ti, tj)))
        if score > found.get(key, {"score": 0.0})["score"]:
            found[key] = {"a_title": key[0], "b_title": key[1], "score": score}

    return sorted(found.values(), key=lambda d: d["score"], reverse=True)


def contradiction_instruction(contradictions: List[Dict[str, Any]]) -> str:
    """Prompt fragment telling the model to present both sides, or '' if none."""
    if not contradictions:
        return ""
    pairs = "; ".join(f'"{c["a_title"]}" vs "{c["b_title"]}"' for c in contradictions)
    return (
        "\n\nNOTE — the retrieved sources appear to DISAGREE on some points "
        f"({pairs}). Where the passages conflict, do not silently pick one: state "
        "both positions explicitly and cite the source [N] for each side."
    )


if __name__ == "__main__":
    # Self-check: detection logic is correct without loading a real NLI model.
    logging.basicConfig(level=logging.INFO)
    import contradiction as cd

    class _FakeModel:
        """predict() → high contradiction logit iff the two texts differ."""
        def predict(self, pairs):
            out = []
            for premise, hypothesis in pairs:
                if premise.strip() != hypothesis.strip():
                    logits = [0.0, 0.0, 0.0]
                    logits[config.NLI_CONTRADICTION_INDEX] = 6.0  # contradiction
                else:
                    logits = [0.0, 0.0, 0.0]
                    logits[config.NLI_ENTAILMENT_INDEX] = 6.0
                out.append(logits)
            return np.array(out)

    verify._model = _FakeModel()  # bypass _load()

    chunks = ["antenna resonates at 2.4 GHz", "antenna resonates at 5 GHz", "sky is blue"]
    metas = [{"title": "Paper A"}, {"title": "Paper B"}, {"title": "Paper A"}]
    cons = cd.find_contradictions(chunks, metas, threshold=0.6)
    # A-vs-B and B-vs-A(sky) contradict; A-vs-A pair is skipped (same title).
    titles = {tuple(sorted((c["a_title"], c["b_title"]))) for c in cons}
    assert ("Paper A", "Paper B") in titles, titles
    assert all(c["a_title"] != c["b_title"] for c in cons)
    assert all(c["score"] >= 0.6 for c in cons)

    # Identical passages → no contradiction.
    same = cd.find_contradictions(["x", "x"], [{"title": "A"}, {"title": "B"}], threshold=0.6)
    assert same == [], same

    # Instruction fragment.
    assert contradiction_instruction([]) == ""
    frag = contradiction_instruction([{"a_title": "A", "b_title": "B", "score": 0.9}])
    assert "DISAGREE" in frag and '"A" vs "B"' in frag

    print("contradiction self-check passed")
