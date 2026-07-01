"""Unit tests for colbert_rerank.py MaxSim fusion (no real BGE-M3 model loaded)."""

from unittest.mock import patch

import numpy as np
import pytest

import colbert_rerank


def test_maxsim_identical_vectors_score_high():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert colbert_rerank._maxsim(q, d) == pytest.approx(2.0)


def test_maxsim_orthogonal_vectors_score_zero():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    d = np.array([[0.0, 1.0]], dtype=np.float32)
    assert colbert_rerank._maxsim(q, d) == pytest.approx(0.0)


def test_rerank_empty_docs_returns_empty():
    assert colbert_rerank.rerank("q", [], [], [], top_k=5) == ([], [], [])


def test_rerank_reorders_by_fused_score():
    fake_model = type("FakeModel", (), {})()

    def fake_encode(texts, **kwargs):
        if texts == ["query"]:
            return {"colbert_vecs": [np.array([[1.0, 0.0]], dtype=np.float32)]}
        # doc 0: orthogonal to query (low colbert score); doc 1: aligned (high)
        return {
            "colbert_vecs": [
                np.array([[0.0, 1.0]], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
            ]
        }

    fake_model.encode = fake_encode

    with patch("colbert_rerank._load", return_value=fake_model):
        docs, metas, scores = colbert_rerank.rerank(
            "query",
            ["doc_low", "doc_high"],
            [{"id": 0}, {"id": 1}],
            dense_similarities=[0.5, 0.5],  # tie on dense so colbert decides order
            top_k=2,
            weight=0.0,  # pure colbert signal for this test
        )

    assert docs[0] == "doc_high"
    assert docs[1] == "doc_low"
    assert scores[0] > scores[1]
