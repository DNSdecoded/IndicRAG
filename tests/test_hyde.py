"""Unit tests for HyDE retrieval path (rag._hyde_embedding, retrieve_context use_hyde)."""

from unittest.mock import MagicMock, patch

import numpy as np

import rag


def test_hyde_embedding_uses_hypothetical_answer_on_success():
    fake_response = MagicMock()
    fake_response.text = "Diabetes is treated with insulin and lifestyle changes."
    fake_vec = np.array([0.1, 0.2, 0.3])

    with patch("llm_client.generate_with_failover", return_value=fake_response) as mock_gen, \
         patch("embeddings.embed_query", return_value=fake_vec) as mock_embed:
        result = rag._hyde_embedding("How is diabetes treated?")

    mock_gen.assert_called_once()
    mock_embed.assert_called_once_with("Diabetes is treated with insulin and lifestyle changes.")
    assert np.array_equal(result, fake_vec)


def test_hyde_embedding_falls_back_on_llm_failure():
    fake_vec = np.array([0.4, 0.5, 0.6])

    with patch("llm_client.generate_with_failover", side_effect=RuntimeError("LLM down")), \
         patch("embeddings.embed_query", return_value=fake_vec) as mock_embed:
        result = rag._hyde_embedding("How is diabetes treated?")

    mock_embed.assert_called_once_with("How is diabetes treated?")
    assert np.array_equal(result, fake_vec)


def test_hyde_embedding_falls_back_on_empty_response():
    fake_response = MagicMock()
    fake_response.text = ""
    fake_response.candidates = []
    fake_vec = np.array([0.7, 0.8, 0.9])

    with patch("llm_client.generate_with_failover", return_value=fake_response), \
         patch("embeddings.embed_query", return_value=fake_vec) as mock_embed:
        result = rag._hyde_embedding("How is diabetes treated?")

    mock_embed.assert_called_once_with("How is diabetes treated?")
    assert np.array_equal(result, fake_vec)


def test_retrieve_context_use_hyde_false_skips_hyde_draft():
    with patch("rag._hyde_embedding") as mock_hyde, \
         patch("embeddings.embed_query", return_value=np.array([0.1])) as mock_embed, \
         patch("vector_store.search", return_value={
             "ids": [], "documents": [], "metadatas": [], "distances": []
         }):
        rag.retrieve_context("q", top_k=3, collection=MagicMock(), use_hyde=False)

    mock_hyde.assert_not_called()
    mock_embed.assert_called_once()


def test_retrieve_context_use_hyde_true_calls_hyde_draft():
    with patch("rag._hyde_embedding", return_value=np.array([0.1])) as mock_hyde, \
         patch("vector_store.search", return_value={
             "ids": [], "documents": [], "metadatas": [], "distances": []
         }):
        rag.retrieve_context("q", top_k=3, collection=MagicMock(), use_hyde=True)

    mock_hyde.assert_called_once_with("q")
