"""Unit tests for vector_store.py — delete count and timeout wrapper."""
import time
import pytest
from unittest.mock import MagicMock


def test_delete_returns_count():
    """delete_by_paper_id returns the number of chunks deleted (fetched before delete)."""
    import vector_store

    coll = MagicMock()
    coll.get.return_value = {"ids": ["a", "b"]}

    result = vector_store.delete_by_paper_id("paper-x", collection=coll)

    assert result == 2
    coll.delete.assert_called_once()


def test_timeout_wrapper():
    """_chroma_call raises TimeoutError when the ChromaDB operation exceeds the timeout."""
    from vector_store import _chroma_call

    def slow_fn():
        time.sleep(0.5)  # longer than the timeout below

    with pytest.raises(TimeoutError):
        _chroma_call(slow_fn, timeout=0.1)
