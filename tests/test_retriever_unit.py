import numpy as np
import pytest

from app.rag.retriever import _as_faiss_query


def test_as_faiss_query_from_list_1d():
    vec = [0.1, 0.2, 0.3]
    arr = _as_faiss_query(vec)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (1, 3)


def test_as_faiss_query_from_numpy_1d():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    arr = _as_faiss_query(vec)
    assert arr.dtype == np.float32
    assert arr.shape == (1, 3)


def test_as_faiss_query_rejects_bad_shape():
    bad = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        _as_faiss_query(bad)


class DummyEmbedding:
    def __init__(self, values):
        self.values = values


def test_as_faiss_query_from_object_values():
    vec = DummyEmbedding([0.5, 0.6])
    arr = _as_faiss_query(vec)
    assert arr.shape == (1, 2)
