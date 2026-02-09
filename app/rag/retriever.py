from __future__ import annotations

from typing import Any

import numpy as np

from app.models import Citation
from app.rag.vector_store import load_index, load_meta, embed_query_vertex


def _as_faiss_query(vec: Any) -> np.ndarray:
    """
    Ensure the embedding is a float32 numpy array with shape (1, d).
    Vertex embedding outputs can vary (list, np.ndarray, or object with .values).
    """
    if vec is None:
        raise ValueError("Embedding is None")

    if hasattr(vec, "values"):
        vec = vec.values

    arr = np.array(vec, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ValueError(f"Embedding has unexpected shape: {arr.shape}")

    return arr


class NG12Retriever:
    def __init__(self) -> None:
        self.index = load_index()
        self.meta = load_meta()

    def retrieve(self, query: str, top_k: int = 10) -> list[Citation]:
        emb = embed_query_vertex(query)
        q = _as_faiss_query(emb)

        _, ids = self.index.search(q, top_k)

        citations: list[Citation] = []
        for idx in ids[0]:
            if idx < 0:
                continue
            m: dict[str, Any] = self.meta[idx]
            citations.append(
                Citation(
                    source=m.get("source", "NG12 PDF"),
                    page=int(m.get("page", 0)),
                    chunk_id=str(m.get("chunk_id", "")),
                    excerpt=str(m.get("text", ""))[:500],
                )
            )
        return citations
