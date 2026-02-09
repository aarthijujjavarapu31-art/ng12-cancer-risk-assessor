import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import settings


def load_index() -> faiss.Index:
    idx_path = Path(settings.vector_index_dir) / "faiss.index"
    if not idx_path.exists():
        raise FileNotFoundError("FAISS index not found. Run ingestion first.")
    return faiss.read_index(str(idx_path))


def load_meta() -> list[dict[str, Any]]:
    meta_path = Path(settings.vector_index_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Meta not found. Run ingestion first.")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def embed_query_vertex(query: str) -> np.ndarray:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel

    if not settings.vertex_project_id:
        raise RuntimeError("VERTEX_PROJECT_ID not set")

    vertexai.init(project=settings.vertex_project_id, location=settings.vertex_location)

    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    vec = model.get_embeddings([query])[0].values
    return np.array(vec, dtype="float32").reshape(1, -1)
