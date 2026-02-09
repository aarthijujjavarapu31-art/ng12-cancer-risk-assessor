from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from pypdf import PdfReader

import vertexai
from vertexai.language_models import TextEmbeddingModel


# -----------------------------
# Config
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]  # repo root (app/rag -> repo)
DEFAULT_PDF = ROOT / "data" / "ng12.pdf"
OUT_DIR = ROOT / "data" / "index"
FAISS_PATH = OUT_DIR / "faiss.index"
META_PATH = OUT_DIR / "meta.json"

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID") or ""
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
EMBED_MODEL_NAME = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004")

# Safe defaults
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))          # characters
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))     # characters
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH", "16"))     # keep small to avoid token limits


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    page: int
    text: str


# -----------------------------
# PDF extraction + chunking
# -----------------------------
def extract_pdf_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(pages: List[str]) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0
    for page_i, page_text in enumerate(pages):
        for piece in chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(Chunk(chunk_id=f"c{idx:04d}", page=page_i, text=piece))
            idx += 1
    return chunks


# -----------------------------
# Vertex embeddings
# -----------------------------
def embed_texts_vertex(texts: List[str]) -> np.ndarray:
    if not PROJECT_ID:
        raise RuntimeError(
            "Missing PROJECT_ID / GOOGLE_CLOUD_PROJECT env var. "
            "Set it to your GCP project id."
        )

    # Uses GOOGLE_APPLICATION_CREDENTIALS automatically
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    model = TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)

    vectors: List[List[float]] = []
    total = len(texts)

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        embs = model.get_embeddings(batch)
        vectors.extend([e.values for e in embs])
        print(f"   embedded {min(i + EMBED_BATCH_SIZE, total)}/{total} chunks")

    arr = np.array(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")
    return arr


# -----------------------------
# FAISS build + save
# -----------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_outputs(index: faiss.Index, chunks: List[Chunk]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))

    meta = [asdict(c) for c in chunks]
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    pdf_path = Path(os.getenv("NG12_PDF_PATH", str(DEFAULT_PDF)))

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("1) Extracting PDF text and chunking...")
    pages = extract_pdf_pages(pdf_path)
    chunks = build_chunks(pages)
    print(f"   chunks: {len(chunks)}")

    print("2) Creating embeddings (Vertex AI)...")
    embeddings = embed_texts_vertex([c.text for c in chunks])

    print("3) Building FAISS index...")
    index = build_faiss_index(embeddings)

    save_outputs(index, chunks)

    print("✅ Ingestion complete.")
    print(f"   Saved FAISS index: {FAISS_PATH}")
    print(f"   Saved metadata:    {META_PATH}")


if __name__ == "__main__":
    main()
