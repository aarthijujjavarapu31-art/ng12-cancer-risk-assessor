from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    text: str


def simple_chunk_text(text: str, page: int, chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunk_text = clean[start:end].strip()

        if chunk_text:
            chunk_id = f"ng12_{page:04d}_{idx:02d}"
            chunks.append(Chunk(chunk_id=chunk_id, page=page, text=chunk_text))
            idx += 1

        if end == len(clean):
            break

        start = max(0, end - overlap)

    return chunks
