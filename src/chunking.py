from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str


def extract_text(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_path.read_text(encoding="utf-8")


def chunk_text(text: str, source: str, chunk_size: int = 700, overlap: int = 120) -> list[DocumentChunk]:
    normalized = " ".join(text.split())
    chunks: list[DocumentChunk] = []
    start = 0
    index = 1
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(DocumentChunk(f"{Path(source).stem}-{index}", source, normalized[start:end]))
        if end == len(normalized):
            break
        start = max(0, end - overlap)
        index += 1
    return chunks


def load_documents(directory: str | Path) -> list[DocumentChunk]:
    docs: list[DocumentChunk] = []
    for path in sorted(Path(directory).glob("*")):
        if path.suffix.lower() not in {".pdf", ".txt"}:
            continue
        docs.extend(chunk_text(extract_text(path), path.name))
    return docs
