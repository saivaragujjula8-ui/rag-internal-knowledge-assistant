from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np

from .chunking import DocumentChunk
from .embeddings import HashEmbeddingModel


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


class ManualIndex:
    def __init__(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embeddings = embeddings.astype("float32")

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        model = HashEmbeddingModel(dimensions=self.embeddings.shape[1])
        query_vector = model.embed(query)
        scores = self.embeddings @ query_vector
        top_indexes = np.argsort(scores)[::-1][:top_k]
        return [RetrievedChunk(self.chunks[index], float(scores[index])) for index in top_indexes]

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "ManualIndex":
        with Path(path).open("rb") as handle:
            data = pickle.load(handle)
        return cls(data["chunks"], data["embeddings"])


def build_index(chunks: list[DocumentChunk]) -> ManualIndex:
    model = HashEmbeddingModel()
    embeddings = np.vstack([model.embed(chunk.text) for chunk in chunks])
    return ManualIndex(chunks, embeddings)
