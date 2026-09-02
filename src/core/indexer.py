from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel


class IndexedChunk(BaseModel):
    content: str
    source_doc: str
    chunk_index: int


class CorpusIndexer:
    """Embeds text documents into a FAISS index for retrieval-based verification."""

    def __init__(self, corpus_dir: str = "data/corpus", chunk_size: int = 500, chunk_overlap: int = 50):
        self.corpus_dir = Path(corpus_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[IndexedChunk] = []

    def _chunk_text(self, text: str, source: str) -> list[IndexedChunk]:
        words = text.split()
        chunks = []
        i = 0
        idx = 0
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk_text = " ".join(words[i:end])
            chunks.append(IndexedChunk(content=chunk_text, source_doc=source, chunk_index=idx))
            i += self.chunk_size - self.chunk_overlap
            idx += 1
        return chunks

    def build_index(self) -> None:
        all_chunks: list[IndexedChunk] = []
        for fpath in sorted(self.corpus_dir.glob("*.txt")):
            text = fpath.read_text()
            all_chunks.extend(self._chunk_text(text, fpath.name))

        if not all_chunks:
            raise ValueError(f"No .txt files found in {self.corpus_dir}")

        texts = [c.content for c in all_chunks]
        vectors = self.embeddings.embed_documents(texts)
        matrix = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)

        dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(matrix)
        self.chunks = all_chunks
        print(f"Indexed {len(all_chunks)} chunks from {len(list(self.corpus_dir.glob('*.txt')))} documents")

    def search(self, query: str, top_k: int = 3) -> list[tuple[IndexedChunk, float]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_vec = np.array(self.embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results
