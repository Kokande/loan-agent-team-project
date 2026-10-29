"""Публичный интерфейс RAG для подключения к агенту."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .documents import chunk_pages, load_pages
from .embeddings import Embedder, validate_vectors
from .store import DocumentMetadata, PgVectorStore, SearchResult


@dataclass(frozen=True)
class IngestResult:
    source: str
    chunks: int
    updated: bool


class RAGService:
    def __init__(
        self, store: PgVectorStore, embedder: Embedder,
        chunk_size: int = 1600, overlap: int = 200, batch_size: int = 16,
    ):
        if (store.model, store.dimensions) != (embedder.model, embedder.dimensions):
            raise ValueError("Модель и размерность embedder должны совпадать с хранилищем")
        if chunk_size <= 0 or not 0 <= overlap < chunk_size or batch_size <= 0:
            raise ValueError("Некорректные настройки чанкинга или размера пакета")
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size

    def ingest(
        self, path: str | Path, source: str, title: str | None = None,
        revision: str | None = None, effective_date: date | None = None,
    ) -> IngestResult:
        source = source.strip()
        if not source:
            raise ValueError("Нужен непустой стабильный идентификатор source")
        path = Path(path)
        pages = load_pages(path)
        metadata = DocumentMetadata(source, title or path.stem, revision, effective_date)
        payload = {
            "metadata": asdict(metadata), "pages": [asdict(page) for page in pages],
            "chunk_size": self.chunk_size, "overlap": self.overlap,
            "chunker_version": 1,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(),
        ).hexdigest()
        status = self.store.document_status(source)
        if status and status[0] == fingerprint:
            return IngestResult(source, status[1], updated=False)
        chunks = chunk_pages(pages, self.chunk_size, self.overlap)
        if not chunks:
            raise ValueError("Документ не содержит текста для индексации")
        vectors = []
        for start in range(0, len(chunks), self.batch_size):
            texts = [chunk.text for chunk in chunks[start:start + self.batch_size]]
            batch = self.embedder.embed(texts)
            validate_vectors(batch, len(texts), self.embedder.dimensions)
            vectors.extend(batch)
        self.store.replace_document(metadata, fingerprint, chunks, vectors)
        return IngestResult(source, len(chunks), updated=True)

    def retrieve(
        self, query: str, top_k: int = 5, min_score: float | None = None,
    ) -> list[SearchResult]:
        if not query.strip() or top_k <= 0:
            raise ValueError("Нужны непустой запрос и top_k > 0")
        if min_score is not None and not -1 <= min_score <= 1:
            raise ValueError("min_score должен быть в диапазоне [-1, 1]")
        vectors = self.embedder.embed([query.strip()])
        validate_vectors(vectors, 1, self.embedder.dimensions)
        return self.store.search(vectors[0], top_k, min_score)
