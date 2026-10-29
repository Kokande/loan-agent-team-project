"""Сборка RAG из переменных окружения для CLI и основного backend."""

import os

from .api_embeddings import APIEmbedder
from .embeddings import OllamaEmbedder
from .service import RAGService
from .store import PgVectorStore


def create_rag() -> RAGService:
    database_url = os.environ.get("RAG_DATABASE_URL", "")
    if not database_url:
        raise ValueError("Укажите RAG_DATABASE_URL для подключения к PostgreSQL")
    provider = os.environ.get("RAG_EMBEDDING_PROVIDER", "ollama").strip().lower()
    dimensions = int(os.environ.get("RAG_EMBEDDING_DIMENSIONS", "1024"))
    timeout = float(os.environ.get("RAG_EMBEDDING_TIMEOUT", "120"))
    if provider == "ollama":
        embedder = OllamaEmbedder(
            base_url=os.environ.get("RAG_OLLAMA_URL", "http://localhost:11434"),
            model=os.environ.get("RAG_EMBEDDING_MODEL", "").strip() or "bge-m3",
            dimensions=dimensions, timeout=timeout,
        )
    elif provider == "api":
        embedder = APIEmbedder(
            base_url=os.environ.get("RAG_EMBEDDING_API_BASE_URL", ""),
            api_key=os.environ.get("RAG_EMBEDDING_API_KEY", ""),
            model=os.environ.get("RAG_EMBEDDING_MODEL", ""),
            dimensions=dimensions, timeout=timeout,
        )
    else:
        raise ValueError("RAG_EMBEDDING_PROVIDER должен быть ollama или api")
    return RAGService(
        PgVectorStore(database_url, embedder.model, dimensions), embedder,
        chunk_size=int(os.environ.get("RAG_CHUNK_SIZE", "1600")),
        overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", "200")),
        batch_size=int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "16")),
    )
