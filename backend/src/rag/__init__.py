"""RAG: нормативные документы → pgvector → релевантные чанки."""

from .api_embeddings import APIEmbedder
from .config import create_rag
from .embeddings import Embedder, OllamaEmbedder
from .service import IngestResult, RAGService
from .store import PgVectorStore, SearchResult

__all__ = [
    "APIEmbedder", "Embedder", "IngestResult", "OllamaEmbedder", "PgVectorStore",
    "RAGService", "SearchResult", "create_rag",
]
