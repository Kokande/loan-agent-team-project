import os

import pytest

from src.rag import OllamaEmbedder, create_rag
from src.rag.api_embeddings import APIEmbedder


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    for key in os.environ:
        if key.startswith("RAG_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("RAG_DATABASE_URL", "postgresql://test@localhost/test")


def test_default_provider_preserves_existing_ollama_index():
    rag = create_rag()
    assert isinstance(rag.embedder, OllamaEmbedder)
    assert rag.store.model == "bge-m3"


def test_empty_model_from_compose_uses_local_default(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "")
    assert create_rag().store.model == "bge-m3"


def test_api_provider_configured_without_connecting(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "api")
    monkeypatch.setenv("RAG_EMBEDDING_API_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "test-secret")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "text-embedding-v4")
    rag = create_rag()
    assert isinstance(rag.embedder, APIEmbedder)
    assert rag.store.model == rag.embedder.model
    assert rag.store.model != "text-embedding-v4"


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "typo")
    with pytest.raises(ValueError, match="PROVIDER"):
        create_rag()


@pytest.mark.parametrize("missing", [
    "RAG_EMBEDDING_API_BASE_URL", "RAG_EMBEDDING_API_KEY", "RAG_EMBEDDING_MODEL",
])
def test_incomplete_api_config_rejected(monkeypatch, missing):
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "api")
    monkeypatch.setenv("RAG_EMBEDDING_API_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "test-secret")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError):
        create_rag()


def test_empty_api_model_from_compose_rejected(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "api")
    monkeypatch.setenv("RAG_EMBEDDING_API_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "test-secret")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "")
    with pytest.raises(ValueError):
        create_rag()
