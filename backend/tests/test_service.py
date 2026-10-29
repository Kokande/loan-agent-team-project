from unittest.mock import Mock

import pytest

from src.rag.documents import Page
from src.rag.service import RAGService


@pytest.fixture
def service():
    embedder = Mock(model="test", dimensions=3)
    embedder.embed.return_value = [[1, 0, 0]]
    store = Mock(model="test", dimensions=3)
    store.document_status.return_value = None
    return RAGService(store, embedder, chunk_size=100, overlap=10)


def test_ingest_returns_source_and_chunk_count(service, tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("Тестовый документ")
    result = service.ingest(path, source="policy-1", title="Правила")
    assert result.source == "policy-1"
    assert result.chunks == 1
    assert result.updated is True


def test_unchanged_document_skips_embedding_and_write(service, tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("Тестовый документ")
    # Первый вызов получает hash из реального пути индексации.
    service.ingest(path, source="policy-1")
    fingerprint = service.store.replace_document.call_args.args[1]
    service.store.document_status.return_value = (fingerprint, 1)
    service.embedder.embed.side_effect = AssertionError("Повторная векторизация")
    service.store.replace_document.side_effect = AssertionError("Повторная запись")
    result = service.ingest(path, source="policy-1")
    assert result.updated is False
    assert result.chunks == 1


def test_changed_chunk_settings_trigger_reindex(service, tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("Тестовый документ")
    service.ingest(path, source="policy-1")
    fingerprint = service.store.replace_document.call_args.args[1]
    service.store.document_status.return_value = (fingerprint, 1)
    changed = RAGService(service.store, service.embedder, chunk_size=90, overlap=10)
    assert changed.ingest(path, source="policy-1").updated is True


def test_failed_embeddings_do_not_write_document(service, tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("Тестовый документ")
    service.embedder.embed.return_value = [[0, 0, 0]]
    service.store.replace_document.side_effect = AssertionError("Запись некорректных векторов")
    with pytest.raises(ValueError, match="вектор"):
        service.ingest(path, source="policy-1")


@pytest.mark.parametrize("query,top_k,min_score", [
    (" ", 5, None), ("Вопрос", 0, None), ("Вопрос", 5, 2),
    ("Вопрос", 5, float("nan")),
])
def test_invalid_retrieval_rejected_before_embedding(service, query, top_k, min_score):
    service.embedder.embed.side_effect = AssertionError("Векторизация недопустимого запроса")
    with pytest.raises(ValueError):
        service.retrieve(query, top_k=top_k, min_score=min_score)


def test_incompatible_embedder_rejected(service):
    embedder = Mock(model="different", dimensions=3)
    with pytest.raises(ValueError):
        RAGService(service.store, embedder)


def test_empty_pages_have_no_chunks():
    from src.rag.documents import chunk_pages

    assert chunk_pages([Page("   ")]) == []
