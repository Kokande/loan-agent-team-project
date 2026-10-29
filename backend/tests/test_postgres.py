"""Тесты используют только отдельную БД: тестовая схема rag удаляется."""

import json
import os
import subprocess
import sys
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import psycopg
import pytest

from src.rag.api_embeddings import APIEmbedder
from src.rag.documents import Chunk
from src.rag.store import DocumentMetadata, PgVectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    url = os.environ.get("RAG_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Нужен RAG_TEST_DATABASE_URL с отдельной тестовой БД")
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS rag CASCADE")
    store = PgVectorStore(url, model="test-only", dimensions=3)
    store.initialize()
    yield store
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS rag CASCADE")


def seed(store, source="policy", text="Документ", vector=None):
    store.replace_document(
        DocumentMetadata(source, "Правила", "v1", date(2026, 1, 1)), "fingerprint-v1",
        [Chunk(0, text, 4)], [vector or [1, 0, 0]],
    )


def test_cosine_search_returns_ranked_chunks_and_citations(store):
    seed(store, "relevant", "Нужное положение", [1, 0, 0])
    seed(store, "other", "Другое положение", [0, 1, 0])
    results = store.search([1, 0, 0], top_k=2)
    assert [result.source for result in results] == ["relevant", "other"]
    assert results[0].score == pytest.approx(1)
    assert results[0].page == 4
    assert results[0].revision == "v1"
    assert results[0].effective_date == date(2026, 1, 1)
    assert results[0].text == "Нужное положение"
    assert len(store.search([1, 0, 0], min_score=0.5)) == 1


def test_new_revision_replaces_old_chunks(store):
    seed(store)
    store.replace_document(
        DocumentMetadata("policy", "Правила", "v2"), "fingerprint-v2",
        [Chunk(0, "Новая редакция", 2)], [[0, 1, 0]],
    )
    results = store.search([0, 1, 0], top_k=10)
    assert len(results) == 1
    assert results[0].text == "Новая редакция"
    assert results[0].revision == "v2"
    assert store.document_status("policy") == ("fingerprint-v2", 1)


def test_failed_insert_rolls_back_replacement(store):
    seed(store)
    with pytest.raises(psycopg.errors.CheckViolation):
        store.replace_document(
            DocumentMetadata("policy", "Правила", "v2"), "fingerprint-v2",
            [Chunk(0, "Новый текст", -1)], [[0, 1, 0]],
        )
    assert store.document_status("policy") == ("fingerprint-v1", 1)
    assert store.search([1, 0, 0])[0].text == "Документ"


def test_model_change_rejected_even_with_same_dimensions(store):
    other = PgVectorStore(store.database_url, model="another-model", dimensions=3)
    with pytest.raises(ValueError):
        other.initialize()
    with pytest.raises(ValueError):
        other.search([1, 0, 0])
    assert store.search([1, 0, 0]) == []


def test_quoted_source_is_safe_and_delete_cascades(store):
    source = "правила'; DROP TABLE rag.documents; --"
    seed(store, source)
    assert store.search([1, 0, 0])[0].source == source
    assert store.delete_document(source) is True
    assert store.document_status(source) is None
    assert store.search([1, 0, 0]) == []
    assert store.delete_document(source) is False


@pytest.mark.parametrize("provider", ["ollama", "api"])
def test_cli_ingest_query_eval_and_delete(store, tmp_path, provider):
    # HTTP заменяет только тяжёлую модель; CLI, адаптер и PostgreSQL работают реально.
    class EmbeddingHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if provider == "api":
                if (
                    self.path != "/v1/embeddings"
                    or self.headers.get("Authorization") != "Bearer test-secret"
                    or body.get("model") != "test-only"
                    or body.get("dimensions") != 3
                    or body.get("encoding_format") != "float"
                ):
                    self.send_error(400)
                    return
                response = {
                    "object": "list", "model": "test-only",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [1, 0, 0]}
                        for i in range(len(body["input"]))
                    ], "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            else:
                if self.path != "/api/embed" or body.get("truncate") is not False:
                    self.send_error(400)
                    return
                response = {
                    "model": "test-only", "embeddings": [[1, 0, 0] for _ in body["input"]],
                    "total_duration": 1, "load_duration": 0, "prompt_eval_count": 1,
                }
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = {
        **os.environ, "RAG_DATABASE_URL": store.database_url,
        "RAG_EMBEDDING_PROVIDER": provider,
        "RAG_OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
        "RAG_EMBEDDING_API_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        "RAG_EMBEDDING_API_KEY": "test-secret",
        "RAG_EMBEDDING_MODEL": "test-only", "RAG_EMBEDDING_DIMENSIONS": "3",
        "RAG_CHUNK_SIZE": "1600", "RAG_CHUNK_OVERLAP": "200",
    }

    def cli(*args):
        result = subprocess.run(
            [sys.executable, "-m", "src.rag", *args], env=environment,
            capture_output=True, text=True, check=True, timeout=20,
        )
        return json.loads(result.stdout)

    try:
        if provider == "api":
            # Отдельное векторное пространство API намеренно несовместимо с Ollama.
            with psycopg.connect(store.database_url, autocommit=True) as conn:
                conn.execute("DROP SCHEMA rag CASCADE")
        assert cli("init-db")["initialized"] is True
        path = tmp_path / "synthetic.txt"
        path.write_text("Синтетический документ для проверки программного интерфейса.")
        args = ("ingest", str(path), "--source", "synthetic-policy",
                "--revision", "v1", "--effective-date", "2026-10-01")
        assert cli(*args) == {"source": "synthetic-policy", "chunks": 1, "updated": True}
        assert cli(*args)["updated"] is False
        results = cli("query", "Тестовый вопрос", "--top-k", "1")
        assert results[0]["source"] == "synthetic-policy"
        assert results[0]["revision"] == "v1"
        assert results[0]["effective_date"] == "2026-10-01"
        assert results[0]["text"] == path.read_text()
        dataset = tmp_path / "eval.jsonl"
        dataset.write_text(json.dumps({
            "query": "Тестовый вопрос", "expected_sources": ["synthetic-policy"],
        }))
        metrics = cli("eval", str(dataset))
        assert metrics["recall_at_k"] == 1
        assert metrics["mrr"] == 1
        assert cli("delete", "--source", "synthetic-policy")["deleted"] is True
        assert cli("query", "Тестовый вопрос") == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cloud_endpoint_change_rejected_by_index(store):
    first = APIEmbedder("https://first.example/v1", "test-secret", "same-model", 3)
    second = APIEmbedder("https://second.example/v1", "test-secret", "same-model", 3)
    with psycopg.connect(store.database_url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA rag CASCADE")
    initial = PgVectorStore(store.database_url, model=first.model, dimensions=3)
    initial.initialize()
    initial.search([1, 0, 0])
    changed = PgVectorStore(store.database_url, model=second.model, dimensions=3)
    with pytest.raises(ValueError):
        changed.search([1, 0, 0])


def test_api_redirect_does_not_forward_authorization(store):
    leaked = []

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", "/leak")
            self.end_headers()

        def do_GET(self):
            leaked.append(self.headers.get("Authorization"))
            self.send_error(500)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        embedder = APIEmbedder(
            f"http://127.0.0.1:{server.server_port}/v1", "test-secret", "test-model", 3,
        )
        with pytest.raises(RuntimeError, match="302"):
            embedder.embed(["Текст"])
        assert leaked == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
