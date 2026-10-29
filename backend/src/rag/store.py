"""Транзакционное хранилище нормативных документов в PostgreSQL + pgvector."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from .documents import Chunk
from .embeddings import validate_vectors


@dataclass(frozen=True)
class DocumentMetadata:
    source: str
    title: str
    revision: str | None = None
    effective_date: date | None = None


@dataclass(frozen=True)
class SearchResult:
    source: str
    title: str
    revision: str | None
    effective_date: date | None
    chunk_index: int
    text: str
    page: int | None
    score: float


class PgVectorStore:
    def __init__(self, database_url: str, model: str = "bge-m3", dimensions: int = 1024):
        if not database_url.strip() or not model.strip() or not 1 <= dimensions <= 16000:
            raise ValueError("Нужны строка подключения, модель и размерность от 1 до 16000")
        self.database_url = database_url
        self.model = model
        self.dimensions = dimensions

    def _check_model(self, connection):
        settings = connection.execute("SELECT model, dimensions FROM rag.settings").fetchone()
        if settings != {"model": self.model, "dimensions": self.dimensions}:
            raise ValueError(
                "Модель или размерность не совпадает с индексом RAG. "
                "Используйте прежнюю модель либо отдельную БД и повторную индексацию."
            )

    @contextmanager
    def _connection(self):
        with psycopg.connect(self.database_url, connect_timeout=10, row_factory=dict_row) as conn:
            self._check_model(conn)
            register_vector(conn)
            yield conn

    def initialize(self) -> None:
        with psycopg.connect(self.database_url, connect_timeout=10, row_factory=dict_row) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("CREATE SCHEMA IF NOT EXISTS rag")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag.settings (
                    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
                    model text NOT NULL,
                    dimensions integer NOT NULL CHECK (dimensions BETWEEN 1 AND 16000)
                )
            """)
            conn.execute(
                "INSERT INTO rag.settings (model, dimensions) VALUES (%s, %s) "
                "ON CONFLICT (singleton) DO NOTHING", (self.model, self.dimensions),
            )
            self._check_model(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag.documents (
                    source text PRIMARY KEY CHECK (length(btrim(source)) > 0),
                    title text NOT NULL,
                    revision text,
                    effective_date date,
                    fingerprint text NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
            """)
            conn.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS rag.chunks (
                    source text NOT NULL REFERENCES rag.documents(source) ON DELETE CASCADE,
                    chunk_index integer NOT NULL CHECK (chunk_index >= 0),
                    text text NOT NULL CHECK (length(btrim(text)) > 0),
                    page integer CHECK (page > 0),
                    embedding vector({dimensions}) NOT NULL,
                    PRIMARY KEY (source, chunk_index)
                )
            """).format(dimensions=sql.Literal(self.dimensions)))

    def document_status(self, source: str) -> tuple[str, int] | None:
        with self._connection() as conn:
            row = conn.execute("""
                SELECT d.fingerprint, count(c.chunk_index) AS chunks
                FROM rag.documents d LEFT JOIN rag.chunks c USING (source)
                WHERE d.source = %s GROUP BY d.source
            """, (source,)).fetchone()
            return (row["fingerprint"], row["chunks"]) if row else None

    def replace_document(
        self, metadata: DocumentMetadata, fingerprint: str,
        chunks: list[Chunk], vectors: list[list[float]],
    ) -> None:
        if not chunks:
            raise ValueError("Документ не содержит чанков")
        validate_vectors(vectors, len(chunks), self.dimensions)
        with self._connection() as conn:
            # Сериализует обновления одного source, включая самую первую загрузку.
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (metadata.source,),
            )
            conn.execute("""
                INSERT INTO rag.documents (source, title, revision, effective_date, fingerprint)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source) DO UPDATE SET title = EXCLUDED.title,
                    revision = EXCLUDED.revision, effective_date = EXCLUDED.effective_date,
                    fingerprint = EXCLUDED.fingerprint, updated_at = now()
            """, (metadata.source, metadata.title, metadata.revision,
                  metadata.effective_date, fingerprint))
            conn.execute("DELETE FROM rag.chunks WHERE source = %s", (metadata.source,))
            with conn.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO rag.chunks (source, chunk_index, text, page, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                """, [
                    (metadata.source, chunk.index, chunk.text, chunk.page, Vector(vector))
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ])

    def search(
        self, vector: list[float], top_k: int = 5, min_score: float | None = None,
    ) -> list[SearchResult]:
        validate_vectors([vector], 1, self.dimensions)
        if top_k <= 0 or (min_score is not None and not -1 <= min_score <= 1):
            raise ValueError("Нужны top_k > 0 и min_score в диапазоне [-1, 1]")
        with self._connection() as conn:
            rows = conn.execute("""
                WITH nearest AS (
                    SELECT source, chunk_index, text, page, embedding <=> %s AS distance
                    FROM rag.chunks
                    ORDER BY distance, source, chunk_index LIMIT %s
                )
                SELECT n.source, d.title, d.revision, d.effective_date,
                    n.chunk_index, n.text, n.page,
                    greatest(-1.0, least(1.0, 1 - n.distance)) AS score
                FROM nearest n JOIN rag.documents d USING (source)
                WHERE greatest(-1.0, least(1.0, 1 - n.distance)) >= %s
                ORDER BY n.distance, n.source, n.chunk_index
            """, (Vector(vector), top_k, min_score if min_score is not None else -1.0)).fetchall()
            return [SearchResult(**row) for row in rows]

    def delete_document(self, source: str) -> bool:
        with self._connection() as conn:
            return conn.execute(
                "DELETE FROM rag.documents WHERE source = %s RETURNING source", (source,),
            ).fetchone() is not None
