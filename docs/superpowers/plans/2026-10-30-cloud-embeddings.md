# Cloud Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Переключение RAG между Ollama и облачным API эмбеддингов.
**Architecture:** Новый HTTP-адаптер реализует существующий Embedder; фабрика выбирает провайдера по окружению. БД и RAGService используют прежние интерфейсы.
**Tech Stack:** Python standard library, существующий PostgreSQL + pgvector, pytest.
**Spec:** `docs/superpowers/specs/2026-10-30-cloud-embeddings-design.md`.

## Global Constraints

- Ollama остаётся режимом по умолчанию; прежние индексы работают без миграции.
- Не вызывать платный API и не сохранять ключи в исходниках, логах или идентификаторе индекса.
- Сохранять порядок соответствия текстов и эмбеддингов; проверять каждый пакет до записи.

## Review Focus

- Неполные, дублирующиеся и переставленные `data[].index`.
- Ограничение 10 текстов на HTTP-запрос при существующем batch_size=16.
- 401/403, 429, 5xx, таймауты и перенаправления с ключом.
- Смена endpoint при одинаковом имени модели; ротация ключа.
- Запуск Compose с облаком без зависимости от Ollama.

### Task 1: API-адаптер и выбор провайдера

**Files:** `backend/src/rag/api_embeddings.py`, `backend/src/rag/config.py`, `backend/src/rag/__init__.py`, `backend/tests/test_api_embeddings.py`, `backend/tests/test_config.py`.
**Interfaces:** `APIEmbedder(base_url, api_key, model, dimensions=1024, timeout=120).embed(texts) -> list[list[float]]`; `create_rag()` сохраняет сигнатуру.

- [x] Написать тесты API и конфигурации, запустить и увидеть отсутствие реализации.
- [x] Реализовать HTTP-контракт, пакетирование, проверки, повторы и идентификатор индекса.
- [x] Проверить новые тесты и весь набор без БД.

### Task 2: Запуск, интеграция и документация

**Files:** `backend/tests/test_postgres.py`, `docker-compose.yaml`, `.env.example`, `docs/RAG.md`, `README.md`.
**Interfaces:** прежние команды CLI работают с обоими провайдерами; `docker compose run --rm rag ...` в облаке запускает только БД.

- [x] Расширить тест пути CLI обоими провайдерами и проверить реальный HTTP + pgvector.
- [x] Обновить Compose, примеры настроек и инструкцию Alibaba.
- [x] Выполнить pytest с временной БД, ruff, проверку Compose и diff; провести проверку кода.

## Результат проверки

- 88 тестов прошли, включая полный путь CLI для Ollama и API на временном PostgreSQL 16.2 + pgvector 0.6.2 и локальных HTTP-серверах.
- Проверены запрет перенаправления ключа, смена endpoint и совместимость прежнего индекса Ollama.
- Независимая проверка кода выявила fallback `bge-m3` в Compose для облака. Исправлено: Compose передаёт пустую модель, локальный default применяется только для Ollama. Регрессионный тест прошёл RED → GREEN; конфигурация проверена через реальный `docker compose config` без обращения к API/БД.
- `ruff check backend`, `docker compose --profile local --profile tools config --quiet`, `git diff --check` и проверки ссылок прошли.
- Реальный API провайдера не вызывался: ключ не предоставлен. Сборка Docker не проверялась, daemon недоступен; новых зависимостей не добавлено.
