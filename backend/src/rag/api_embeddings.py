"""Облачные эмбеддинги через совместимый API /embeddings."""

import hashlib
import json
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .embeddings import validate_vectors


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Ключ должен отправляться только на явно настроенный endpoint.
        return None


class APIEmbedder:
    def __init__(
        self, base_url: str, api_key: str, model: str,
        dimensions: int = 1024, timeout: float = 120,
    ):
        base_url = base_url.strip().rstrip("/")
        parts = urlsplit(base_url)
        if (
            not parts.hostname or parts.scheme not in {"https", "http"}
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or any(char.isspace() for char in base_url)
            or parts.path.endswith("/embeddings")
        ):
            raise ValueError("Укажите базовый API URL без ключа, query и суффикса /embeddings")
        if parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Для облачного API требуется HTTPS; HTTP разрешён только для loopback")
        if not api_key.strip() or "\n" in api_key.strip() or "\r" in api_key.strip():
            raise ValueError("Укажите корректный RAG_EMBEDDING_API_KEY")
        if not model.strip() or not 1 <= dimensions <= 16000:
            raise ValueError("Укажите модель и размерность эмбеддингов от 1 до 16000")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Таймаут API должен быть положительным конечным числом")
        self.base_url = base_url
        self.api_key = api_key.strip()
        self.model_name = model.strip()
        # Одинаковые имена у разных поставщиков не означают одинаковые векторы.
        endpoint_id = hashlib.sha256(base_url.encode()).hexdigest()
        self.model = f"api:{endpoint_id}:{self.model_name}"
        self.dimensions = dimensions
        self.timeout = timeout
        self._opener = build_opener(_RejectRedirects())

    @staticmethod
    def _retry_delay(error: HTTPError, attempt: int) -> float | None:
        header = error.headers.get("Retry-After") if error.headers else None
        if header is None:
            return float(2 ** attempt)
        try:
            delay = float(header)
        except ValueError:
            try:
                until = parsedate_to_datetime(header)
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                delay = (until - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None
        # Не повторяем запрос раньше Retry-After и не блокируем CLI надолго.
        return max(0.0, delay) if math.isfinite(delay) and delay <= 30 else None

    def _request(self, texts: list[str]) -> dict:
        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({
                "model": self.model_name, "input": texts,
                "dimensions": self.dimensions, "encoding_format": "float",
            }).encode(),
            headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
            }, method="POST",
        )
        for attempt in range(3):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                    delay = self._retry_delay(exc, attempt)
                    if delay is not None:
                        exc.close()
                        sleep(delay)
                        continue
                status = exc.code
                exc.close()
                raise RuntimeError(
                    f"API эмбеддингов вернул HTTP {status}. "
                    "Проверьте ключ, модель, лимиты и доступность сервиса."
                ) from None
            except (URLError, TimeoutError, ConnectionError):
                raise RuntimeError(
                    "API эмбеддингов недоступен: проверьте соединение, URL и timeout"
                ) from None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Для эмбеддингов нужны непустые строки")
        vectors = []
        # text-embedding-v4 принимает максимум 10 строк за один вызов.
        for start in range(0, len(texts), 10):
            batch = texts[start:start + 10]
            payload = self._request(batch)
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list) or len(data) != len(batch):
                raise ValueError("API вернул неверное количество embeddings в data")
            ordered = [None] * len(batch)
            seen = set()
            for item in data:
                index = item.get("index") if isinstance(item, dict) else None
                if (
                    not isinstance(index, int) or isinstance(index, bool)
                    or not 0 <= index < len(batch) or index in seen
                ):
                    raise ValueError("API вернул некорректные или повторяющиеся индексы embeddings")
                seen.add(index)
                ordered[index] = item.get("embedding")
            validate_vectors(ordered, len(batch), self.dimensions)
            vectors.extend(ordered)
        return vectors
