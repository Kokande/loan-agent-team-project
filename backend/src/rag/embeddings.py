"""Векторизация через локальный Ollama без SDK и платного API."""

import json
import math
import struct
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def validate_vectors(vectors: list, count: int, dimensions: int) -> None:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Количество векторов не совпадает с количеством текстов")
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or len(vector) != dimensions:
            raise ValueError("Неверная размерность вектора")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in vector
        ):
            raise ValueError("Некорректные значения вектора")
        # pgvector хранит float32: проверяем именно записываемые значения,
        # иначе маленький float64 может превратиться в нулевой вектор.
        try:
            stored = [struct.unpack("f", struct.pack("f", value))[0] for value in vector]
        except (OverflowError, struct.error) as exc:
            raise ValueError("Значения вектора выходят за диапазон float32") from exc
        if not all(math.isfinite(value) for value in stored):
            raise ValueError("Некорректные значения вектора float32")
        if not any(stored):
            raise ValueError("Нулевой вектор нельзя использовать для косинусного поиска")


class OllamaEmbedder:
    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "bge-m3",
        dimensions: int = 1024, timeout: float = 120,
    ):
        if not model.strip() or not 1 <= dimensions <= 16000 or timeout <= 0:
            raise ValueError("Некорректные настройки модели эмбеддингов")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        request = Request(
            f"{self.base_url}/api/embed",
            data=json.dumps({"model": self.model, "input": texts, "truncate": False}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(
                f"Ollama вернул HTTP {exc.code}. Проверьте наличие модели и размер чанков."
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                "Ollama недоступен: проверьте URL, запуск сервиса и timeout"
            ) from exc
        if not isinstance(payload, dict) or "embeddings" not in payload:
            raise ValueError("Ответ Ollama не содержит embeddings")
        vectors = payload["embeddings"]
        validate_vectors(vectors, len(texts), self.dimensions)
        return vectors
