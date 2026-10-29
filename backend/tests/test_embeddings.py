import io
import json

import pytest

from src.rag.embeddings import OllamaEmbedder, validate_vectors


@pytest.mark.parametrize("vectors", [
    [], [[1, 0]], [[0, 0, 0]], [[float("nan"), 0, 1]],
    [[float("inf"), 0, 1]], [["1", 0, 1]], [[True, 0, 1]],
    [[1e-100, 0, 0]], [[1e100, 0, 1]],
])
def test_invalid_embeddings_rejected(vectors):
    with pytest.raises(ValueError):
        validate_vectors(vectors, count=1, dimensions=3)


def test_ollama_request_preserves_full_text(monkeypatch):
    def respond(request, timeout):
        assert request.full_url == "http://localhost:11434/api/embed"
        body = json.loads(request.data)
        assert body == {"model": "example", "input": ["Документ", "Вопрос"], "truncate": False}
        return io.BytesIO(json.dumps({
            "model": "example", "embeddings": [[1, 0, 0], [0, 1, 0]],
            "total_duration": 1, "load_duration": 0, "prompt_eval_count": 2,
        }).encode())

    monkeypatch.setattr("src.rag.embeddings.urlopen", respond)
    embedder = OllamaEmbedder(model="example", dimensions=3)
    assert embedder.embed(["Документ", "Вопрос"]) == [[1, 0, 0], [0, 1, 0]]


def test_malformed_ollama_response_rejected(monkeypatch):
    monkeypatch.setattr("src.rag.embeddings.urlopen", lambda *a, **kw: io.BytesIO(b'{}'))
    with pytest.raises(ValueError, match="embeddings"):
        OllamaEmbedder().embed(["Документ"])
