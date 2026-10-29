import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from src.rag.api_embeddings import APIEmbedder


@pytest.fixture
def http(monkeypatch):
    requests = []

    def install(responses):
        replies = iter(responses)

        def send(request, timeout):
            requests.append(request)
            response = next(replies)
            if isinstance(response, Exception):
                raise response
            return io.BytesIO(json.dumps(response).encode())

        monkeypatch.setattr(
            "src.rag.api_embeddings.build_opener", lambda *a: SimpleNamespace(open=send),
        )
        return requests

    return install


def client(**kwargs):
    return APIEmbedder(
        base_url="https://api.example/v1", api_key="test-secret",
        model="test-model", dimensions=3, **kwargs,
    )


def response(*vectors):
    return {
        "object": "list", "model": "test-model",
        "data": [
            {"object": "embedding", "index": i, "embedding": vector}
            for i, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


def test_api_request_and_restoring_response_order(http):
    payload = response([1, 0, 0], [0, 1, 0])
    payload["data"].reverse()
    requests = http([payload])
    assert client().embed(["Документ", "Вопрос"]) == [[1, 0, 0], [0, 1, 0]]
    request = requests[0]
    assert request.full_url == "https://api.example/v1/embeddings"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert json.loads(request.data) == {
        "model": "test-model", "input": ["Документ", "Вопрос"],
        "dimensions": 3, "encoding_format": "float",
    }


def test_api_splits_large_batch_without_losing_vectors(http):
    requests = http([response(*([[1, 0, 0]] * 10)), response(*([[0, 1, 0]] * 6))])
    vectors = client().embed([f"Текст {i}" for i in range(16)])
    assert vectors == [[1, 0, 0]] * 10 + [[0, 1, 0]] * 6
    assert [len(json.loads(request.data)["input"]) for request in requests] == [10, 6]


@pytest.mark.parametrize("payload", [
    {}, {"data": []}, {"data": "not-a-list"},
    {"data": [{"index": 0, "embedding": [1, 0, 0]}] * 2},
    {"data": [{"index": 2, "embedding": [1, 0, 0]}]},
    {"data": [{"index": True, "embedding": [1, 0, 0]}]},
    {"data": [{"embedding": [1, 0, 0]}]},
    {"data": [{"index": 0}]}, {"data": [None]},
    response([1, 0]), response([0, 0, 0]), response([float("nan"), 0, 1]),
])
def test_malformed_api_embeddings_rejected(http, payload):
    http([payload])
    with pytest.raises(ValueError):
        client().embed(["Текст"])


@pytest.mark.parametrize("status", [401, 403, 400, 302])
def test_permanent_http_errors_are_safe_and_not_retried(http, status):
    error = HTTPError("https://api.example/v1/embeddings", status, "test-secret", {}, None)
    requests = http([error])
    with pytest.raises(RuntimeError) as caught:
        client().embed(["Текст"])
    assert str(status) in str(caught.value)
    assert "test-secret" not in str(caught.value)
    assert len(requests) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_error_retried(http, monkeypatch, status):
    monkeypatch.setattr("src.rag.api_embeddings.sleep", lambda _: None)
    error = HTTPError("https://api.example/v1/embeddings", status, "temporary", {}, None)
    requests = http([error, response([1, 0, 0])])
    assert client().embed(["Текст"]) == [[1, 0, 0]]
    assert len(requests) == 2


def test_retry_budget_exhausted(http, monkeypatch):
    monkeypatch.setattr("src.rag.api_embeddings.sleep", lambda _: None)
    error = HTTPError("https://api.example/v1/embeddings", 429, "limit", {}, None)
    requests = http([error, error, error])
    with pytest.raises(RuntimeError, match="429"):
        client().embed(["Текст"])
    assert len(requests) == 3


def test_long_retry_after_fails_without_early_retry(http, monkeypatch):
    monkeypatch.setattr(
        "src.rag.api_embeddings.sleep", lambda _: pytest.fail("Нельзя ждать вместо отказа"),
    )
    error = HTTPError(
        "https://api.example/v1/embeddings", 429, "limit", {"Retry-After": "120"}, None,
    )
    requests = http([error])
    with pytest.raises(RuntimeError, match="429"):
        client().embed(["Текст"])
    assert len(requests) == 1


def test_network_error_does_not_expose_details(http):
    http([URLError("test-secret")])
    with pytest.raises(RuntimeError) as caught:
        client().embed(["Текст"])
    assert "test-secret" not in str(caught.value)


@pytest.mark.parametrize("url", [
    "", "ftp://api.example/v1", "http://api.example/v1",
    "https://user:secret@api.example/v1", "https://api.example/v1?key=secret",
    "https://api.example/v1#secret", "https://api.example/v1/embeddings",
])
def test_invalid_or_unsafe_api_url_rejected(url):
    with pytest.raises(ValueError):
        APIEmbedder(url, "test-secret", "test-model", dimensions=3)


def test_endpoint_and_model_define_index_identity_without_key():
    first = client()
    same = APIEmbedder("https://api.example/v1/", "rotated-secret", "test-model", 3)
    other = APIEmbedder("https://another.example/v1", "test-secret", "test-model", 3)
    assert first.model == same.model
    assert first.model != other.model
    assert "test-secret" not in first.model
    assert first.model != "test-model"
