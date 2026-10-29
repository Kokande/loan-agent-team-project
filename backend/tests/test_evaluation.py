from types import SimpleNamespace

import pytest

from src.rag.evaluation import evaluate


def test_source_recall_and_mrr_ignore_duplicate_chunks():
    cases = [
        {"query": "Первый вопрос", "expected_sources": ["a", "b"]},
        {"query": "Второй вопрос", "expected_sources": ["c"]},
    ]

    def retrieve(query, top_k):
        sources = ["x", "a", "a", "b"] if query == "Первый вопрос" else ["x"]
        return [SimpleNamespace(source=source) for source in sources[:top_k]]

    result = evaluate(cases, retrieve, top_k=4)
    assert result["queries"] == 2
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] == 0.25


@pytest.mark.parametrize("cases", [
    [], [{"query": "Вопрос", "expected_sources": []}],
    [{"query": " ", "expected_sources": ["a"]}],
    [{"query": "Вопрос", "expected_sources": "a"}],
    [{"query": "Вопрос", "expected_sources": [""]}],
])
def test_invalid_eval_dataset_rejected(cases):
    with pytest.raises(ValueError):
        evaluate(cases, lambda *a, **kw: [], top_k=5)
