"""Source-level Recall@k и MRR для размеченных вопросов."""

from collections.abc import Callable

from .store import SearchResult


def evaluate(
    cases: list[dict], retrieve: Callable[..., list[SearchResult]], top_k: int = 5,
) -> dict:
    if not isinstance(cases, list) or not cases or top_k <= 0:
        raise ValueError("Нужны непустой набор вопросов и top_k > 0")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Каждый пример eval должен быть объектом")
        query = case.get("query")
        sources = case.get("expected_sources")
        if (
            not isinstance(query, str) or not query.strip()
            or not isinstance(sources, list) or not sources
            or any(not isinstance(source, str) or not source.strip() for source in sources)
        ):
            raise ValueError("Нужны query и непустой список expected_sources")
    details = []
    for case in cases:
        results = retrieve(case["query"], top_k=top_k)
        expected = set(case["expected_sources"])
        sources = [result.source for result in results[:top_k]]
        first_hit = next((i for i, source in enumerate(sources, 1) if source in expected), None)
        details.append({
            "query": case["query"], "retrieved_sources": sources,
            "recall_at_k": len(expected.intersection(sources)) / len(expected),
            "reciprocal_rank": 1 / first_hit if first_hit else 0.0,
        })
    return {
        "queries": len(details), "top_k": top_k,
        "recall_at_k": sum(item["recall_at_k"] for item in details) / len(details),
        "mrr": sum(item["reciprocal_rank"] for item in details) / len(details),
        "details": details,
    }
