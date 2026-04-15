from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any

from packages.core.config import settings
from packages.rag.reranker import Reranker


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model)


async def rerank_node(state: dict[str, Any]) -> dict[str, Any]:
    query = state.get("query", "")
    docs = state.get("retrieved_docs", []) or []

    if not settings.reranker_enabled:
        return {
            **state,
            "retrieved_docs": docs,
            "retrieval_meta": {
                **state.get("retrieval_meta", {}),
                "reranker_enabled": False,
                "reranked": False,
            },
        }

    if not query or not docs:
        return {
            **state,
            "retrieved_docs": docs,
            "retrieval_meta": {
                **state.get("retrieval_meta", {}),
                "reranker_enabled": False,
                "reranked": False,
                "reason": "empty_query_or_docs",
            },
        }

    start = perf_counter()

    try:
        candidate_k = min(settings.reranker_candidate_k, len(docs))
        candidates = docs[:candidate_k]

        reranker = get_reranker()
        reranked_docs = reranker.rerank(
            query=query,
            docs=candidates,
            top_k=min(settings.reranker_top_k, candidate_k),
        )

        latency_ms = int((perf_counter() - start) * 1000)

        return {
            **state,
            "retrieved_docs": reranked_docs,
            "retrieval_meta": {
                **state.get("retrieval_meta", {}),
                "reranker_enabled": True,
                "reranked": True,
                "candidate_k": candidate_k,
                "final_k": len(reranked_docs),
                "reranker_model": settings.reranker_model,
                "rerank_latency_ms": latency_ms,
            },
        }
    except Exception as e:
        latency_ms = int((perf_counter() - start) * 1000)

        return {
            **state,
            "retrieved_docs": docs,
            "retrieval_meta": {
                **state.get("retrieval_meta", {}),
                "reranker_enabled": True,
                "reranked": False,
                "rerank_latency_ms": latency_ms,
                "error": str(e),
                "fallback": "original_retrieval_order",
            },
        }