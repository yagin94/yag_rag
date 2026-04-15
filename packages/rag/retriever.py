from sentence_transformers import SentenceTransformer

from packages.core.config import settings
from packages.rag.keyword_scorer import keyword_score
from packages.rag.qdrant_store import get_client
import logging
from time import perf_counter

logger = logging.getLogger("rag.retrieve")

model = SentenceTransformer(settings.embedding_model)

MIN_DENSE_SCORE = 0.35
MIN_FINAL_SCORE = 0.35


def embed_text(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()


def retrieve(query: str, limit: int = 5, request_id: str = "no_request_id") -> list[dict]:
    start = perf_counter()

    client = get_client()
    normalized_query = query.strip().lower()

    if not normalized_query:
        logger.warning(f"[{request_id}] Retrieval skipped | reason=empty_query")
        return []

    vector = embed_text(normalized_query)

    query_limit = settings.hybrid_candidate_k if settings.hybrid_enabled else limit

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=query_limit,
    )

    docs = []
    for r in results.points:
        payload = r.payload or {}
        title = payload.get("title", "")
        text = payload.get("text", "")

        dense_score = float(r.score)
        lexical_score = keyword_score(normalized_query, title, text) if settings.hybrid_enabled else 0.0
        final_score = (
            settings.hybrid_dense_weight * dense_score
            + settings.hybrid_lexical_weight * lexical_score
            if settings.hybrid_enabled
            else dense_score
        )

        docs.append({
            "id": r.id,
            "score": final_score,
            "dense_score": dense_score,
            "lexical_score": lexical_score,
            "payload": payload,
        })

    docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    filtered_docs = [
        doc for doc in docs
        if doc.get("dense_score", 0.0) >= MIN_DENSE_SCORE
        and doc.get("score", 0.0) >= MIN_FINAL_SCORE
    ]

    docs = filtered_docs[:limit]

    latency_ms = int((perf_counter() - start) * 1000)
    if settings.log_queries:
        logger.info(
            "retrieval_done request_id=%s latency_ms=%s retrieved=%s raw_candidates=%s hybrid=%s normalized_query=%s",
            request_id,
            latency_ms,
            len(docs),
            len(results.points),
            settings.hybrid_enabled,
            normalized_query,
        )
    else:
        logger.info(
            "retrieval_done request_id=%s latency_ms=%s retrieved=%s raw_candidates=%s hybrid=%s",
            request_id,
            latency_ms,
            len(docs),
            len(results.points),
            settings.hybrid_enabled,
        )

    return docs