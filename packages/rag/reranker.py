from __future__ import annotations

from typing import Any
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not docs:
            return []

        pairs = []
        for doc in docs:
            text = (
                doc.get("payload", {}).get("text")
                or doc.get("text")
                or ""
            )
            pairs.append([query, text])

        scores = self.model.predict(pairs)

        rescored = []
        for doc, score in zip(docs, scores):
            item = dict(doc)
            item["rerank_score"] = float(score)
            rescored.append(item)

        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return rescored[:top_k]