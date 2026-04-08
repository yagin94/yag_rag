from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from packages.rag.qdrant_store import ensure_collection, get_client
from packages.core.config import settings


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    return model.encode(text).tolist()


def main():
    ensure_collection()
    client = get_client()

    docs = [
        {
            "id": 1,
            "title": "Docker Basics",
            "text": "Docker lets you package applications into containers."
        },
        {
            "id": 2,
            "title": "LangGraph Basics",
            "text": "LangGraph helps orchestrate stateful workflows for LLM applications."
        },
        {
            "id": 3,
            "title": "Qdrant Basics",
            "text": "Qdrant stores vectors and payload metadata for retrieval."
        },
        {
            "id": 4,
            "title": "LangChain Basics",
            "text": "LangChain is a framework for building applications powered by language models."
        },
    ]

    points = []
    for d in docs:
        points.append(
            PointStruct(
                id=d["id"],
                vector=embed_text(d["text"]),
                payload=d
            )
        )

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )
    print("Ingestion completed.")


if __name__ == "__main__":
    main()