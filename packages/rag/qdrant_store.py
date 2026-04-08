from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from packages.core.config import settings


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection() -> None:
    client = get_client()
    collections = client.get_collections().collections
    names = {c.name for c in collections}

    if settings.qdrant_collection not in names:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.vector_size,
                distance=Distance.COSINE,
            ),
        )