from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.qdrant import qdrant_client
from app.rag.embeddings import generate_embeddings


COLLECTION_NAME = "enterprise_documents"
VECTOR_SIZE = 384


def create_collection():
    collections = qdrant_client.get_collections()

    existing_collections = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME not in existing_collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        return "Collection created"

    return "Collection already exists"


def store_chunks(
    chunks: list[str],
    filename: str,
):
    embeddings = generate_embeddings(chunks)

    points = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        point = PointStruct(
            id=index,
            vector=embedding,
            payload={
                "filename": filename,
                "chunk_id": index,
                "text": chunk,
            },
        )

        points.append(point)

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return len(points)
