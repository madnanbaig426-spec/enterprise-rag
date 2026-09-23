from app.core.qdrant import qdrant_client
from app.rag.embeddings import generate_embedding
from app.rag.vector_store import COLLECTION_NAME


def search_documents(query: str, limit: int = 5):

    query_embedding = generate_embedding(query)

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
        with_payload=True,
    ).points

    return results
