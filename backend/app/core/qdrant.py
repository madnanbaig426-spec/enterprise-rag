import os

from qdrant_client import QdrantClient
from dotenv import load_dotenv


load_dotenv()


QDRANT_HOST = os.getenv(
    "QDRANT_HOST",
    "localhost",
)

QDRANT_PORT = int(
    os.getenv(
        "QDRANT_PORT",
        "6333",
    )
)


qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)
