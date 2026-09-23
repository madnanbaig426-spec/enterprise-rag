from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException

from app.core.qdrant import qdrant_client
from app.rag.vector_store import create_collection, store_chunks
from app.rag.chunker import chunk_text
from app.rag.search import search_documents
from app.api.documents import router as documents_router
from app.services.document_service import extract_text_from_pdf
from app.services.llm_service import generate_answer


app = FastAPI(
    title="Enterprise RAG API",
    description="Enterprise Retrieval-Augmented Generation Platform",
    version="1.0.0",
)


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


app.include_router(documents_router)


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Enterprise RAG API is running"
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-rag-api"
    }


# --------------------------------------------------
# Qdrant Health Check
# --------------------------------------------------

@app.get("/qdrant-health")
def qdrant_health():

    collections = qdrant_client.get_collections()

    return {
        "status": "connected",
        "collections": [
            collection.name
            for collection in collections.collections
        ]
    }


# --------------------------------------------------
# Create Qdrant Collection
# --------------------------------------------------

@app.post("/qdrant/create-collection")
def create_qdrant_collection():

    result = create_collection()

    return {
        "status": "success",
        "message": result
    }


# --------------------------------------------------
# Test PDF Chunking
# --------------------------------------------------

@app.post("/documents/test-chunking")
async def test_chunking(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    file_path = UPLOAD_DIR / file.filename

    file_content = await file.read()

    file_path.write_bytes(file_content)

    text = extract_text_from_pdf(
        str(file_path)
    )

    chunks = chunk_text(text)

    return {
        "filename": file.filename,
        "total_characters": len(text),
        "total_chunks": len(chunks),
        "chunks": chunks
    }


# --------------------------------------------------
# Index PDF Document
# --------------------------------------------------

@app.post("/documents/index")
async def index_document(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    file_path = UPLOAD_DIR / file.filename

    file_content = await file.read()

    file_path.write_bytes(file_content)

    text = extract_text_from_pdf(
        str(file_path)
    )

    chunks = chunk_text(text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the PDF"
        )

    stored_vectors = store_chunks(
        chunks=chunks,
        filename=file.filename
    )

    return {
        "status": "success",
        "filename": file.filename,
        "characters": len(text),
        "chunks": len(chunks),
        "stored_vectors": stored_vectors
    }


# --------------------------------------------------
# Semantic Search
# --------------------------------------------------

@app.get("/search")
def search(
    query: str,
    limit: int = 5,
):

    results = search_documents(
        query=query,
        limit=limit,
    )

    return {
        "query": query,
        "results": [
            {
                "score": result.score,
                "filename": result.payload.get("filename"),
                "chunk_id": result.payload.get("chunk_id"),
                "text": result.payload.get("text"),
            }
            for result in results
        ],
    }


# --------------------------------------------------
# RAG Ask Endpoint
# --------------------------------------------------

@app.get("/ask")
def ask(
    question: str,
    limit: int = 5,
):

    results = search_documents(
        query=question,
        limit=limit,
    )

    context = "\n\n".join(
        result.payload.get("text", "")
        for result in results
    )

    answer = generate_answer(
        question=question,
        context=context,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "score": result.score,
                "filename": result.payload.get("filename"),
                "chunk_id": result.payload.get("chunk_id"),
            }
            for result in results
        ],
    }
