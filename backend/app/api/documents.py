from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.services.document_service import extract_text_from_pdf


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    file_path = UPLOAD_DIR / file.filename

    file_content = await file.read()

    file_path.write_bytes(file_content)

    text = extract_text_from_pdf(str(file_path))

    return {
        "status": "success",
        "filename": file.filename,
        "characters": len(text),
        "text_preview": text[:500],
    }
