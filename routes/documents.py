import os
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from database import get_db
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from auth import get_current_user

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR        = os.getenv("UPLOAD_DIR", "./uploads")
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".eml", ".msg", ".txt"}
MAX_FILE_MB       = 20

# Make sure upload folder exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

chunker  = ChunkingService()
embedder = EmbeddingService()


# ── UPLOAD ────────────────────────────────────────────────────────────────────
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chat_id: str = "",
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    # Validate file extension
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"File type '{ext}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file bytes
    content = await file.read()

    # Validate file size
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {MAX_FILE_MB}MB limit.")

    # Save file to disk
    doc_id     = str(uuid.uuid4())
    saved_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

    with open(saved_path, "wb") as f:
        f.write(content)

    try:
        # ── Chunk ──────────────────────────────────────────────────────────
        chunks = chunker.process_file(saved_path, filename, doc_id)
        if not chunks:
            raise ValueError("No text could be extracted from this file.")

        # ── Embed + store in ChromaDB ───────────────────────────────────────
        embedder.embed_chunks(chunks, chat_id=chat_id, user_id=user["user_id"])

        # ── Save metadata to MongoDB ────────────────────────────────────────
        doc_record = {
            "_id":             doc_id,
            "user_id":         user["user_id"],
            "chat_id":         chat_id,
            "filename":        filename,
            "file_type":       ext.lstrip("."),
            "file_size_bytes": len(content),
            "chunk_count":     len(chunks),
            "upload_date":     datetime.now(timezone.utc).isoformat(),
            "status":          "ready",
        }
        await db["documents"].insert_one(doc_record)

        return JSONResponse({
            "doc_id":      doc_id,
            "filename":    filename,
            "chunk_count": len(chunks),
            "status":      "ready",
        })

    except Exception as e:
        # Clean up file on failure
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(500, f"Processing failed: {str(e)}")


# ── LIST ──────────────────────────────────────────────────────────────────────
@router.get("/list")
async def list_documents(chat_id: str = "", db=Depends(get_db), user=Depends(get_current_user)):
    query = {"user_id": user["user_id"]}
    if chat_id:
        query["chat_id"] = chat_id
    docs = await db["documents"].find(query).to_list(200)
    return [
        {
            "doc_id":      d["_id"],
            "chat_id":     d.get("chat_id", ""),
            "filename":    d["filename"],
            "file_type":   d["file_type"],
            "chunk_count": d["chunk_count"],
            "upload_date": d["upload_date"],
        }
        for d in docs
    ]


# ── DELETE ────────────────────────────────────────────────────────────────────
@router.delete("/{doc_id}")
async def delete_document(doc_id: str, chat_id: str = "", db=Depends(get_db), user=Depends(get_current_user)):
    query = {"_id": doc_id, "user_id": user["user_id"]}
    if chat_id:
        query["chat_id"] = chat_id
    doc = await db["documents"].find_one(query)
    if not doc:
        raise HTTPException(404, "Document not found.")

    # Remove from ChromaDB
    embedder.delete_document(doc_id)

    # Remove from MongoDB
    await db["documents"].delete_one(query)

    # Remove file from disk
    for ext in ALLOWED_EXTENSIONS:
        path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")
        if os.path.exists(path):
            os.remove(path)
            break

    return {"status": "deleted", "doc_id": doc_id}


# ── STATS (optional, useful for debugging) ────────────────────────────────────
@router.get("/stats")
async def stats(chat_id: str = "", db=Depends(get_db), user=Depends(get_current_user)):
    query = {"user_id": user["user_id"]}
    if chat_id:
        query["chat_id"] = chat_id
    doc_count   = await db["documents"].count_documents(query)
    chunk_count = embedder.collection_count()
    return {
        "total_documents":     doc_count,
        "total_chunks":        chunk_count,
        "documents_in_mongo":  doc_count,
        "chunks_in_chromadb":  chunk_count,
        "upload_dir":          UPLOAD_DIR,
        "chroma_path":         os.getenv("CHROMA_PATH", "./chroma_db"),
    }
