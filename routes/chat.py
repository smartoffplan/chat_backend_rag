import os
import uuid
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone

from database import get_db
from schemas import ChatRequest, CreateSessionRequest
from services.rag_service import RAGService
from auth import get_current_user

router      = APIRouter(prefix="/chat", tags=["chat"])
rag_service = RAGService()


# ── CREATE SESSION ───────────────────────────────────────────────────────────
@router.post("/session")
async def create_session(
    request: CreateSessionRequest = None,
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    """Create a new chat session with a unique session_id."""
    now = datetime.now(timezone.utc).isoformat()

    # Use provided session_id if given and not already taken by this user
    if request and request.session_id:
        existing = await db["chats"].find_one({"session_id": request.session_id, "user_id": user["user_id"]})
        if existing:
            raise HTTPException(409, "Session ID already exists.")
        session_id = request.session_id
    else:
        session_id = str(uuid.uuid4())

    doc_ids = request.doc_ids if request and request.doc_ids else []
    title = request.title if request and request.title else "New Chat"
    persona_id = request.persona_id if request and request.persona_id else None

    session_doc = {
        "session_id": session_id,
        "user_id": user["user_id"],
        "title": title,
        "messages": [],
        "doc_ids": doc_ids,
        "persona_id": persona_id,
        "created_at": now,
        "updated_at": now,
    }
    await db["chats"].insert_one(session_doc)

    return {
        "session_id": session_id,
        "title": title,
        "doc_ids": doc_ids,
        "created_at": now,
    }


# ── SEND MESSAGE ─────────────────────────────────────────────────────────────
@router.post("/message")
async def send_message(
    request: ChatRequest,
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    # Load existing session from MongoDB (scoped to user)
    session = await db["chats"].find_one({"session_id": request.session_id, "user_id": user["user_id"]})
    chat_history = session.get("messages", []) if session else []

    # Merge request doc_ids with session doc_ids
    session_doc_ids = session.get("doc_ids", []) if session else []
    request_doc_ids = request.doc_ids if request.doc_ids else []
    all_doc_ids = list(dict.fromkeys(session_doc_ids + request_doc_ids))  # preserve order, remove duplicates

    # Load persona — explicit request persona_id overrides session persona_id
    effective_persona_id = request.persona_id or (session.get("persona_id") if session else None)
    persona = None
    if effective_persona_id:
        persona_doc = await db["personas"].find_one(
            {"_id": effective_persona_id, "user_id": user["user_id"]}
        )
        if persona_doc:
            persona = persona_doc
    # Run RAG pipeline with strict chat_id + user_id isolation
    result = await rag_service.answer_with_rag(
        query=request.message,
        chat_history=chat_history,
        chat_id=request.session_id,
        doc_ids=all_doc_ids if all_doc_ids else None,
        user_id=user["user_id"],
        persona=persona,
    )

    now = datetime.now(timezone.utc).isoformat()

    user_msg = {
        "role":      "user",
        "content":   request.message,
        "timestamp": now,
    }

    assistant_msg = {
        "role":            "assistant",
        "content":         result["answer"],
        "sources":         result["sources"],
        "rewritten_query": result.get("rewritten_query"),
        "timestamp":       now,
    }

    # Update title from first user message if still default
    update_fields = {
        "$push": {"messages": {"$each": [user_msg, assistant_msg]}},
        "$set": {"updated_at": now},
    }
    if session and (not session.get("title") or session.get("title") == "New Chat"):
        update_fields["$set"]["title"] = request.message[:50] + "..." if len(request.message) > 50 else request.message

    if session:
        await db["chats"].update_one(
            {"session_id": request.session_id, "user_id": user["user_id"]},
            update_fields,
        )
    else:
        await db["chats"].insert_one({
            "session_id": request.session_id,
            "user_id": user["user_id"],
            "title": request.message[:50] + "..." if len(request.message) > 50 else request.message,
            "messages":   [user_msg, assistant_msg],
            "doc_ids": request_doc_ids,
            "persona_id": effective_persona_id,
            "created_at": now,
            "updated_at": now,
        })

    return {
        "answer":          result["answer"],
        "sources":         result["sources"],
        "session_id":      request.session_id,
        "rewritten_query": result.get("rewritten_query"),
    }


# ── GET HISTORY ───────────────────────────────────────────────────────────────
@router.get("/history/{session_id}")
async def get_history(session_id: str, db=Depends(get_db), user=Depends(get_current_user)):
    session = await db["chats"].find_one({"session_id": session_id, "user_id": user["user_id"]})
    if not session:
        return {"session_id": session_id, "messages": []}
    return {
        "session_id": session_id,
        "title": session.get("title", "Chat"),
        "messages":   session.get("messages", []),
        "doc_ids": session.get("doc_ids", []),
        "persona_id": session.get("persona_id"),
    }


# ── LIST SESSIONS ─────────────────────────────────────────────────────────────
@router.get("/sessions")
async def list_sessions(db=Depends(get_db), user=Depends(get_current_user)):
    sessions = await db["chats"].find(
        {"user_id": user["user_id"]}, {"session_id": 1, "title": 1, "created_at": 1, "_id": 0}
    ).sort("updated_at", -1).to_list(100)
    return sessions


# ── UPDATE SESSION DOC_IDS ──────────────────────────────────────────────────
@router.put("/session/{session_id}/docs")
async def update_session_docs(
    session_id: str,
    request: CreateSessionRequest,
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    """Update the document IDs linked to a chat session."""
    result = await db["chats"].update_one(
        {"session_id": session_id, "user_id": user["user_id"]},
        {"$set": {"doc_ids": request.doc_ids or [], "updated_at": datetime.now(timezone.utc).isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Session not found.")
    return {"status": "updated", "session_id": session_id, "doc_ids": request.doc_ids or []}


# ── UPDATE SESSION PERSONA ──────────────────────────────────────────────────
@router.put("/session/{session_id}/persona")
async def update_session_persona(
    session_id: str,
    request: CreateSessionRequest,
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    """Update the persona linked to a chat session."""
    result = await db["chats"].update_one(
        {"session_id": session_id, "user_id": user["user_id"]},
        {"$set": {"persona_id": request.persona_id, "updated_at": datetime.now(timezone.utc).isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Session not found.")
    return {"status": "updated", "session_id": session_id, "persona_id": request.persona_id}


# ── DELETE SESSION ────────────────────────────────────────────────────────────
@router.delete("/session/{session_id}")
async def delete_session(session_id: str, db=Depends(get_db), user=Depends(get_current_user)):
    # 1. Find all documents associated with this chat to clean up disk files
    docs = await db["documents"].find({"chat_id": session_id, "user_id": user["user_id"]}).to_list(None)
    
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    allowed_extensions = {".pdf", ".docx", ".csv", ".eml", ".msg", ".txt"} # Same as in documents.py

    for doc in docs:
        doc_id = doc["_id"]
        # Try to delete file from disk for each possible extension
        for ext in allowed_extensions:
            path = os.path.join(upload_dir, f"{doc_id}{ext}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"[DeleteSession] Error removing file {path}: {e}")
                break

    # 2. Delete associated embeddings from ChromaDB
    await rag_service.delete_chat_data(session_id)

    # 3. Delete document records from MongoDB (associated with this chat)
    await db["documents"].delete_many({"chat_id": session_id, "user_id": user["user_id"]})

    # 4. Delete the chat session itself from MongoDB
    result = await db["chats"].delete_one({"session_id": session_id, "user_id": user["user_id"]})
    
    if result.deleted_count == 0:
        raise HTTPException(404, "Session not found.")
    
    return {"status": "deleted", "session_id": session_id}
