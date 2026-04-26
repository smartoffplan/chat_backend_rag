import os
import uuid
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone

from database import get_db
from schemas import ChatRequest, CreateSessionRequest
from services.rag_service import RAGService

router      = APIRouter(prefix="/chat", tags=["chat"])
rag_service = RAGService()


# ── CREATE SESSION ───────────────────────────────────────────────────────────
@router.post("/session")
async def create_session(
    request: CreateSessionRequest = None,
    db=Depends(get_db),
):
    """Create a new chat session with a unique session_id."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    doc_ids = request.doc_ids if request and request.doc_ids else []
    title = request.title if request and request.title else "New Chat"

    session_doc = {
        "session_id": session_id,
        "title": title,
        "messages": [],
        "doc_ids": doc_ids,
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
):
    # Load existing session from MongoDB
    session      = await db["chats"].find_one({"session_id": request.session_id})
    chat_history = session.get("messages", []) if session else []

    # Merge request doc_ids with session doc_ids
    session_doc_ids = session.get("doc_ids", []) if session else []
    request_doc_ids = request.doc_ids if request.doc_ids else []
    all_doc_ids = list(dict.fromkeys(session_doc_ids + request_doc_ids))  # preserve order, remove duplicates

    # Run RAG pipeline with strict chat_id isolation
    result = await rag_service.answer_with_rag(
        query=request.message,
        chat_history=chat_history,
        chat_id=request.session_id,
        doc_ids=all_doc_ids if all_doc_ids else None,
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
            {"session_id": request.session_id},
            update_fields,
        )
    else:
        await db["chats"].insert_one({
            "session_id": request.session_id,
            "title": request.message[:50] + "..." if len(request.message) > 50 else request.message,
            "messages":   [user_msg, assistant_msg],
            "doc_ids": request_doc_ids,
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
async def get_history(session_id: str, db=Depends(get_db)):
    session = await db["chats"].find_one({"session_id": session_id})
    if not session:
        return {"session_id": session_id, "messages": []}
    return {
        "session_id": session_id,
        "title": session.get("title", "Chat"),
        "messages":   session.get("messages", []),
        "doc_ids": session.get("doc_ids", []),
    }


# ── LIST SESSIONS ─────────────────────────────────────────────────────────────
@router.get("/sessions")
async def list_sessions(db=Depends(get_db)):
    sessions = await db["chats"].find(
        {}, {"session_id": 1, "title": 1, "created_at": 1, "_id": 0}
    ).sort("updated_at", -1).to_list(100)
    return sessions


# ── UPDATE SESSION DOC_IDS ──────────────────────────────────────────────────
@router.put("/session/{session_id}/docs")
async def update_session_docs(
    session_id: str,
    request: CreateSessionRequest,
    db=Depends(get_db),
):
    """Update the document IDs linked to a chat session."""
    result = await db["chats"].update_one(
        {"session_id": session_id},
        {"$set": {"doc_ids": request.doc_ids or [], "updated_at": datetime.now(timezone.utc).isoformat()}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Session not found.")
    return {"status": "updated", "session_id": session_id, "doc_ids": request.doc_ids or []}


# ── DELETE SESSION ────────────────────────────────────────────────────────────
@router.delete("/session/{session_id}")
async def delete_session(session_id: str, db=Depends(get_db)):
    result = await db["chats"].delete_one({"session_id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Session not found.")
    return {"status": "deleted", "session_id": session_id}
