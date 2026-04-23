import os
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone

from database import get_db
from schemas import ChatRequest
from services.rag_service import RAGService

router      = APIRouter(prefix="/chat", tags=["chat"])
rag_service = RAGService()


# ── SEND MESSAGE ─────────────────────────────────────────────────────────────
@router.post("/message")
async def send_message(
    request: ChatRequest,
    db=Depends(get_db),
):
    # Load existing session from MongoDB
    session      = await db["chats"].find_one({"session_id": request.session_id})
    chat_history = session.get("messages", []) if session else []

    # Run RAG pipeline
    result = await rag_service.answer_with_rag(
        query=request.message,
        chat_history=chat_history,
        doc_ids=request.doc_ids if request.doc_ids else None,
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

    if session:
        await db["chats"].update_one(
            {"session_id": request.session_id},
            {"$push": {"messages": {"$each": [user_msg, assistant_msg]}}},
        )
    else:
        await db["chats"].insert_one({
            "session_id": request.session_id,
            "messages":   [user_msg, assistant_msg],
            "created_at": now,
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
        "messages":   session.get("messages", []),
    }


# ── LIST SESSIONS ─────────────────────────────────────────────────────────────
@router.get("/sessions")
async def list_sessions(db=Depends(get_db)):
    sessions = await db["chats"].find(
        {}, {"session_id": 1, "created_at": 1, "_id": 0}
    ).to_list(100)
    return sessions


# ── DELETE SESSION ────────────────────────────────────────────────────────────
@router.delete("/session/{session_id}")
async def delete_session(session_id: str, db=Depends(get_db)):
    result = await db["chats"].delete_one({"session_id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Session not found.")
    return {"status": "deleted", "session_id": session_id}
