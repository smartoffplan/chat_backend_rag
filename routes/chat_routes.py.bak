from fastapi import APIRouter, HTTPException, Body
from typing import List
from datetime import datetime, timezone
from bson import ObjectId

from database import get_database
from schemas import ChatSessionResponse, MessageResponse, SendMessageRequest, SendMessageResponse
from services.ai_service import ask_ai

router = APIRouter()

@router.post("/session", response_model=ChatSessionResponse)
async def create_session():
    db = get_database()
    now = datetime.now(timezone.utc)
    
    session = {
        "title": "New Chat",
        "createdAt": now,
        "updatedAt": now
    }
    
    result = await db.chatsessions.insert_one(session)
    session["_id"] = result.inserted_id
    
    return session

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_sessions():
    db = get_database()
    cursor = db.chatsessions.find().sort("updatedAt", -1)
    sessions = await cursor.to_list(length=1000)
    return sessions

@router.get("/messages/{session_id}", response_model=List[MessageResponse])
async def get_messages(session_id: str):
    db = get_database()
    
    if not ObjectId.is_valid(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
        
    cursor = db.messages.find({"sessionId": ObjectId(session_id)}).sort("createdAt", 1)
    messages = await cursor.to_list(length=1000)
    return messages

@router.post("/message", response_model=SendMessageResponse)
async def send_message(payload: SendMessageRequest = Body(...)):
    db = get_database()
    
    if not ObjectId.is_valid(payload.sessionId):
        raise HTTPException(status_code=400, detail="Invalid session id")
        
    session_oid = ObjectId(payload.sessionId)
    now = datetime.now(timezone.utc)
    
    # Check if session exists
    session = await db.chatsessions.find_one({"_id": session_oid})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Save user message
    user_msg = {
        "sessionId": session_oid,
        "role": "user",
        "content": payload.message,
        "createdAt": now,
        "updatedAt": now
    }
    await db.messages.insert_one(user_msg)
    
    # Update session updatedAt
    await db.chatsessions.update_one(
        {"_id": session_oid},
        {"$set": {"updatedAt": now}}
    )
    
    # Fetch history
    cursor = db.messages.find({"sessionId": session_oid}).sort("createdAt", 1)
    history = await cursor.to_list(length=1000)
    
    # Map for AI
    ai_messages = [{"role": m["role"], "content": m["content"]} for m in history]
    
    try:
        reply = await ask_ai(ai_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Save assistant message
    now = datetime.now(timezone.utc)
    assistant_msg = {
        "sessionId": session_oid,
        "role": "assistant",
        "content": reply,
        "createdAt": now,
        "updatedAt": now
    }
    await db.messages.insert_one(assistant_msg)
    
    return {"reply": reply}

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    db = get_database()
    
    if not ObjectId.is_valid(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
        
    session_oid = ObjectId(session_id)
    
    delete_result = await db.chatsessions.delete_one({"_id": session_oid})
    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
        
    await db.messages.delete_many({"sessionId": session_oid})
    
    return {"message": "Session and associated messages deleted successfully"}
