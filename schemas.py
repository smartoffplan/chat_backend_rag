from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from models import PyObjectId

class ChatSessionResponse(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    title: str = "New Chat"
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

    class Config:
        populate_by_name = True
        json_encoders = {PyObjectId: str}

class MessageResponse(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    sessionId: PyObjectId
    role: str
    content: str
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

    class Config:
        populate_by_name = True
        json_encoders = {PyObjectId: str}

class SendMessageRequest(BaseModel):
    sessionId: str
    message: str

class SendMessageResponse(BaseModel):
    reply: str
