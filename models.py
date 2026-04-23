from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DocumentModel(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    upload_date: str
    status: str


class MessageModel(BaseModel):
    role: str                        # "user" or "assistant"
    content: str
    sources: Optional[list] = []
    rewritten_query: Optional[str] = None
    timestamp: str


class ChatSessionModel(BaseModel):
    session_id: str
    messages: list[MessageModel] = []
    created_at: str
