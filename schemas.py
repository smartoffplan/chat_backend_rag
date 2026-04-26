from pydantic import BaseModel
from typing import Optional


# ── CHAT ────────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"
    doc_ids: Optional[list[str]] = []

class ChatRequest(BaseModel):
    session_id: str
    message: str
    doc_ids: Optional[list[str]] = []   # empty = search ALL uploaded docs


class SourceCitation(BaseModel):
    reference_id: int
    source: str
    page: Optional[str] = None
    doc_id: str
    text_excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    session_id: str
    rewritten_query: Optional[str] = None


# ── DOCUMENTS ────────────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    upload_date: str
