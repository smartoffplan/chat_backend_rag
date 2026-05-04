from pydantic import BaseModel, EmailStr
from typing import Optional


# ── AUTH ────────────────────────────────────────────────────────────────────

class UserSignup(BaseModel):
    email: EmailStr
    password: str


class UserSignin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str


# ── CHAT ────────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = None
    title: Optional[str] = "New Chat"
    doc_ids: Optional[list[str]] = []
    persona_id: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str
    doc_ids: Optional[list[str]] = []   # empty = search ALL uploaded docs
    persona_id: Optional[str] = None
    persona_name: Optional[str] = None
    persona_color: Optional[str] = None


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


# ── PERSONAS ────────────────────────────────────────────────────────────────

class PersonaCreateRequest(BaseModel):
    persona_name: str
    profession: str
    purpose: str
    domain: Optional[str] = None
    knowledge_level: Optional[str] = "Beginner"
    preferred_language: Optional[str] = "English"
    tone: Optional[str] = "Professional"
    answer_style: Optional[str] = "Detailed"
    output_format: Optional[str] = "Paragraphs"
    citation_preference: Optional[str] = "Cite relevant document sections"
    document_behavior: Optional[str] = "Use uploaded documents first"
    restrictions: Optional[str] = None
    color: Optional[str] = "#F97316"


class PersonaUpdateRequest(BaseModel):
    persona_name: Optional[str] = None
    profession: Optional[str] = None
    purpose: Optional[str] = None
    domain: Optional[str] = None
    knowledge_level: Optional[str] = None
    preferred_language: Optional[str] = None
    tone: Optional[str] = None
    answer_style: Optional[str] = None
    output_format: Optional[str] = None
    citation_preference: Optional[str] = None
    document_behavior: Optional[str] = None
    restrictions: Optional[str] = None
    color: Optional[str] = None


class PersonaResponse(BaseModel):
    persona_id: str
    user_id: str
    persona_name: str
    profession: str
    purpose: str
    domain: Optional[str] = None
    knowledge_level: Optional[str] = None
    preferred_language: Optional[str] = None
    tone: Optional[str] = None
    answer_style: Optional[str] = None
    output_format: Optional[str] = None
    citation_preference: Optional[str] = None
    document_behavior: Optional[str] = None
    restrictions: Optional[str] = None
    color: Optional[str] = "#F97316"
    is_default: bool = False
    created_at: Optional[str] = None
