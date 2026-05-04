# RAGChat Backend Documentation

This document provides a comprehensive guide to the **RAGChat Backend**, a Retrieval-Augmented Generation (RAG) system built with FastAPI. It handles multi-format document processing, high-speed semantic search, and persona-driven AI interactions.

---

## 🏗 Architecture Overview

The backend follows a modular service-oriented architecture designed for scalability and clear separation of concerns.

### 🛠 Core Technology Stack
- **Framework**: FastAPI (Asynchronous Python)
- **Primary Database**: MongoDB (Chat history, Metadata, User Profiles, Personas)
- **Vector Database**: ChromaDB (Document Embeddings & Semantic Search)
- **LLM Engine**: Groq (Llama 3.3 70B)
- **Embeddings**: Local `Sentence-Transformers` (running on CPU/GPU)
- **Auth**: JWT (JSON Web Tokens) with bcrypt password hashing

---

## 📁 Directory Structure

```text
chat_backend_rag/
├── main.py            # Application entry point & middleware
├── auth.py            # JWT authentication & password security
├── database.py        # MongoDB connection management
├── models.py          # MongoDB Pydantic models (Data Persistence)
├── schemas.py         # API Request/Response Pydantic schemas (Validation)
├── routes/            # API Route Handlers (Controllers)
│   ├── auth.py        # /auth (Signup/Login)
│   ├── chat.py        # /chat (Sessions, Messages, History)
│   ├── documents.py   # /documents (Upload, List, Delete)
│   └── personas.py    # /personas (CRUD for AI behaviors)
├── services/          # Business Logic (The "Brain")
│   ├── rag_service.py        # Orchestrates the RAG pipeline
│   ├── embedding_service.py  # ChromaDB interface & Vector generation
│   ├── chunking_service.py   # Document parsing & text splitting
│   └── query_rewrite_service.py # LLM-based query optimization
├── uploads/           # Temporary storage for processed files
└── chroma_db/         # Local persistent storage for vector data
```

---

## 🚀 The RAG Pipeline (Data Flow)

The core strength of this backend is how it connects user questions to document data.

### 1. Document Ingestion Flow
When a user uploads a file (`.pdf`, `.docx`, `.csv`, `.txt`):
1. **Upload**: File is saved to `uploads/`.
2. **Chunking**: `ChunkingService` extracts text and splits it into small, overlapping chunks (e.g., 1000 characters).
3. **Embedding**: `EmbeddingService` converts each chunk into a 384-dimensional vector.
4. **Storage**: Vectors are stored in **ChromaDB**, and metadata (filename, size) is stored in **MongoDB**.

### 2. Chat & Retrieval Flow
When a user sends a message:
1. **Rewrite**: `QueryRewriteService` uses the chat history to turn "Tell me more" into "Tell me more about [Specific Topic]".
2. **Search**: The rewritten query is embedded and searched against **ChromaDB** to find the top-5 most relevant chunks.
3. **Context Construction**: These chunks are combined into a context block with source markers like `[SOURCE 1]`.
4. **LLM Generation**: The **Groq LLM** receives the context, the user's message, and their **Persona** instructions.
5. **Response**: The LLM generates an answer with citations. The backend returns the answer + the source references.

---

## 🛡 Security & Isolation

- **User Isolation**: All database queries and vector searches are strictly filtered by `user_id`. One user can never see another user's documents or chats.
- **Chat Isolation**: Document retrieval can be filtered by `chat_id` if specific documents are linked to a single conversation.
- **JWT Auth**: Every sensitive endpoint requires a valid `Authorization: Bearer <token>` header.

---

## 👤 Persona System

The backend supports "Personas" which act as a **System Prompt** for the AI.
- **Attributes**: Profession, Purpose, Tone, Knowledge Level, Language.
- **Logic**: The `rag_service.py` dynamically builds a prompt that forces the AI to adopt these traits while strictly adhering to the retrieved document facts.

---

## 📡 API Endpoints Summary

### Authentication
- `POST /auth/signup`: Create a new account.
- `POST /auth/signin`: Get access token.

### Chat Management
- `POST /chat/session`: Initialize a new conversation.
- `POST /chat/message`: Send a message and get a RAG-augmented response.
- `GET /chat/sessions`: List all previous chats.
- `GET /chat/history/{session_id}`: Retrieve all messages in a session.

### Document Management
- `POST /documents/upload`: Process a file and store its embeddings.
- `GET /documents/list`: See all documents accessible by the user.
- `DELETE /documents/{doc_id}`: Remove document from MongoDB, ChromaDB, and Disk.

### Personas
- `POST /personas/`: Create a custom AI behavior.
- `GET /personas/`: List all personas.

---

## 📊 Database Schemas

### MongoDB (Chats)
```json
{
  "session_id": "uuid",
  "user_id": "uuid",
  "title": "Chat Title",
  "messages": [
    {
      "role": "user/assistant",
      "content": "...",
      "sources": [...],
      "timestamp": "ISO-8601"
    }
  ],
  "doc_ids": ["uuid1", "uuid2"],
  "persona_id": "uuid"
}
```

### ChromaDB (Metadata)
Each vector in ChromaDB is stored with:
- `user_id`: For strict multi-tenancy.
- `chat_id`: For session-specific context.
- `doc_id`: To link back to the source file.
- `source`: Filename.
- `page`: Page number (if PDF).
