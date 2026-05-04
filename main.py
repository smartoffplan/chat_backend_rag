import os
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from routes import chat, documents, auth, personas

# Becomes True once the model + ChromaDB finish loading in the background
_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    # Load model in a thread so the event loop (and port 8080) stays responsive.
    # /health returns 503 while loading; Cloud Run startup probe retries until 200.
    loop = asyncio.get_event_loop()
    from services.embedding_service import EmbeddingService
    await loop.run_in_executor(None, EmbeddingService()._ensure_ready)
    _ready = True
    print("[Startup] All services ready.")
    yield


app = FastAPI(title="RAG Chatbot API", version="2.0.0", lifespan=lifespan)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "https://chat-frontend-rag.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(personas.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/")
async def root():
    return {"status": "running", "version": "2.0.0 (RAG enabled)"}


@app.get("/health")
async def health():
    # 503 while model is loading → Cloud Run startup probe keeps retrying.
    # 200 once ready → probe passes and traffic is allowed in.
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok"}
