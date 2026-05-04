import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ── Readiness flag — False until the model and DB are loaded ─────────────────
_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    # Import routes here so the sentence-transformer model and ChromaDB
    # load BEFORE uvicorn starts accepting traffic (fixes Cloud Run startup timeout)
    from routes import chat, documents, auth, personas
    app.include_router(auth.router)
    app.include_router(personas.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    _ready = True
    print("[Startup] All services ready.")
    yield
    # Shutdown
    _ready = False


app = FastAPI(title="RAG Chatbot API", version="2.0.0", lifespan=lifespan)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "https://chat-frontend-rag.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "running", "version": "2.0.0 (RAG enabled)"}


@app.get("/health")
async def health():
    # Returns 503 until model + ChromaDB finish loading, so Cloud Run
    # startup probe only passes once the app is truly ready
    if not _ready:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok"}
