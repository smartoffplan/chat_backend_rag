import os
import threading
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from routes import chat, documents, auth, personas

# Set to True once the embedding model + ChromaDB finish loading
_ready = False


_startup_error: str = ""


def _warm_services():
    """Runs in a daemon thread — loads model without blocking uvicorn startup."""
    global _ready, _startup_error
    try:
        from services.embedding_service import EmbeddingService
        EmbeddingService()._ensure_ready()
        print("[Startup] All services ready.")
    except Exception as exc:
        _startup_error = str(exc)
        print(f"[Startup] WARNING: service warm-up failed: {exc}")
    finally:
        # Always mark ready so /health returns 200 and Cloud Run keeps the
        # revision alive. Model-dependent endpoints handle missing state gracefully.
        _ready = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fire model loading in a background thread.
    # The yield returns IMMEDIATELY so uvicorn opens port 8080 right away.
    # Cloud Run TCP probe passes within ~1 second regardless of model load time.
    t = threading.Thread(target=_warm_services, daemon=True)
    t.start()
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
    # Always 200 — tells Cloud Run the container is alive.
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    # Detailed readiness: use this to check if the AI model finished loading.
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "starting"})
    if _startup_error:
        return JSONResponse(status_code=500, content={"status": "error", "detail": _startup_error})
    return {"status": "ready"}
