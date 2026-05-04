import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "rag_documents"


class EmbeddingService:
    """
    Singleton with fully deferred initialization.
    Heavy imports (torch, sentence_transformers, chromadb) happen ONLY inside
    _ensure_ready(), never at module import time.  This keeps the Python startup
    phase fast and memory-light so Cloud Run's TCP probe passes immediately.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def _ensure_ready(self):
        if self._ready:
            return

        # Heavy imports deferred to here — not at module level
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings

        print("[EmbeddingService] Loading sentence-transformer model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("[EmbeddingService] Model loaded.")

        try:
            print(f"[EmbeddingService] Connecting to ChromaDB at: {CHROMA_PATH}")
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            print("[EmbeddingService] ChromaDB ready.")
        except Exception as e:
            print(f"[EmbeddingService] WARNING: ChromaDB init failed: {type(e).__name__}: {e}. Queries will return no context.")
            self.chroma_client = None
            self.collection = None

        self._ready = True
        print("[EmbeddingService] Ready.")

    # ── STORE ────────────────────────────────────────────────────────────────
    def embed_chunks(self, chunks: list[dict], chat_id: str | None = None, user_id: str | None = None) -> None:
        self._ensure_ready()
        if not chunks or self.collection is None:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
        ids = [f"{c['doc_id']}_chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                "source":      c.get("source", ""),
                "doc_id":      c.get("doc_id", ""),
                "page":        str(c.get("page") or ""),
                "chunk_index": str(c.get("chunk_index", 0)),
                "file_type":   c.get("file_type", ""),
                "chat_id":     chat_id if chat_id else "",
                "user_id":     user_id or "",
            }
            for c in chunks
        ]
        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas,
        )
        print(f"[EmbeddingService] Stored {len(chunks)} chunks.")

    # ── QUERY ────────────────────────────────────────────────────────────────
    def query(
        self,
        query_text: str,
        chat_id: str | None = None,
        doc_ids: list[str] | None = None,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        self._ensure_ready()
        if self.collection is None:
            return []
        query_embedding = self.model.encode([query_text]).tolist()[0]

        conditions = []
        if chat_id:
            conditions.append({"chat_id": chat_id})
        if doc_ids:
            conditions.append({"doc_id": {"$in": doc_ids}})
        if user_id:
            conditions.append({"user_id": user_id})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}
        else:
            where_filter = None

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"[EmbeddingService] ChromaDB query failed: {type(e).__name__}: {e}")
            return []

        output = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({
                    "text":        doc,
                    "source":      meta.get("source", ""),
                    "page":        meta.get("page", ""),
                    "doc_id":      meta.get("doc_id", ""),
                    "chunk_index": meta.get("chunk_index", ""),
                    "score":       round(1 - dist, 4),
                })
        return output

    # ── DELETE ───────────────────────────────────────────────────────────────
    def delete_document(self, doc_id: str) -> None:
        self._ensure_ready()
        if self.collection is None:
            return
        self.collection.delete(where={"doc_id": doc_id})
        print(f"[EmbeddingService] Deleted chunks for doc {doc_id}")

    def delete_by_chat_id(self, chat_id: str) -> None:
        self._ensure_ready()
        if self.collection is None:
            return
        self.collection.delete(where={"chat_id": chat_id})
        print(f"[EmbeddingService] Deleted chunks for chat {chat_id}")

    # ── STATS ────────────────────────────────────────────────────────────────
    def collection_count(self) -> int:
        self._ensure_ready()
        if self.collection is None:
            return 0
        return self.collection.count()
