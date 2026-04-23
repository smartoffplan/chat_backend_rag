import os
from groq import Groq
from dotenv import load_dotenv
from .embedding_service import EmbeddingService
from .query_rewrite_service import rewrite_query

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

class RAGService:
    def __init__(self):
        self.embedder = EmbeddingService()

    # ── MAIN PIPELINE ────────────────────────────────────────────────────────
    async def answer_with_rag(
        self,
        query: str,
        chat_history: list[dict],
        doc_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> dict:
        """
        Full RAG pipeline:
          1. Rewrite query
          2. Retrieve top-k chunks from ChromaDB
          3. Build prompt with [SOURCE N] markers
          4. Call LLM
          5. Return answer + citations

        Returns:
        {
            "answer": str,
            "sources": [ { reference_id, source, page, doc_id, text_excerpt, score } ],
            "rewritten_query": str
        }
        """

        # ── Step 1: Query rewriting ──────────────────────────────────────────
        rewritten_query = await rewrite_query(query, chat_history)

        # ── Step 2: Retrieve relevant chunks ────────────────────────────────
        chunks = self.embedder.query(
            rewritten_query,
            doc_ids=doc_ids if doc_ids else None,
            top_k=top_k,
        )

        # ── Step 3: No chunks found → plain LLM answer ──────────────────────
        if not chunks:
            answer = await self._call_llm(query, chat_history, context="")
            return {
                "answer": answer,
                "sources": [],
                "rewritten_query": rewritten_query,
            }

        # ── Step 4: Build context string with source markers ────────────────
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            page_info = f", page {chunk['page']}" if chunk.get("page") else ""
            context_parts.append(
                f"[SOURCE {i}: {chunk['source']}{page_info}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # ── Step 5: Generate answer ──────────────────────────────────────────
        answer = await self._call_llm(query, chat_history, context)

        # ── Step 6: Build citation objects ───────────────────────────────────
        sources = []
        seen_keys = set()
        for i, chunk in enumerate(chunks, start=1):
            key = f"{chunk['doc_id']}_{chunk['chunk_index']}"
            if key not in seen_keys:
                seen_keys.add(key)
                excerpt = chunk["text"]
                sources.append({
                    "reference_id": i,
                    "source":       chunk["source"],
                    "page":         chunk.get("page") or None,
                    "doc_id":       chunk["doc_id"],
                    "text_excerpt": excerpt[:200] + "..." if len(excerpt) > 200 else excerpt,
                    "score":        chunk["score"],
                })

        return {
            "answer":          answer,
            "sources":         sources,
            "rewritten_query": rewritten_query,
        }

    # ── LLM CALL ─────────────────────────────────────────────────────────────
    async def _call_llm(self, query, chat_history, context):

        system_prompt = (
            "You are a helpful assistant. "
            "Use provided document context when available. "
            "Cite sources like [SOURCE 1]."
        )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in chat_history[-6:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        if context:
            user_content = f"""
Use this context:

{context}

Question:
{query}
"""
        else:
            user_content = query

        messages.append({"role": "user", "content": user_content})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Groq error: {str(e)}"