import os
from groq import Groq
from dotenv import load_dotenv
from .embedding_service import EmbeddingService
from .query_rewrite_service import rewrite_query

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_groq_client = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


class RAGService:
    def __init__(self):
        self.embedder = EmbeddingService()

    # ── MAIN PIPELINE ────────────────────────────────────────────────────────
    async def answer_with_rag(
        self,
        query: str,
        chat_history: list[dict],
        chat_id: str | None = None,
        doc_ids: list[str] | None = None,
        user_id: str | None = None,
        persona: dict | None = None,
        top_k: int = 5,
    ) -> dict:
        rewritten_query = await rewrite_query(query, chat_history)

        chunks = self.embedder.query(
            rewritten_query,
            chat_id=chat_id,
            doc_ids=doc_ids if doc_ids else None,
            user_id=user_id,
            top_k=top_k,
        )

        if not chunks:
            answer = await self._call_llm(query, chat_history, context="", persona=persona)
            return {"answer": answer, "sources": [], "rewritten_query": rewritten_query}

        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            page_info = f", page {chunk['page']}" if chunk.get("page") else ""
            context_parts.append(
                f"[SOURCE {i}: {chunk['source']}{page_info}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        answer = await self._call_llm(query, chat_history, context, persona=persona)

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

        return {"answer": answer, "sources": sources, "rewritten_query": rewritten_query}

    # ── LLM CALL ─────────────────────────────────────────────────────────────
    async def _call_llm(self, query, chat_history, context, persona=None):
        if persona:
            system_prompt = self._build_persona_prompt(persona)
        else:
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

        user_content = (
            f"Use this context:\n\n{context}\n\nQuestion:\n{query}"
            if context else query
        )
        messages.append({"role": "user", "content": user_content})

        try:
            response = _get_groq().chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq error: {str(e)}"

    async def delete_chat_data(self, chat_id: str) -> None:
        self.embedder.delete_by_chat_id(chat_id)

    # ── PERSONA PROMPT BUILDER ───────────────────────────────────────────────
    def _build_persona_prompt(self, persona: dict) -> str:
        lines = [
            "You are a personalized AI assistant.",
            "",
            "User Persona:",
            f"- Persona Name: {persona.get('persona_name', 'Assistant')}",
            f"- Profession: {persona.get('profession', 'General')}",
            f"- Purpose: {persona.get('purpose', 'Assist the user')}",
        ]
        for field, label in [
            ("domain", "Domain"), ("knowledge_level", "Knowledge Level"),
            ("preferred_language", "Preferred Language"), ("tone", "Tone"),
            ("answer_style", "Answer Style"), ("output_format", "Output Format"),
            ("citation_preference", "Citation Preference"),
            ("document_behavior", "Document Behavior"), ("restrictions", "Restrictions"),
        ]:
            if persona.get(field):
                lines.append(f"- {label}: {persona[field]}")

        lines.extend([
            "",
            "Behavior Rules:",
            "1. Answer according to the user's persona.",
            "2. Use the retrieved document context when available.",
            "3. Do not invent facts that are not present in the documents.",
            "4. If the answer is not found in the documents, clearly say so.",
            "5. Follow the requested tone, language, and output format.",
            "6. Cite sources like [SOURCE 1] when using document context.",
        ])
        return "\n".join(lines)
