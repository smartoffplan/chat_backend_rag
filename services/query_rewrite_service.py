import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


async def rewrite_query(
    original_query: str,
    chat_history: list[dict] | None = None,
) -> str:
    history_text = ""
    if chat_history:
        recent = chat_history[-4:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent
        )

    prompt = f"""You are a search query optimizer.

Rewrite the user's query into a clear, specific, self-contained search query for a vector database.

Rules:
- Output ONLY the rewritten query
- No explanation, no quotes
- Max 20 words
- Make it precise and searchable

Conversation:
{history_text if history_text else "(no history)"}

Original query:
{original_query}

Rewritten query:"""

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You rewrite search queries."},
                {"role": "user", "content": prompt},
            ],
        )

        rewritten = response.choices[0].message.content.strip()

        if rewritten and 5 < len(rewritten) < 200 and "\n" not in rewritten:
            print(f"[QueryRewrite] '{original_query}' → '{rewritten}'")
            return rewritten

    except Exception as e:
        print(f"[QueryRewrite] Groq call failed: {e}. Using original query.")

    return original_query
