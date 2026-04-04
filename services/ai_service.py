import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

async def ask_ai(messages: list) -> str:
    completion = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return completion.choices[0].message.content
