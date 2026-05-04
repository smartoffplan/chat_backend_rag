import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "chatbot_db")

_client = None
_db = None


def _get_db():
    global _client, _db
    if _client is None:
        _client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
        _db = _client[DB_NAME]
    return _db


async def get_db():
    return _get_db()
