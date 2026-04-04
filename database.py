import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_to_mongo():
    try:
        db.client = AsyncIOMotorClient(MONGO_URI)
        try:
            db.db = db.client.get_default_database()
        except:
            db.db = db.client["chattesting"] 
        print("MongoDB connected")
    except Exception as e:
        print(f"MongoDB connection error: {e}")

async def close_mongo_connection():
    if db.client:
        db.client.close()
        print("MongoDB connection closed")

def get_database():
    return db.db
