import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends

from database import get_db
from schemas import UserSignup, UserSignin, TokenResponse
from auth import get_password_hash, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse)
async def signup(request: UserSignup, db=Depends(get_db)):
    """Register a new user and return a JWT token."""
    existing = await db["users"].find_one({"email": request.email})
    if existing:
        raise HTTPException(400, "Email already registered.")

    user_id = str(uuid.uuid4())
    user_doc = {
        "_id": user_id,
        "email": request.email,
        "password_hash": get_password_hash(request.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db["users"].insert_one(user_doc)

    token = create_access_token(user_id)
    return TokenResponse(access_token=token, user_id=user_id, email=request.email)


@router.post("/signin", response_model=TokenResponse)
async def signin(request: UserSignin, db=Depends(get_db)):
    """Authenticate and return a JWT token."""
    user = await db["users"].find_one({"email": request.email})
    if not user:
        raise HTTPException(401, "Invalid email or password.")

    if not verify_password(request.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password.")

    token = create_access_token(user["_id"])
    return TokenResponse(access_token=token, user_id=user["_id"], email=user["email"])
