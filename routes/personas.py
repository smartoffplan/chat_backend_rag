import uuid
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone

from database import get_db
from schemas import PersonaCreateRequest, PersonaUpdateRequest
from auth import get_current_user

router = APIRouter(prefix="/personas", tags=["personas"])


# ── HELPERS ─────────────────────────────────────────────────────────────────
def _serialize(doc):
    """Convert a MongoDB persona doc into a clean dict."""
    if not doc:
        return None
    return {
        "persona_id": str(doc.get("_id")),
        "user_id": doc.get("user_id"),
        "persona_name": doc.get("persona_name"),
        "profession": doc.get("profession"),
        "purpose": doc.get("purpose"),
        "domain": doc.get("domain"),
        "knowledge_level": doc.get("knowledge_level"),
        "preferred_language": doc.get("preferred_language"),
        "tone": doc.get("tone"),
        "answer_style": doc.get("answer_style"),
        "output_format": doc.get("output_format"),
        "citation_preference": doc.get("citation_preference"),
        "document_behavior": doc.get("document_behavior"),
        "restrictions": doc.get("restrictions"),
        "color": doc.get("color", "#F97316"),
        "is_default": doc.get("is_default", False),
        "created_at": doc.get("created_at"),
    }


# ── CREATE ──────────────────────────────────────────────────────────────────
@router.post("")
async def create_persona(
    request: PersonaCreateRequest,
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    persona_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # If this is the user's first persona, make it default
    existing_count = await db["personas"].count_documents({"user_id": user["user_id"]})
    is_default = existing_count == 0

    doc = {
        "_id": persona_id,
        "user_id": user["user_id"],
        "persona_name": request.persona_name,
        "profession": request.profession,
        "purpose": request.purpose,
        "domain": request.domain,
        "knowledge_level": request.knowledge_level,
        "preferred_language": request.preferred_language,
        "tone": request.tone,
        "answer_style": request.answer_style,
        "output_format": request.output_format,
        "citation_preference": request.citation_preference,
        "document_behavior": request.document_behavior,
        "restrictions": request.restrictions,
        "color": request.color or "#F97316",
        "is_default": is_default,
        "created_at": now,
        "updated_at": now,
    }

    await db["personas"].insert_one(doc)
    return _serialize(doc)


# ── LIST ────────────────────────────────────────────────────────────────────
@router.get("")
async def list_personas(db=Depends(get_db), user=Depends(get_current_user)):
    cursor = db["personas"].find({"user_id": user["user_id"]}).sort("created_at", -1)
    docs = await cursor.to_list(100)
    return [_serialize(d) for d in docs]


# ── GET ONE ─────────────────────────────────────────────────────────────────
@router.get("/{persona_id}")
async def get_persona(persona_id: str, db=Depends(get_db), user=Depends(get_current_user)):
    doc = await db["personas"].find_one({"_id": persona_id, "user_id": user["user_id"]})
    if not doc:
        raise HTTPException(404, "Persona not found.")
    return _serialize(doc)


# ── UPDATE ──────────────────────────────────────────────────────────────────
@router.put("/{persona_id}")
async def update_persona(
    persona_id: str,
    request: PersonaUpdateRequest,
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    update_fields = {k: v for k, v in request.model_dump().items() if v is not None}
    if not update_fields:
        raise HTTPException(400, "No fields provided for update.")

    update_fields["updated_at"] = datetime.now(timezone.utc).isoformat()

    result = await db["personas"].update_one(
        {"_id": persona_id, "user_id": user["user_id"]},
        {"$set": update_fields},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Persona not found.")

    doc = await db["personas"].find_one({"_id": persona_id})
    return _serialize(doc)


# ── DELETE ──────────────────────────────────────────────────────────────────
@router.delete("/{persona_id}")
async def delete_persona(persona_id: str, db=Depends(get_db), user=Depends(get_current_user)):
    result = await db["personas"].delete_one({"_id": persona_id, "user_id": user["user_id"]})
    if result.deleted_count == 0:
        raise HTTPException(404, "Persona not found.")

    # If the deleted persona was default, set the oldest remaining as default
    was_default = await db["personas"].find_one({"_id": persona_id})
    if was_default and was_default.get("is_default"):
        oldest = await db["personas"].find_one(
            {"user_id": user["user_id"]}, sort=[("created_at", 1)]
        )
        if oldest:
            await db["personas"].update_one(
                {"_id": oldest["_id"]}, {"$set": {"is_default": True}}
            )

    return {"status": "deleted", "persona_id": persona_id}


# ── SET DEFAULT ─────────────────────────────────────────────────────────────
@router.post("/{persona_id}/set-default")
async def set_default_persona(
    persona_id: str,
    db=Depends(get_db),
    user=Depends(get_current_user),
):
    # Clear existing default
    await db["personas"].update_many(
        {"user_id": user["user_id"]}, {"$set": {"is_default": False}}
    )
    # Set new default
    result = await db["personas"].update_one(
        {"_id": persona_id, "user_id": user["user_id"]},
        {"$set": {"is_default": True}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Persona not found.")

    doc = await db["personas"].find_one({"_id": persona_id})
    return _serialize(doc)
