from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session
import json
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field   
from services.api.vectorstore import add_message_embedding, search_messages

from .db import Base, engine, SessionLocal
from .models import Character, Property
from .schemas import CharacterIn, CharacterOut, PropertyIn, PropertyResponse, SearchRequest, SearchResult, SearchResultItem
from .vectorstore import add_or_update, delete, search
from services.api.chunker import json_to_documents, chunk_documents
from services.api.vectorstore_langchain import upsert_property_docs, delete_property, search_properties
app = FastAPI(title="Real Estate API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

class MessageIn(BaseModel):
    conversation_id: str
    user_id: str
    role: str = "user"           # "user" | "assistant" | "system"
    text: str
    intent: Optional[str] = None
    entities: list[dict] = Field(default_factory=list)
    slots: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)

class MessageAck(BaseModel):
    success: bool
    message: str
    id: str
    
class PropertyIn(BaseModel):
    id: str
    unitId: str | None = None
    physical_features: Dict[str, Any] = Field(default_factory=dict)
    design_and_layout: Dict[str, Any] = Field(default_factory=dict)
    living_experience: Dict[str, Any] = Field(default_factory=dict)
    equipment_and_handover_materials: Dict[str, Any] = Field(default_factory=dict)
    legal_and_product_status: Dict[str, Any] = Field(default_factory=dict)
    description: str | None = ""
    property_groups: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    additionalProp1: Dict[str, Any] = Field(default_factory=dict)

class APIResp(BaseModel):
    message: str
    unitId: Optional[str] = None
    success: bool
    additional: Dict[str, Any] = Field(default_factory=dict)

class SearchReq(BaseModel):
    query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5

class SearchOut(BaseModel):
    items: List[Dict[str, Any]]
    
@app.post("/api/v2/conversation/message", response_model=MessageAck)
def store_message(msg: MessageIn):
    # A stable id: conv:user:timestamp or you can pass one in extra
    import time
    ts = int(time.time() * 1000)
    msg_id = msg.extra.get("id") or f"{msg.conversation_id}:{msg.role}:{ts}"
    meta = {
        "conversation_id": msg.conversation_id,
        "user_id": msg.user_id,
        "role": msg.role,
        "intent": msg.intent,
        "entities": msg.entities,
        "slots": msg.slots,
        **msg.extra
    }
    add_message_embedding(msg_id=msg_id, text=msg.text, metadata=meta)
    return MessageAck(success=True, message="Message embedded", id=msg_id)

# Optional: search conversation messages (for memory / retrieval)
class MsgSearchReq(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    top_k: int = 5

class MsgSearchRes(BaseModel):
    items: list[dict]

@app.post("/api/v2/conversation/message/search", response_model=MsgSearchRes)
def search_conv_messages(req: MsgSearchReq):
    filters = {}
    if req.conversation_id: filters["conversation_id"] = req.conversation_id
    if req.user_id: filters["user_id"] = req.user_id
    items = search_messages(req.query, req.top_k, filters=filters)
    return MsgSearchRes(items=items)
# Characters CRUD
@app.get("/api/v2/prompt-config/agent/character", response_model=list[CharacterOut])
def list_characters():
    with SessionLocal() as db:
        rows = db.execute(select(Character)).scalars().all()
        return [CharacterOut(id=r.id, name=r.name, style=r.style, objective=r.objective) for r in rows]

@app.post("/api/v2/prompt-config/agent/character", response_model=CharacterOut)
def create_character(body: CharacterIn):
    with SessionLocal() as db:
        ch = Character(name=body.name, style=body.style, objective=body.objective)
        db.add(ch); db.commit(); db.refresh(ch)
        return CharacterOut(id=ch.id, name=ch.name, style=ch.style, objective=ch.objective)

@app.get("/api/v2/prompt-config/agent/character/{cid}", response_model=CharacterOut)
def get_character(cid: int):
    with SessionLocal() as db:
        ch = db.get(Character, cid)
        if not ch: raise HTTPException(404, "not found")
        return CharacterOut(id=ch.id, name=ch.name, style=ch.style, objective=ch.objective)

@app.put("/api/v2/prompt-config/agent/character/{cid}", response_model=CharacterOut)
def update_character(cid: int, body: CharacterIn):
    with SessionLocal() as db:
        ch = db.get(Character, cid)
        if not ch: raise HTTPException(404, "not found")
        ch.name, ch.style, ch.objective = body.name, body.style, body.objective
        db.commit(); db.refresh(ch)
        return CharacterOut(id=ch.id, name=ch.name, style=ch.style, objective=ch.objective)

@app.delete("/api/v2/prompt-config/agent/character/{cid}")
def delete_character(cid: int):
    with SessionLocal() as db:
        ch = db.get(Character, cid)
        if not ch: raise HTTPException(404, "not found")
        db.delete(ch); db.commit();
        return {"ok": True}

# # Property Embedding CRUD
# @app.post("/api/v2/property/embedding", response_model=PropertyResponse)
# def upsert_property(body: PropertyIn):
#     meta = {
#         "title": body.id,
#         "location": body.design_and_layout.get("location") or body.physical_features.get("location") or "",
#         "property_type": body.design_and_layout.get("type") or body.physical_features.get("type") or "",
#         "bedrooms": body.design_and_layout.get("bedrooms") or body.physical_features.get("bedrooms") or 0,
#         "price": body.design_and_layout.get("price") or body.physical_features.get("price") or 0,
#         "description": body.description or "",
#     }
#     doc = body.model_dump()
#     doc["_meta"] = meta

#     with SessionLocal() as db:
#         # Save relational row for quick reads
#         pr = db.get(Property, body.id)
#         if pr is None:
#             pr = Property(
#                 id=body.id,
#                 title=meta["title"],
#                 location=meta["location"],
#                 property_type=meta["property_type"],
#                 bedrooms=float(meta.get("bedrooms") or 0),
#                 price=float(meta.get("price") or 0),
#                 description=meta["description"],
#                 raw_json=json.dumps(body.model_dump(), ensure_ascii=False),
#             )
#             db.add(pr)
#         else:
#             pr.title=meta["title"]; pr.location=meta["location"]; pr.property_type=meta["property_type"]
#             pr.bedrooms=float(meta.get("bedrooms") or 0); pr.price=float(meta.get("price") or 0)
#             pr.description=meta["description"]; pr.raw_json=json.dumps(body.model_dump(), ensure_ascii=False)
#         db.commit()

#     add_or_update(doc)
#     return PropertyResponse(message="Embedding Property Successfully", unitId=body.unitId, success=True, additional={"id": body.id})

# @app.put("/api/v2/property/embedding")
# def update_property(body: PropertyIn):
#     return upsert_property(body)

# @app.delete("/api/v2/property/embedding/{pid}")
# def remove_property(pid: str):
#     with SessionLocal() as db:
#         pr = db.get(Property, pid)
#         if pr: db.delete(pr); db.commit()
#     delete(pid)
#     return {"ok": True}

# # Search
# @app.post("/api/v2/property/search", response_model=SearchResult)
# def search_properties(req: SearchRequest):
#     items = search(req.query, req.filters, req.top_k)
#     return SearchResult(items=[SearchResultItem(id=i["id"], score=i["score"], metadata=i["metadata"]) for i in items])

@app.post("/api/v2/property/embedding", response_model=APIResp)
def create_or_upsert_property(prop: PropertyIn):
    try:
        base_docs = json_to_documents(prop.model_dump())
        chunks = chunk_documents(base_docs, chunk_size=800, chunk_overlap=120)
        n_chunks, _ = upsert_property_docs(chunks)
        return APIResp(
            message=f"Embedded {n_chunks} chunks for {prop.id}",
            unitId=prop.unitId,
            success=True,
            additional={"id": prop.id, "chunks": n_chunks}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

@app.put("/api/v2/property/embedding", response_model=APIResp)
def update_property(prop: PropertyIn):
    return create_or_upsert_property(prop)
@app.get("/api/v2/property/vector/{property_id}")
def get_property_vectors(property_id: str):
    from .vectorstore_langchain import _chroma
    vs = _chroma()
    data = vs.get(where={"property_id": property_id})
    return {
        "total": len(data["ids"]),
        "chunks": [
            {
                "id": id_,
                "metadata": meta,
                "content_preview": doc
            }
            for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]
    }

@app.delete("/api/v2/property/embedding/{property_id}", response_model=APIResp)
def remove_property(property_id: str):
    try:
        deleted = delete_property(property_id)
        return APIResp(
            message=f"Deleted property {property_id}",
            unitId=property_id,
            success=bool(deleted),
            additional={}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

@app.post("/api/v2/property/search", response_model=SearchOut)
def search_endpoint(req: SearchReq):
    try:
        items = search_properties(req.query, req.filters, req.top_k)
        return SearchOut(items=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
@app.post("/api/v2/property/search/natural", response_model=SearchResult)
def natural_language_search(req: SearchRequest):
    # For now identical to /search. Keep this route to match your logs.
    items = search(req.query, req.filters, req.top_k)
    return SearchResult(items=[SearchResultItem(id=i["id"], score=i["score"], metadata=i["metadata"]) for i in items])