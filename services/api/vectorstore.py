from __future__ import annotations
from typing import Dict, Any, List
import os, json, logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PERSIST_DIR = os.getenv("PERSIST_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "vectorstore_data")))
COLLECTION = "properties"
logging.getLogger(__name__).warning(f"[Chroma] PERSIST_DIR = {PERSIST_DIR}")


_model = None
_client = None
_collection = None

# --- add alongside your property helpers ---
MSG_COLLECTION = "messages"

def _msg_collection():
    global _client
    client = _client or chromadb.Client(Settings(
        is_persistent=True, persist_directory=PERSIST_DIR, anonymized_telemetry=False
    ))
    return client.get_or_create_collection(MSG_COLLECTION, metadata={"hnsw:space":"cosine"})

def embed_text(text: str) -> list[float]:
    model = get_model()  # or _model_instance()
    return model.encode([text])[0].tolist()

def add_message_embedding(msg_id: str, text: str, metadata: dict) -> None:
    coll = _msg_collection()
    emb = embed_text(text)
    coll.upsert(ids=[msg_id], documents=[text], embeddings=[emb], metadatas=[metadata])

def search_messages(query: str, top_k: int = 5, filters: dict | None = None):
    coll = _msg_collection()
    q_emb = embed_text(query)
    res = coll.query(query_embeddings=[q_emb], n_results=top_k*2, include=["ids","metadatas","distances"])
    items = []
    ids = res.get("ids", [[]])[0]; metas = res.get("metadatas", [[]])[0]; dists = res.get("distances", [[]])[0]
    for i, mid in enumerate(ids):
        meta = metas[i] or {}
        if filters:
            ok = True
            for k,v in filters.items():
                if v is None: continue
                ok &= str(meta.get(k,"")).lower().find(str(v).lower()) >= 0
            if not ok: continue
        items.append({"id": mid, "score": 1.0 - float(dists[i]), "metadata": meta})
        if len(items) >= top_k: break
    return items

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.Client(Settings(is_persistent=True, persist_directory=PERSIST_DIR))
        _collection = _client.get_or_create_collection(COLLECTION, metadata={"hnsw:space":"cosine"})
    return _collection


def _flatten_property(doc: Dict[str, Any]) -> str:
# Turn the complex schema into a searchable paragraph
    parts = []
    for k in [
    "description","unitId","physical_features","design_and_layout","living_experience",
    "equipment_and_handover_materials","legal_and_product_status","property_groups"
    ]:
        v = doc.get(k)
        if not v: continue
        if isinstance(v, (dict, list)):
            parts.append(json.dumps(v, ensure_ascii=False))
        else:
            parts.append(str(v))    
    # add light meta if available
    meta = doc.get("_meta", {})
    title = meta.get("title")
    location = meta.get("location")
    ptype = meta.get("property_type")
    price = meta.get("price")
    br = meta.get("bedrooms")
    parts.append(f"title:{title} location:{location} type:{ptype} bedrooms:{br} price:{price}")
    return "\n".join(parts)

def add_or_update(doc: Dict[str, Any]):
    coll = get_collection()
    model = get_model()
    text = _flatten_property(doc)
    emb = model.encode([text])[0].tolist()
    pid = doc["id"]
    meta = doc.get("_meta", {})
    # Upsert
    coll.upsert(documents=[text], embeddings=[emb], ids=[pid], metadatas=[meta])


def delete(pid: str):
    coll = get_collection()
    coll.delete(ids=[pid])


def search(query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    coll = get_collection()
    model = get_model()
    q_emb = model.encode([query])[0].tolist()
# Apply simple metadata filtering in-app    
    res = coll.query(query_embeddings=[q_emb], n_results=top_k, include=["metadatas", "distances"])   # ← no "ids" here)

    items = []
    for i, pid in enumerate(res.get("ids", [[]])[0]):
        meta = res["metadatas"][0][i] or {}
        # naive filter
        ok = True
        if "location" in filters and filters["location"]:
            ok &= (str(meta.get("location",""))).lower().find(str(filters["location"]).lower()) >= 0
        if "property_type" in filters and filters["property_type"]:
            ok &= (str(meta.get("property_type",""))).lower().find(str(filters["property_type"]).lower()) >= 0
        if "bedrooms" in filters and filters["bedrooms"]:
            try:
                ok &= float(meta.get("bedrooms", 0)) >= float(filters["bedrooms"]) - 0.1
            except: pass
        if "budget_max" in filters and filters["budget_max"]:
            try:
                ok &= float(meta.get("price", 1e12)) <= float(filters["budget_max"]) + 1e-6
            except: pass        
        if not ok:
            continue
        items.append({
        "id": pid,
        "score": 1.0 - float(res["distances"][0][i]),
        "metadata": meta,
        })
        if len(items) >= top_k:
            break
    return items