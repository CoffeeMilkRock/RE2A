from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

PERSIST_DIR = os.getenv("PERSIST_DIR", r"C:\REA\vectorstore_data")
COLL_PROPERTIES = "properties"
_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def _chroma() -> Chroma:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    return Chroma(
        collection_name=COLL_PROPERTIES,
        embedding_function=_embeddings,
        persist_directory=PERSIST_DIR,
    )

def inspect_collection(collection_name="real_estate_embeddings"):
    vs = Chroma(
        collection_name=collection_name,
        persist_directory="C:/REA/vectorstore_data"  # đường dẫn bạn đang dùng
    )

    data = vs.get()
    print(f"📦 Tổng số vectors: {len(data['ids'])}")
    for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
        print("="*80)
        print(f"🧩 Chunk {i+1}: {meta.get('property_id')}::{meta.get('section')}::{meta.get('chunk_index')}")
        print(f"📄 Nội dung:\n{doc}\n")
        print(f"🧾 Metadata:\n{meta}\n")
def upsert_property_docs(chunks: List[Document]) -> Tuple[int, int]:
    vs = _chroma()
    ids, texts, metas = [], [], []
    pids = list({d.metadata.get("property_id") for d in chunks if d.metadata.get("property_id")})
    # xóa cũ theo property_id để "update"
    if pids:
        for pid in pids:
            try: vs._collection.delete(where={"property_id": pid})
            except: pass
    for d in chunks:
        pid = d.metadata.get("property_id") or "UNKNOWN"
        sec = d.metadata.get("section") or "misc"
        idx = d.metadata.get("chunk_index") or 0
        ids.append(f"{pid}::{sec}::{idx}")
        texts.append(d.page_content)
        metas.append(d.metadata)
    vs.add_texts(texts=texts, metadatas=metas, ids=ids)
    vs.persist()
    return len(ids), len(pids)

def delete_property(property_id: str) -> int:
    vs = _chroma()
    try:
        vs._collection.delete(where={"id": property_id})
        vs.persist()
        return 1
    except:
        return 0

def search_properties(query: str, filters: Dict[str, Any], top_k: int = 5):
    vs = _chroma()
    hard_where: Dict[str, Any] = {}
    if filters.get("property_type"):
        hard_where["property_type"] = str(filters["property_type"]).lower()

    docs_scores = vs.similarity_search_with_score(query, k=top_k * 5, filter=hard_where or None)

    print(f"search_properties: query={query}, filters={filters}, found={len(docs_scores)}")
    print(f"  hard_where={hard_where}")
    print(f"  docs_scores[0:3]={docs_scores[0:3]}")
    out = []
    q_loc = str(filters.get("location") or "").lower()
    q_type = str(filters.get("property_type") or "").lower()
    q_bed = filters.get("bedrooms")
    q_budget = filters.get("budget_max")

    for doc, score in docs_scores:
        m = doc.metadata or {}
        ok = True
        if q_type:
            ok &= str(m.get("property_type","")).lower() == q_type
        if q_loc:
            ok &= q_loc in str(m.get("location","")).lower()
        if q_bed is not None:
            try: ok &= float(m.get("bedrooms") or 0) >= float(q_bed) - 1e-6
            except: pass
        if q_budget is not None:
            try: ok &= float(m.get("price") or 1e12) <= float(q_budget) + 1e-6
            except: pass

        if ok:
            out.append({
                "id": f'{m.get("property_id")}::{m.get("section")}::{m.get("chunk_index")}',
                "score": float(score),
                "metadata": m,
                "page_content": doc.page_content
            })
            if len(out) >= top_k:
                break
    return out
