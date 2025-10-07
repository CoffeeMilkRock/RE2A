from __future__ import annotations
from typing import Any, Dict, List
from langchain.schema import Document
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
# Các key KHÔNG đưa vào text (đi khắp JSON theo path)
NOISE_KEYS = {
    "misc.created_at", "misc.updated_at",
    "design_and_layout.id", "design_and_layout.code", "design_and_layout.uuid",
    "physical_features.id", "physical_features.code",
    "legal_and_product_status.id", "legal_and_product_status.code",
    "equipment_and_handover_materials.id",
    # Mẫu tổng quát ở dưới xử lý thêm: .*id, .*uuid, .*code
}

# Allowlist theo section (nếu section có trong ALLOW_SECTIONS thì CHỈ giữ các key có prefix này)
ALLOW_SECTIONS = {
    # Chỉ là ví dụ—tùy dữ liệu thực tế của bạn
    "design_and_layout": {
        "design_and_layout.location",
        "design_and_layout.type",
        "design_and_layout.property_type",
        "design_and_layout.bedrooms",
        "design_and_layout.bathrooms",
        "design_and_layout.area",
        "design_and_layout.price",
        "design_and_layout.floor",
    },
    "living_experience": set(),  # rỗng = cho phép tất cả (trừ noise)
    "physical_features": set(),
    "legal_and_product_status": {
        "legal_and_product_status.status",
        "legal_and_product_status.red_book",
        "legal_and_product_status.ownership",
    },
}

# Heuristic lọc giá trị
MIN_VALUE_CHARS = 4           # bỏ các mẩu quá ngắn (vd: "ok", "N/A")
MAX_VALUE_CHARS = 600         # cắt bớt giá trị quá dài  (per value)
MAX_LIST_ITEMS_PER_SECTION = 50  # list quá dài thì cắt bớt

# Regex để bỏ chuỗi có ít ngữ nghĩa
RE_MOSTLY_NUMERIC = re.compile(r"^[\d\W_]+$")  # toàn số/ký tự không chữ
RE_UUID_LIKE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
RE_URL = re.compile(r"https?://\S+")
RE_CODEY = re.compile(r"[{}[\];<>]{3,}")
RE_KEY_NOISE_SUFFIX = re.compile(r"(?:^|\.)(?:id|uuid|code|created_at|updated_at)(?:\[.*\])?$", re.I)

def _flatten(obj: Any, prefix: str = "", section: str | None = None) -> List[tuple[str, str]]:
    """
    Ép JSON thành list (key_path, text_value) với lọc noise theo key/value.
    """
    out: List[tuple[str, str]] = []

    def to_str(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, (str, int, float, bool)):
            return str(x)
        # stringify gọn cho object phức tạp
        try:
            s = json.dumps(x, ensure_ascii=False)
            return s
        except Exception:
            return str(x)

    def rec(node: Any, path: str, lvl_section: str | None):
        # Lọc theo key ngay tại đây: nếu path trúng noise key thì bỏ
        if path and _is_noise_key(path, lvl_section):
            return
        
        if isinstance(node, dict):
            for k, v in node.items():
                child_path = f"{path}.{k}" if path else k
                rec(v, child_path, lvl_section)
        elif isinstance(node, list):
            # cắt bớt list dài
            for i, v in enumerate(node[:MAX_LIST_ITEMS_PER_SECTION]):
                child_path = f"{path}[{i}]"
                rec(v, child_path, lvl_section)
        else:
            text = to_str(node).strip()
            # Lọc theo giá trị
            if not text or _is_noise_value(text):
                return
            text = _truncate_value(text)
            out.append((path, text))

    rec(obj, prefix, section)
    return out

def _safe_float(v: Any) -> float | None:
    try:
        if v is None: return None
        if isinstance(v, (int, float)): return float(v)
        s = str(v).replace(",", "").strip()
        return float(s) if s else None
    except: return None

def _first_non_empty(data: dict, keys: List[str]) -> str | None:
    def get_path(d, path: str):
        cur = d
        for part in path.split("."):
            if "[" in part and part.endswith("]"):
                name = part[:part.index("[")]
                idx = int(part[part.index("[")+1:-1])
                cur = (cur.get(name) or [])[idx] if isinstance(cur, dict) else None
            else:
                cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None: return None
        return cur
    for k in keys:
        val = get_path(data, k)
        if val not in (None, "", []): return str(val)
    return None

def json_to_documents(property_json: Dict[str, Any]) -> List[Document]:
    pid = property_json.get("id") or property_json.get("unitId") or "UNKNOWN"
    buckets = {
        "description": property_json.get("description", ""),
        "design_and_layout": property_json.get("design_and_layout", {}),
        "living_experience": property_json.get("living_experience", {}),
        "physical_features": property_json.get("physical_features", {}),
        "equipment_and_handover_materials": property_json.get("equipment_and_handover_materials", {}),
        "legal_and_product_status": property_json.get("legal_and_product_status", {}),
        "property_groups": property_json.get("property_groups", []),
        "misc": {
            "unitId": property_json.get("unitId"),
            "created_at": property_json.get("created_at"),
            "updated_at": property_json.get("updated_at"),
        },
    }

    docs: List[Document] = []
    for section, content in buckets.items():
        parts = _flatten(content, prefix=section, section=section)  # <-- truyền section
        if not parts:
            continue

        # Ghép các cặp (key_path, value) thành text “sạch” hơn
        # Bạn có thể bỏ prefix [key] nếu muốn, mình giữ lại để truy vết
        lines = [f"[{k}] {v}" for k, v in parts]
        text_block = "\n".join(lines)

        metadata = {
            "property_id": pid,
            "section": section,
            "unitId": property_json.get("unitId"),
            "location": _first_non_empty(property_json, ["design_and_layout.location","physical_features.location"]),
            "property_type": _first_non_empty(property_json, ["design_and_layout.type","design_and_layout.property_type"]),
            "bedrooms": _safe_float(_first_non_empty(property_json, ["design_and_layout.bedrooms"])),
            "price": _safe_float(_first_non_empty(property_json, ["design_and_layout.price"])),
        }

        # 🔧 1) ép kiểu về primitive và bỏ None
        cleaned = {}
        for k, v in metadata.items():
            if v is None:
                continue
            # chuẩn hoá string
            if k in ("location", "property_type", "unitId", "section", "property_id"):
                cleaned[k] = str(v)
            # numeric giữ float
            elif k in ("bedrooms", "price"):
                try:
                    cleaned[k] = float(v)
                except Exception:
                    continue
            else:
                # fallback: stringify
                cleaned[k] = str(v)

        metadata = cleaned
        docs.append(Document(page_content=text_block, metadata=metadata))   
    return docs
def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n","\n",". "," ",""],
    )
    out: List[Document] = []
    for d in docs:
        chs = splitter.split_documents([d])
        for i, c in enumerate(chs):
            c.metadata = dict(c.metadata or {})
            c.metadata["chunk_index"] = i
        out.extend(chs)
    return out

def _is_noise_key(path: str, section: str | None = None) -> bool:
    # Bỏ nếu trong NOISE_KEYS hoặc có hậu tố "id/uuid/code/created_at/updated_at"
    if path in NOISE_KEYS:
        return True
    if RE_KEY_NOISE_SUFFIX.search(path):
        return True

    # Nếu section có allowlist: chỉ giữ key bắt đầu đúng allow
    if section and section in ALLOW_SECTIONS and ALLOW_SECTIONS[section]:
        allowed = ALLOW_SECTIONS[section]
        # Cho qua nếu path bắt đầu bằng bất kỳ prefix trong allowed
        return not any(path.startswith(p) for p in allowed)

    return False

def _is_noise_value(text: str) -> bool:
    if not text or len(text) < MIN_VALUE_CHARS:
        return True
    if RE_MOSTLY_NUMERIC.match(text):
        return True
    if RE_UUID_LIKE.search(text):
        return True
    if RE_URL.search(text):
        # URL thường không hữu ích cho semsearch (trừ khi bạn muốn giữ)
        return True
    if RE_CODEY.search(text):
        return True
    return False

def _truncate_value(text: str) -> str:
    if not text:
        return text
    if len(text) > MAX_VALUE_CHARS:
        return text[:MAX_VALUE_CHARS] + "…"
    return text