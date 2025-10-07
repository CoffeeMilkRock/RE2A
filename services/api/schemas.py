from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class CharacterIn(BaseModel):
    name: str
    style: str
    objective: str


class CharacterOut(CharacterIn):
    id: int


class PropertyIn(BaseModel):
    id: str
    unitId: str | None = None
    description: str = ""
    property_groups: List[Dict[str, Any]] = Field(default_factory=list)
    physical_features: Dict[str, Any] = Field(default_factory=dict)
    design_and_layout: Dict[str, Any] = Field(default_factory=dict)
    living_experience: Dict[str, Any] = Field(default_factory=dict)
    equipment_and_handover_materials: Dict[str, Any] = Field(default_factory=dict)
    legal_and_product_status: Dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

class PropertyResponse(BaseModel):
    message: str
    unitId: str | None = None
    success: bool
    additional: dict[str, Any] = Field(default_factory=dict)

class SearchRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = {}
    top_k: int = 5


class SearchResultItem(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResult(BaseModel):
    items: List[SearchResultItem]