from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from apps.symptoms.schemas import SymptomRead


# -------------------------------
# Base schema
# -------------------------------
class CaseBase(BaseModel):
    user_id: str = Field(..., example="USER5678")
    age: int = Field(..., example=30)
    gender: str = Field(..., example="male")
    confidence: float = Field(..., example=0.95)


# -------------------------------
# Create schema
# -------------------------------
class CaseCreate(CaseBase):
    pass


# -------------------------------
# Update schema
# -------------------------------
class CaseUpdate(BaseModel):
    user_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    confidence: Optional[float] = None



class CaseRead(BaseModel):
    id: int
    uuid: UUID
    user_id: str
    age: int
    gender: str
    diagnosis: str | None
    confidence: float
    symptoms: list[SymptomRead] = []   # ← nested list

    class Config:
        from_attributes = True