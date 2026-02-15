from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime


# -------------------------------
# Base schema
# -------------------------------
class CaseBase(BaseModel):
    case_id: str = Field(..., example="CASE1234")
    user_id: str = Field(..., example="USER5678")
    age: int = Field(..., example=30)
    gender: str = Field(..., example="male")
    diagnosis: str = Field(..., example="migraine")
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
    case_id: Optional[str] = None
    user_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    confidence: Optional[float] = None


# -------------------------------
# Read / Response schema
# -------------------------------
class CaseRead(CaseBase):
    id: int
    uuid: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
