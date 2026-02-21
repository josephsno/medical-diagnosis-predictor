from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime

# -------------------------------
# Base schema (shared fields)
# -------------------------------
class SymptomBase(BaseModel):
    case_id: Optional[str] = None
    symptom: Optional[str] = None
    severity: int
    duration_hours: float
    body_location: Optional[str] = None


# -------------------------------
# Schema for creating a new symptom
# -------------------------------
class SymptomCreate(SymptomBase):
    pass  # Inherits all fields; case_id can be optional if handled elsewhere


# -------------------------------
# Schema for updating a symptom
# -------------------------------
class SymptomUpdate(SymptomBase):
    pass  # Optional: you can make all fields Optional if partial updates are allowed


# -------------------------------
# Schema for reading (response)
# -------------------------------
class SymptomRead(SymptomBase):
    id: int
    uuid: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
       from_attributes = True  # Tells Pydantic to read data from ORM objects
