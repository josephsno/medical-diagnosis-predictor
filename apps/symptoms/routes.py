from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from database.database import get_db
from .schemas import SymptomCreate, SymptomRead, SymptomUpdate
from .crud import SymptomCRUD

router = APIRouter()


# -------------------------------
# LIST & CREATE
# -------------------------------
@router.get("/", response_model=List[SymptomRead])
def list_symptoms(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    crud = SymptomCRUD(db)
    return crud.list(skip=skip, limit=limit)


@router.post("/{case_uuid}/symptoms", response_model=SymptomRead)
def create_symptom(case_uuid: UUID, symptom_in: SymptomCreate, db: Session = Depends(get_db)):
    crud = SymptomCRUD(db)
    return crud.create(symptom_in, case_uuid=case_uuid)


# -------------------------------
# RETRIEVE / UPDATE / PATCH / DELETE BY UUID
# -------------------------------
@router.get("/{uuid}", response_model=SymptomRead)
def get_symptom(uuid: UUID, db: Session = Depends(get_db)):
    crud = SymptomCRUD(db)
    symptom = crud.get(uuid)
    if not symptom:
        raise HTTPException(status_code=404, detail="Symptom not found")
    return symptom


@router.put("/{uuid}", response_model=SymptomRead)
def update_symptom(
    uuid: UUID, symptom_in: SymptomUpdate, db: Session = Depends(get_db)
):
    crud = SymptomCRUD(db)
    updated = crud.update(uuid, symptom_in)
    if not updated:
        raise HTTPException(status_code=404, detail="Symptom not found")
    return updated


@router.patch("/{uuid}", response_model=SymptomRead)
def patch_symptom(uuid: UUID, symptom_in: SymptomUpdate, db: Session = Depends(get_db)):
    # PATCH works same as PUT but allows partial updates
    crud = SymptomCRUD(db)
    updated = crud.update(uuid, symptom_in)
    if not updated:
        raise HTTPException(status_code=404, detail="Symptom not found")
    return updated
