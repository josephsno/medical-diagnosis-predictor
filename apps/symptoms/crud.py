from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID, uuid4

from .models import Symptom
from .schemas import SymptomCreate, SymptomUpdate


class SymptomCRUD:
    def __init__(self, db: Session):
        self.db = db

    # -------------------------------
    # CREATE
    # -------------------------------
    def create(self, symptom_in: SymptomCreate) -> Symptom:
        db_symptom = Symptom(
            uuid=uuid4(),
            case_id=symptom_in.case_id,
            symptom=symptom_in.symptom,
            severity=symptom_in.severity,
            duration_hours=symptom_in.duration_hours,
            body_location=symptom_in.body_location
        )
        self.db.add(db_symptom)
        self.db.commit()
        self.db.refresh(db_symptom)
        return db_symptom

    # -------------------------------
    # LIST ALL
    # -------------------------------
    def list(self, skip: int = 0, limit: int = 100) -> List[Symptom]:
        return self.db.query(Symptom).offset(skip).limit(limit).all()

    # -------------------------------
    # GET BY UUID
    # -------------------------------
    def get(self, uuid: UUID) -> Optional[Symptom]:
        return self.db.query(Symptom).filter(Symptom.uuid == uuid).first()

    # -------------------------------
    # GET BY CASE UUID
    # -------------------------------
    def get_by_case(self, case_id: str) -> List[Symptom]:
        return self.db.query(Symptom).filter(Symptom.case_id == case_id).all()

    # -------------------------------
    # UPDATE (PUT/PATCH)
    # -------------------------------
    def update(self, uuid: UUID, symptom_in: SymptomUpdate) -> Optional[Symptom]:
        db_symptom = self.get(uuid)
        if not db_symptom:
            return None
        for field, value in symptom_in.dict(exclude_unset=True).items():
            setattr(db_symptom, field, value)
        self.db.commit()
        self.db.refresh(db_symptom)
        return db_symptom

  
