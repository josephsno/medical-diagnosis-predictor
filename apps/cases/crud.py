from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID, uuid4

from .models import Case
from .schemas import CaseCreate, CaseUpdate


class CaseCRUD:
    def __init__(self, db: Session):
        self.db = db

    # -------------------------------
    # CREATE
    # -------------------------------
    def create(self, case_in: CaseCreate) -> Case:
        db_case = Case(
            uuid=uuid4(),
            case_id=case_in.case_id,
            user_id=case_in.user_id,
            age=case_in.age,
            gender=case_in.gender,
            diagnosis=case_in.diagnosis,
            confidence=case_in.confidence
        )
        self.db.add(db_case)
        self.db.commit()
        self.db.refresh(db_case)
        return db_case

    # -------------------------------
    # LIST ALL
    # -------------------------------
    def list(self, skip: int = 0, limit: int = 100) -> List[Case]:
        return self.db.query(Case).offset(skip).limit(limit).all()

    # -------------------------------
    # GET BY UUID
    # -------------------------------
    def get(self, uuid: UUID) -> Optional[Case]:
        return self.db.query(Case).filter(Case.uuid == uuid).first()

    # -------------------------------
    # UPDATE (PUT/PATCH)
    # -------------------------------
    def update(self, uuid: UUID, case_in: CaseUpdate) -> Optional[Case]:
        db_case = self.get(uuid)
        if not db_case:
            return None
        for field, value in case_in.dict(exclude_unset=True).items():
            setattr(db_case, field, value)
        self.db.commit()
        self.db.refresh(db_case)
        return db_case
