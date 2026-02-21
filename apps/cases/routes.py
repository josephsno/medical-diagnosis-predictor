from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from database.database import get_db
from utils.utils import CustomPagination
from .schemas import CaseCreate, CaseRead, CaseUpdate
from .crud import CaseCRUD

router = APIRouter()


# -------------------------------
# LIST & CREATE
# -------------------------------
@router.get("/")
def list_cases(
    request: Request,
    pagination: CustomPagination = Depends(),
    db: Session = Depends(get_db)
):
    crud = CaseCRUD(db)
    total = crud.count()  # you'll need to add a count() method to your CRUD
    items = crud.list(skip=pagination.offset, limit=pagination.per_page)
    
    return pagination.paginate(
        data=items,
        total=total,
        request=request,
    )


@router.post("/", response_model=CaseRead)
def create_case(case_in: CaseCreate, db: Session = Depends(get_db)):
    crud = CaseCRUD(db)
    return crud.create(case_in)


# -------------------------------
# RETRIEVE / UPDATE / PATCH / DELETE BY UUID
# -------------------------------
@router.get("/{uuid}", response_model=CaseRead)
def get_case(uuid: UUID, db: Session = Depends(get_db)):
    crud = CaseCRUD(db)
    case = crud.get(uuid)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.put("/{uuid}", response_model=CaseRead)
def update_case(uuid: UUID, case_in: CaseUpdate, db: Session = Depends(get_db)):
    crud = CaseCRUD(db)
    updated = crud.update(uuid, case_in)
    if not updated:
        raise HTTPException(status_code=404, detail="Case not found")
    return updated


@router.patch("/{uuid}", response_model=CaseRead)
def patch_case(uuid: UUID, case_in: CaseUpdate, db: Session = Depends(get_db)):
    crud = CaseCRUD(db)
    updated = crud.update(uuid, case_in)
    if not updated:
        raise HTTPException(status_code=404, detail="Case not found")
    return updated

