from database.base_model import BaseORMModel
from sqlalchemy import Column, String, Integer, Float, ForeignKey

class Symptom(BaseORMModel):
    __tablename__ = "symptoms"

    case_id = Column(String, ForeignKey("cases.case_id"))
    symptom = Column(String, nullable=True)
    severity = Column(Integer, nullable=False)
    duration_hours = Column(Float, nullable=False)
    body_location = Column(String)
