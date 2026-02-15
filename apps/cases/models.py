from database.base_model import BaseORMModel
from sqlalchemy import Column, String, Integer, Float, DateTime

class Case(BaseORMModel):
    __tablename__ = "cases"

    case_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    diagnosis = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    
