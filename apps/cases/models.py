from database.base_model import BaseORMModel
from sqlalchemy import Column, String, Integer, Float

class Case(BaseORMModel):
    __tablename__ = "cases"

    user_id = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    diagnosis = Column(String, nullable=True)
    confidence = Column(Float, nullable=False)

    