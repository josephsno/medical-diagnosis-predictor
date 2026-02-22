from database.base_model import BaseORMModel
from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.orm import relationship

class Case(BaseORMModel):
    __tablename__ = "cases"

    user_id = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    diagnosis = Column(String, nullable=True)
    confidence = Column(Float, nullable=False)
    symptoms = relationship("Symptom", back_populates="case", lazy="selectin")


    