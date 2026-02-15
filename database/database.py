import os
from dotenv import dotenv_values
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# -------------------------------
# Load .env directly, bypassing shell environment
# -------------------------------
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env")
config = dotenv_values(dotenv_path)  # returns a dict of key/value pairs

DATABASE_URL = config.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(f"DATABASE_URL not found in {dotenv_path}")


# -------------------------------
# SQLAlchemy setup
# -------------------------------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# -------------------------------
# Dependency for FastAPI endpoints
# -------------------------------
def get_db():
    """
    Dependency to inject SQLAlchemy session into FastAPI endpoints
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
