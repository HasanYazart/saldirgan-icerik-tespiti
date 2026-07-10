from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    warning_count = Column(Integer, default=0)
    is_banned = Column(Boolean, default=False)

class MessageLog(Base):
    __tablename__ = "message_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    original_text = Column(String)
    censored_text = Column(String)
    is_toxic = Column(Boolean, default=False)
    toxicity_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
