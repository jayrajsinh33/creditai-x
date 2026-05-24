from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from app.core.database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    prediction_type = Column(String, default="manual")  # manual, batch
    
    # Input data
    applicant_name = Column(String, nullable=True)
    age = Column(Integer)
    income = Column(Float)
    loan_amount = Column(Float)
    credit_score = Column(Integer)
    employment_years = Column(Float)
    late_payments = Column(Integer)
    existing_debt = Column(Float)
    
    # Prediction results
    creditworthiness_score = Column(Float)
    approval_probability = Column(Float)
    risk_level = Column(String)
    default_probability = Column(Float)
    
    # Full data storage
    full_response = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self):
        return {
            "id": self.id,
            "applicant_name": self.applicant_name,
            "creditworthiness_score": self.creditworthiness_score,
            "approval_probability": self.approval_probability,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }