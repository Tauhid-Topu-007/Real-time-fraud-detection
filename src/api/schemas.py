from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import numpy as np

class TransactionCreate(BaseModel):
    step: int = Field(..., description="Time step (hour)")
    type: str = Field(..., description="Transaction type")
    amount: float = Field(..., gt=0, description="Transaction amount")
    nameOrig: str = Field(..., min_length=1, description="Origin account")
    oldbalanceOrg: float = Field(..., ge=0, description="Origin balance before")
    newbalanceOrig: float = Field(..., ge=0, description="Origin balance after")
    nameDest: str = Field(..., min_length=1, description="Destination account")
    oldbalanceDest: float = Field(..., ge=0, description="Destination balance before")
    newbalanceDest: float = Field(..., ge=0, description="Destination balance after")
    isFlaggedFraud: Optional[int] = Field(0, ge=0, le=1)
    transaction_id: Optional[str] = None
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        if v not in valid_types:
            raise ValueError(f"Invalid transaction type. Must be one of {valid_types}")
        return v
    
    @validator('transaction_id', pre=True, always=True)
    def generate_transaction_id(cls, v):
        if v is None:
            return f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
        return v

class FraudPrediction(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: str
    decision: str
    timestamp: datetime
    model_version: str
    features_used: int
    top_features: Optional[List[dict]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_20240101120000_1234",
                "fraud_probability": 0.94,
                "risk_level": "HIGH",
                "decision": "BLOCK",
                "timestamp": "2024-01-01T12:00:00",
                "model_version": "v1.0.0",
                "features_used": 45
            }
        }