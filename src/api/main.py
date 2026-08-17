from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.inference.predictor import FraudPredictor
from src.api.schemas import TransactionCreate, FraudPrediction

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Predict fraud probability for bank transactions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = FraudPredictor()

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "feature_engineer_loaded": predictor.feature_engineer is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=FraudPrediction)
async def predict(transaction: TransactionCreate, background_tasks: BackgroundTasks):
    try:
        txn_dict = transaction.dict()
        result = predictor.predict_single(txn_dict)
        background_tasks.add_task(predictor.log_prediction, txn_dict, result)
        return FraudPrediction(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    try:
        return predictor.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh")
async def refresh_model():
    try:
        predictor.load_models()
        return {"status": "success", "message": "Model refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000)