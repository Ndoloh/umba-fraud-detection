import pandas as pd
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Fraud Detection API",
    version="1.0"
)

# Load the trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "final_model.pkl")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }

ALARM_THRESHOLD = 0.7201


# ---- Required fields validated strictly; everything else passes through ----
class Transaction(BaseModel):
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float

    class Config:
        extra = "allow"  # allows all your other engineered columns through untyped


@app.post("/predict")
def predict(transactions: List[Transaction]):
    if len(transactions) == 0:
        raise HTTPException(status_code=422, detail="No transactions provided")

    df = pd.DataFrame([t.dict() for t in transactions])

    transaction_ids = df["TransactionID"].tolist()
    df = df.drop(columns=["TransactionID"])

    try:
        probabilities = model.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    results = []
    for transaction_id, prob in zip(transaction_ids, probabilities):
        results.append({
            "TransactionID": transaction_id,
            "isFraud_prob": float(prob),
            "alarm": bool(prob >= ALARM_THRESHOLD)
        })

    return {
        "threshold": ALARM_THRESHOLD,
        "predictions": results
    }