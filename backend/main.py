import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import Client, create_client

# 1. Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Load Environment Variables
load_dotenv(os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in backend/.env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 3. Load Machine Learning Artifacts using absolute paths
try:
    model_path = os.path.join(BASE_DIR, "model.pkl")
    iso_path = os.path.join(BASE_DIR, "iso_model.pkl")

    # Check for scalar.pkl vs scaler.pkl
    if os.path.exists(os.path.join(BASE_DIR, "scaler.pkl")):
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    elif os.path.exists(os.path.join(BASE_DIR, "scalar.pkl")):
        scaler_path = os.path.join(BASE_DIR, "scalar.pkl")
    else:
        raise FileNotFoundError(
            "Neither scaler.pkl nor scalar.pkl found in backend folder"
        )

    model = joblib.load(model_path)
    iso_model = joblib.load(iso_path)
    scaler = joblib.load(scaler_path)
    print("SUCCESS: Loaded model.pkl, iso_model.pkl, and scaler.pkl!")
except Exception as e:
    print(f"CRITICAL ERROR loading models: {e}")
    model, iso_model, scaler = None, None, None

app = FastAPI(title="Fraud Detection API Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionSchema(BaseModel):
    amount: float = Field(..., gt=0)
    v_features: list[float] = Field(..., min_items=28, max_items=28)


@app.post("/api/v1/assess-risk")
async def assess_risk(payload: TransactionSchema):
    if scaler is None or model is None or iso_model is None:
        raise HTTPException(
            status_code=500,
            detail="ML models failed to load at server startup. Check terminal logs.",
        )

    try:
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        raw_features = payload.v_features + [payload.amount]

        input_df = pd.DataFrame([raw_features], columns=feature_names)
        input_df["Amount"] = scaler.transform(input_df[["Amount"]])

        prob = float(model.predict_proba(input_df)[0][1])
        is_anomaly = bool(iso_model.predict(input_df)[0] == -1)
        flagged = prob > 0.5 or is_anomaly

        importances = model.feature_importances_
        contribs = dict(
            zip(feature_names, importances * np.abs(input_df.iloc[0].values))
        )
        top_reasons = sorted(contribs.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]
        reasons = [
            f"{feat} impact score: {val:.3f}" for feat, val in top_reasons
        ]

        db_record = {
            "amount": payload.amount,
            "risk_score": round(prob, 4),
            "is_anomaly": is_anomaly,
            "flagged": flagged,
            "reasons": reasons,
        }
        supabase.table("transactions").insert(db_record).execute()

        return {
            "risk_score": round(prob, 4),
            "flagged": flagged,
            "is_anomaly": is_anomaly,
            "reasons": reasons,
        }
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history")
async def get_history():
    response = (
        supabase.table("transactions")
        .select("*")
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )
    return response.data