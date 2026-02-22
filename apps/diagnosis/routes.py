import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID
from database.database import get_db
from apps.cases.models import Case
from apps.symptoms.models import Symptom
from training_model import MedicalDiagnosisCleaner

router = APIRouter()

# Load model once at startup (not on every request)
MODEL_PATH = "./best_diagnosis_model.joblib"
loaded_model_data = joblib.load(MODEL_PATH)

SAVED_MODEL = loaded_model_data["model"]
SAVED_SCALER = loaded_model_data["scaler"]
SAVED_LABEL_ENCODER = loaded_model_data["label_encoder"]
SAVED_FEATURE_NAMES = loaded_model_data[
    "feature_names"
]  # critical — must match exactly


@router.get("/cases/{case_uuid}/predict")
async def predict_diagnosis(case_uuid: UUID, db: Session = Depends(get_db)):

    # ── 1. Fetch case ──────────────────────────────────────────────────────────
    case = db.query(Case).filter(Case.uuid == case_uuid).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_uuid} not found")

    symptoms = db.query(Symptom).filter(Symptom.case_id == case.id).all()
    if not symptoms:
        raise HTTPException(
            status_code=404, detail=f"No symptoms found for case {case_uuid}"
        )

    # ── 2. Build DataFrames matching what the cleaner expects ──────────────────
    df_cases = pd.DataFrame(
        [
            {
                "case_id": str(case.id),
                "user_id": str(case.user_id),
                "age": case.age,
                "gender": (
                    "M" if case.gender.upper() in ["M", "MALE"] else "F"
                ),  # ← normalize here
                "timestamp": case.created_at,  # adjust to your actual timestamp field name
                "diagnosis": case.diagnosis
                or "unknown",  # cleaner needs this col to exist
                "confidence": case.confidence,
            }
        ]
    )

    df_symptoms = pd.DataFrame(
        [
            {
                "case_id": str(case.id),
                "symptom": s.symptom,
                "severity": s.severity,
                "duration_hours": s.duration_hours,
                "body_location": s.body_location,
            }
            for s in symptoms
        ]
    )

    # ── 3. Run through the cleaning pipeline ───────────────────────────────────
    cleaner = MedicalDiagnosisCleaner()
    try:
        df_prepared = cleaner.clean_and_prepare(df_cases, df_symptoms)
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Cleaning pipeline failed: {str(e)}"
        )

    # ── 4. Get features (X), drop diagnosis column ─────────────────────────────
    X, _ = cleaner.get_X_y()

    # ── 5. Align columns to exactly match training features ────────────────────
    # Add any missing columns (symptoms not seen in this case) as 0
    # Drop any extra columns that weren't in training
    X = X.reindex(columns=SAVED_FEATURE_NAMES, fill_value=0)

    # ── 6. Scale + predict ─────────────────────────────────────────────────────
    X_scaled = SAVED_SCALER.transform(X)
    pred_encoded = SAVED_MODEL.predict(X_scaled)
    diagnosis = SAVED_LABEL_ENCODER.inverse_transform(pred_encoded)[0]

    # Probabilities (if model supports it)
    top_predictions = None
    if hasattr(SAVED_MODEL, "predict_proba"):
        proba = SAVED_MODEL.predict_proba(X_scaled)[0]
        top_3_idx = np.argsort(proba)[::-1][:3]
        top_predictions = [
            {
                "diagnosis": SAVED_LABEL_ENCODER.classes_[i],
                "confidence": round(float(proba[i]), 4),
            }
            for i in top_3_idx
        ]
    case.diagnosis = diagnosis
    db.commit()

    return {
        "case_uuid": case_uuid,
        "diagnosis": diagnosis,
        "top_predictions": top_predictions,
        "symptoms_used": len(symptoms),
        "model_used": loaded_model_data.get("model_name"),
    }
