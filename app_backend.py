

import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from xgboost import XGBClassifier

# Optional Gemini LLM import (gracefully falls back if key is not configured)
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

app = FastAPI(title="Micron ADTS - Live XGBoost & AI Drift Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Data & Metadata
df_stream = pd.read_csv("scored_production_drift_stream.csv")
df_labeled = pd.read_csv("clean_labeled_wafers.csv")
df_unlabeled_full = pd.read_csv("clean_unlabeled_stream.csv")

with open("golden_baseline_metadata.json", "r") as f:
    metadata = json.load(f)

TOP_10_SENSORS = metadata["top_10_sensors"]
TOP_5_SENSORS = TOP_10_SENSORS[:5]
TOP_30_SENSORS = metadata.get("top_30_sensors", TOP_10_SENSORS)
OPTIMAL_THRESH = metadata["optimal_threshold"]

# 2. Load the Real Trained XGBoost Models from Disk
teacher_xgb = XGBClassifier()
teacher_xgb.load_model("xgb_teacher_model.json")

student_xgb = XGBClassifier()
student_xgb.load_model("xgb_student_model.json")

z_cols = [f"{s}_Z_Score" for s in TOP_5_SENSORS]
df_stream["Multivariate_Drift_Index"] = np.sqrt((df_stream[z_cols] ** 2).mean(axis=1)).round(3)


class LiveWaferPayload(BaseModel):
    base_wafer_id: str = "Wafer-100"
    sensor_overrides: Dict[str, float] = {}


@app.get("/api/overview")
def get_overview():
    total_wafers = len(df_stream)
    excursion_count = int((df_stream["Predicted_Status"] == "Excursion_Alert").sum())
    severe_drift_count = int((df_stream["Multivariate_Drift_Index"] > 3.0).sum())

    batch_summary = (
        df_stream.groupby("Batch_File")
        .agg(
            wafer_count=("Wafer", "count"),
            avg_defect_prob=("Defect_Probability", "mean"),
            avg_drift_index=("Multivariate_Drift_Index", "mean"),
            timestamp=("Batch_Timestamp", "first"),
        )
        .reset_index()
        .sort_values("timestamp")
    )

    return {
        "kpis": {
            "total_production_wafers": total_wafers,
            "excursion_alerts": excursion_count,
            "excursion_rate_pct": round((excursion_count / total_wafers) * 100, 1),
            "severe_3sigma_drift_wafers": severe_drift_count,
            "active_monitored_sensors": 257,
            "model_roc_auc": 0.7441,
            "kaggle_baseline_roc_auc": 0.6190,
            "live_model_loaded": True,
        },
        "top_sensors": [
            {
                "sensor": s,
                "importance": round(metadata["feature_importance"][s], 4),
                "stage_type": "Downstream Probe/Cell Metric" if int(s.split("-")[1]) > 500 else "Upstream Inline Process",
                "golden_mean": round(metadata["baseline_limits"][s]["golden_mean"], 3),
                "ucl_3sigma": round(metadata["baseline_limits"][s]["ucl_3sigma"], 3),
                "lcl_3sigma": round(metadata["baseline_limits"][s]["lcl_3sigma"], 3),
            }
            for s in TOP_10_SENSORS
        ],
        "batches": batch_summary.replace({np.nan: None}).to_dict(orient="records"),
    }


@app.get("/api/spc-chart/{sensor_id}")
def get_spc_chart(
    sensor_id: str,
    limit: int = Query(200),
    batch_file: str = Query(None),
):
    if sensor_id not in TOP_10_SENSORS:
        return {"error": f"{sensor_id} is not in top 10 monitored sensors."}

    sub_df = df_stream.copy()
    if batch_file and batch_file != "ALL":
        sub_df = sub_df[sub_df["Batch_File"] == batch_file]

    sub_df = sub_df.head(limit).copy()
    ewma_col = f"{sensor_id}_EWMA"
    if ewma_col not in sub_df.columns:
        sub_df[ewma_col] = sub_df[sensor_id].ewm(alpha=0.2, adjust=False).mean()

    limits = metadata["baseline_limits"][sensor_id]

    records = []
    for _, row in sub_df.iterrows():
        val = float(row[sensor_id])
        ewma_val = float(row[ewma_col])
        is_breach = val > limits["ucl_3sigma"] or val < limits["lcl_3sigma"]
        records.append({
            "wafer": row["Wafer"],
            "batch": row["Batch_File"],
            "raw_value": round(val, 4),
            "ewma_value": round(ewma_val, 4),
            "defect_prob": round(float(row["Defect_Probability"]), 3),
            "is_3sigma_breach": bool(is_breach),
        })

    return {"sensor": sensor_id, "limits": limits, "series": records}


@app.get("/api/inline-probe-correlation")
def get_inline_probe_correlation():
    corr_matrix = df_labeled[TOP_10_SENSORS].corr().round(3)
    pairs = []
    for i, s1 in enumerate(TOP_10_SENSORS):
        for j, s2 in enumerate(TOP_10_SENSORS):
            if i < j:
                pairs.append({
                    "sensor_a": s1,
                    "sensor_b": s2,
                    "correlation": float(corr_matrix.loc[s1, s2]),
                })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return {"matrix": corr_matrix.to_dict(), "strongest_pairs": pairs[:10]}


@app.post("/api/predict-live")
def predict_live_wafer(payload: LiveWaferPayload):
    """
    Runs real-time inference using the loaded XGBoost Teacher-Student binary models
    when an engineer adjusts inline sensor sliders on the dashboard.
    """
    base_row = df_unlabeled_full[df_unlabeled_full["Wafer"] == payload.base_wafer_id]
    if base_row.empty:
        base_row = df_unlabeled_full.iloc[[0]]

    input_df = base_row[TOP_30_SENSORS].copy()
    for sensor, new_val in payload.sensor_overrides.items():
        if sensor in input_df.columns:
            input_df[sensor] = float(new_val)

    # Real-time XGBoost Ensemble Inference
    prob_teacher = float(teacher_xgb.predict_proba(input_df)[:, 1][0])
    prob_student = float(student_xgb.predict_proba(input_df)[:, 1][0])
    ensemble_prob = round(0.65 * prob_teacher + 0.35 * prob_student, 4)

    # Real-time Multivariate Drift Index (RMS Z-Score)
    z_scores = {}
    for s in TOP_5_SENSORS:
        g_mean = metadata["baseline_limits"][s]["golden_mean"]
        g_std = metadata["baseline_limits"][s]["golden_std"]
        z_scores[s] = round(float((input_df[s].iloc[0] - g_mean) / g_std), 2)

    live_mdi = round(float(np.sqrt(np.mean([z ** 2 for z in z_scores.values()]))), 3)

    return {
        "ensemble_defect_probability": ensemble_prob,
        "teacher_prob": round(prob_teacher, 4),
        "student_prob": round(prob_student, 4),
        "predicted_status": "Excursion_Alert" if ensemble_prob >= OPTIMAL_THRESH else "In_Control",
        "live_multivariate_drift_index": live_mdi,
        "top_5_z_scores": z_scores,
    }


@app.get("/api/ai-copilot-diagnosis/{wafer_id}")
def get_wafer_diagnosis(wafer_id: str):
    """
    Calls the Gemini 2.5 Flash LLM to generate a physics-informed semiconductor root-cause
    report from the wafer's live XGBoost probability and Z-score telemetry.
    """
    row_df = df_stream[df_stream["Wafer"] == wafer_id]
    if row_df.empty:
        return {"error": "Wafer ID not found"}

    row = row_df.iloc[0]
    deviations = []
    for s in TOP_5_SENSORS:
        z_val = float(row[f"{s}_Z_Score"])
        deviations.append({
            "sensor": s,
            "raw_value": round(float(row[s]), 3),
            "golden_mean": round(metadata["baseline_limits"][s]["golden_mean"], 3),
            "z_score": round(z_val, 2),
            "status": "CRITICAL_DRIFT" if abs(z_val) > 3.0 else ("WARNING_DRIFT" if abs(z_val) > 2.0 else "NORMAL"),
        })

    deviations.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    primary_culprit = deviations[0]

    # Default deterministic fallback summary
    narrative = (
        f"Wafer {wafer_id} (Batch {row['Batch_File']}) has a live XGBoost excursion probability of "
        f"{float(row['Defect_Probability'])*100:.1f}% and a Multivariate Drift Index of {row['Multivariate_Drift_Index']}σ. "
        f"Primary root-cause indicator is {primary_culprit['sensor']} shifting {primary_culprit['z_score']}σ "
        f"from the Golden Wafer baseline ({primary_culprit['raw_value']} vs target {primary_culprit['golden_mean']}). "
        f"Recommended Action: Inspect upstream inline chamber calibration at Stage {primary_culprit['sensor'].split('-')[1]} "
        f"before downstream probe metrics (Sensor-572) breach specification."
    )
    ai_source = "Deterministic Rule Engine (Set GEMINI_API_KEY for Live LLM)"

    # If GEMINI_API_KEY is present, call Gemini 2.5 Flash for real AI analysis
    api_key = os.environ.get("GEMINI_API_KEY")
    if GENAI_AVAILABLE and api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                f"You are a Senior Semiconductor Yield & Process Integration Engineer at Micron Technology. "
                f"Write a concise, 3-sentence root-cause diagnosis and corrective action for Wafer {wafer_id} "
                f"(Batch {row['Batch_File']}).\n"
                f"- Live XGBoost Ensemble Excursion Probability: {float(row['Defect_Probability'])*100:.1f}%\n"
                f"- Multivariate Drift Index (RMS Z-Score): {row['Multivariate_Drift_Index']} sigma\n"
                f"- Top 5 Root-Cause Sensor Deviations vs Golden Baseline: {json.dumps(deviations)}\n"
                f"Note that Sensors 1-500 are upstream inline chamber parameters (etch/deposition/litho) and "
                f"Sensors 501-590 (like Sensor-548 and Sensor-572) are downstream electrical probe/cell metrics. "
                f"Be specific, technical, and actionable."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if response and response.text:
                narrative = response.text.strip()
                ai_source = "Gemini 2.5 Flash Live LLM"
        except Exception as e:
            ai_source = f"Rule Engine Fallback (LLM Error: {str(e)[:40]})"

    return {
        "wafer": wafer_id,
        "batch": row["Batch_File"],
        "defect_probability": round(float(row["Defect_Probability"]), 3),
        "multivariate_drift_index": float(row["Multivariate_Drift_Index"]),
        "sensor_deviations": deviations,
        "ai_engineering_summary": narrative,
        "ai_engine_source": ai_source,
    }