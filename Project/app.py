"""
app.py  –  FastAPI backend for the Student GPA Predictor
---------------------------------------------------------
Start with:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /            – serves the frontend HTML
    GET  /health      – health check
    POST /predict     – predict GPA from student features
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

# ── Model definition ──────────────────────────────────────────────────────────
class GPARegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.output(x)


# ── Load artifacts ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifacts():
    model_path   = os.path.join(BASE_DIR, "model.pt")
    scaler_path  = os.path.join(BASE_DIR, "scaler.pkl")
    feature_path = os.path.join(BASE_DIR, "feature_columns.pkl")

    missing = [p for p in [model_path, scaler_path, feature_path] if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Missing artifact(s): {missing}. "
            "Run  python train_and_save.py  first."
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GPARegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler   = joblib.load(scaler_path)
    features = joblib.load(feature_path)
    return model, scaler, features, device


model, scaler, FEATURE_COLUMNS, device = load_artifacts()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student GPA Predictor",
    description="MLP-based GPA prediction using AI usage and academic features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    # Numeric fields
    pre_semester_gpa: float = Field(..., ge=0.0, le=4.0, description="GPA before semester (0–4)")
    weekly_genai_hours: float = Field(..., ge=0.0, le=40.0, description="Weekly GenAI usage hours")
    tool_diversity: int = Field(..., ge=1, le=5, description="Number of distinct AI tools used (1–5)")
    paid_subscription: bool = Field(..., description="Has a paid AI subscription?")
    traditional_study_hours: float = Field(..., ge=0.0, le=40.0, description="Weekly traditional study hours")
    perceived_ai_dependency: int = Field(..., ge=1, le=10, description="Self-rated AI dependency (1–10)")
    anxiety_level_during_exams: int = Field(..., ge=1, le=10, description="Exam anxiety level (1–10)")
    skill_retention_score: float = Field(..., ge=0.0, le=100.0, description="Skill retention score (0–100)")

    # Categorical fields (raw values; one-hot encoding happens server-side)
    major_category: Literal["Arts", "Business", "Humanities", "Medical", "STEM"]
    year_of_study: Literal["Freshman", "Graduate", "Junior", "Senior", "Sophomore"]
    primary_use_case: Literal[
        "Copywriting/Drafting",
        "Debugging/Troubleshooting",
        "Direct_Answer_Generation",
        "Ideation",
        "Summarizing_Reading",
    ]
    prompt_engineering_skill: Literal["Advanced", "Beginner", "Intermediate"]
    institutional_policy: Literal[
        "Actively_Encouraged",
        "Allowed_With_Citation",
        "Strict_Ban",
    ]
    burnout_risk_level: Literal["High", "Low", "Medium"]


class PredictResponse(BaseModel):
    predicted_gpa: float
    gpa_letter: str
    confidence_band: dict


# ── Helper: build feature vector ──────────────────────────────────────────────
MAJOR_CATS   = ["Arts", "Business", "Humanities", "Medical", "STEM"]
YEARS        = ["Freshman", "Graduate", "Junior", "Senior", "Sophomore"]
USE_CASES    = [
    "Copywriting/Drafting",
    "Debugging/Troubleshooting",
    "Direct_Answer_Generation",
    "Ideation",
    "Summarizing_Reading",
]
PE_SKILLS    = ["Advanced", "Beginner", "Intermediate"]
POLICIES     = ["Actively_Encouraged", "Allowed_With_Citation", "Strict_Ban"]
BURNOUT_LVLS = ["High", "Low", "Medium"]


def one_hot(value: str, choices: list) -> list:
    return [1.0 if c == value else 0.0 for c in choices]


def build_feature_vector(req: PredictRequest) -> np.ndarray:
    row = [
        req.pre_semester_gpa,
        req.weekly_genai_hours,
        req.tool_diversity,
        1.0 if req.paid_subscription else 0.0,
        req.traditional_study_hours,
        req.perceived_ai_dependency,
        req.anxiety_level_during_exams,
        req.skill_retention_score,
    ]
    row += one_hot(req.major_category,          MAJOR_CATS)
    row += one_hot(req.year_of_study,            YEARS)
    row += one_hot(req.primary_use_case,         USE_CASES)
    row += one_hot(req.prompt_engineering_skill, PE_SKILLS)
    row += one_hot(req.institutional_policy,     POLICIES)
    row += one_hot(req.burnout_risk_level,       BURNOUT_LVLS)

    assert len(row) == 32, f"Feature vector length mismatch: {len(row)} vs 32"
    return np.array(row, dtype=np.float32)


def gpa_to_letter(gpa: float) -> str:
    if gpa >= 3.7: return "A / A+"
    if gpa >= 3.3: return "A-"
    if gpa >= 3.0: return "B+"
    if gpa >= 2.7: return "B"
    if gpa >= 2.3: return "B-"
    if gpa >= 2.0: return "C+"
    if gpa >= 1.7: return "C"
    if gpa >= 1.3: return "C-"
    if gpa >= 1.0: return "D"
    return "F"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_frontend():
    index = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Frontend not found. Place index.html in the frontend/ directory."}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "GPARegressor", "features": len(FEATURE_COLUMNS)}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        features = build_feature_vector(req)
        scaled   = scaler.transform(features.reshape(1, -1))
        tensor   = torch.tensor(scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            raw_pred = model(tensor).item()

        predicted_gpa = float(np.clip(raw_pred, 0.0, 4.0))
        letter        = gpa_to_letter(predicted_gpa)

        # Simple ±0.15 confidence band (clamped to 0–4)
        confidence_band = {
            "lower": round(max(0.0, predicted_gpa - 0.15), 2),
            "upper": round(min(4.0, predicted_gpa + 0.15), 2),
        }

        return PredictResponse(
            predicted_gpa=round(predicted_gpa, 3),
            gpa_letter=letter,
            confidence_band=confidence_band,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
