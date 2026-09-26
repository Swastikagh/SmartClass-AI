"""
schemas.py — Pydantic request / response models
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, model_validator


class ClassInput(BaseModel):
    """Input features for a single live-class session."""

    # ── Class metadata ──────────────────────────────────────────────────────
    class_size:                int   = Field(..., ge=1, le=500,  description="Number of students enrolled")
    duration_min:              int   = Field(..., ge=5, le=300,  description="Class duration in minutes")
    session_hour:              int   = Field(..., ge=0, le=23,   description="Hour of day (24h)")
    day_of_week:               int   = Field(..., ge=0, le=6,    description="0=Monday … 6=Sunday")
    subject:                   str   = Field(...,                description="Subject name")
    grade_level:               int   = Field(..., ge=1, le=12,   description="Grade 1–12")
    teacher_experience_years:  int   = Field(..., ge=0, le=50,   description="Teacher experience in years")

    # ── Engagement ──────────────────────────────────────────────────────────
    attendance_pct:            float = Field(..., ge=0, le=100)
    chat_messages:             Optional[float] = Field(None, ge=0)
    polls_answered:            Optional[float] = Field(None, ge=0)
    hand_raises:               int   = Field(..., ge=0)
    screen_shares:             int   = Field(0,   ge=0)
    avg_response_sec:          float = Field(..., ge=0)
    camera_on_pct:             float = Field(..., ge=0, le=100)
    mic_on_pct:                float = Field(..., ge=0, le=100)

    # ── Technical ───────────────────────────────────────────────────────────
    network_drops:             int   = Field(0,   ge=0)
    avg_latency_ms:            float = Field(..., ge=0)
    reconnects:                int   = Field(0,   ge=0)
    platform:                  str   = Field("Zoom")

    # ── Assessment ──────────────────────────────────────────────────────────
    quiz_completion:           float = Field(..., ge=0, le=100)
    avg_quiz_score:            Optional[float] = Field(None, ge=0, le=100)
    assignment_submission_pct: Optional[float] = Field(None, ge=0, le=100)

    # ── Historical ──────────────────────────────────────────────────────────
    prev_class_score:          float = Field(..., ge=0, le=100)
    streak_good_sessions:      int   = Field(0,   ge=0)

    model_config = {"json_schema_extra": {"example": {
        "class_size": 35, "duration_min": 60, "session_hour": 10,
        "day_of_week": 1, "subject": "Math", "grade_level": 9,
        "teacher_experience_years": 8, "attendance_pct": 82.5,
        "chat_messages": 28, "polls_answered": 4, "hand_raises": 9,
        "screen_shares": 2, "avg_response_sec": 4.2,
        "camera_on_pct": 65.0, "mic_on_pct": 30.0,
        "network_drops": 1, "avg_latency_ms": 65.0, "reconnects": 0,
        "platform": "Zoom", "quiz_completion": 78.0, "avg_quiz_score": 72.5,
        "assignment_submission_pct": 80.0, "prev_class_score": 73.0,
        "streak_good_sessions": 3,
    }}}


class PredictionResponse(BaseModel):
    prediction:         int
    label:              str
    confidence:         float
    probability_good:   float
    probability_needs:  float
    risk_flag:          bool
    risk_level:         str
    engagement_signals: Dict[str, Any]
    recommendations:    List[str]
    model_version:      str
    latency_ms:         float


class BatchInput(BaseModel):
    sessions: List[ClassInput]


class BatchPredictionResponse(BaseModel):
    total:       int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    accuracy:      float
    uptime_sec:    float
