"""
inference.py — loads model artefacts and runs predictions
"""
from __future__ import annotations
import json, time, logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelService:
    """Singleton that holds the loaded model + preprocessor."""

    _instance: "ModelService | None" = None

    def __init__(self):
        self.model       = None
        self.preprocessor = None
        self.meta        = {}
        self.report      = {}
        self.loaded      = False
        self._load()

    @classmethod
    def get(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Load ─────────────────────────────────────────────────────────────
    def _load(self):
        try:
            self.model        = joblib.load(MODEL_DIR / "best_model.joblib")
            self.preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
            with open(MODEL_DIR / "feature_meta.json") as f:
                self.meta = json.load(f)
            with open(MODEL_DIR / "evaluation_report.json") as f:
                self.report = json.load(f)
            self.loaded = True
            logger.info(f"Model loaded: {self.report.get('best_model')}  "
                        f"accuracy={self.report.get('best_accuracy'):.4f}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise

    @property
    def version(self) -> str:
        return self.report.get("best_model", "unknown")

    @property
    def accuracy(self) -> float:
        return float(self.report.get("best_accuracy", 0.0))

    # ── Feature engineering (mirrors train_91.py) ────────────────────────
    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # fill optionals before engineering
        for col in ["avg_quiz_score", "assignment_submission_pct",
                    "chat_messages", "polls_answered"]:
            if col in d.columns:
                d[col] = d[col].fillna(d[col].median() if len(d) > 1 else 0)

        d["engagement_proxy"]    = (d["attendance_pct"] * d["camera_on_pct"]) / 100
        d["tech_penalty"]        = d["network_drops"] * d["avg_latency_ms"] / 1000
        d["quiz_efficiency"]     = d["quiz_completion"] * d["avg_quiz_score"] / 100
        d["participation_index"] = (d["chat_messages"].fillna(0) +
                                    d["polls_answered"].fillna(0) +
                                    d["hand_raises"]) / (d["class_size"] + 1)
        d["cam_x_att"]           = d["camera_on_pct"] * d["attendance_pct"] / 100
        d["quiz_x_att"]          = d["quiz_completion"] * d["attendance_pct"] / 100
        d["is_weekend"]          = (d["day_of_week"] >= 5).astype(int)
        d["is_morning"]          = ((d["session_hour"] >= 9) & (d["session_hour"] < 12)).astype(int)
        d["is_evening"]          = (d["session_hour"] >= 18).astype(int)
        d["low_latency"]         = (d["avg_latency_ms"] < 80).astype(int)
        d["experienced_teacher"] = (d["teacher_experience_years"] >= 10).astype(int)
        d["high_quiz"]           = (d["quiz_completion"] > 75).astype(int)
        d["high_attendance"]     = (d["attendance_pct"] > 80).astype(int)
        d["time_slot"]           = pd.cut(d["session_hour"],
                                          bins=[0, 9, 12, 17, 20, 24],
                                          labels=["early","morning","afternoon","evening","night"],
                                          right=False).astype(str)
        return d

    # ── Recommendations engine ────────────────────────────────────────────
    @staticmethod
    def _recommendations(row: Dict[str, Any], label: str) -> list[str]:
        recs = []
        if row.get("attendance_pct", 100) < 70:
            recs.append("📢 Attendance is low — send session reminders before class starts.")
        if row.get("camera_on_pct", 100) < 40:
            recs.append("🎥 Camera participation is low — encourage students to turn on video.")
        if row.get("quiz_completion", 100) < 60:
            recs.append("📝 Many students skipped the quiz — consider shorter, ungraded knowledge checks.")
        if row.get("avg_quiz_score", 100) is not None and row.get("avg_quiz_score", 100) < 55:
            recs.append("📚 Quiz scores are below average — revisit key concepts in next session.")
        if row.get("network_drops", 0) > 3:
            recs.append("🌐 High network drops detected — advise students to switch to a stable connection.")
        if row.get("avg_latency_ms", 0) > 200:
            recs.append("⚡ High latency observed — consider reducing video quality or switching platform.")
        if row.get("chat_messages") is not None and row.get("chat_messages", 100) < 5:
            recs.append("💬 Low chat activity — use icebreakers or Q&A prompts to drive interaction.")
        if label == "GoodClass" and not recs:
            recs.append("✅ Class is performing well — keep up the engagement strategies!")
        if not recs:
            recs.append("⚠️ Multiple engagement signals are weak — consider a mid-class check-in.")
        return recs[:4]  # cap to 4 recommendations

    # ── Predict single ────────────────────────────────────────────────────
    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()

        df = pd.DataFrame([data])
        df = self._engineer(df)

        expected_cols = self.meta["all_input_cols"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        X = self.preprocessor.transform(df)
        pred  = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]

        label_map = {"0": "NeedsImprovement", "1": "GoodClass"}
        label     = label_map[str(pred)]
        prob_good = float(proba[1])
        prob_need = float(proba[0])
        confidence = float(max(proba))

        risk_flag  = prob_good < 0.45 or data.get("network_drops", 0) > 5
        risk_level = "HIGH" if prob_good < 0.35 else ("MEDIUM" if prob_good < 0.55 else "LOW")

        eng_signals = {
            "engagement_proxy":    round(float(df["engagement_proxy"].iloc[0]), 2),
            "participation_index": round(float(df["participation_index"].iloc[0]), 2),
            "tech_penalty":        round(float(df["tech_penalty"].iloc[0]), 2),
            "quiz_efficiency":     round(float(df["quiz_efficiency"].iloc[0]), 2),
        }

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "prediction":        pred,
            "label":             label,
            "confidence":        round(confidence, 4),
            "probability_good":  round(prob_good, 4),
            "probability_needs": round(prob_need, 4),
            "risk_flag":         risk_flag,
            "risk_level":        risk_level,
            "engagement_signals": eng_signals,
            "recommendations":   self._recommendations(data, label),
            "model_version":     self.version,
            "latency_ms":        round(latency_ms, 3),
        }

    def predict_batch(self, records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        return [self.predict_single(r) for r in records]
