"""
logger.py — structured JSON logging + prediction log store
"""
import json, logging, time
from collections import deque
from datetime import datetime, timezone
from pythonjsonlogger import jsonlogger
from typing import Any, Dict

# ── Prediction log ring buffer (last 1000 entries, in-memory) ─────────────
_prediction_log: deque = deque(maxlen=1000)


def setup_logging():
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def log_prediction(input_data: Dict[str, Any], result: Dict[str, Any]):
    entry = {
        "ts":         datetime.now(timezone.utc).isoformat(),
        "prediction": result.get("prediction"),
        "label":      result.get("label"),
        "confidence": result.get("confidence"),
        "risk_level": result.get("risk_level"),
        "latency_ms": result.get("latency_ms"),
        "input_summary": {
            "class_size":     input_data.get("class_size"),
            "attendance_pct": input_data.get("attendance_pct"),
            "quiz_completion": input_data.get("quiz_completion"),
            "network_drops":  input_data.get("network_drops"),
        },
    }
    _prediction_log.append(entry)
    logging.getLogger("prediction").info("prediction", extra=entry)


def get_recent_predictions(n: int = 50) -> list:
    log_list = list(_prediction_log)
    return log_list[-n:]


def get_stats() -> Dict[str, Any]:
    log_list = list(_prediction_log)
    if not log_list:
        return {"total": 0}
    preds = [e["prediction"] for e in log_list]
    lats  = [e["latency_ms"] for e in log_list if e.get("latency_ms")]
    return {
        "total":            len(log_list),
        "good_class_count": sum(p == 1 for p in preds),
        "needs_impr_count": sum(p == 0 for p in preds),
        "good_class_pct":   round(sum(p==1 for p in preds)/len(preds)*100, 2),
        "avg_latency_ms":   round(sum(lats)/len(lats), 2) if lats else 0,
        "max_latency_ms":   round(max(lats), 2) if lats else 0,
    }
