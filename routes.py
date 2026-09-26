"""
routes.py — all API endpoints
"""
import time, logging
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
import pandas as pd, io

from app.schemas import (ClassInput, PredictionResponse, BatchInput,
                          BatchPredictionResponse, HealthResponse)
from app.services.inference import ModelService
from app.utils.logger import log_prediction, get_recent_predictions, get_stats

router = APIRouter()
logger = logging.getLogger(__name__)

_START = time.time()


# ── Health ────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health():
    svc = ModelService.get()
    return HealthResponse(
        status       = "ok" if svc.loaded else "degraded",
        model_loaded = svc.loaded,
        model_version= svc.version,
        accuracy     = svc.accuracy,
        uptime_sec   = round(time.time() - _START, 1),
    )


# ── Single prediction ─────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(payload: ClassInput, background_tasks: BackgroundTasks):
    svc  = ModelService.get()
    data = payload.model_dump()
    try:
        result = svc.predict_single(data)
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(log_prediction, data, result)
    return result


# ── Batch prediction ──────────────────────────────────────────────────────
@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
async def predict_batch(payload: BatchInput, background_tasks: BackgroundTasks):
    svc = ModelService.get()
    results = []
    for session in payload.sessions:
        data = session.model_dump()
        try:
            r = svc.predict_single(data)
        except Exception as e:
            logger.warning(f"Batch item failed: {e}")
            continue
        background_tasks.add_task(log_prediction, data, r)
        results.append(r)

    return BatchPredictionResponse(total=len(results), predictions=results)


# ── CSV upload prediction ─────────────────────────────────────────────────
@router.post("/predict/csv", tags=["inference"])
async def predict_csv(file: UploadFile = File(...)):
    """Upload a CSV file and get predictions for all rows."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    svc     = ModelService.get()
    records = df.to_dict(orient="records")
    results = []
    errors  = []

    for i, row in enumerate(records):
        try:
            r = svc.predict_single(row)
            results.append({"row": i + 1, **r})
        except Exception as e:
            errors.append({"row": i + 1, "error": str(e)})

    return {
        "total_rows": len(records),
        "successful": len(results),
        "failed":     len(errors),
        "predictions": results,
        "errors":      errors,
    }


# ── Monitoring / Logs ─────────────────────────────────────────────────────
@router.get("/logs", tags=["monitoring"])
async def recent_logs(n: int = 50):
    return {"logs": get_recent_predictions(n)}


@router.get("/stats", tags=["monitoring"])
async def stats():
    return get_stats()


# ── Model info ────────────────────────────────────────────────────────────
@router.get("/model/info", tags=["model"])
async def model_info():
    svc = ModelService.get()
    return {
        "model":          svc.version,
        "accuracy":       svc.accuracy,
        "target":         "binary: GoodClass / NeedsImprovement",
        "label_map":      svc.meta.get("label_map"),
        "feature_count":  svc.report.get("feature_count"),
        "trained_at":     svc.report.get("trained_at"),
        "all_model_results": svc.report.get("all_results"),
    }
