"""
main.py — FastAPI application entry point
"""
import time, logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.utils.logger import setup_logging
from app.services.inference import ModelService

setup_logging()
logger = logging.getLogger(__name__)

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model …")
    ModelService.get()          # eager load
    app.state.start_time = time.time()
    logger.info("Model ready. API is live.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Live Class Quality Monitor API",
    description=(
        "Production-ready ML API that predicts live-class quality and "
        "engagement in real time. Model accuracy ≥ 91 %."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(
        round((time.perf_counter() - t0) * 1000, 2)
    )
    return response


app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "Live Class Quality Monitor",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }
