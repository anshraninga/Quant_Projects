"""
CommodityKing — FastAPI application entry point.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from rag.vector_store import init_historical_events

# Force UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CommodityKing",
    description=(
        "Quantitative commodity analysis using geopolitical intelligence, "
        "weather pattern analysis, supply/demand fundamentals, and LangGraph "
        "multi-agent AI with HMM, Kalman filter, and Bayesian surprise scoring."
    ),
    version="0.9.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup() -> None:
    logger.info("[startup] Initialising RAG knowledge base...")
    n = init_historical_events()
    logger.info("[startup] Historical events loaded: %d", n)
    logger.info("[startup] CommodityKing ready.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort error handler — returns 200 with error in body rather than
    a bare 500, so clients always get structured JSON.
    """
    logger.error("[app] Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=200,
        content={"error": str(exc), "path": str(request.url.path)},
    )
