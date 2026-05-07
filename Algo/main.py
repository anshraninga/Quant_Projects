"""
main.py
-------
FastAPI application entry point for Alpha-Seeker.

Start:  uvicorn main:app --reload
Docker: docker build -t alpha-seeker . && docker run -p 8000:8000 alpha-seeker
"""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config  # noqa: F401 — ensures ROOT in sys.path before any existing_modules import
from alpha_seeker.memory import trade_memory
from alpha_seeker.rag.ingester import refresh_rag
from alpha_seeker.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────
    logger.info("Alpha-Seeker starting up…")
    trade_memory.init_db()

    # Initial RAG refresh in background so startup is fast
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, refresh_rag, None)
    logger.info("RAG background refresh scheduled")

    yield

    # ── Shutdown ─────────────────────────────────────────
    logger.info("Alpha-Seeker shutting down.")


app = FastAPI(
    title="Alpha-Seeker",
    description="Multi-agent crypto research system — LangGraph + RAG + memory",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "Alpha-Seeker",
        "docs":    "/docs",
        "health":  "/health",
        "analyse": "POST /analyse/{symbol}",
        "report":  "GET  /report/{symbol}",
    }
