"""
routes.py
---------
FastAPI routes for Alpha-Seeker.

POST /analyse/{symbol}   → run full analysis, return JSON
GET  /report/{symbol}    → HTML portfolio demo page
GET  /memory/{symbol}    → agent accuracy + recent records
GET  /health             → status check
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

import llm_client
from models import (
    AnalysisResponse, AgentOut, DebateOut, DebateRoundOut,
    FinalOut, MemoryOut, MetadataOut,
    MemoryStatsResponse, AgentAccuracyOut, MemoryRecordOut,
    HealthResponse,
)
from alpha_seeker.graph.graph import run_analysis
from alpha_seeker.memory import trade_memory
from alpha_seeker.rag.vector_store import doc_count

logger   = logging.getLogger(__name__)
router   = APIRouter()
_TMPL    = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)

# ── In-memory cache ────────────────────────────────────────
import config
_cache: dict[str, tuple[float, AnalysisResponse]] = {}


def _is_cached(symbol: str) -> Optional[AnalysisResponse]:
    entry = _cache.get(symbol.upper())
    if entry is None:
        return None
    ts, result = entry
    if time.time() - ts < config.ANALYSIS_CACHE_MINUTES * 60:
        return result
    return None


def _set_cache(symbol: str, result: AnalysisResponse) -> None:
    _cache[symbol.upper()] = (time.time(), result)


# ── State → response model ─────────────────────────────────

def _state_to_response(state: dict, elapsed: float) -> AnalysisResponse:
    from alpha_seeker.rag.vector_store import doc_count as _dc

    llm_stats = llm_client.get_session_stats()

    def _agent_out(thesis, verifications) -> AgentOut:
        if thesis is None:
            return AgentOut(
                direction="NEUTRAL", confidence=0.0,
                reasoning="No data", key_signals=[],
                rag_status="INSUFFICIENT_EVIDENCE", revision_applied=False,
            )
        rag_status = thesis.rag_status if thesis.rag_status != "PENDING" else "INSUFFICIENT_EVIDENCE"
        return AgentOut(
            direction=thesis.direction,
            confidence=round(thesis.confidence, 3),
            reasoning=thesis.reasoning,
            key_signals=thesis.key_signals,
            rag_status=rag_status,
            revision_applied=thesis.revision_flag,
        )

    rec = state.get("final_recommendation")
    if rec is None:
        final_out = FinalOut(
            direction="STAY OUT", conviction=0.0,
            reasoning="Analysis incomplete.", key_risks=[],
            conditions_to_revisit="", agent_agreement="SPLIT",
        )
    else:
        final_out = FinalOut(
            direction=rec.direction,
            conviction=round(rec.conviction, 3),
            reasoning=rec.reasoning,
            key_risks=rec.key_risks,
            conditions_to_revisit=rec.conditions_to_revisit,
            agent_agreement=rec.agent_agreement,
        )

    debate = state.get("debate_transcript")
    debate_out = DebateOut(
        rounds=[
            DebateRoundOut(
                challenger=r.challenger, challenged=r.challenged,
                challenge=r.challenge, response=r.response,
                confidence_delta=r.confidence_delta,
            )
            for r in (debate.rounds if debate else [])
        ],
        consensus_reached=debate.consensus_reached if debate else False,
    )

    rag_docs = sum(
        len(v) for v in [
            state.get("tech_verification") or [],
            state.get("sent_verification") or [],
            state.get("fund_verification") or [],
        ]
    )

    return AnalysisResponse(
        symbol=state["symbol"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        final=final_out,
        agents={
            "technical":   _agent_out(state.get("tech_thesis"),  state.get("tech_verification")),
            "sentiment":   _agent_out(state.get("sent_thesis"),  state.get("sent_verification")),
            "fundamental": _agent_out(state.get("fund_thesis"),  state.get("fund_verification")),
        },
        debate=debate_out,
        memory=MemoryOut(
            context_summary=state.get("memory_context") or "",
            record_id=state.get("memory_record_id"),
        ),
        metadata=MetadataOut(
            duration_seconds=round(elapsed, 2),
            llm_calls=llm_stats["calls"],
            estimated_cost_usd=round(llm_stats["cost_usd"], 5),
            rag_docs_checked=rag_docs,
            errors=state.get("errors", []),
        ),
    )


# ── Routes ─────────────────────────────────────────────────

@router.post("/analyse/{symbol}", response_model=AnalysisResponse)
async def analyse(symbol: str, force_refresh: bool = False):
    symbol = symbol.upper()

    if not force_refresh:
        cached = _is_cached(symbol)
        if cached:
            logger.info("Cache hit for %s", symbol)
            return cached

    llm_client.reset_session_stats()
    t0 = time.time()

    try:
        state = await run_analysis(symbol)
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", symbol, exc)
        # Never return 500 — return a degraded response
        raise HTTPException(status_code=200, detail=str(exc))

    elapsed  = time.time() - t0
    response = _state_to_response(state, elapsed)
    _set_cache(symbol, response)
    return response


@router.get("/report/{symbol}", response_class=HTMLResponse)
async def report(request: Request, symbol: str, force_refresh: bool = False):
    symbol = symbol.upper()
    cached = None if force_refresh else _is_cached(symbol)

    if cached is None:
        llm_client.reset_session_stats()
        t0    = time.time()
        state = await run_analysis(symbol)
        result = _state_to_response(state, time.time() - t0)
        _set_cache(symbol, result)
    else:
        result = cached

    return _TMPL.TemplateResponse(
        request=request,
        name="report.html",
        context={"r": result, "symbol": symbol},
    )


@router.get("/memory/{symbol}", response_model=MemoryStatsResponse)
async def memory(symbol: str):
    symbol = symbol.upper()
    try:
        stats = trade_memory.get_memory_stats(symbol)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    agent_acc = {
        agent: AgentAccuracyOut(**data)
        for agent, data in stats["agent_accuracy"].items()
    }
    recent = [
        MemoryRecordOut(**{k: row[k] for k in MemoryRecordOut.model_fields})
        for row in stats["recent"]
    ]
    return MemoryStatsResponse(
        symbol=symbol,
        total_analyses=stats["total_analyses"],
        agent_accuracy=agent_acc,
        recent=recent,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    try:
        n_docs = doc_count()
    except Exception:
        n_docs = -1
    try:
        stats = trade_memory.get_memory_stats("BTC")
        n_mem = stats["total_analyses"]
    except Exception:
        n_mem = -1
    return HealthResponse(status="ok", rag_docs=n_docs, memory_records=n_mem)
