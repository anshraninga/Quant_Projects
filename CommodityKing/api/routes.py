"""
FastAPI route handlers for CommodityKing.

Endpoints:
  POST /analyse       — run analysis for one or more commodity symbols
  GET  /commodities   — list all supported commodities
  GET  /health        — service health check
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

import config
from commodities_config import COMMODITIES, VALID_SYMBOLS
from graph.graph import run_analysis, run_analyses
from models import (
    AgentResponse,
    AnalyseRequest,
    AnalyseResponse,
    CommodityAnalysisResponse,
    CommodityListItem,
    CommodityListResponse,
    CommodityReport,
    DebateResponse,
    FinalRecommendationResponse,
    HealthResponse,
    HistoricalAnalogResponse,
    MetadataResponse,
    PriceContextResponse,
    QuantitativeResponse,
)
from rag.vector_store import collection_counts

logger = logging.getLogger(__name__)

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# Serialisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _agent_response(thesis) -> AgentResponse:
    if thesis is None:
        return AgentResponse(direction="NEUTRAL", confidence=0.0,
                             headline="Agent did not run", rag_status="NOT_CHECKED")
    return AgentResponse(
        direction=thesis.direction,
        confidence=thesis.confidence,
        headline=thesis.headline,
        rag_status=thesis.rag_status,
        error=thesis.error,
        # Geopolitical
        analysis_24h=thesis.analysis_24h,
        analysis_7d=thesis.analysis_7d,
        historical_analog=thesis.historical_analog,
        key_risks=thesis.key_risks,
        surprise_score=thesis.surprise_score,
        n_articles=thesis.n_articles,
        # Weather
        critical_regions=thesis.critical_regions,
        forecast_outlook=thesis.forecast_outlook,
        supply_impact=thesis.supply_impact,
        max_zscore=thesis.max_zscore,
        weather_relevant=thesis.weather_relevant,
        # Fundamentals
        supply_analysis=thesis.supply_analysis,
        demand_analysis=thesis.demand_analysis,
        inventory_analysis=thesis.inventory_analysis,
        ssi=thesis.ssi_result.get("ssi") if thesis.ssi_result else None,
        ssi_level=thesis.ssi_result.get("level") if thesis.ssi_result else None,
        regime=thesis.hmm_result.get("regime_label") if thesis.hmm_result else None,
    )


def _price_context(report: CommodityReport) -> PriceContextResponse:
    pd = report.price_context  # dict from fetch_price_data()
    return PriceContextResponse(
        current_price=pd.get("current_price", 0.0),
        unit=COMMODITIES.get(report.symbol, {}).get("unit", ""),
        change_24h_pct=pd.get("change_24h_pct", 0.0),
        change_7d_pct=pd.get("change_7d_pct", 0.0),
        kalman_trend=report.kalman.trend if report.kalman else "unknown",
        signal_sigma=report.kalman.signal_sigma if report.kalman else 0.0,
        regime=report.hmm.regime_label if report.hmm else "unknown",
    )


def _quant_response(report: CommodityReport) -> QuantitativeResponse:
    return QuantitativeResponse(
        kalman={
            "latest_signal": report.kalman.latest_signal,
            "is_signal":     report.kalman.is_signal,
            "trend":         report.kalman.trend,
            "signal_sigma":  report.kalman.signal_sigma,
        } if report.kalman else {},
        hmm={
            "current_regime":     report.hmm.current_regime,
            "regime_label":       report.hmm.regime_label,
            "confidence":         report.hmm.confidence,
            "is_high_volatility": report.hmm.is_high_volatility,
        } if report.hmm else {},
        ssi={
            "ssi":                   report.ssi.ssi,
            "level":                 report.ssi.level,
            "direction":             report.ssi.direction,
            "weather_component":     report.ssi.weather_component,
            "geo_component":         report.ssi.geo_component,
            "inventory_component":   report.ssi.inventory_component,
        } if report.ssi else {},
        bayesian_surprise={
            "surprise_score":       report.bayesian_surprise.surprise_score,
            "interpretation":       report.bayesian_surprise.interpretation,
            "top_surprise_terms":   report.bayesian_surprise.top_surprise_terms,
            "kl_divergence":        report.bayesian_surprise.kl_divergence,
            "history_source":       report.bayesian_surprise.history_source,
        } if report.bayesian_surprise else {},
        cointegration=[
            {
                "pair":          c.pair,
                "cointegrated":  c.cointegrated,
                "pvalue":        c.pvalue,
                "spread_zscore": c.spread_zscore,
                "signal":        c.signal,
                "mean_reverting": c.mean_reverting,
            }
            for c in report.cointegration
        ],
    )


def _debate_response(report: CommodityReport) -> DebateResponse:
    if not report.debate:
        return DebateResponse(occurred=False, rounds=0)
    d = report.debate
    return DebateResponse(
        occurred=d.occurred,
        rounds=d.rounds,
        skip_reason=d.skip_reason,
        transcript=[
            {
                "challenger":        r.challenger,
                "challenged":        r.challenged,
                "challenge_text":    r.challenge_text,
                "response_text":     r.response_text,
                "confidence_before": r.confidence_before,
                "confidence_after":  r.confidence_after,
                "revised":           r.revised,
            }
            for r in d.transcript
        ],
    )


def _analog_response(report: CommodityReport) -> HistoricalAnalogResponse:
    a = report.historical_analog
    if not a or not a.found:
        return HistoricalAnalogResponse(found=False)
    return HistoricalAnalogResponse(
        found=True,
        event=a.event,
        date=a.date,
        similarity=a.similarity,
        price_impact_then=a.price_impact_then,
        context=a.context,
        resolution=a.resolution,
    )


def _final_rec_response(report: CommodityReport) -> FinalRecommendationResponse:
    r = report.final_recommendation
    if not r:
        return FinalRecommendationResponse(
            direction="NEUTRAL", conviction=0.0, time_horizon="MEDIUM", reasoning="")
    return FinalRecommendationResponse(
        direction=r.direction,
        conviction=r.conviction,
        time_horizon=r.time_horizon,
        reasoning=r.reasoning,
        upside_risks=r.upside_risks,
        downside_risks=r.downside_risks,
        watch_list=r.watch_list,
    )


def _serialise_report(report: CommodityReport) -> CommodityAnalysisResponse:
    return CommodityAnalysisResponse(
        symbol=report.symbol,
        commodity_name=report.commodity_name,
        generated_at=report.generated_at,
        duration_seconds=report.duration_seconds,
        executive_summary=report.executive_summary,
        price_context=_price_context(report),
        agents={
            "geopolitical": _agent_response(report.geo_thesis),
            "weather":      _agent_response(report.weather_thesis),
            "fundamentals": _agent_response(report.fund_thesis),
        },
        debate=_debate_response(report),
        quantitative=_quant_response(report),
        historical_analog=_analog_response(report),
        final_recommendation=_final_rec_response(report),
        full_report_text=report.full_report_text,
        metadata=MetadataResponse(
            data_sources_used=report.data_sources_used,
            llm_calls=report.llm_calls,
            estimated_cost_usd=report.estimated_cost_usd,
            errors=report.errors,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# POST /analyse
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/analyse", response_model=AnalyseResponse)
async def analyse(request: AnalyseRequest) -> AnalyseResponse:
    """
    Run commodity analysis for one or more symbols in parallel.
    Multiple symbols are analysed concurrently via asyncio.gather().
    Unknown symbols return HTTP 400 before any analysis runs.
    """
    # Validate all symbols upfront
    unknown = [s for s in request.symbols if s not in VALID_SYMBOLS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unknown commodity symbols",
                "unknown": unknown,
                "valid_symbols": sorted(VALID_SYMBOLS),
            },
        )

    logger.info("[API] /analyse request: symbols=%s", request.symbols)
    start = time.time()

    # Run all symbols in parallel — never raises, errors captured in report.errors
    reports = await run_analyses(request.symbols)

    total_duration = round(time.time() - start, 2)
    logger.info("[API] /analyse complete: %d commodities in %.1fs", len(reports), total_duration)

    return AnalyseResponse(
        analyses=[_serialise_report(r) for r in reports],
        total_duration_seconds=total_duration,
        commodities_analysed=len(reports),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /commodities
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/commodities", response_model=CommodityListResponse)
def list_commodities() -> CommodityListResponse:
    """Return all supported commodity symbols with their metadata."""
    items = []
    for symbol, cfg in COMMODITIES.items():
        items.append(CommodityListItem(
            symbol=symbol,
            name=cfg["name"],
            ticker=cfg["ticker"],
            unit=cfg["unit"],
            category=cfg["category"],
            weather_relevant=cfg.get("weather_relevant", False),
            usda_relevant=cfg.get("usda_relevant", False),
            eia_relevant=cfg.get("eia_relevant", False),
            n_weather_regions=len(cfg.get("weather_regions", [])),
            cointegrated_with=cfg.get("cointegrated_with", []),
        ))
    return CommodityListResponse(commodities=items, total=len(items))


# ══════════════════════════════════════════════════════════════════════════════
# GET /health
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check — includes ChromaDB doc count and config warnings."""
    try:
        counts  = collection_counts()
        chroma_docs = counts.get("historical_events", 0) + counts.get("commodity_news", 0)
    except Exception as exc:
        logger.warning("[API] /health ChromaDB check failed: %s", exc)
        chroma_docs = -1

    warnings = config.validate()

    return HealthResponse(
        status="ok",
        chroma_docs=chroma_docs,
        commodities_supported=len(COMMODITIES),
        config_warnings=warnings,
    )
