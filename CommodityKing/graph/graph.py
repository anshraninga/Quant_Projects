"""
CommodityKing LangGraph — full analysis pipeline.

Graph topology:
  validate_node
    → fan_out_geo_weather          (conditional: returns [Send(geo), Send(weather)])
    → geo_node | weather_node      (parallel)
    → collect_and_fund_node        (barrier: waits for both, then runs fundamentals)
    → rag_node
    → analog_node
    → debate_node
    → synthesis_node
    → format_node
    → END

Public interface:
  run_analysis(symbol)       -> CommodityReport
  run_analyses(symbols)      -> list[CommodityReport]   (parallel across commodities)
  build_graph()              -> compiled graph (for inspection / reuse)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from graph.nodes import (
    analog_node,
    collect_and_fund_node,
    debate_node,
    fan_out_geo_weather,
    format_node,
    geo_node,
    rag_node,
    synthesis_node,
    validate_node,
    weather_node,
)
from graph.state import CommodityState
from models import CommodityReport

logger = logging.getLogger(__name__)


def build_graph():
    """Compile and return the LangGraph StateGraph."""
    g = StateGraph(CommodityState)

    # Register nodes
    g.add_node("validate_node",         validate_node)
    g.add_node("geo_node",              geo_node)
    g.add_node("weather_node",          weather_node)
    g.add_node("collect_and_fund_node", collect_and_fund_node)
    g.add_node("rag_node",              rag_node)
    g.add_node("analog_node",           analog_node)
    g.add_node("debate_node",           debate_node)
    g.add_node("synthesis_node",        synthesis_node)
    g.add_node("format_node",           format_node)

    # Entry point
    g.set_entry_point("validate_node")

    # validate → fan-out (conditional edge returns Send objects)
    g.add_conditional_edges(
        "validate_node",
        fan_out_geo_weather,
        ["geo_node", "weather_node"],
    )

    # Both parallel branches → collect_and_fund_node (barrier)
    g.add_edge("geo_node",     "collect_and_fund_node")
    g.add_edge("weather_node", "collect_and_fund_node")

    # Sequential post-collection pipeline
    g.add_edge("collect_and_fund_node", "rag_node")
    g.add_edge("rag_node",              "analog_node")
    g.add_edge("analog_node",           "debate_node")
    g.add_edge("debate_node",           "synthesis_node")
    g.add_edge("synthesis_node",        "format_node")
    g.add_edge("format_node",           END)

    return g.compile()


# Module-level compiled graph (built once)
_graph = build_graph()


async def run_analysis(symbol: str) -> CommodityReport:
    """
    Run the full CommodityKing pipeline for one commodity symbol.
    Returns CommodityReport (never raises — errors are captured in report.errors).
    """
    initial_state: dict[str, Any] = {
        "symbol":           symbol,
        "commodity_config": {},
        "start_time":       0.0,
        "agent_theses":     [],
        "geo_thesis":       None,
        "weather_thesis":   None,
        "fund_thesis":      None,
        "rag_status":       {},
        "historical_analog":    None,
        "debate_transcript":    None,
        "executive_summary":    "",
        "full_report_text":     "",
        "final_recommendation": None,
        "price_context":        {},
        "report":               None,
        "llm_calls":            0,
        "llm_cost_usd":         0.0,
        "errors":               [],
    }

    logger.info("[graph] Starting analysis for %s", symbol)
    try:
        final_state = await _graph.ainvoke(initial_state)
        report = final_state.get("report")
        if report is None:
            from datetime import datetime, timezone
            import time
            report = CommodityReport(
                symbol=symbol,
                commodity_name=symbol,
                generated_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=0.0,
                errors=final_state.get("errors", ["Graph completed without a report"]),
            )
        logger.info("[graph] Completed %s in %.1fs", symbol, report.duration_seconds)
        return report
    except Exception as exc:
        logger.error("[graph] Unhandled exception for %s: %s", symbol, exc)
        from datetime import datetime, timezone
        return CommodityReport(
            symbol=symbol,
            commodity_name=symbol,
            generated_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=0.0,
            errors=[str(exc)],
        )


async def run_analyses(symbols: list[str]) -> list[CommodityReport]:
    """
    Run analysis for multiple commodities in parallel.
    Each commodity gets an independent graph run.
    """
    logger.info("[graph] Running parallel analyses for: %s", symbols)
    tasks = [run_analysis(s) for s in symbols]
    reports = await asyncio.gather(*tasks, return_exceptions=False)
    return list(reports)
