"""
CommodityState — the single TypedDict that flows through the LangGraph.

Parallel agent nodes (geo + weather) write into `agent_theses` via the
operator.add reducer. The collect node unpacks them into typed fields.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from models import AgentThesis, CommodityReport, DebateTranscript, FinalRecommendation, HistoricalAnalog


class CommodityState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    symbol:           str
    commodity_config: dict
    start_time:       float   # unix timestamp, set in validate_node

    # ── Parallel agent results (reducer concatenates lists from each branch) ──
    agent_theses: Annotated[list[AgentThesis], operator.add]

    # ── Typed references populated by collect_node ────────────────────────────
    geo_thesis:     Optional[AgentThesis]
    weather_thesis: Optional[AgentThesis]
    fund_thesis:    Optional[AgentThesis]

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_status:        dict            # agent_name -> RAGStatus string
    historical_analog: Optional[HistoricalAnalog]

    # ── Debate ────────────────────────────────────────────────────────────────
    debate_transcript: Optional[DebateTranscript]

    # ── Synthesis outputs (written by synthesis_node, read by format_node) ────
    executive_summary:    str
    full_report_text:     str
    final_recommendation: Optional[FinalRecommendation]

    # ── Price scalars (set by synthesis_node, read by format_node) ──────────
    price_context: dict

    # ── Final report ──────────────────────────────────────────────────────────
    report: Optional[CommodityReport]

    # ── Cost tracking (accumulated across all nodes) ─────────────────────────
    llm_calls:     Annotated[int,   operator.add]
    llm_cost_usd:  Annotated[float, operator.add]

    # ── Error tracking ────────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]
