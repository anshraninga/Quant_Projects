"""
graph.py
--------
Builds and compiles the Alpha-Seeker LangGraph.

Parallel execution of the three agent nodes uses LangGraph's fan-out
pattern: check_outcomes → memory → [technical, sentiment, fundamental]
fan-in → rag_verification → debate → synthesis → output.
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from models import AlphaState
from alpha_seeker.graph.nodes import (
    check_outcomes_node,
    memory_node,
    technical_node,
    sentiment_node,
    fundamental_node,
    rag_verification_node,
    debate_node,
    synthesis_node,
    output_node,
)
import config

logger = logging.getLogger(__name__)

_graph = None


def _build_graph():
    builder = StateGraph(AlphaState)

    # Register nodes
    builder.add_node("check_outcomes",    check_outcomes_node)
    builder.add_node("memory",            memory_node)
    builder.add_node("technical",         technical_node)
    builder.add_node("sentiment",         sentiment_node)
    builder.add_node("fundamental",       fundamental_node)
    builder.add_node("rag_verification",  rag_verification_node)
    builder.add_node("debate",            debate_node)
    builder.add_node("synthesis",         synthesis_node)
    builder.add_node("output",            output_node)

    # Entry point
    builder.set_entry_point("check_outcomes")

    # Sequential: outcomes → memory
    builder.add_edge("check_outcomes", "memory")

    # Fan-out: memory → all three agents in parallel
    builder.add_edge("memory", "technical")
    builder.add_edge("memory", "sentiment")
    builder.add_edge("memory", "fundamental")

    # Fan-in: all three agents → rag_verification
    builder.add_edge("technical",   "rag_verification")
    builder.add_edge("sentiment",   "rag_verification")
    builder.add_edge("fundamental", "rag_verification")

    # Sequential remainder
    builder.add_edge("rag_verification", "debate")
    builder.add_edge("debate",           "synthesis")
    builder.add_edge("synthesis",        "output")
    builder.add_edge("output",           END)

    return builder.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
        logger.info("Alpha-Seeker graph compiled")
    return _graph


async def run_analysis(symbol: str) -> AlphaState:
    """Run a full analysis for the given symbol and return the final state."""
    from datetime import datetime, timezone

    graph = get_graph()

    initial: AlphaState = {
        "symbol":               symbol.upper(),
        "tech_thesis":          None,
        "sent_thesis":          None,
        "fund_thesis":          None,
        "tech_verification":    None,
        "sent_verification":    None,
        "fund_verification":    None,
        "debate_transcript":    None,
        "memory_context":       None,
        "final_recommendation": None,
        "memory_record_id":     None,
        "started_at":           datetime.now(timezone.utc).isoformat(),
        "errors":               [],
    }

    final_state = await graph.ainvoke(initial)
    return final_state
