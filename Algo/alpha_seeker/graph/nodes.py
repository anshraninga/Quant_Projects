"""
nodes.py
--------
Every node function in the Alpha-Seeker LangGraph.

Execution order (from graph.py):
  check_outcomes_node
  memory_node
  technical_node  ──┐
  sentiment_node  ──┼── parallel
  fundamental_node──┘
  rag_verification_node
  debate_node
  synthesis_node
  output_node
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import llm_client
from models import (
    AgentThesis, DebateRound, DebateTranscript,
    FinalRecommendation, AlphaState,
)
from alpha_seeker.agents.technical_agent   import run_technical_agent
from alpha_seeker.agents.sentiment_agent   import run_sentiment_agent
from alpha_seeker.agents.fundamental_agent import run_fundamental_agent
from alpha_seeker.rag.verifier             import verify_thesis
from alpha_seeker.memory                   import trade_memory
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _safe_update(state: AlphaState, **kwargs) -> AlphaState:
    return {**state, **kwargs}


def _direction_score(direction: str) -> float:
    """Numeric representation for measuring agent disagreement."""
    return {"BULLISH": 1.0, "LONG": 1.0,
            "NEUTRAL": 0.0, "STAY OUT": 0.0,
            "BEARISH": -1.0, "SHORT": -1.0}.get(direction.upper(), 0.0)


def _agent_agreement(theses: list[AgentThesis]) -> str:
    scores = [_direction_score(t.direction) for t in theses]
    spread = max(scores) - min(scores)
    if spread <= 0.5:
        return "CONSENSUS"
    if spread >= 1.5:
        return "OPPOSED"
    return "SPLIT"


# ─────────────────────────────────────────────────────────
# NODE 1 — check_outcomes_node
# ─────────────────────────────────────────────────────────

def check_outcomes_node(state: AlphaState) -> AlphaState:
    """Update pending trade outcomes from yfinance. Never fails the graph."""
    try:
        trade_memory.init_db()
        n = trade_memory.check_pending_outcomes()
        if n:
            logger.info("Outcome tracker: updated %d records", n)
    except Exception as exc:
        logger.warning("check_outcomes_node error: %s", exc)
    return state


# ─────────────────────────────────────────────────────────
# NODE 2 — memory_node
# ─────────────────────────────────────────────────────────

def memory_node(state: AlphaState) -> AlphaState:
    try:
        context = trade_memory.get_memory_context(state["symbol"])
    except Exception as exc:
        logger.warning("memory_node error: %s", exc)
        context = f"Memory unavailable: {exc}"
    return _safe_update(state, memory_context=context)


# ─────────────────────────────────────────────────────────
# NODE 3 — technical_node
# ─────────────────────────────────────────────────────────

def technical_node(state: AlphaState) -> dict:
    # Returns only its own keys — parallel fan-out requires no full-state spread
    try:
        thesis = run_technical_agent(state["symbol"])
        return {"tech_thesis": thesis}
    except Exception as exc:
        logger.error("technical_node failed: %s", exc)
        return {
            "tech_thesis": AgentThesis(
                agent="technical", direction="NEUTRAL", confidence=0.30,
                reasoning="Technical analysis unavailable.", key_signals=[],
            ),
            "errors": [f"technical: {exc}"],
        }


# ─────────────────────────────────────────────────────────
# NODE 4 — sentiment_node
# ─────────────────────────────────────────────────────────

def sentiment_node(state: AlphaState) -> dict:
    try:
        thesis = run_sentiment_agent(state["symbol"])
        return {"sent_thesis": thesis}
    except Exception as exc:
        logger.error("sentiment_node failed: %s", exc)
        return {
            "sent_thesis": AgentThesis(
                agent="sentiment", direction="NEUTRAL", confidence=0.30,
                reasoning="Sentiment analysis unavailable.", key_signals=[],
            ),
            "errors": [f"sentiment: {exc}"],
        }


# ─────────────────────────────────────────────────────────
# NODE 5 — fundamental_node
# ─────────────────────────────────────────────────────────

def fundamental_node(state: AlphaState) -> dict:
    try:
        thesis = run_fundamental_agent(state["symbol"])
        return {"fund_thesis": thesis}
    except Exception as exc:
        logger.error("fundamental_node failed: %s", exc)
        return {
            "fund_thesis": AgentThesis(
                agent="fundamental", direction="NEUTRAL", confidence=0.30,
                reasoning="Fundamental analysis unavailable.", key_signals=[],
            ),
            "errors": [f"fundamental: {exc}"],
        }


# ─────────────────────────────────────────────────────────
# NODE 6 — rag_verification_node
# ─────────────────────────────────────────────────────────

def rag_verification_node(state: AlphaState) -> AlphaState:
    symbol = state["symbol"]
    errors = list(state.get("errors", []))
    updates: dict = {}

    for field, thesis_key in [
        ("tech_thesis",  "tech_verification"),
        ("sent_thesis",  "sent_verification"),
        ("fund_thesis",  "fund_verification"),
    ]:
        thesis = state.get(field)
        if thesis is None or not thesis.key_signals:
            updates[thesis_key] = []
            if thesis is not None:
                thesis.rag_status = "INSUFFICIENT_EVIDENCE"
                updates[field] = thesis
            continue
        try:
            revised_thesis, results = verify_thesis(thesis, symbol)
            updates[field]     = revised_thesis
            updates[thesis_key] = results
        except Exception as exc:
            logger.warning("RAG verification failed for %s: %s", field, exc)
            errors.append(f"rag_{field}: {exc}")
            updates[thesis_key] = []

    return _safe_update(state, **updates, errors=errors)


# ─────────────────────────────────────────────────────────
# NODE 7 — debate_node
# ─────────────────────────────────────────────────────────

def debate_node(state: AlphaState) -> AlphaState:
    theses = {
        "technical":   state.get("tech_thesis"),
        "sentiment":   state.get("sent_thesis"),
        "fundamental": state.get("fund_thesis"),
    }
    theses = {k: v for k, v in theses.items() if v is not None}

    # Skip if all agents agree (within 0.15 on direction score)
    scores = [_direction_score(t.direction) for t in theses.values()]
    if max(scores) - min(scores) <= 0.15:
        transcript = DebateTranscript(consensus_reached=True)
        return _safe_update(state, debate_transcript=transcript)

    rounds: list[DebateRound] = []
    confidence_changes: dict  = {}

    # Find the two most opposing agents
    items   = list(theses.items())
    pairs   = [(a, b) for i, (a, _) in enumerate(items)
               for b, _ in items[i+1:]]
    pair    = max(pairs, key=lambda p: abs(
        _direction_score(theses[p[0]].direction) -
        _direction_score(theses[p[1]].direction)
    ))

    for round_n in range(config.MAX_DEBATE_ROUNDS):
        challenger_name, challenged_name = pair
        challenger = theses[challenger_name]
        challenged = theses[challenged_name]

        ch_prompt = (
            f"You are the {challenger_name} analyst. "
            f"The {challenged_name} analyst says: '{challenged.reasoning}'. "
            f"Write one specific, evidence-based challenge in 1-2 sentences."
        )
        challenge_text = llm_client.call_llm(ch_prompt, max_tokens=150)

        resp_prompt = (
            f"You are the {challenged_name} analyst. "
            f"Challenge: '{challenge_text}'. "
            f"Your original view: '{challenged.reasoning}'. "
            f"Defend or acknowledge and revise your confidence. 1-2 sentences."
        )
        response_text = llm_client.call_llm(resp_prompt, max_tokens=150)

        # Parse any confidence revision mentioned in the response
        import re
        delta = 0.0
        m = re.search(r"(\d+\.?\d*)\s*(?:to|->|→)\s*(\d+\.?\d*)", response_text)
        if m:
            old_c = float(m.group(1))
            new_c = float(m.group(2))
            if 0 < old_c <= 1 and 0 < new_c <= 1:
                delta = round(new_c - old_c, 2)
                theses[challenged_name].confidence = max(0.10, new_c)

        rounds.append(DebateRound(
            challenger=challenger_name,
            challenged=challenged_name,
            challenge=challenge_text,
            response=response_text,
            confidence_delta=delta,
        ))

        if delta != 0.0:
            confidence_changes[challenged_name] = delta

        # Only run Round 2 if meaningful revision happened
        if round_n == 0 and abs(delta) > 0.10 and len(theses) == 3:
            mediator = [k for k in theses if k not in pair][0]
            pair = (mediator, challenged_name)
        else:
            break

    transcript = DebateTranscript(
        rounds=rounds,
        confidence_changes=confidence_changes,
        consensus_reached=False,
    )

    # Write revised theses back to state
    return _safe_update(
        state,
        tech_thesis=theses.get("technical", state.get("tech_thesis")),
        sent_thesis=theses.get("sentiment", state.get("sent_thesis")),
        fund_thesis=theses.get("fundamental", state.get("fund_thesis")),
        debate_transcript=transcript,
    )


# ─────────────────────────────────────────────────────────
# NODE 8 — synthesis_node
# ─────────────────────────────────────────────────────────

def synthesis_node(state: AlphaState) -> AlphaState:
    symbol = state["symbol"]
    tech   = state.get("tech_thesis")
    sent   = state.get("sent_thesis")
    fund   = state.get("fund_thesis")
    debate = state.get("debate_transcript")
    memory = state.get("memory_context", "No history.")

    theses = [t for t in [tech, sent, fund] if t]
    agreement = _agent_agreement(theses) if theses else "SPLIT"

    def _fmt(t: AgentThesis | None, label: str) -> str:
        if t is None:
            return f"{label}: no data"
        return (
            f"{label} ({t.agent}): {t.direction} confidence={t.confidence:.2f}\n"
            f"  {t.reasoning}"
        )

    debate_summary = "No debate (agents agreed)."
    if debate and debate.rounds:
        parts = []
        for r in debate.rounds:
            parts.append(
                f"  {r.challenger} challenged {r.challenged}: \"{r.challenge[:100]}\" "
                f"→ delta={r.confidence_delta:+.2f}"
            )
        debate_summary = "\n".join(parts)

    prompt = f"""You are a senior crypto analyst making a final trade recommendation.
Three specialist agents have analysed {symbol}:

{_fmt(tech, 'Technical')}

{_fmt(sent, 'Sentiment')}

{_fmt(fund, 'Fundamental')}

Debate outcome:
{debate_summary}

Memory context:
{memory}

Produce your final recommendation in this exact format:
DIRECTION: LONG | SHORT | STAY OUT
CONVICTION: (0.0-1.0)
REASONING: (3-4 sentences addressing agent disagreements and key risks)
KEY_RISKS: risk 1 | risk 2
CONDITIONS_TO_REVISIT: (one sentence)

Be decisive. Do not simply average the agents — synthesise."""

    response = llm_client.call_llm(prompt, max_tokens=500)
    rec      = _parse_final(response, agreement)

    # Save to memory DB
    record_id = None
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}-USD")
        hist   = ticker.history("1d")
        price  = float(hist["Close"].iloc[-1]) if not hist.empty else None

        record_id = trade_memory.save_analysis(
            symbol=symbol,
            tech_direction=tech.direction if tech else None,
            tech_confidence=tech.confidence if tech else None,
            sent_direction=sent.direction if sent else None,
            sent_confidence=sent.confidence if sent else None,
            fund_direction=fund.direction if fund else None,
            fund_confidence=fund.confidence if fund else None,
            final_direction=rec.direction,
            final_conviction=rec.conviction,
            entry_price=price,
        )
    except Exception as exc:
        logger.warning("Memory save failed: %s", exc)

    return _safe_update(state, final_recommendation=rec, memory_record_id=record_id)


def _parse_final(text: str, agreement: str) -> FinalRecommendation:
    import re

    direction = "STAY OUT"
    for d in ("LONG", "SHORT", "STAY OUT"):
        m = re.search(rf"DIRECTION\s*:\s*{d}", text, re.IGNORECASE)
        if m:
            direction = d
            break

    conviction = 0.50
    m = re.search(r"CONVICTION\s*:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if m:
        conviction = min(max(float(m.group(1)), 0.0), 1.0)

    reasoning = ""
    m = re.search(r"REASONING\s*:\s*(.+?)(?=KEY_RISKS|CONDITIONS|$)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()

    key_risks = []
    m = re.search(r"KEY_RISKS\s*:\s*(.+?)(?=CONDITIONS|$)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        key_risks = [s.strip() for s in re.split(r"[|\n]", m.group(1)) if s.strip()][:2]

    conditions = ""
    m = re.search(r"CONDITIONS_TO_REVISIT\s*:\s*(.+?)$",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        conditions = m.group(1).strip()

    if not reasoning:
        reasoning = text[:400]

    return FinalRecommendation(
        direction=direction,
        conviction=conviction,
        reasoning=reasoning,
        key_risks=key_risks,
        conditions_to_revisit=conditions,
        agent_agreement=agreement,
    )


# ─────────────────────────────────────────────────────────
# NODE 9 — output_node
# ─────────────────────────────────────────────────────────

def output_node(state: AlphaState) -> AlphaState:
    """Final pass-through — state is already fully populated."""
    return state
