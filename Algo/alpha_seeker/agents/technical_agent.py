"""
technical_agent.py
------------------
Calls TechnicalAnalysis, CandlestickAnalysis, ChartPatterns then
passes outputs to the LLM for a structured thesis.
"""

import logging

from existing_modules.TechnicalAnalysis   import run_technical_analysis
from existing_modules.CandlestickAnalysis import run_candlestick_analysis
from existing_modules.ChartPatterns       import run_chart_analysis
from models import AgentThesis
from alpha_seeker.agents.base import call_agent_llm, parse_llm_thesis, _neutral_thesis

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a senior technical analyst specialising in crypto markets. "
    "Be specific, cite indicator names and values, and be decisive."
)


def run_technical_agent(symbol: str) -> AgentThesis:
    raw: dict = {}
    try:
        raw["technical"]   = run_technical_analysis(symbol)
        raw["candlestick"] = run_candlestick_analysis(symbol)
        raw["chart"]       = run_chart_analysis(symbol)
    except Exception as exc:
        logger.error("Technical tools failed for %s: %s", symbol, exc)
        return _neutral_thesis("technical", raw, str(exc))

    summary = _format_raw(raw)

    prompt = f"""You are a technical analyst. Here are the indicator signals for {symbol}:

{summary}

Write a 2-3 sentence thesis. Include:
DIRECTION: BULLISH | BEARISH | NEUTRAL
CONFIDENCE: (0.0-1.0)
REASONING: (2-3 sentences citing specific indicators and values)
KEY_SIGNALS: signal 1 | signal 2

Be specific about which indicators drive your view. If fewer than 3 indicators agree, cap CONFIDENCE at 0.55."""

    response = call_agent_llm(_SYSTEM, prompt)
    return parse_llm_thesis("technical", response, raw)


def _format_raw(raw: dict) -> str:
    lines = []
    for module, data in raw.items():
        if not isinstance(data, dict):
            continue
        timeframes = data.get("timeframes", {})
        if timeframes:
            for tf, tf_data in timeframes.items():
                agg  = tf_data.get("aggregate", {})
                sigs = tf_data.get("signals", [])
                direction  = agg.get("overall", "?").upper()
                bull_pct   = agg.get("bullish_pct", "?")
                bear_pct   = agg.get("bearish_pct", "?")
                price      = tf_data.get("current_price", "?")
                lines.append(
                    f"[{module.upper()} {tf}]  direction={direction}  "
                    f"bull={bull_pct}%  bear={bear_pct}%  price={price}"
                )
                for s in sigs[:4]:
                    if isinstance(s, dict) and s.get("signal") not in (None, "neutral"):
                        name = s.get("indicator", s.get("pattern", ""))
                        sig  = s.get("signal", "")
                        val  = s.get("value", s.get("reason", ""))
                        lines.append(f"  {name}: {sig}  ({str(val)[:60]})")
        else:
            direction = data.get("overall", data.get("direction", "?"))
            lines.append(f"[{module.upper()}]  direction={direction}")
    return "\n".join(lines) if lines else str(raw)[:800]
