"""
base.py
-------
Shared LLM parsing utilities used by all three specialist agents.
"""

import re
import logging

import llm_client
from models import AgentThesis

logger = logging.getLogger(__name__)

_DIRECTIONS = {"BULLISH", "BEARISH", "NEUTRAL"}


def _neutral_thesis(agent: str, raw_data: dict, reason: str) -> AgentThesis:
    return AgentThesis(
        agent=agent,
        direction="NEUTRAL",
        confidence=0.30,
        reasoning=f"Analysis unavailable: {reason}",
        key_signals=[],
        raw_data=raw_data,
        rag_status="INSUFFICIENT_EVIDENCE",
    )


def parse_llm_thesis(agent: str, response: str, raw_data: dict) -> AgentThesis:
    """
    Parse a free-form LLM response into an AgentThesis.

    Expected format (flexible — we scan for keywords):
      DIRECTION: BULLISH | BEARISH | NEUTRAL
      CONFIDENCE: 0.72
      REASONING: ...
      KEY_SIGNALS: signal 1 | signal 2
    """
    if not response:
        return _neutral_thesis(agent, raw_data, "empty LLM response")

    direction  = _extract_direction(response)
    confidence = _extract_confidence(response)
    reasoning  = _extract_reasoning(response)
    key_signals = _extract_key_signals(response)

    return AgentThesis(
        agent=agent,
        direction=direction,
        confidence=confidence,
        reasoning=reasoning,
        key_signals=key_signals,
        raw_data=raw_data,
    )


def _extract_direction(text: str) -> str:
    # 1. Look for "DIRECTION: X" label
    m = re.search(r"DIRECTION\s*:\s*(BULLISH|BEARISH|NEUTRAL)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2. Scan for any occurrence
    upper = text.upper()
    for d in _DIRECTIONS:
        if d in upper:
            return d
    return "NEUTRAL"


def _extract_confidence(text: str) -> float:
    # Look for "CONFIDENCE: 0.72" or bare decimal near "confidence"
    m = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if m:
        return min(max(float(m.group(1)), 0.0), 1.0)
    # Fallback: any decimal in [0.1, 0.99] range
    matches = re.findall(r"\b(0\.[1-9][0-9]?)\b", text)
    if matches:
        return float(matches[0])
    return 0.50


def _clean(text: str) -> str:
    """Strip markdown bold/italic markers left by the LLM."""
    return re.sub(r"\*+", "", text).strip(" \n\t-")


def _extract_reasoning(text: str) -> str:
    m = re.search(r"REASONING\s*:\s*(.+?)(?=KEY_SIGNALS|$)", text,
                  re.IGNORECASE | re.DOTALL)
    if m:
        return _clean(m.group(1))
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not re.match(r"(DIRECTION|CONFIDENCE|KEY_SIGNALS)\s*:",
                                           l, re.IGNORECASE)]
    return _clean(" ".join(lines))[:500] or _clean(text)[:500]


def _extract_key_signals(text: str) -> list[str]:
    m = re.search(r"KEY_SIGNALS\s*:\s*(.+?)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        raw = m.group(1).strip()
        signals = [_clean(s) for s in re.split(r"[|\n,]", raw) if _clean(s)]
        return [s for s in signals if len(s) > 3][:3]  # skip empty/noise entries
    return []


def call_agent_llm(system: str, prompt: str) -> str:
    return llm_client.call_llm(prompt, system=system, max_tokens=400)
