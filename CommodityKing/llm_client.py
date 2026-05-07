"""
Central LLM client for CommodityKing.

All LLM calls in the system must go through this module.
Provides:
  - Provider abstraction (Anthropic / OpenAI)
  - Cheap agent model vs expensive synthesis model
  - Per-call logging with tokens and cost estimate
  - Accumulated cost tracking across one analysis run
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import config

logger = logging.getLogger(__name__)

# ── Per-model pricing (USD per million tokens) ────────────────────────────────
# Sources: Anthropic & OpenAI pricing pages (May 2026)
_COST_TABLE: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    # OpenAI
    "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "gpt-4o":                    {"input": 2.50,  "output": 10.00},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _COST_TABLE.get(model, {"input": 3.00, "output": 15.00})
    return (
        input_tokens  / 1_000_000 * pricing["input"] +
        output_tokens / 1_000_000 * pricing["output"]
    )


# ── Call record (one entry per LLM call) ─────────────────────────────────────

@dataclass
class CallRecord:
    model:         str
    purpose:       str    # e.g. "geo_agent", "synthesis"
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    duration_s:    float
    symbol:        str    = ""


# ── Main client ───────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around Anthropic (default) or OpenAI.

    Usage:
        client = LLMClient(symbol="wheat")
        text   = await client.agent(system="You are...", prompt="Analyse...")
        text   = await client.synthesis(system="You are...", prompt="Write...")
        records = client.records          # all calls for this symbol
        cost    = client.total_cost_usd   # total spend so far
    """

    def __init__(self, symbol: str = "") -> None:
        self.symbol   = symbol
        self.records:  list[CallRecord] = []
        self._provider = config.LLM_PROVIDER  # "anthropic" | "openai"

        if self._provider == "anthropic":
            self._agent_model     = config.ANTHROPIC_AGENT_MODEL
            self._synthesis_model = (
                config.ANTHROPIC_SYNTHESIS_MODEL
                if config.SYNTHESIS_MODEL_OVERRIDE
                else config.ANTHROPIC_AGENT_MODEL
            )
        else:
            self._agent_model     = config.OPENAI_AGENT_MODEL
            self._synthesis_model = (
                config.OPENAI_SYNTHESIS_MODEL
                if config.SYNTHESIS_MODEL_OVERRIDE
                else config.OPENAI_AGENT_MODEL
            )

    # ── Public interface ──────────────────────────────────────────────────────

    async def agent(
        self,
        prompt:     str,
        system:     str  = "",
        purpose:    str  = "agent",
        max_tokens: int  = 1024,
    ) -> str:
        """Call the cheap agent model. Returns response text."""
        return await self._call(
            model=self._agent_model,
            system=system,
            prompt=prompt,
            purpose=purpose,
            max_tokens=max_tokens,
        )

    async def synthesis(
        self,
        prompt:     str,
        system:     str = "",
        purpose:    str = "synthesis",
        max_tokens: int = 4096,
    ) -> str:
        """Call the expensive synthesis model. Returns response text."""
        return await self._call(
            model=self._synthesis_model,
            system=system,
            prompt=prompt,
            purpose=purpose,
            max_tokens=max_tokens,
        )

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 6)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    def cost_summary(self) -> dict:
        return {
            "total_calls":         self.total_calls,
            "total_cost_usd":      self.total_cost_usd,
            "total_input_tokens":  sum(r.input_tokens  for r in self.records),
            "total_output_tokens": sum(r.output_tokens for r in self.records),
            "by_purpose": {
                rec.purpose: {
                    "cost_usd":      rec.cost_usd,
                    "input_tokens":  rec.input_tokens,
                    "output_tokens": rec.output_tokens,
                    "duration_s":    rec.duration_s,
                }
                for rec in self.records
            },
        }

    # ── Internal dispatch ─────────────────────────────────────────────────────

    async def _call(
        self,
        model:      str,
        system:     str,
        prompt:     str,
        purpose:    str,
        max_tokens: int,
    ) -> str:
        t0 = time.perf_counter()
        try:
            if self._provider == "anthropic":
                text, in_tok, out_tok = await self._call_anthropic(
                    model, system, prompt, max_tokens
                )
            else:
                text, in_tok, out_tok = await self._call_openai(
                    model, system, prompt, max_tokens
                )
        except Exception as exc:
            logger.error(
                "[LLM] %s | %s | FAILED | %s",
                self.symbol, purpose, exc
            )
            raise

        duration = round(time.perf_counter() - t0, 2)
        cost     = _estimate_cost(model, in_tok, out_tok)

        rec = CallRecord(
            model=model,
            purpose=purpose,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            duration_s=duration,
            symbol=self.symbol,
        )
        self.records.append(rec)

        logger.info(
            "[LLM] %s | %-20s | model=%-30s | in=%5d out=%5d | $%.5f | %.1fs",
            self.symbol or "—",
            purpose,
            model,
            in_tok,
            out_tok,
            cost,
            duration,
        )
        return text

    # ── Anthropic backend ─────────────────────────────────────────────────────

    async def _call_anthropic(
        self,
        model:      str,
        system:     str,
        prompt:     str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        text      = response.content[0].text
        in_tok    = response.usage.input_tokens
        out_tok   = response.usage.output_tokens
        return text, in_tok, out_tok

    # ── OpenAI backend ────────────────────────────────────────────────────────

    async def _call_openai(
        self,
        model:      str,
        system:     str,
        prompt:     str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        from openai import AsyncOpenAI

        client   = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response  = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        text    = response.choices[0].message.content or ""
        in_tok  = response.usage.prompt_tokens
        out_tok = response.usage.completion_tokens
        return text, in_tok, out_tok
