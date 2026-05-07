"""
llm_client.py
-------------
Single gateway for all LLM calls in Alpha-Seeker.

All agents and nodes must use call_llm() — never instantiate the SDK
client directly elsewhere.  This centralises token counting, cost
estimation, and session logging.

Supported providers (set LLM_PROVIDER in .env):
  anthropic  → claude-haiku-4-5-20251001
  openai     → gpt-4o-mini

Cost estimates (as of May 2026, USD per 1M tokens):
  Haiku 4.5  input  $0.80   output  $4.00
  gpt-4o-mini input $0.15   output  $0.60
"""

import time
import logging
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)

# ── Cost table (USD per token) ─────────────────────────────
_COST = {
    "anthropic": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    "openai":    {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
}

# ── Session totals ─────────────────────────────────────────
_session_calls    = 0
_session_cost_usd = 0.0
_session_input_tokens  = 0
_session_output_tokens = 0


def get_session_stats() -> dict:
    return {
        "calls":          _session_calls,
        "input_tokens":   _session_input_tokens,
        "output_tokens":  _session_output_tokens,
        "cost_usd":       round(_session_cost_usd, 6),
    }


def reset_session_stats() -> None:
    global _session_calls, _session_cost_usd
    global _session_input_tokens, _session_output_tokens
    _session_calls = _session_cost_usd = 0
    _session_input_tokens = _session_output_tokens = 0


# ── Anthropic client (lazy) ────────────────────────────────
_anthropic_client = None

def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


# ── OpenAI client (lazy) ───────────────────────────────────
_openai_client = None

def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


# ── Main public function ───────────────────────────────────

def call_llm(
    prompt:     str,
    system:     str = "",
    max_tokens: int = 500,
) -> str:
    """
    Send a prompt to the configured LLM and return the text response.

    Args:
        prompt:     The user message.
        system:     Optional system prompt.
        max_tokens: Maximum tokens in the response.

    Returns:
        The model's text response (stripped).

    Never raises — returns an empty string and logs on error so callers
    can degrade gracefully rather than crashing the graph.
    """
    global _session_calls, _session_cost_usd
    global _session_input_tokens, _session_output_tokens

    provider = config.LLM_PROVIDER.lower()
    t0 = time.time()

    try:
        if provider == "anthropic":
            text, in_tok, out_tok = _call_anthropic(prompt, system, max_tokens)
        elif provider == "openai":
            text, in_tok, out_tok = _call_openai(prompt, system, max_tokens)
        else:
            logger.error("Unknown LLM_PROVIDER: %s", provider)
            return ""
    except Exception as exc:
        logger.error("LLM call failed (%s): %s", provider, exc)
        return ""

    cost = in_tok * _COST[provider]["input"] + out_tok * _COST[provider]["output"]

    _session_calls += 1
    _session_input_tokens  += in_tok
    _session_output_tokens += out_tok
    _session_cost_usd      += cost

    elapsed = round(time.time() - t0, 2)
    logger.info(
        "[LLM] %s | in=%d out=%d | $%.5f | %.2fs",
        provider, in_tok, out_tok, cost, elapsed,
    )

    return text.strip()


def _call_anthropic(prompt: str, system: str, max_tokens: int) -> tuple[str, int, int]:
    client = _get_anthropic()
    kwargs: dict = {
        "model":      config.ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages":   [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    resp     = client.messages.create(**kwargs)
    text     = resp.content[0].text
    in_tok   = resp.usage.input_tokens
    out_tok  = resp.usage.output_tokens
    return text, in_tok, out_tok


def _call_openai(prompt: str, system: str, max_tokens: int) -> tuple[str, int, int]:
    client   = _get_openai()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp    = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
    )
    text    = resp.choices[0].message.content or ""
    in_tok  = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    return text, in_tok, out_tok
