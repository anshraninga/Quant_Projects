"""
fundamental_agent.py
--------------------
Fetches Binance futures market structure data and CryptoCompare
on-chain proxy, then synthesises via LLM.

All HTTP calls are best-effort — partial data is returned rather
than raising so the graph never crashes on a bad endpoint.
"""

import logging
import time
from typing import Any

import requests

import config
from models import AgentThesis
from alpha_seeker.agents.base import call_agent_llm, parse_llm_thesis, _neutral_thesis

logger = logging.getLogger(__name__)

_FUTURES_BASE = "https://fapi.binance.com"
_CC_BASE      = "https://min-api.cryptocompare.com"
_TIMEOUT      = 8

# CryptoCompare coin IDs for common symbols
_CC_COIN_IDS: dict[str, int] = {
    "BTC": 1182, "ETH": 7605, "SOL": 5426, "BNB": 1839,
    "XRP": 52,   "ADA": 321992, "DOGE": 74,  "AVAX": 9462,
    "LINK": 1975, "DOT": 28301, "ICP": 8916, "SUI": 29255,
    "APT": 27153, "TON": 28321,
}

_SYSTEM = (
    "You are a crypto fundamental analyst specialising in derivatives "
    "market structure. Be precise about funding rates, OI changes, and "
    "long/short ratios. Identify leverage risk explicitly."
)


# ── Binance futures data ───────────────────────────────────

def _get(url: str, params: dict) -> Any:
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug("Binance futures request failed %s: %s", url, exc)
        return None


def fetch_binance_fundamentals(symbol: str) -> dict:
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym_usdt = sym + "USDT"
    else:
        sym_usdt = sym

    data: dict = {"symbol": sym_usdt, "errors": []}

    # Funding rate (last 8 readings)
    fr = _get(f"{_FUTURES_BASE}/fapi/v1/fundingRate",
              {"symbol": sym_usdt, "limit": 8})
    if fr and isinstance(fr, list):
        rates = [float(r["fundingRate"]) for r in fr if "fundingRate" in r]
        data["funding_rates"]     = rates
        data["funding_rate_last"] = rates[-1] if rates else None
        data["funding_rate_avg"]  = round(sum(rates) / len(rates), 6) if rates else None

    # Current open interest
    oi = _get(f"{_FUTURES_BASE}/fapi/v1/openInterest", {"symbol": sym_usdt})
    if oi and "openInterest" in oi:
        data["open_interest"] = float(oi["openInterest"])

    # OI history (24 h, 1 h bars)
    oih = _get(f"{_FUTURES_BASE}/futures/data/openInterestHist",
               {"symbol": sym_usdt, "period": "1h", "limit": 24})
    if oih and isinstance(oih, list) and len(oih) >= 2:
        oi_vals = [float(r["sumOpenInterest"]) for r in oih if "sumOpenInterest" in r]
        if oi_vals:
            pct_chg = (oi_vals[-1] - oi_vals[0]) / (oi_vals[0] + 1e-10) * 100
            data["oi_change_24h_pct"] = round(pct_chg, 2)

    # Long/short ratio
    ls = _get(f"{_FUTURES_BASE}/futures/data/globalLongShortAccountRatio",
              {"symbol": sym_usdt, "period": "1h", "limit": 8})
    if ls and isinstance(ls, list):
        ratios = [float(r["longShortRatio"]) for r in ls if "longShortRatio" in r]
        data["long_short_ratio_last"] = ratios[-1] if ratios else None
        data["long_short_ratio_avg"]  = round(sum(ratios) / len(ratios), 4) if ratios else None

    # Mark vs index price (basis)
    pi = _get(f"{_FUTURES_BASE}/fapi/v1/premiumIndex", {"symbol": sym_usdt})
    if pi and "markPrice" in pi:
        mark  = float(pi["markPrice"])
        index = float(pi.get("indexPrice", mark))
        basis = (mark - index) / (index + 1e-10) * 100
        data["mark_price"]        = mark
        data["index_price"]       = index
        data["basis_pct"]         = round(basis, 4)

    return data


# ── CryptoCompare on-chain proxy ──────────────────────────

def fetch_onchain_proxy(symbol: str) -> dict:
    coin_id = _CC_COIN_IDS.get(symbol.upper())
    if coin_id is None:
        return {"error": f"Unknown symbol {symbol} for CryptoCompare"}
    try:
        r = requests.get(
            f"{_CC_BASE}/data/social/coin/latest",
            params={"coinId": coin_id},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json().get("Data", {})
        return {
            "reddit_posts_per_hour":   raw.get("Reddit", {}).get("posts_per_hour"),
            "reddit_comments_per_hour":raw.get("Reddit", {}).get("comments_per_hour"),
            "twitter_followers":       raw.get("Twitter", {}).get("followers"),
            "twitter_statuses_24h":    raw.get("Twitter", {}).get("statuses"),
        }
    except Exception as exc:
        logger.debug("CryptoCompare on-chain failed for %s: %s", symbol, exc)
        return {"error": str(exc)}


# ── Agent entry point ─────────────────────────────────────

def run_fundamental_agent(symbol: str) -> AgentThesis:
    raw: dict = {}
    try:
        raw["binance"]  = fetch_binance_fundamentals(symbol)
        raw["onchain"]  = fetch_onchain_proxy(symbol)
    except Exception as exc:
        logger.error("Fundamental tools failed for %s: %s", symbol, exc)
        return _neutral_thesis("fundamental", raw, str(exc))

    summary = _format_raw(raw)

    prompt = f"""You are a crypto fundamental analyst specialising in derivatives market structure.

Market structure data for {symbol}:
{summary}

Write a 2-3 sentence thesis. Include:
DIRECTION: BULLISH | BEARISH | NEUTRAL
CONFIDENCE: (0.0-1.0)
REASONING: (2-3 sentences citing specific values)
KEY_SIGNALS: signal 1 | signal 2

Pay particular attention to:
- Funding rate >0.05% per 8h (longs over-leveraged, bearish risk) or <-0.05% (shorts squeezable, bullish risk)
- OI change >10% in 24h (elevated leverage risk in either direction)
- Long/short ratio >0.65 (longs dominating, squeeze risk) or <0.35 (shorts dominating)
If data is missing or symbol is not on futures, state NEUTRAL with low confidence."""

    response = call_agent_llm(_SYSTEM, prompt)
    return parse_llm_thesis("fundamental", response, raw)


def _format_raw(raw: dict) -> str:
    b = raw.get("binance", {})
    o = raw.get("onchain", {})
    lines = []

    if b.get("funding_rate_last") is not None:
        lines.append(f"Funding rate (last): {b['funding_rate_last']:.6f} per 8h")
    if b.get("funding_rate_avg") is not None:
        lines.append(f"Funding rate (8-period avg): {b['funding_rate_avg']:.6f}")
    if b.get("oi_change_24h_pct") is not None:
        lines.append(f"Open interest change 24h: {b['oi_change_24h_pct']:+.2f}%")
    if b.get("long_short_ratio_last") is not None:
        lines.append(f"Long/short ratio (last): {b['long_short_ratio_last']:.4f}")
    if b.get("basis_pct") is not None:
        lines.append(f"Mark/index basis: {b['basis_pct']:+.4f}%")
    if o.get("reddit_posts_per_hour") is not None:
        lines.append(f"Reddit posts/h: {o['reddit_posts_per_hour']}")

    if not lines:
        lines.append("No futures data available (symbol may not be listed on Binance futures).")

    return "\n".join(lines)
