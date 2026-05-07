"""
Supply/Demand Fundamentals Agent.

Data flow:
  1. Fetch 5-year weekly price data (yfinance) → HMM regime
  2. If agricultural: fetch USDA NASS production + stocks (replaces discontinued WASDE)
  3. If energy: fetch EIA weekly inventory → compute deviation
  4. Fetch Baltic Dry Index (skip if unavailable)
  5. Run hmm_regime, supply_shock_index, cointegration_spread
  6. Build structured LLM prompt
  7. Parse response into AgentThesis

Note on SSI: weather_zscores and geo_surprise are passed in from the graph
node after all three agents complete. When called directly (e.g. in tests),
they default to neutral values.
"""

from __future__ import annotations

import logging

import numpy as np
import yfinance as yf

from agents.base import (
    compute_inventory_deviation,
    fetch_baltic_dry_index,
    fetch_eia_crude_inventory,
    fetch_eia_natgas_storage,
    fetch_nass_supply_data,
    fetch_price_data,
    format_nass_summary,
    parse_confidence,
    parse_direction,
    parse_section,
    strip_markdown,
    _first_meaningful_line,
)
from commodities_config import COMMODITIES
from llm_client import LLMClient
from models import AgentThesis
from models_quant.quant_models import (
    cointegration_spread,
    hmm_regime,
    supply_shock_index,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a commodity supply/demand analyst with deep expertise in "
    "fundamental market structure. You interpret inventory data, production "
    "estimates, and mathematical models. You are quantitative, specific, and "
    "honest about data quality. You clearly separate structural trends from "
    "short-term noise."
)

_ALL_HEADERS = [
    "DIRECTION", "CONFIDENCE", "HEADLINE",
    "SUPPLY ANALYSIS", "DEMAND ANALYSIS", "STOCKS/INVENTORY",
    "SSI INTERPRETATION", "CROSS-COMMODITY SIGNALS",
]


async def run_fundamentals_agent(
    commodity_config: dict,
    llm_client:       LLMClient,
    weather_zscores:  list[float] | None = None,
    geo_surprise:     float = 0.5,
) -> AgentThesis:
    """
    Run the supply/demand fundamentals agent.
    weather_zscores and geo_surprise are passed by the graph node after
    parallel agent execution; they default to neutral when not available.
    Never raises.
    """
    symbol   = commodity_config.get("symbol", "")
    name     = commodity_config["name"]
    ticker   = commodity_config["ticker"]
    category = commodity_config["category"]
    drivers  = commodity_config["key_drivers"]
    weights  = commodity_config["shock_weights"]
    coint_partners = commodity_config.get("cointegrated_with", [])

    logger.info("[FundAgent] Starting for %s (category=%s)", name, category)

    # ── 1. Price data ─────────────────────────────────────────────────────────
    price_data = fetch_price_data(ticker)
    weekly_returns = price_data["weekly_returns"]
    prices_1wk     = price_data["prices_1wk"]
    sources_used   = ["yfinance"]

    # ── 2. HMM regime ─────────────────────────────────────────────────────────
    hmm_result = hmm_regime(weekly_returns)
    regime_label = hmm_result.get("regime_label", "unknown")
    regime_conf  = hmm_result.get("confidence", 0.5)

    # ── 3. Category-specific fundamental data ─────────────────────────────────
    fundamental_text = ""
    inventory_dev    = 0.0

    if category == "agricultural":
        # NASS QuickStats replaces the discontinued WASDE plain-text endpoint.
        # Maps wheat/corn/soybeans/sugar → US production + ending stocks (3 years).
        nass_commodity = symbol if symbol in ("wheat", "corn", "soybeans", "sugar") else None
        if nass_commodity:
            nass_data = fetch_nass_supply_data(nass_commodity)
            if not nass_data.get("error") and (nass_data["production"] or nass_data["stocks"]):
                unit = "TONS" if nass_commodity == "sugar" else "BU"
                fundamental_text = format_nass_summary(nass_data, nass_commodity, unit)
                sources_used.append("USDA NASS")
                logger.info("[FundAgent] NASS data fetched for %s (%d prod rows, %d stock rows)",
                            symbol, len(nass_data["production"]), len(nass_data["stocks"]))
            else:
                fundamental_text = (
                    f"USDA NASS data unavailable for {symbol}: "
                    f"{nass_data.get('error', 'no data')}"
                )
                logger.warning("[FundAgent] NASS fetch failed for %s: %s",
                               symbol, nass_data.get("error"))
        else:
            fundamental_text = f"No USDA supply data available for {symbol} (not an NASS-tracked commodity)."

    elif category == "energy":
        if symbol == "oil_brent":
            inv_data = fetch_eia_crude_inventory()
            if inv_data["values"] and not inv_data["error"]:
                inventory_dev = compute_inventory_deviation(inv_data["values"])
                vals_str = ", ".join(f"{v:.0f}" for v in inv_data["values"][:5])
                fundamental_text = (
                    f"EIA US Crude Oil Inventories (weekly, MMbbl, most recent first):\n"
                    f"  Values: {vals_str}\n"
                    f"  Deviation from recent average: {inventory_dev:+.1%}\n"
                    f"  {'Below average (bullish supply)' if inventory_dev < 0 else 'Above average (bearish supply)'}"
                )
                sources_used.append("EIA")
            else:
                fundamental_text = f"EIA inventory data unavailable: {inv_data.get('error', 'unknown error')}"

        elif symbol == "natural_gas":
            inv_data = fetch_eia_natgas_storage()
            if inv_data["values"] and not inv_data["error"]:
                inventory_dev = compute_inventory_deviation(inv_data["values"])
                vals_str = ", ".join(f"{v:.0f}" for v in inv_data["values"][:5])
                fundamental_text = (
                    f"EIA US Natural Gas Storage (weekly, Bcf, most recent first):\n"
                    f"  Values: {vals_str}\n"
                    f"  Deviation from recent average: {inventory_dev:+.1%}\n"
                    f"  {'Below 5-period avg (bullish)' if inventory_dev < 0 else 'Above 5-period avg (bearish)'}"
                )
                sources_used.append("EIA")
            else:
                fundamental_text = f"EIA storage data unavailable: {inv_data.get('error', 'unknown error')}"

    elif category == "industrial_metals":
        fundamental_text = (
            "LME warehouse data: Proxy via news search. "
            "Key driver: China PMI and property sector activity."
        )

    elif category == "precious_metals":
        fundamental_text = (
            f"Supply/demand fundamentals for {name} are primarily driven by "
            "investment flows, central bank policy, and ETF positioning rather "
            "than physical inventories. See geopolitical agent for flow analysis."
        )

    # ── 4. Baltic Dry Index ───────────────────────────────────────────────────
    bdi_value = fetch_baltic_dry_index()
    bdi_text  = f"Baltic Dry Index: {bdi_value:.0f}" if bdi_value else "Baltic Dry Index: unavailable"
    if bdi_value:
        sources_used.append("Baltic Dry Index")

    # ── 5. Supply Shock Index ─────────────────────────────────────────────────
    w_zscores = weather_zscores or []
    ssi_result = supply_shock_index(
        weather_zscores=w_zscores,
        geo_event_rate=geo_surprise,
        inventory_dev=inventory_dev,
        weights=weights,
    )

    # ── 6. Cointegration ──────────────────────────────────────────────────────
    coint_results: list[dict] = []
    for partner_sym in coint_partners:
        if partner_sym not in COMMODITIES:
            continue
        partner_cfg    = COMMODITIES[partner_sym]
        partner_ticker = partner_cfg["ticker"]
        try:
            partner_data  = fetch_price_data(partner_ticker)
            partner_1wk   = partner_data["prices_1wk"]
            result        = cointegration_spread(
                prices_a=prices_1wk,
                prices_b=partner_1wk,
                symbol_a=symbol,
                symbol_b=partner_sym,
            )
            coint_results.append(result)
            if result.get("cointegrated") and result.get("mean_reverting"):
                logger.info(
                    "[FundAgent] Cointegration signal: %s vs %s z=%.2f — %s",
                    symbol, partner_sym,
                    result.get("spread_zscore", 0), result.get("signal", "")
                )
        except Exception as exc:
            logger.warning("[FundAgent] Cointegration(%s, %s) failed: %s",
                           symbol, partner_sym, exc)

    # ── 7. Format cointegration summary for LLM ───────────────────────────────
    def _fmt_coint(r: dict) -> str:
        if not r.get("cointegrated"):
            return f"  {r.get('pair', '?')}: Not cointegrated (p={r.get('pvalue', 1.0):.3f})"
        z   = r.get("spread_zscore", 0.0)
        sig = r.get("signal", "within_normal_range")
        mr  = "MEAN REVERSION SIGNAL" if r.get("mean_reverting") else "within normal range"
        return (
            f"  {r.get('pair', '?')}: Cointegrated (p={r.get('pvalue', 0):.4f}) | "
            f"Spread z={z:+.2f} | {sig} | {mr}"
        )

    coint_text = "\n".join(_fmt_coint(r) for r in coint_results) or "  No cointegrated pairs configured."

    # ── 8. LLM prompt ─────────────────────────────────────────────────────────
    prompt = f"""You are a commodity supply/demand analyst covering {name}.

PRICE CONTEXT:
  Current price:  {price_data['current_price']} {commodity_config['unit']}
  24h change:     {price_data['change_24h_pct']:+.2f}%
  7d change:      {price_data['change_7d_pct']:+.2f}%

MARKET REGIME (HMM — 2-state model on 5-year weekly returns):
  Regime:       {regime_label}
  Confidence:   {regime_conf:.1%}
  Implication:  {'Elevated volatility — news events have larger price impact' if 'high' in regime_label else 'Low volatility — markets are range-bound, news is quickly absorbed'}

SUPPLY SHOCK INDEX: {ssi_result['ssi']:.3f} ({ssi_result['level'].upper()})
  Direction:              {ssi_result['direction'].upper()}
  Weather component:      {ssi_result['weather_component']:+.3f}
  Geopolitical component: {ssi_result['geo_component']:+.3f}
  Inventory component:    {ssi_result['inventory_component']:+.3f}
  Scale: |SSI|<1 normal, 1-2 elevated, 2-3 high, >3 extreme (2022 wheat shock = ~3.5)

SUPPLY/DEMAND FUNDAMENTAL DATA:
{fundamental_text}

{bdi_text}

CROSS-COMMODITY COINTEGRATION:
{coint_text}

KEY FUNDAMENTAL DRIVERS:
{chr(10).join(f'  - {d}' for d in drivers)}

Produce your analysis with EXACTLY these labelled sections:

DIRECTION: [BULLISH | BEARISH | NEUTRAL]
CONFIDENCE: [0.0-1.0]

HEADLINE:
[One sentence — the most important fundamental development right now]

SUPPLY ANALYSIS:
[Current supply conditions, production estimates, any disruptions]

DEMAND ANALYSIS:
[Demand trends, key buyers/consumers, substitution dynamics]

STOCKS/INVENTORY:
[Current inventory levels vs historical, implications for price sensitivity. \
'Low stocks = high price sensitivity to any supply shock']

SSI INTERPRETATION:
[What does an SSI of {ssi_result['ssi']:.2f} ({ssi_result['level']}) mean for \
{name}? How does the current market regime affect this reading?]

CROSS-COMMODITY SIGNALS:
[Any cointegration divergences worth noting? If pairs are within normal range, \
say so briefly.]

Be specific with numbers. Clearly state when data is unavailable. \
Do not invent inventory figures."""

    # ── 9. LLM call ───────────────────────────────────────────────────────────
    try:
        response_text = await llm_client.agent(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            purpose="fundamentals_agent",
            max_tokens=1100,
        )
    except Exception as exc:
        logger.error("[FundAgent] LLM call failed for %s: %s", name, exc)
        return _neutral_thesis(name, str(exc))

    # ── 10. Parse response ────────────────────────────────────────────────────
    direction  = parse_direction(response_text)
    confidence = parse_confidence(response_text)
    headline   = _first_meaningful_line(parse_section(response_text, "HEADLINE", _ALL_HEADERS))
    supply_txt = parse_section(response_text, "SUPPLY ANALYSIS", _ALL_HEADERS)
    demand_txt = parse_section(response_text, "DEMAND ANALYSIS", _ALL_HEADERS)
    inv_txt    = parse_section(response_text, "STOCKS/INVENTORY", _ALL_HEADERS)
    ssi_txt    = parse_section(response_text, "SSI INTERPRETATION", _ALL_HEADERS)
    cross_txt  = parse_section(response_text, "CROSS-COMMODITY SIGNALS", ["END", ""])

    logger.info(
        "[FundAgent] %s → %s (conf=%.2f, SSI=%.3f %s, regime=%s)",
        name, direction, confidence,
        ssi_result["ssi"], ssi_result["level"], regime_label
    )

    return AgentThesis(
        agent="fundamentals",
        direction=direction,
        confidence=confidence,
        headline=headline or f"Fundamentals analysis of {name}",
        supply_analysis=supply_txt,
        demand_analysis=demand_txt,
        inventory_analysis=inv_txt + "\n\nSSI: " + ssi_txt,
        ssi_result=ssi_result,
        hmm_result=hmm_result,
        cointegration_results=coint_results,
    )


def _neutral_thesis(name: str, error: str) -> AgentThesis:
    return AgentThesis(
        agent="fundamentals",
        direction="NEUTRAL",
        confidence=0.0,
        headline=f"Fundamental data unavailable for {name}",
        error=error,
    )
