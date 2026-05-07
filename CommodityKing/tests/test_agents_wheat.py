"""
Step 7 review test — run all three agents individually on wheat.
Requires a populated .env file with ANTHROPIC_API_KEY (and optionally NEWSAPI_KEY).

Usage:
    python tests/test_agents_wheat.py
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force UTF-8 output on Windows consoles that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from commodities_config import COMMODITIES
from llm_client import LLMClient
from agents.geo_agent import run_geo_agent
from agents.weather_agent import run_weather_agent
from agents.fundamentals_agent import run_fundamentals_agent
from rag.vector_store import init_historical_events


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _print_thesis(thesis) -> None:
    print(f"  agent:      {thesis.agent}")
    print(f"  direction:  {thesis.direction}")
    print(f"  confidence: {thesis.confidence:.2f}")
    print(f"  headline:   {thesis.headline}")
    if thesis.error:
        print(f"  ERROR:      {thesis.error}")
    if thesis.analysis_24h:
        print(f"\n  24H ANALYSIS:\n  {thesis.analysis_24h[:300]}")
    if thesis.analysis_7d:
        print(f"\n  7D NARRATIVE:\n  {thesis.analysis_7d[:300]}")
    if thesis.forecast_outlook:
        print(f"\n  FORECAST:\n  {thesis.forecast_outlook[:300]}")
    if thesis.supply_analysis:
        print(f"\n  SUPPLY:\n  {thesis.supply_analysis[:300]}")
    if thesis.demand_analysis:
        print(f"\n  DEMAND:\n  {thesis.demand_analysis[:300]}")
    if thesis.ssi_result:
        s = thesis.ssi_result
        print(f"\n  SSI: {s.get('ssi'):.3f} ({s.get('level')}) — {s.get('direction')}")
    if thesis.hmm_result:
        h = thesis.hmm_result
        print(f"  HMM: {h.get('regime_label')} (conf={h.get('confidence'):.2f})")
    if thesis.cointegration_results:
        print("  COINTEGRATION:")
        for r in thesis.cointegration_results:
            if r.get("cointegrated"):
                print(f"    {r.get('pair')}: z={r.get('spread_zscore'):+.2f} — {r.get('signal')}")
    if thesis.kalman_result:
        k = thesis.kalman_result
        print(f"  KALMAN: trend={k.get('trend')} signal_sigma={k.get('signal_sigma'):.2f}σ is_signal={k.get('is_signal')}")
    print(f"  surprise:   {thesis.surprise_score:.3f}")
    print(f"  n_articles: {thesis.n_articles}")
    if thesis.key_risks:
        print("  risks:")
        for r in thesis.key_risks:
            print(f"    - {r[:100]}")
    if thesis.critical_regions:
        print("  critical regions:")
        for r in thesis.critical_regions:
            print(f"    - {r[:100]}")
    if thesis.weather_zscores:
        print("  weather z-scores:")
        for region, zs in list(thesis.weather_zscores.items())[:3]:
            print(f"    {region}: temp_z={zs.get('temp_z',0):+.2f} precip_z={zs.get('precip_z',0):+.2f}")


async def main():
    _banner("CommodityKing — Agent Tests (Wheat)")

    # Validate env
    import config
    warnings = config.validate()
    if warnings:
        print("\nCONFIG WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    if not config.ANTHROPIC_API_KEY:
        print("\nFATAL: ANTHROPIC_API_KEY not set in .env — cannot run LLM agents")
        sys.exit(1)

    # Load commodity config
    cfg = dict(COMMODITIES["wheat"])
    cfg["symbol"] = "wheat"

    # Init RAG
    print("\nInitialising RAG knowledge base...")
    n = init_historical_events()
    print(f"  Historical events: {n} documents")

    # One LLM client shared across all agents (tracks cumulative cost)
    llm = LLMClient(symbol="wheat")

    # ── GEOPOLITICAL AGENT ────────────────────────────────────────────────────
    _banner("1. Geopolitical & Macro Agent")
    geo = await run_geo_agent(cfg, llm)
    _print_thesis(geo)

    # ── WEATHER AGENT ─────────────────────────────────────────────────────────
    _banner("2. Weather & Climate Agent")
    weather = await run_weather_agent(cfg, llm)
    _print_thesis(weather)

    # ── FUNDAMENTALS AGENT ────────────────────────────────────────────────────
    _banner("3. Supply/Demand Fundamentals Agent")

    # Collect weather z-scores for SSI
    w_zscores = []
    for region_data in weather.weather_zscores.values():
        pz = region_data.get("precip_z", 0.0)
        w_zscores.append(pz)

    fund = await run_fundamentals_agent(
        cfg, llm,
        weather_zscores=w_zscores,
        geo_surprise=geo.surprise_score,
    )
    _print_thesis(fund)

    # ── COST SUMMARY ──────────────────────────────────────────────────────────
    _banner("Cost Summary")
    summary = llm.cost_summary()
    print(f"  Total calls:    {summary['total_calls']}")
    print(f"  Total cost:     ${summary['total_cost_usd']:.5f}")
    print(f"  Input tokens:   {summary['total_input_tokens']:,}")
    print(f"  Output tokens:  {summary['total_output_tokens']:,}")
    print("\n  Per call:")
    for purpose, rec in summary["by_purpose"].items():
        print(f"    {purpose:25s}  ${rec['cost_usd']:.5f}  {rec['duration_s']:.1f}s")

    print("\n\nAll three agents completed successfully.")
    print("Review outputs above, then confirm to proceed to Step 8 (graph layer).")


if __name__ == "__main__":
    asyncio.run(main())
