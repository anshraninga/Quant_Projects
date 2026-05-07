"""
Step 8 end-to-end test — run the full LangGraph pipeline on wheat.
Requires a populated .env file with ANTHROPIC_API_KEY.

Usage:
    python tests/test_graph_wheat.py
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

from rag.vector_store import init_historical_events
from graph.graph import run_analysis


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_report(report) -> None:
    _banner(f"COMMODITY REPORT: {report.commodity_name} ({report.symbol.upper()})")
    print(f"  Generated:    {report.generated_at}")
    print(f"  Duration:     {report.duration_seconds:.1f}s")

    if report.errors:
        print(f"\n  ERRORS: {report.errors}")

    print(f"\n  EXECUTIVE SUMMARY:")
    print(f"  {report.executive_summary[:500]}")

    if report.final_recommendation:
        rec = report.final_recommendation
        print(f"\n  FINAL RECOMMENDATION:")
        print(f"    Direction:   {rec.direction}")
        print(f"    Conviction:  {rec.conviction:.2f}")
        print(f"    Horizon:     {rec.time_horizon}")
        print(f"    Upside risks:  {rec.upside_risks[:2]}")
        print(f"    Downside risks:{rec.downside_risks[:2]}")
        print(f"    Watch list:    {rec.watch_list[:2]}")

    print(f"\n  AGENT RESULTS:")
    for agent_name, thesis in [
        ("Geopolitical", report.geo_thesis),
        ("Weather",      report.weather_thesis),
        ("Fundamentals", report.fund_thesis),
    ]:
        if thesis:
            print(f"    {agent_name}: {thesis.direction} conf={thesis.confidence:.2f} "
                  f"RAG={thesis.rag_status}")
            print(f"      Headline: {thesis.headline[:100]}")

    if report.debate:
        d = report.debate
        print(f"\n  DEBATE: occurred={d.occurred} rounds={d.rounds}")
        if d.occurred:
            for r in d.transcript:
                print(f"    {r.challenger} vs {r.challenged}: {'REVISED' if r.revised else 'MAINTAINED'}")
        else:
            print(f"    Skipped: {d.skip_reason}")

    if report.historical_analog and report.historical_analog.found:
        a = report.historical_analog
        print(f"\n  HISTORICAL ANALOG: {a.event} ({a.date}) sim={a.similarity:.3f}")

    print(f"\n  QUANT SIGNALS:")
    if report.kalman:
        k = report.kalman
        print(f"    Kalman: trend={k.trend} sigma={k.signal_sigma:.2f} is_signal={k.is_signal}")
    if report.hmm:
        h = report.hmm
        print(f"    HMM: {h.regime_label} conf={h.confidence:.0%}")
    if report.ssi:
        s = report.ssi
        print(f"    SSI: {s.ssi:.3f} ({s.level}) direction={s.direction}")
    if report.bayesian_surprise:
        b = report.bayesian_surprise
        print(f"    Bayesian surprise: {b.surprise_score:.3f}")
    if report.cointegration:
        print(f"    Cointegration ({len(report.cointegration)} pairs):")
        for c in report.cointegration:
            if c.cointegrated:
                print(f"      {c.pair}: z={c.spread_zscore:+.2f} {c.signal}")

    print(f"\n  FULL REPORT (first 600 chars):")
    print(f"  {report.full_report_text[:600]}")


async def main():
    import config
    _banner("CommodityKing — End-to-End Graph Test (Wheat)")

    warnings = config.validate()
    if warnings:
        print("\nCONFIG WARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
    if not config.ANTHROPIC_API_KEY:
        print("\nFATAL: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print("\nInitialising RAG knowledge base...")
    n = init_historical_events()
    print(f"  Historical events loaded: {n}")

    print("\nRunning full graph analysis for WHEAT...")
    report = await run_analysis("wheat")

    _print_report(report)

    print("\n\nEnd-to-end graph test complete.")
    print("Review the report above, then confirm to proceed to Step 9 (FastAPI layer).")


if __name__ == "__main__":
    asyncio.run(main())
