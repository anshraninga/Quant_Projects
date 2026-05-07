"""
main.py
-------
The brain of the algo. Runs all 5 modules in parallel and combines
signals into a final analysis with trade suggestions.

Usage:
  python main.py

  Then type any symbol: ICP, BTC, NEAR, LINEA etc.

Output per timeframe (1h, 4h, 1d):
  - Bullish % / Bearish % / Neutral %
  - Overall confidence score
  - Trade suggestion: LONG / SHORT / STAY OUT
  - Stop Loss & Take Profit prices
  - Sentiment summary
  - Recent news headlines
  - Key reasons behind the call

Module weights (loaded from weights.json if available,
otherwise falls back to these defaults):
  Technical Analysis:    35%
  Candlestick Patterns:  20%
  Chart Patterns:        20%
  News Sentiment:        12.5%
  Social Sentiment:      12.5%

NOTE: Effective weights are adjusted dynamically at runtime by
signal_normalizer.py — uncertain modules self-attenuate automatically.

Setup:
  pip install requests pandas numpy python-dotenv transformers torch tweepy
"""

import os
import concurrent.futures
from dotenv import load_dotenv

from TechnicalAnalysis   import run_technical_analysis
from CandlestickAnalysis import run_candlestick_analysis
from ChartPatterns       import run_chart_analysis
from NewsAnalysis        import run_news_analysis
# FIX #1: was "run_sentiment_analysisa" (trailing 'a') — caused NameError at runtime,
# silently returning None for sentiment on every single run.
from SentimentAnalysis   import run_sentiment_analysis
from SignalNormalizer    import (
    normalize_technical,
    combine_signal_vectors,
    print_normalization_report,
    SignalVector,
)

load_dotenv()


# ─────────────────────────────────────────────────────────
# DEFAULT WEIGHTS — used as fallback if weights.json missing
# ─────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "technical":   0.35,
    "candlestick": 0.20,
    "chart":       0.20,
    "news":        0.125,
    "sentiment":   0.125,
}


def _load_weights() -> dict:
    """
    Loads per-timeframe optimized weights from weights.json.
    Falls back to default weights if file not found.
    Validates all weights on load — raises ValueError on bad data.
    """
    try:
        from weight_optimizer import load_weights
        weights_by_tf = load_weights()
        for tf, w in weights_by_tf.items():
            total = sum(w.values())
            if abs(total - 1.0) > 1e-3:
                raise ValueError(f"Weights for {tf} sum to {total:.4f}, not 1.0")
        return weights_by_tf
    except (ImportError, FileNotFoundError):
        return {"1h": dict(DEFAULT_WEIGHTS), "4h": dict(DEFAULT_WEIGHTS), "1d": dict(DEFAULT_WEIGHTS)}
    except ValueError as e:
        print(f"  ⚠️  Weight validation failed: {e} — using defaults")
        return {"1h": dict(DEFAULT_WEIGHTS), "4h": dict(DEFAULT_WEIGHTS), "1d": dict(DEFAULT_WEIGHTS)}


WEIGHTS_BY_TF = _load_weights()

# Minimum directional gap (after confidence adjustment) to make a trade call.
CONVICTION_THRESHOLD = 15.0


# ─────────────────────────────────────────────────────────
# PARALLEL RUNNER
# ─────────────────────────────────────────────────────────

def run_all_modules(symbol: str) -> dict:
    """
    Runs all 5 analysis modules in parallel using ThreadPoolExecutor.
    Returns dict with results from each module.
    ~15-20s total vs ~60s sequential.
    """
    print(f"\n{'='*60}")
    print(f"  🚀 Running full analysis for: {symbol.upper()}")
    print(f"{'='*60}")
    print("  Running all modules in parallel...\n")

    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            "technical":   executor.submit(run_technical_analysis,   symbol),
            "candlestick": executor.submit(run_candlestick_analysis,  symbol),
            "chart":       executor.submit(run_chart_analysis,        symbol),
            "news":        executor.submit(run_news_analysis,         symbol),
            "sentiment":   executor.submit(run_sentiment_analysis,    symbol),
        }

        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=120)
            except Exception as e:
                print(f"  ⚠️  {name} module failed: {e}")
                results[name] = None

    return results


# ─────────────────────────────────────────────────────────
# SIGNAL COMBINER
# ─────────────────────────────────────────────────────────

def _get_weights(timeframe: str) -> dict:
    """
    Returns the weight dict for a given timeframe.
    FIX #6: uses .get() with DEFAULT_WEIGHTS fallback so a missing key
    in weights.json never causes a KeyError downstream.
    """
    tf_weights = WEIGHTS_BY_TF.get(timeframe, WEIGHTS_BY_TF.get("1h", DEFAULT_WEIGHTS))
    # Ensure all 5 module keys are always present — guards against stale weights.json
    return {mod: tf_weights.get(mod, DEFAULT_WEIGHTS[mod]) for mod in DEFAULT_WEIGHTS}


def _safe_technical_vec(results: dict, module: str,
                         timeframe: str, weight: float) -> SignalVector:
    """
    Extracts bull/bear from a price-based module (technical/candlestick/chart)
    and wraps in a SignalVector via normalize_technical().
    Falls back to 50/50 neutral on any error.
    """
    try:
        r = results.get(module)
        if r is None:
            raise ValueError("module returned None")
        tf_data = r.get("timeframes", {}).get(timeframe, {})
        if "error" in tf_data:
            raise ValueError(tf_data["error"])
        agg = tf_data.get("aggregate", {})
        bull = agg.get("bullish_pct", 50.0)
        bear = agg.get("bearish_pct", 50.0)
        n    = len(tf_data.get("signals", []))
        return normalize_technical(bull, bear, weight, source=module, n_samples=n)
    except Exception:
        return normalize_technical(50.0, 50.0, weight, source=module)


def combine_signals(results: dict, timeframe: str) -> dict:
    """
    Builds a SignalVector for each module and combines them.

    Key improvements over the original:
      - BERT modules use pre-computed signal_vectors with temperature
        scaling and confidence weighting already applied.
      - Technical modules derive neutral mass from signal decisiveness.
      - Effective weights replace raw weights — uncertain modules
        self-attenuate so they can't inject noise at full weight.
      - Neutral is tracked explicitly — the masking issue where both
        bull and bear scores are low is now visible in the output.
    """
    weights = _get_weights(timeframe)

    vectors = {}

    # ── Price-based modules ──────────────────────────────
    vectors["technical"]   = _safe_technical_vec(results, "technical",   timeframe, weights["technical"])
    vectors["candlestick"] = _safe_technical_vec(results, "candlestick", timeframe, weights["candlestick"])
    vectors["chart"]       = _safe_technical_vec(results, "chart",       timeframe, weights["chart"])

    # ── News (FinBERT) ───────────────────────────────────
    try:
        news = results.get("news") or {}
        if "signal_vector" in news:
            sv = news["signal_vector"]
            # FIX #5: was hardcoded / 0.125 — now reads the base_weight the vector
            # was actually created with, stored in metadata by normalize_news().
            # Falls back to 0.125 if metadata key absent (backward compat).
            original_base = sv.metadata.get("base_weight", 0.125)
            if original_base > 0:
                sv.effective_weight = sv.effective_weight / original_base * weights["news"]
            vectors["news"] = sv
        else:
            ns   = news.get("news_sentiment", {})
            bull = ns.get("bullish_pct", 50.0)
            bear = ns.get("bearish_pct", 50.0)
            vectors["news"] = normalize_technical(bull, bear, weights["news"], source="news")
    except Exception:
        vectors["news"] = normalize_technical(50.0, 50.0, weights["news"], source="news")

    # ── Sentiment (CryptoBERT) ───────────────────────────
    try:
        sent = results.get("sentiment") or {}
        if "signal_vector" in sent:
            sv = sent["signal_vector"]
            # FIX #5: same fix as news — use stored base_weight from metadata.
            original_base = sv.metadata.get("base_weight", 0.125)
            if original_base > 0:
                sv.effective_weight = sv.effective_weight / original_base * weights["sentiment"]
            vectors["sentiment"] = sv
        else:
            bull = sent.get("combined_bullish_pct", 50.0)
            bear = sent.get("combined_bearish_pct", 50.0)
            vectors["sentiment"] = normalize_technical(bull, bear, weights["sentiment"], source="sentiment")
    except Exception:
        vectors["sentiment"] = normalize_technical(50.0, 50.0, weights["sentiment"], source="sentiment")

    return combine_signal_vectors(vectors, CONVICTION_THRESHOLD)


# ─────────────────────────────────────────────────────────
# TP / SL EXTRACTOR
# ─────────────────────────────────────────────────────────

def get_tp_sl(results: dict, timeframe: str, direction: str = "LONG") -> dict:
    """
    Gets Stop Loss and Take Profit levels, direction-aware.

    LONG:  SL below current price, TP above current price
    SHORT: SL above current price, TP below current price

    Primary source: ATR from TechnicalAnalysis.
    Fallback: chart pattern targets adjusted for direction.
    """
    is_short = direction == "SHORT"

    try:
        tech = results.get("technical")
        if tech:
            tf_data = tech.get("timeframes", {}).get(timeframe, {})
            atr     = tf_data.get("atr", {})
            price   = tf_data.get("current_price", 0)
            atr_val = atr.get("atr", 0)

            if price and atr_val:
                if is_short:
                    sl = round(price + 1.5 * atr_val, 6)
                    tp = round(price - 2.5 * atr_val, 6)
                else:
                    sl = round(price - 1.5 * atr_val, 6)
                    tp = round(price + 2.5 * atr_val, 6)
                return {"stop_loss": sl, "take_profit": tp, "source": "ATR"}
    except Exception:
        pass

    # Fallback: chart pattern target, adjusted for direction
    try:
        chart = results.get("chart")
        if chart:
            tf_data  = chart.get("timeframes", {}).get(timeframe, {})
            patterns = tf_data.get("chart_patterns", [])
            price    = tf_data.get("current_price", 0)
            for p in patterns:
                if p.get("target") and price:
                    target  = p["target"]
                    sl_dist = abs(target - price) * 0.4
                    if is_short:
                        return {
                            "stop_loss":   round(price + sl_dist, 6),
                            "take_profit": round(target, 6),
                            "source":      f"Chart Pattern ({p['pattern']})",
                        }
                    else:
                        return {
                            "stop_loss":   round(price - sl_dist, 6),
                            "take_profit": round(target, 6),
                            "source":      f"Chart Pattern ({p['pattern']})",
                        }
    except Exception:
        pass

    return {"stop_loss": "N/A", "take_profit": "N/A", "source": "unavailable"}


# ─────────────────────────────────────────────────────────
# REASON BUILDER
# ─────────────────────────────────────────────────────────

def build_reasons(results: dict, timeframe: str, direction: str) -> list[str]:
    """
    Pulls the strongest aligned signals from each module.
    Filters to only show signals that agree with the final direction.
    """
    reasons = []
    target_signal = "bullish" if direction == "LONG" else \
                    "bearish" if direction == "SHORT" else None

    try:
        tech_tf = results["technical"]["timeframes"].get(timeframe, {})
        for sig in tech_tf.get("signals", []):
            if target_signal is None or sig.get("signal") == target_signal:
                reasons.append(f"[{sig['indicator']}] {sig['reason']}")
    except Exception:
        pass

    try:
        cs_tf = results["candlestick"]["timeframes"].get(timeframe, {})
        for p in cs_tf.get("patterns_detected", [])[:3]:
            if target_signal is None or p.get("signal") == target_signal:
                reasons.append(f"[{p['pattern']}] {p['reason']}")
    except Exception:
        pass

    try:
        cp_tf = results["chart"]["timeframes"].get(timeframe, {})
        for p in cp_tf.get("chart_patterns", []):
            if target_signal is None or p.get("signal") == target_signal:
                reasons.append(f"[{p['pattern']}] {p['reason']}")
        sd = cp_tf.get("supply_demand", {})
        if sd.get("signal") == target_signal or target_signal is None:
            reasons.append(f"[Supply/Demand] {sd.get('reason', '')}")
    except Exception:
        pass

    try:
        news = results["news"]
        sv = news.get("signal_vector")
        if sv:
            reasons.append(
                f"[News/FinBERT] {sv.source} — "
                f"Bull: {sv.bullish_pct()}% | Bear: {sv.bearish_pct()}% | "
                f"Confidence: {sv.confidence:.2f} | EffW: {sv.effective_weight:.3f}"
            )
        else:
            ns = news.get("news_sentiment", {})
            reasons.append(
                f"[News] {ns.get('overall','').upper()} — "
                f"Bull: {ns.get('bullish_pct',0)}% | Bear: {ns.get('bearish_pct',0)}%"
            )
        for h in news.get("top_headlines", [])[:2]:
            reasons.append(f"  → [{h.get('label','').upper()}] {h.get('title','')[:70]}")
    except Exception:
        pass

    try:
        sent = results["sentiment"]
        sv   = sent.get("signal_vector")
        if sv:
            reasons.append(
                f"[Social/CryptoBERT] {sv.source} — "
                f"Bull: {sv.bullish_pct()}% | Bear: {sv.bearish_pct()}% | "
                f"Confidence: {sv.confidence:.2f} | EffW: {sv.effective_weight:.3f}"
            )
        else:
            reasons.append(
                f"[Social] Combined {sent.get('combined_overall','').upper()} — "
                f"Bull: {sent.get('combined_bullish_pct',0)}% | "
                f"Bear: {sent.get('combined_bearish_pct',0)}%"
            )
        reddit = sent.get("reddit", {})
        if reddit.get("posts_analyzed", 0) > 0:
            subs = ", ".join(f"r/{s}" for s in reddit.get("subreddits", []))
            reasons.append(
                f"  → Reddit: Bull {reddit['bullish_pct']}% | "
                f"Bear {reddit['bearish_pct']}% | "
                f"{reddit.get('posts_raw_count', 0)} posts ({subs})"
            )
            for p in reddit.get("sample_posts", [])[:2]:
                icon = "↑" if p["label"] == "bullish" else "↓" if p["label"] == "bearish" else "-"
                reasons.append(f"    · [{icon}] {p['text'][:70]}")
        else:
            st = sent.get("stocktwits", {})
            if st.get("posts_analyzed", 0) > 0:
                reasons.append(
                    f"  → StockTwits (fallback): Bull {st['bullish_pct']}% | "
                    f"Bear {st['bearish_pct']}% ({st['posts_analyzed']} posts)"
                )
    except Exception:
        pass

    return reasons[:12]


# ─────────────────────────────────────────────────────────
# OUTPUT FORMATTER
# ─────────────────────────────────────────────────────────

DIRECTION_EMOJI = {
    "LONG":     "🟢",
    "SHORT":    "🔴",
    "STAY OUT": "🟡",
}

TIMEFRAME_LABEL = {
    "1h": "1 HOUR",
    "4h": "4 HOUR",
    "1d": "1 DAY",
}


def print_timeframe_result(tf: str, combined: dict, tp_sl: dict,
                            reasons: list[str], price: float,
                            vectors: dict = None):
    emoji = DIRECTION_EMOJI.get(combined["direction"], "⚪")
    label = TIMEFRAME_LABEL.get(tf, tf.upper())

    print(f"\n{'─'*60}")
    print(f"  {emoji}  {label} ANALYSIS")
    print(f"{'─'*60}")
    print(f"  Current Price   : {price}")
    print(f"  Bullish         : {combined['bullish_pct']}%")
    print(f"  Bearish         : {combined['bearish_pct']}%")
    print(f"  Neutral         : {combined.get('neutral_pct', 0)}%")
    print(f"  Confidence      : {combined.get('overall_confidence', 0):.2f}")
    print(f"  Adj. Gap        : {abs(combined.get('adjusted_gap', combined['gap']))}%"
          f"  (raw gap: {abs(combined['gap'])}%)")
    print(f"  Suggestion      : {combined['direction']}")
    print(f"  Stop Loss       : {tp_sl['stop_loss']}")
    print(f"  Take Profit     : {tp_sl['take_profit']}")

    print(f"\n  📊 Module Breakdown (effective weight %):")
    total_eff = combined.get("total_eff_weight", 1)
    for mod, data in combined["module_breakdown"].items():
        conf      = data.get("confidence", 1.0)
        bull_bar  = "█" * int(data["bullish_pct"] / 10)
        bear_bar  = "█" * int(data["bearish_pct"] / 10)
        eff_share = data.get("weight_used", 0)
        print(f"    {mod:12} | Bull: {str(data['bullish_pct'])+'%':7} {bull_bar}")
        print(f"    {'':12} | Bear: {str(data['bearish_pct'])+'%':7} {bear_bar}")
        print(f"    {'':12} | Conf: {conf:.2f}  EffW share: {eff_share:.1f}%")

    if reasons:
        print(f"\n  💡 Key Reasons:")
        for r in reasons:
            print(f"    • {r}")

    if combined["direction"] == "STAY OUT":
        print(f"\n  ⚠️  Adjusted gap of {abs(combined.get('adjusted_gap', combined['gap']))}% "
              f"is below the {CONVICTION_THRESHOLD}% conviction threshold.")
        print(f"     Overall confidence: {combined.get('overall_confidence', 0):.2f} "
              f"— signals are too mixed or weak to take a position.")


def print_full_report(symbol: str, results: dict):
    """Prints the complete analysis report for all 3 timeframes."""

    print(f"\n{'═'*60}")
    print(f"  📋 FULL ANALYSIS REPORT — {symbol.upper()}")
    print(f"{'═'*60}")

    prices = {}
    try:
        for tf in ["1h", "4h", "1d"]:
            tf_data    = results["technical"]["timeframes"].get(tf, {})
            prices[tf] = tf_data.get("current_price", 0)
    except Exception:
        prices = {"1h": 0, "4h": 0, "1d": 0}

    timeframe_results = {}
    for tf in ["1h", "4h", "1d"]:
        combined = combine_signals(results, tf)
        tp_sl    = get_tp_sl(results, tf, combined["direction"])
        reasons  = build_reasons(results, tf, combined["direction"])
        timeframe_results[tf] = combined
        print_timeframe_result(tf, combined, tp_sl, reasons, prices.get(tf, "N/A"))

    print(f"\n{'═'*60}")
    print(f"  📌 SUMMARY")
    print(f"{'═'*60}")
    for tf, combined in timeframe_results.items():
        emoji = DIRECTION_EMOJI.get(combined["direction"], "⚪")
        print(
            f"  {TIMEFRAME_LABEL[tf]:8} → {emoji} {combined['direction']:10} "
            f"| Bull: {combined['bullish_pct']}% "
            f"| Bear: {combined['bearish_pct']}% "
            f"| Neut: {combined.get('neutral_pct',0)}% "
            f"| Conf: {combined.get('overall_confidence',0):.2f}"
        )

    print(f"\n{'═'*60}\n")


# ─────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  🤖 CRYPTO ANALYSIS ALGO")
    print("  Powered by FinBERT + CryptoBERT + Technical Analysis")
    print("  Signal normalization: Temperature scaling + Confidence weighting")
    print("═"*60)
    print("\n  Supports: any crypto symbol vs USDT")
    print("  Examples: ICP | BTC | NEAR | LINEA | ETHUSDT")
    print("  Type 'quit' to exit\n")

    try:
        import json, os
        if os.path.exists("weights.json"):
            with open("weights.json") as f:
                meta = json.load(f)
            print(f"  ✓ Using optimized weights from weights.json "
                  f"(generated {meta.get('generated_at','unknown')[:10]})\n")
        else:
            print("  ℹ️  weights.json not found — using default weights.\n"
                  "     Run weight_optimizer.py to generate optimized weights.\n")
    except Exception:
        pass

    while True:
        symbol = input("  Enter symbol: ").strip()

        if symbol.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 👋\n")
            break

        if not symbol:
            print("  ⚠️  Please enter a symbol.\n")
            continue

        try:
            results = run_all_modules(symbol)
            print_full_report(symbol, results)
        except KeyboardInterrupt:
            print("\n\n  Analysis interrupted.")
            break
        except Exception as e:
            print(f"\n  ❌ Error during analysis: {e}\n")

        print("\n" + "─"*60)
        another = input("  Analyse another symbol? (y/n): ").strip().lower()
        if another != "y":
            print("\n  Goodbye! 👋\n")
            break


if __name__ == "__main__":
    main()