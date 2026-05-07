"""
candlestick_patterns.py
-----------------------
Detects candlestick patterns from OHLCV data across all 3 timeframes.
Uses the same multi-exchange fallback chain as technical_analysis.py.

Patterns detected:
  Single candle:
    - Doji (indecision)
    - Hammer (bullish reversal)
    - Inverted Hammer (bullish reversal)
    - Shooting Star (bearish reversal)
    - Hanging Man (bearish reversal)
    - Marubozu Bullish (strong bull)
    - Marubozu Bearish (strong bear)
    - Spinning Top (indecision)

  Double candle:
    - Bullish Engulfing
    - Bearish Engulfing
    - Bullish Harami
    - Bearish Harami
    - Tweezer Bottom (bullish)
    - Tweezer Top (bearish)
    - Piercing Line (bullish)
    - Dark Cloud Cover (bearish)

  Triple candle:
    - Morning Star (bullish reversal)
    - Evening Star (bearish reversal)
    - Three White Soldiers (strong bullish)
    - Three Black Crows (strong bearish)
    - Three Inside Up (bullish)
    - Three Inside Down (bearish)

Setup:
  pip install requests pandas numpy python-dotenv

No API key needed.
"""

import pandas as pd
import numpy as np
from TechnicalAnalysis import fetch_ohlcv, normalize_symbol, TIMEFRAMES


# ─────────────────────────────────────────────────────────
# CANDLE MATH HELPERS
# ─────────────────────────────────────────────────────────

def body(o, c):
    """Absolute candle body size."""
    return abs(c - o)

def upper_shadow(o, h, c):
    """Upper wick size."""
    return h - max(o, c)

def lower_shadow(o, l, c):
    """Lower wick size."""
    return min(o, c) - l

def is_bullish(o, c):
    return c > o

def is_bearish(o, c):
    return c < o

def candle_range(h, l):
    return h - l

def avg_body(df: pd.DataFrame, lookback: int = 14) -> float:
    """Average body size over last N candles — used to filter noise."""
    bodies = abs(df["close"] - df["open"]).tail(lookback)
    return bodies.mean()


# ─────────────────────────────────────────────────────────
# SINGLE CANDLE PATTERNS
# ─────────────────────────────────────────────────────────

def detect_doji(o, h, l, c, avg_body_size: float) -> dict | None:
    """
    Doji: open ≈ close (body very small relative to range).
    Signals indecision — potential reversal in either direction.
    Threshold: body < 10% of candle range.
    """
    b = body(o, c)
    r = candle_range(h, l)
    if r == 0:
        return None
    if b / r < 0.1:
        return {
            "pattern": "Doji",
            "type": "single",
            "signal": "neutral",
            "strength": "medium",
            "reason": "Doji — open ≈ close, market indecision, watch for reversal",
        }
    return None


def detect_hammer(o, h, l, c, avg_body_size: float) -> dict | None:
    """
    Hammer: small body at top, long lower wick (≥2x body), tiny upper wick.
    Must appear after a downtrend. Bullish reversal signal.
    """
    b = body(o, c)
    ls = lower_shadow(o, l, c)
    us = upper_shadow(o, h, c)
    if b == 0:
        return None
    if ls >= 2 * b and us <= 0.3 * b and b >= 0.3 * avg_body_size:
        return {
            "pattern": "Hammer",
            "type": "single",
            "signal": "bullish",
            "strength": "strong",
            "reason": "Hammer — long lower wick shows buyers rejected lower prices",
        }
    return None


def detect_inverted_hammer(o, h, l, c, avg_body_size: float) -> dict | None:
    """
    Inverted Hammer: small body at bottom, long upper wick (≥2x body), tiny lower wick.
    Bullish reversal — buyers tried to push up.
    """
    b = body(o, c)
    us = upper_shadow(o, h, c)
    ls = lower_shadow(o, l, c)
    if b == 0:
        return None
    if us >= 2 * b and ls <= 0.3 * b and b >= 0.3 * avg_body_size:
        return {
            "pattern": "Inverted Hammer",
            "type": "single",
            "signal": "bullish",
            "strength": "medium",
            "reason": "Inverted Hammer — buyers pushed up despite opening low",
        }
    return None


def detect_shooting_star(o, h, l, c, avg_body_size: float) -> dict | None:
    """
    Shooting Star: small body at bottom, long upper wick (≥2x body), tiny lower wick.
    Must appear after an uptrend. Bearish reversal signal.
    Same shape as Inverted Hammer but in uptrend context.
    """
    b = body(o, c)
    us = upper_shadow(o, h, c)
    ls = lower_shadow(o, l, c)
    if b == 0:
        return None
    if us >= 2 * b and ls <= 0.3 * b and is_bearish(o, c) and b >= 0.3 * avg_body_size:
        return {
            "pattern": "Shooting Star",
            "type": "single",
            "signal": "bearish",
            "strength": "strong",
            "reason": "Shooting Star — sellers rejected higher prices, bearish reversal likely",
        }
    return None


def detect_hanging_man(o, h, l, c, avg_body_size: float) -> dict | None:
    """
    Hanging Man: same shape as Hammer but appears after uptrend.
    Bearish reversal warning.
    """
    b = body(o, c)
    ls = lower_shadow(o, l, c)
    us = upper_shadow(o, h, c)
    if b == 0:
        return None
    if ls >= 2 * b and us <= 0.3 * b and is_bearish(o, c) and b >= 0.3 * avg_body_size:
        return {
            "pattern": "Hanging Man",
            "type": "single",
            "signal": "bearish",
            "strength": "medium",
            "reason": "Hanging Man — despite recovery, sellers showed up after uptrend",
        }
    return None


def detect_marubozu(o, h, l, c, avg_body_size: float) -> dict | None:
    """
    Marubozu: large body with very small or no wicks.
    Shows strong conviction in one direction.
    Threshold: body > 90% of candle range.
    """
    b = body(o, c)
    r = candle_range(h, l)
    if r == 0:
        return None
    if b / r > 0.9 and b >= avg_body_size:
        if is_bullish(o, c):
            return {
                "pattern": "Bullish Marubozu",
                "type": "single",
                "signal": "bullish",
                "strength": "very strong",
                "reason": "Bullish Marubozu — strong buying pressure, no upper/lower rejection",
            }
        else:
            return {
                "pattern": "Bearish Marubozu",
                "type": "single",
                "signal": "bearish",
                "strength": "very strong",
                "reason": "Bearish Marubozu — strong selling pressure, no upper/lower rejection",
            }
    return None


def detect_spinning_top(o, h, l, c, avg_body_size: float) -> dict | None:
    """
    Spinning Top: small body with upper and lower wicks both larger than body.
    Similar to Doji but body is slightly larger. Signals indecision.
    """
    b = body(o, c)
    us = upper_shadow(o, h, c)
    ls = lower_shadow(o, l, c)
    if b == 0:
        return None
    if us > b and ls > b and b < avg_body_size * 0.5:
        return {
            "pattern": "Spinning Top",
            "type": "single",
            "signal": "neutral",
            "strength": "weak",
            "reason": "Spinning Top — indecision, neither bulls nor bears in control",
        }
    return None


# ─────────────────────────────────────────────────────────
# DOUBLE CANDLE PATTERNS
# ─────────────────────────────────────────────────────────

def detect_engulfing(prev, curr, avg_body_size: float) -> dict | None:
    """
    Engulfing: current candle body completely engulfs previous candle body.
    Bullish engulfing: bearish prev + bullish curr that engulfs it.
    Bearish engulfing: bullish prev + bearish curr that engulfs it.
    """
    po, pc = prev["open"], prev["close"]
    co, cc = curr["open"], curr["close"]

    prev_body_top    = max(po, pc)
    prev_body_bottom = min(po, pc)
    curr_body_top    = max(co, cc)
    curr_body_bottom = min(co, cc)

    engulfs = curr_body_top > prev_body_top and curr_body_bottom < prev_body_bottom

    if engulfs and is_bearish(po, pc) and is_bullish(co, cc):
        return {
            "pattern": "Bullish Engulfing",
            "type": "double",
            "signal": "bullish",
            "strength": "very strong",
            "reason": "Bullish Engulfing — bulls completely overtook previous bearish candle",
        }
    if engulfs and is_bullish(po, pc) and is_bearish(co, cc):
        return {
            "pattern": "Bearish Engulfing",
            "type": "double",
            "signal": "bearish",
            "strength": "very strong",
            "reason": "Bearish Engulfing — bears completely overtook previous bullish candle",
        }
    return None


def detect_harami(prev, curr, avg_body_size: float) -> dict | None:
    """
    Harami: opposite of engulfing — current small body contained within previous large body.
    Bullish harami: large bearish prev + small bullish curr inside it.
    Bearish harami: large bullish prev + small bearish curr inside it.
    """
    po, pc = prev["open"], prev["close"]
    co, cc = curr["open"], curr["close"]

    prev_body_top    = max(po, pc)
    prev_body_bottom = min(po, pc)

    inside = max(co, cc) < prev_body_top and min(co, cc) > prev_body_bottom
    prev_large = body(po, pc) >= avg_body_size

    if inside and prev_large and is_bearish(po, pc) and is_bullish(co, cc):
        return {
            "pattern": "Bullish Harami",
            "type": "double",
            "signal": "bullish",
            "strength": "medium",
            "reason": "Bullish Harami — small bullish candle inside large bearish, momentum slowing",
        }
    if inside and prev_large and is_bullish(po, pc) and is_bearish(co, cc):
        return {
            "pattern": "Bearish Harami",
            "type": "double",
            "signal": "bearish",
            "strength": "medium",
            "reason": "Bearish Harami — small bearish candle inside large bullish, momentum slowing",
        }
    return None


def detect_tweezer(prev, curr, avg_body_size: float) -> dict | None:
    """
    Tweezer Bottom: two candles with matching lows — buyers defended same level twice.
    Tweezer Top: two candles with matching highs — sellers rejected same level twice.
    Threshold: lows/highs within 0.1% of each other.
    """
    threshold = 0.001  # 0.1%
    lows_match  = abs(prev["low"]  - curr["low"])  / max(prev["low"],  1) < threshold
    highs_match = abs(prev["high"] - curr["high"]) / max(prev["high"], 1) < threshold

    if lows_match and is_bearish(prev["open"], prev["close"]) and is_bullish(curr["open"], curr["close"]):
        return {
            "pattern": "Tweezer Bottom",
            "type": "double",
            "signal": "bullish",
            "strength": "medium",
            "reason": f"Tweezer Bottom — buyers defended {round(curr['low'], 4)} twice",
        }
    if highs_match and is_bullish(prev["open"], prev["close"]) and is_bearish(curr["open"], curr["close"]):
        return {
            "pattern": "Tweezer Top",
            "type": "double",
            "signal": "bearish",
            "strength": "medium",
            "reason": f"Tweezer Top — sellers rejected {round(curr['high'], 4)} twice",
        }
    return None


def detect_piercing_line(prev, curr, avg_body_size: float) -> dict | None:
    """
    Piercing Line: bearish prev + bullish curr that opens below prev low
    but closes above prev midpoint. Bullish reversal.
    """
    po, pc = prev["open"], prev["close"]
    co, cc = curr["open"], curr["close"]
    prev_mid = (po + pc) / 2

    if (is_bearish(po, pc) and is_bullish(co, cc) and
            co < prev["low"] and cc > prev_mid and cc < po):
        return {
            "pattern": "Piercing Line",
            "type": "double",
            "signal": "bullish",
            "strength": "strong",
            "reason": "Piercing Line — bulls pushed back above midpoint of bearish candle",
        }
    return None


def detect_dark_cloud_cover(prev, curr, avg_body_size: float) -> dict | None:
    """
    Dark Cloud Cover: bullish prev + bearish curr that opens above prev high
    but closes below prev midpoint. Bearish reversal.
    """
    po, pc = prev["open"], prev["close"]
    co, cc = curr["open"], curr["close"]
    prev_mid = (po + pc) / 2

    if (is_bullish(po, pc) and is_bearish(co, cc) and
            co > prev["high"] and cc < prev_mid and cc > po):
        return {
            "pattern": "Dark Cloud Cover",
            "type": "double",
            "signal": "bearish",
            "strength": "strong",
            "reason": "Dark Cloud Cover — bears pushed back below midpoint of bullish candle",
        }
    return None


# ─────────────────────────────────────────────────────────
# TRIPLE CANDLE PATTERNS
# ─────────────────────────────────────────────────────────

def detect_morning_star(c1, c2, c3, avg_body_size: float) -> dict | None:
    """
    Morning Star: large bearish + small body (gap down) + large bullish.
    Bullish reversal — one of the most reliable 3-candle patterns.
    """
    o1, c_1 = c1["open"], c1["close"]
    o2, c_2 = c2["open"], c2["close"]
    o3, c_3 = c3["open"], c3["close"]

    large_bear = is_bearish(o1, c_1) and body(o1, c_1) >= avg_body_size
    small_mid  = body(o2, c_2) < avg_body_size * 0.5
    large_bull = is_bullish(o3, c_3) and body(o3, c_3) >= avg_body_size
    bull_closes_above_mid = c_3 > (o1 + c_1) / 2

    if large_bear and small_mid and large_bull and bull_closes_above_mid:
        return {
            "pattern": "Morning Star",
            "type": "triple",
            "signal": "bullish",
            "strength": "very strong",
            "reason": "Morning Star — strong bullish reversal after downtrend, high reliability",
        }
    return None


def detect_evening_star(c1, c2, c3, avg_body_size: float) -> dict | None:
    """
    Evening Star: large bullish + small body (gap up) + large bearish.
    Bearish reversal — opposite of Morning Star.
    """
    o1, c_1 = c1["open"], c1["close"]
    o2, c_2 = c2["open"], c2["close"]
    o3, c_3 = c3["open"], c3["close"]

    large_bull = is_bullish(o1, c_1) and body(o1, c_1) >= avg_body_size
    small_mid  = body(o2, c_2) < avg_body_size * 0.5
    large_bear = is_bearish(o3, c_3) and body(o3, c_3) >= avg_body_size
    bear_closes_below_mid = c_3 < (o1 + c_1) / 2

    if large_bull and small_mid and large_bear and bear_closes_below_mid:
        return {
            "pattern": "Evening Star",
            "type": "triple",
            "signal": "bearish",
            "strength": "very strong",
            "reason": "Evening Star — strong bearish reversal after uptrend, high reliability",
        }
    return None


def detect_three_white_soldiers(c1, c2, c3, avg_body_size: float) -> dict | None:
    """
    Three White Soldiers: three consecutive large bullish candles,
    each opening within previous body and closing near highs.
    Very strong bullish continuation.
    """
    candles = [c1, c2, c3]
    for c in candles:
        if not is_bullish(c["open"], c["close"]):
            return None
        if body(c["open"], c["close"]) < avg_body_size * 0.7:
            return None
        if upper_shadow(c["open"], c["high"], c["close"]) > body(c["open"], c["close"]) * 0.3:
            return None

    # Each opens within previous body
    if (c2["open"] > c1["open"] and c2["open"] < c1["close"] and
            c3["open"] > c2["open"] and c3["open"] < c2["close"]):
        return {
            "pattern": "Three White Soldiers",
            "type": "triple",
            "signal": "bullish",
            "strength": "very strong",
            "reason": "Three White Soldiers — three strong bullish candles, powerful uptrend",
        }
    return None


def detect_three_black_crows(c1, c2, c3, avg_body_size: float) -> dict | None:
    """
    Three Black Crows: three consecutive large bearish candles,
    each opening within previous body and closing near lows.
    Very strong bearish continuation.
    """
    candles = [c1, c2, c3]
    for c in candles:
        if not is_bearish(c["open"], c["close"]):
            return None
        if body(c["open"], c["close"]) < avg_body_size * 0.7:
            return None
        if lower_shadow(c["open"], c["low"], c["close"]) > body(c["open"], c["close"]) * 0.3:
            return None

    if (c2["open"] < c1["open"] and c2["open"] > c1["close"] and
            c3["open"] < c2["open"] and c3["open"] > c2["close"]):
        return {
            "pattern": "Three Black Crows",
            "type": "triple",
            "signal": "bearish",
            "strength": "very strong",
            "reason": "Three Black Crows — three strong bearish candles, powerful downtrend",
        }
    return None


def detect_three_inside_up(c1, c2, c3, avg_body_size: float) -> dict | None:
    """
    Three Inside Up: bearish c1 + bullish harami c2 + bullish c3 closing above c1 open.
    Confirmed bullish reversal.
    """
    o1, c_1 = c1["open"], c1["close"]
    o2, c_2 = c2["open"], c2["close"]
    o3, c_3 = c3["open"], c3["close"]

    if (is_bearish(o1, c_1) and is_bullish(o2, c_2) and is_bullish(o3, c_3) and
            max(o2, c_2) < o1 and min(o2, c_2) > c_1 and
            c_3 > o1):
        return {
            "pattern": "Three Inside Up",
            "type": "triple",
            "signal": "bullish",
            "strength": "strong",
            "reason": "Three Inside Up — confirmed bullish reversal with follow-through",
        }
    return None


def detect_three_inside_down(c1, c2, c3, avg_body_size: float) -> dict | None:
    """
    Three Inside Down: bullish c1 + bearish harami c2 + bearish c3 closing below c1 open.
    Confirmed bearish reversal.
    """
    o1, c_1 = c1["open"], c1["close"]
    o2, c_2 = c2["open"], c2["close"]
    o3, c_3 = c3["open"], c3["close"]

    if (is_bullish(o1, c_1) and is_bearish(o2, c_2) and is_bearish(o3, c_3) and
            max(o2, c_2) < c_1 and min(o2, c_2) > o1 and
            c_3 < o1):
        return {
            "pattern": "Three Inside Down",
            "type": "triple",
            "signal": "bearish",
            "strength": "strong",
            "reason": "Three Inside Down — confirmed bearish reversal with follow-through",
        }
    return None


# ─────────────────────────────────────────────────────────
# PATTERN SCANNER — runs all detectors on recent candles
# ─────────────────────────────────────────────────────────

STRENGTH_SCORE = {
    "very strong": 3,
    "strong":      2,
    "medium":      1,
    "weak":        0.5,
}

def scan_patterns(df: pd.DataFrame, lookback: int = 10) -> list[dict]:
    """
    Scans the last `lookback` candles for all patterns.
    Returns list of detected patterns with candle index info.
    """
    recent = df.tail(lookback + 3).reset_index()
    avg_b = avg_body(df)
    detected = []

    for i in range(2, len(recent)):
        c1 = recent.iloc[i - 2]
        c2 = recent.iloc[i - 1]
        c3 = recent.iloc[i]

        candle_time = str(c3["timestamp"])

        # --- Single candle (on c3) ---
        single_fns = [
            detect_doji, detect_hammer, detect_inverted_hammer,
            detect_shooting_star, detect_hanging_man,
            detect_marubozu, detect_spinning_top,
        ]
        for fn in single_fns:
            result = fn(c3["open"], c3["high"], c3["low"], c3["close"], avg_b)
            if result:
                result["candle_time"] = candle_time
                result["candle_index"] = i
                detected.append(result)

        # --- Double candle (c2, c3) ---
        double_fns = [
            detect_engulfing, detect_harami, detect_tweezer,
            detect_piercing_line, detect_dark_cloud_cover,
        ]
        for fn in double_fns:
            result = fn(c2, c3, avg_b)
            if result:
                result["candle_time"] = candle_time
                result["candle_index"] = i
                detected.append(result)

        # --- Triple candle (c1, c2, c3) ---
        triple_fns = [
            detect_morning_star, detect_evening_star,
            detect_three_white_soldiers, detect_three_black_crows,
            detect_three_inside_up, detect_three_inside_down,
        ]
        for fn in triple_fns:
            result = fn(c1, c2, c3, avg_b)
            if result:
                result["candle_time"] = candle_time
                result["candle_index"] = i
                detected.append(result)

    return detected


# ─────────────────────────────────────────────────────────
# AGGREGATE PATTERN SIGNALS
# ─────────────────────────────────────────────────────────

def aggregate_patterns(patterns: list[dict]) -> dict:
    """
    Aggregates all detected patterns into a bullish/bearish score.
    More recent patterns and stronger patterns get higher weight:
      - Recency weight: patterns on last candle = 2x, second last = 1.5x, older = 1x
      - Strength weight: very strong=3, strong=2, medium=1, weak=0.5
    """
    if not patterns:
        return {
            "bullish_pct": 0,
            "bearish_pct": 0,
            "neutral_pct": 100,
            "overall": "neutral",
            "pattern_count": 0,
        }

    max_index = max(p["candle_index"] for p in patterns)
    bull_score = 0
    bear_score = 0
    neut_score = 0

    for p in patterns:
        strength_w = STRENGTH_SCORE.get(p.get("strength", "medium"), 1)

        # Recency weight
        distance = max_index - p["candle_index"]
        recency_w = 2.0 if distance == 0 else 1.5 if distance == 1 else 1.0

        weight = strength_w * recency_w

        if p["signal"] == "bullish":
            bull_score += weight
        elif p["signal"] == "bearish":
            bear_score += weight
        else:
            neut_score += weight

    total = bull_score + bear_score + neut_score or 1
    bull_pct = round(bull_score / total * 100, 1)
    bear_pct = round(bear_score / total * 100, 1)
    neut_pct = round(neut_score / total * 100, 1)

    overall = "bullish" if bull_pct > bear_pct + 15 else \
              "bearish" if bear_pct > bull_pct + 15 else "neutral"

    return {
        "bullish_pct": bull_pct,
        "bearish_pct": bear_pct,
        "neutral_pct": neut_pct,
        "overall": overall,
        "pattern_count": len(patterns),
    }


# ─────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def run_candlestick_analysis(symbol: str) -> dict:
    """
    Full candlestick pattern analysis across all 3 timeframes.
    Uses same multi-exchange fallback as technical_analysis.py.
    Call this from main.py.

    Returns:
    {
      "symbol": "ICPUSDT",
      "timeframes": {
        "1h": {
          "source": "Binance",
          "patterns_detected": [
            { "pattern": "Bullish Engulfing", "signal": "bullish",
              "strength": "very strong", "reason": "...", "candle_time": "..." },
            ...
          ],
          "aggregate": {
            "bullish_pct": 75.0, "bearish_pct": 25.0,
            "overall": "bullish", "pattern_count": 3
          }
        },
        "4h": { ... },
        "1d": { ... }
      }
    }
    """
    normalized = normalize_symbol(symbol)
    print(f"\n🕯️  Candlestick analysis for: {normalized}")

    result = {"symbol": normalized, "timeframes": {}}

    for tf_label, tf_config in TIMEFRAMES.items():
        print(f"  [{tf_label}] Fetching data...", end=" ")
        df, source = fetch_ohlcv(normalized, tf_config["interval"], tf_config["limit"])

        if df.empty or len(df) < 10:
            print("insufficient data.")
            result["timeframes"][tf_label] = {"error": "Insufficient data"}
            continue

        print(f"{len(df)} candles from {source}.")

        patterns = scan_patterns(df, lookback=10)
        aggregate = aggregate_patterns(patterns)

        result["timeframes"][tf_label] = {
            "source": source,
            "patterns_detected": patterns,
            "aggregate": aggregate,
            "current_price": round(df["close"].iloc[-1], 6),
            "last_candle": str(df.index[-1]),
        }

        # Pretty print
        agg = aggregate
        if patterns:
            pattern_names = [p["pattern"] for p in patterns[-3:]]  # last 3
            print(f"  [{tf_label}] {agg['overall'].upper():8} | "
                  f"Bull: {agg['bullish_pct']}% | Bear: {agg['bearish_pct']}% | "
                  f"Patterns: {', '.join(pattern_names)}")
        else:
            print(f"  [{tf_label}] No patterns detected in last 10 candles")

    return result


# ─────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sym = input("Enter symbol (e.g. ICP, BTC, NEAR): ").strip()
    run_candlestick_analysis(sym)