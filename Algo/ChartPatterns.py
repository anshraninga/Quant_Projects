"""
chart_patterns.py
-----------------
Detects chart patterns and supply/demand zones from OHLCV data.
Uses same multi-exchange fallback chain as technical_analysis.py.

Chart Patterns:
  Reversal:
    - Head & Shoulders / Inverse Head & Shoulders
    - Double Top / Double Bottom
    - Triple Top / Triple Bottom
    - Rising Wedge / Falling Wedge
    - Cup & Handle

  Continuation:
    - Ascending Triangle
    - Descending Triangle
    - Symmetrical Triangle
    - Bull Flag / Bear Flag
    - Rectangle / Range

Supply & Demand Zones:
    - Demand zones (institutional buy clusters)
    - Supply zones (institutional sell clusters)
    - Zone strength scoring
    - Multi-timeframe zone analysis (1h, 4h, 1d)

Setup:
  pip install requests pandas numpy python-dotenv
"""

import numpy as np
import pandas as pd
from TechnicalAnalysis import fetch_ohlcv, normalize_symbol, TIMEFRAMES


# ─────────────────────────────────────────────────────────
# PIVOT POINT HELPERS
# Chart patterns are built on pivot highs and lows
# ─────────────────────────────────────────────────────────

def find_pivot_highs(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Find local pivot highs — candles where high is highest in window on both sides.
    window: number of candles to look left and right.
    """
    highs = df["high"]
    pivot = pd.Series(False, index=df.index)
    for i in range(window, len(df) - window):
        if highs.iloc[i] == highs.iloc[i - window: i + window + 1].max():
            pivot.iloc[i] = True
    return pivot


def find_pivot_lows(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Find local pivot lows — candles where low is lowest in window on both sides.
    """
    lows = df["low"]
    pivot = pd.Series(False, index=df.index)
    for i in range(window, len(df) - window):
        if lows.iloc[i] == lows.iloc[i - window: i + window + 1].min():
            pivot.iloc[i] = True
    return pivot


def get_pivot_high_values(df: pd.DataFrame, window: int = 5) -> list[dict]:
    """Returns list of {price, index, time} for each pivot high."""
    mask = find_pivot_highs(df, window)
    result = []
    for i, (idx, is_pivot) in enumerate(mask.items()):
        if is_pivot:
            result.append({"price": df["high"][idx], "index": i, "time": str(idx)})
    return result


def get_pivot_low_values(df: pd.DataFrame, window: int = 5) -> list[dict]:
    """Returns list of {price, index, time} for each pivot low."""
    mask = find_pivot_lows(df, window)
    result = []
    for i, (idx, is_pivot) in enumerate(mask.items()):
        if is_pivot:
            result.append({"price": df["low"][idx], "index": i, "time": str(idx)})
    return result


def prices_similar(a: float, b: float, threshold_pct: float = 0.02) -> bool:
    """Check if two prices are within threshold% of each other."""
    return abs(a - b) / max(a, 1e-10) < threshold_pct


# ─────────────────────────────────────────────────────────
# SUPPLY & DEMAND ZONES
# ─────────────────────────────────────────────────────────

def detect_supply_demand_zones(df: pd.DataFrame, lookback: int = 100) -> dict:
    """
    Identifies supply and demand zones using institutional order flow logic.

    DEMAND ZONE mechanism:
      - Find a base: 1-3 small candles consolidating in tight range
      - Followed immediately by a strong bullish impulse candle (body > 1.5x avg)
      - The base range = demand zone (unfilled buy orders)
      - Stronger if: price hasn't returned since, more recent, sharper impulse

    SUPPLY ZONE mechanism:
      - Find a base: 1-3 small candles consolidating in tight range
      - Followed immediately by a strong bearish impulse candle (body > 1.5x avg)
      - The base range = supply zone (unfilled sell orders)
      - Stronger if: price hasn't returned since, more recent, sharper impulse

    Zone strength scoring (0-100):
      +30  sharp impulse (body > 2x avg)
      +20  zone is fresh (price hasn't returned)
      +20  higher timeframe alignment
      +15  multiple touches/bounces
      +15  recent formation (last 30 candles)
    """
    recent = df.tail(lookback).reset_index()
    avg_body = abs(recent["close"] - recent["open"]).mean()
    current_price = df["close"].iloc[-1]
    n = len(recent)

    demand_zones = []
    supply_zones = []

    for i in range(1, n - 1):
        candle = recent.iloc[i]
        body_size = abs(candle["close"] - candle["open"])

        # ── DEMAND ZONE ──
        # Look for a strong bullish impulse candle
        if candle["close"] > candle["open"] and body_size > 1.5 * avg_body:
            # The base is the 1-2 candles before the impulse
            base_start = max(0, i - 2)
            base_candles = recent.iloc[base_start:i]

            if len(base_candles) > 0:
                zone_top    = base_candles["high"].max()
                zone_bottom = base_candles["low"].min()
                zone_mid    = (zone_top + zone_bottom) / 2

                # Only valid if base is tight (range < 50% of avg body)
                base_range = zone_top - zone_bottom
                if base_range < avg_body * 1.5:

                    # Strength scoring
                    strength = 0
                    impulse_ratio = body_size / avg_body
                    if impulse_ratio > 2.0:
                        strength += 30
                    elif impulse_ratio > 1.5:
                        strength += 15

                    # Fresh zone — price hasn't come back to it
                    future_lows = recent.iloc[i+1:]["low"]
                    is_fresh = (future_lows > zone_bottom).all()
                    if is_fresh:
                        strength += 20

                    # Recent formation
                    candles_ago = n - i
                    if candles_ago < 30:
                        strength += 15

                    # How many times has price bounced from this zone
                    touches = sum(
                        1 for low in recent["low"]
                        if zone_bottom <= low <= zone_top
                    )
                    if touches > 1:
                        strength += min(touches * 5, 15)

                    demand_zones.append({
                        "zone_top":    round(zone_top, 6),
                        "zone_bottom": round(zone_bottom, 6),
                        "zone_mid":    round(zone_mid, 6),
                        "strength":    min(strength, 100),
                        "is_fresh":    is_fresh,
                        "candles_ago": candles_ago,
                        "impulse_ratio": round(impulse_ratio, 2),
                        "type":        "demand",
                        "formed_at":   str(recent.iloc[i]["timestamp"]),
                    })

        # ── SUPPLY ZONE ──
        # Look for a strong bearish impulse candle
        if candle["open"] > candle["close"] and body_size > 1.5 * avg_body:
            base_start = max(0, i - 2)
            base_candles = recent.iloc[base_start:i]

            if len(base_candles) > 0:
                zone_top    = base_candles["high"].max()
                zone_bottom = base_candles["low"].min()
                zone_mid    = (zone_top + zone_bottom) / 2

                base_range = zone_top - zone_bottom
                if base_range < avg_body * 1.5:

                    strength = 0
                    impulse_ratio = body_size / avg_body
                    if impulse_ratio > 2.0:
                        strength += 30
                    elif impulse_ratio > 1.5:
                        strength += 15

                    future_highs = recent.iloc[i+1:]["high"]
                    is_fresh = (future_highs < zone_top).all()
                    if is_fresh:
                        strength += 20

                    candles_ago = n - i
                    if candles_ago < 30:
                        strength += 15

                    touches = sum(
                        1 for high in recent["high"]
                        if zone_bottom <= high <= zone_top
                    )
                    if touches > 1:
                        strength += min(touches * 5, 15)

                    supply_zones.append({
                        "zone_top":    round(zone_top, 6),
                        "zone_bottom": round(zone_bottom, 6),
                        "zone_mid":    round(zone_mid, 6),
                        "strength":    min(strength, 100),
                        "is_fresh":    is_fresh,
                        "candles_ago": candles_ago,
                        "impulse_ratio": round(impulse_ratio, 2),
                        "type":        "supply",
                        "formed_at":   str(recent.iloc[i]["timestamp"]),
                    })

    # Sort by strength
    demand_zones = sorted(demand_zones, key=lambda z: z["strength"], reverse=True)[:5]
    supply_zones  = sorted(supply_zones,  key=lambda z: z["strength"], reverse=True)[:5]

    # Find nearest zones to current price
    nearest_demand = next(
        (z for z in sorted(demand_zones, key=lambda z: abs(z["zone_mid"] - current_price))
         if z["zone_top"] < current_price), None
    )
    nearest_supply = next(
        (z for z in sorted(supply_zones, key=lambda z: abs(z["zone_mid"] - current_price))
         if z["zone_bottom"] > current_price), None
    )

    # Generate signal based on proximity to zones
    signal = "neutral"
    reason = "No nearby supply/demand zones"

    if nearest_demand and nearest_supply:
        dist_to_demand = (current_price - nearest_demand["zone_top"]) / current_price * 100
        dist_to_supply = (nearest_supply["zone_bottom"] - current_price) / current_price * 100

        if dist_to_demand < 1.5 and nearest_demand["strength"] >= 30:
            signal = "bullish"
            reason = (f"Price within {round(dist_to_demand, 2)}% of demand zone "
                      f"[{nearest_demand['zone_bottom']} - {nearest_demand['zone_top']}] "
                      f"(strength: {nearest_demand['strength']})")
        elif dist_to_supply < 1.5 and nearest_supply["strength"] >= 30:
            signal = "bearish"
            reason = (f"Price within {round(dist_to_supply, 2)}% of supply zone "
                      f"[{nearest_supply['zone_bottom']} - {nearest_supply['zone_top']}] "
                      f"(strength: {nearest_supply['strength']})")
        elif dist_to_demand < dist_to_supply:
            signal = "bullish"
            reason = f"Price closer to demand zone than supply zone"
        else:
            signal = "bearish"
            reason = f"Price closer to supply zone than demand zone"

    elif nearest_demand:
        dist = (current_price - nearest_demand["zone_top"]) / current_price * 100
        if dist < 2.0:
            signal = "bullish"
            reason = f"Price approaching demand zone [{nearest_demand['zone_bottom']} - {nearest_demand['zone_top']}]"

    elif nearest_supply:
        dist = (nearest_supply["zone_bottom"] - current_price) / current_price * 100
        if dist < 2.0:
            signal = "bearish"
            reason = f"Price approaching supply zone [{nearest_supply['zone_bottom']} - {nearest_supply['zone_top']}]"

    return {
        "demand_zones":    demand_zones,
        "supply_zones":    supply_zones,
        "nearest_demand":  nearest_demand,
        "nearest_supply":  nearest_supply,
        "current_price":   round(current_price, 6),
        "signal":          signal,
        "reason":          reason,
    }


# ─────────────────────────────────────────────────────────
# CHART PATTERN DETECTORS
# ─────────────────────────────────────────────────────────

def detect_head_and_shoulders(df: pd.DataFrame) -> dict | None:
    """
    Head & Shoulders (bearish reversal) / Inverse H&S (bullish reversal).

    H&S structure:
      Left shoulder (high) → Head (higher high) → Right shoulder (lower high ≈ left shoulder)
      Neckline = line connecting the two troughs between shoulders and head
      Signal: price breaks below neckline → bearish

    Inverse H&S: same but upside down → bullish
    """
    pivot_highs = get_pivot_high_values(df, window=5)
    pivot_lows  = get_pivot_low_values(df,  window=5)

    if len(pivot_highs) < 3 or len(pivot_lows) < 2:
        return None

    # Check last 3 pivot highs for H&S
    ph = pivot_highs[-3:]
    left, head, right = ph[0]["price"], ph[1]["price"], ph[2]["price"]

    # Head must be highest, shoulders roughly equal (within 3%)
    if (head > left and head > right and prices_similar(left, right, 0.03)):
        neckline = (pivot_lows[-2]["price"] + pivot_lows[-1]["price"]) / 2
        current  = df["close"].iloc[-1]
        target   = round(neckline - (head - neckline), 6)

        if current < neckline:
            return {
                "pattern":   "Head & Shoulders",
                "signal":    "bearish",
                "strength":  "very strong",
                "neckline":  round(neckline, 6),
                "target":    target,
                "reason":    f"H&S confirmed — price broke neckline at {round(neckline,4)}. Target: {target}",
            }
        else:
            return {
                "pattern":   "Head & Shoulders (forming)",
                "signal":    "bearish",
                "strength":  "strong",
                "neckline":  round(neckline, 6),
                "target":    target,
                "reason":    f"H&S forming — watch for neckline break at {round(neckline,4)}",
            }

    # Check for Inverse H&S using pivot lows
    pl = pivot_lows[-3:]
    if len(pl) >= 3:
        left_l, head_l, right_l = pl[0]["price"], pl[1]["price"], pl[2]["price"]
        if (head_l < left_l and head_l < right_l and prices_similar(left_l, right_l, 0.03)):
            neckline = (pivot_highs[-2]["price"] + pivot_highs[-1]["price"]) / 2
            current  = df["close"].iloc[-1]
            target   = round(neckline + (neckline - head_l), 6)

            if current > neckline:
                return {
                    "pattern":  "Inverse Head & Shoulders",
                    "signal":   "bullish",
                    "strength": "very strong",
                    "neckline": round(neckline, 6),
                    "target":   target,
                    "reason":   f"Inverse H&S confirmed — broke neckline at {round(neckline,4)}. Target: {target}",
                }
            else:
                return {
                    "pattern":  "Inverse Head & Shoulders (forming)",
                    "signal":   "bullish",
                    "strength": "strong",
                    "neckline": round(neckline, 6),
                    "target":   target,
                    "reason":   f"Inverse H&S forming — watch for neckline break at {round(neckline,4)}",
                }
    return None


def detect_double_top_bottom(df: pd.DataFrame) -> dict | None:
    """
    Double Top (bearish): two highs at roughly same level with a trough between.
    Double Bottom (bullish): two lows at roughly same level with a peak between.
    """
    pivot_highs = get_pivot_high_values(df, window=5)
    pivot_lows  = get_pivot_low_values(df,  window=5)
    current     = df["close"].iloc[-1]

    # Double Top
    if len(pivot_highs) >= 2:
        h1, h2 = pivot_highs[-2], pivot_highs[-1]
        if prices_similar(h1["price"], h2["price"], 0.02):
            # Find trough between the two highs
            between = df.iloc[h1["index"]:h2["index"]]
            if not between.empty:
                trough = between["low"].min()
                target = round(trough - (h1["price"] - trough), 6)
                if current < trough:
                    return {
                        "pattern":  "Double Top",
                        "signal":   "bearish",
                        "strength": "very strong",
                        "level":    round(h1["price"], 6),
                        "target":   target,
                        "reason":   f"Double Top at {round(h1['price'],4)} confirmed. Target: {target}",
                    }
                else:
                    return {
                        "pattern":  "Double Top (forming)",
                        "signal":   "bearish",
                        "strength": "strong",
                        "level":    round(h1["price"], 6),
                        "target":   target,
                        "reason":   f"Double Top forming at {round(h1['price'],4)} — watch for breakdown",
                    }

    # Double Bottom
    if len(pivot_lows) >= 2:
        l1, l2 = pivot_lows[-2], pivot_lows[-1]
        if prices_similar(l1["price"], l2["price"], 0.02):
            between = df.iloc[l1["index"]:l2["index"]]
            if not between.empty:
                peak   = between["high"].max()
                target = round(peak + (peak - l1["price"]), 6)
                if current > peak:
                    return {
                        "pattern":  "Double Bottom",
                        "signal":   "bullish",
                        "strength": "very strong",
                        "level":    round(l1["price"], 6),
                        "target":   target,
                        "reason":   f"Double Bottom at {round(l1['price'],4)} confirmed. Target: {target}",
                    }
                else:
                    return {
                        "pattern":  "Double Bottom (forming)",
                        "signal":   "bullish",
                        "strength": "strong",
                        "level":    round(l1["price"], 6),
                        "target":   target,
                        "reason":   f"Double Bottom forming at {round(l1['price'],4)} — watch for breakout",
                    }
    return None


def detect_triple_top_bottom(df: pd.DataFrame) -> dict | None:
    """
    Triple Top: three highs at same level — stronger than double top.
    Triple Bottom: three lows at same level — stronger than double bottom.
    """
    pivot_highs = get_pivot_high_values(df, window=5)
    pivot_lows  = get_pivot_low_values(df,  window=5)
    current     = df["close"].iloc[-1]

    # Triple Top
    if len(pivot_highs) >= 3:
        h1, h2, h3 = pivot_highs[-3], pivot_highs[-2], pivot_highs[-1]
        if prices_similar(h1["price"], h2["price"], 0.02) and prices_similar(h2["price"], h3["price"], 0.02):
            avg_high = (h1["price"] + h2["price"] + h3["price"]) / 3
            neckline = df.iloc[h1["index"]:h3["index"]]["low"].min()
            target   = round(neckline - (avg_high - neckline), 6)
            signal   = "bearish"
            confirmed = current < neckline
            return {
                "pattern":  "Triple Top" if confirmed else "Triple Top (forming)",
                "signal":   signal,
                "strength": "very strong",
                "level":    round(avg_high, 6),
                "neckline": round(neckline, 6),
                "target":   target,
                "reason":   (f"Triple Top {'confirmed' if confirmed else 'forming'} at "
                             f"{round(avg_high,4)}. Target: {target}"),
            }

    # Triple Bottom
    if len(pivot_lows) >= 3:
        l1, l2, l3 = pivot_lows[-3], pivot_lows[-2], pivot_lows[-1]
        if prices_similar(l1["price"], l2["price"], 0.02) and prices_similar(l2["price"], l3["price"], 0.02):
            avg_low  = (l1["price"] + l2["price"] + l3["price"]) / 3
            neckline = df.iloc[l1["index"]:l3["index"]]["high"].max()
            target   = round(neckline + (neckline - avg_low), 6)
            confirmed = current > neckline
            return {
                "pattern":  "Triple Bottom" if confirmed else "Triple Bottom (forming)",
                "signal":   "bullish",
                "strength": "very strong",
                "level":    round(avg_low, 6),
                "neckline": round(neckline, 6),
                "target":   target,
                "reason":   (f"Triple Bottom {'confirmed' if confirmed else 'forming'} at "
                             f"{round(avg_low,4)}. Target: {target}"),
            }
    return None


def detect_triangles(df: pd.DataFrame) -> dict | None:
    """
    Triangle patterns using linear regression on pivot highs and lows.

    Ascending Triangle:   flat top resistance + rising lows → bullish breakout
    Descending Triangle:  flat bottom support + falling highs → bearish breakdown
    Symmetrical Triangle: falling highs + rising lows → breakout either direction (follow trend)
    """
    pivot_highs = get_pivot_high_values(df, window=4)
    pivot_lows  = get_pivot_low_values(df,  window=4)

    if len(pivot_highs) < 3 or len(pivot_lows) < 3:
        return None

    ph = pivot_highs[-4:]
    pl = pivot_lows[-4:]

    high_prices = [p["price"] for p in ph]
    low_prices  = [p["price"] for p in pl]
    high_idx    = [p["index"] for p in ph]
    low_idx     = [p["index"] for p in pl]

    # Linear regression slopes
    high_slope = np.polyfit(high_idx, high_prices, 1)[0]
    low_slope  = np.polyfit(low_idx,  low_prices,  1)[0]

    flat_threshold = df["close"].mean() * 0.0005   # 0.05% of price = "flat"
    current = df["close"].iloc[-1]

    if abs(high_slope) < flat_threshold and low_slope > flat_threshold:
        resistance = np.mean(high_prices[-2:])
        target = round(resistance + (resistance - np.mean(low_prices[-2:])), 6)
        return {
            "pattern":    "Ascending Triangle",
            "signal":     "bullish",
            "strength":   "strong",
            "resistance": round(resistance, 6),
            "target":     target,
            "reason":     f"Ascending Triangle — flat resistance at {round(resistance,4)}, rising lows. Bullish breakout expected.",
        }

    if abs(low_slope) < flat_threshold and high_slope < -flat_threshold:
        support = np.mean(low_prices[-2:])
        target  = round(support - (np.mean(high_prices[-2:]) - support), 6)
        return {
            "pattern": "Descending Triangle",
            "signal":  "bearish",
            "strength": "strong",
            "support":  round(support, 6),
            "target":   target,
            "reason":   f"Descending Triangle — flat support at {round(support,4)}, falling highs. Bearish breakdown expected.",
        }

    if high_slope < -flat_threshold and low_slope > flat_threshold:
        apex_price = (np.mean(high_prices[-2:]) + np.mean(low_prices[-2:])) / 2
        # Direction follows prior trend
        prior_trend = df["close"].iloc[-20] < df["close"].iloc[-1]
        signal = "bullish" if prior_trend else "bearish"
        return {
            "pattern":  "Symmetrical Triangle",
            "signal":   signal,
            "strength": "medium",
            "apex":     round(apex_price, 6),
            "reason":   f"Symmetrical Triangle converging — breakout imminent, bias {'bullish' if prior_trend else 'bearish'} with trend",
        }

    return None


def detect_flags(df: pd.DataFrame) -> dict | None:
    """
    Bull Flag: strong bullish impulse (pole) → brief consolidation (flag) → continuation up
    Bear Flag: strong bearish impulse (pole) → brief consolidation (flag) → continuation down

    Detection:
      - Pole: 5+ candles moving strongly in one direction (>3% move)
      - Flag: 5-15 candles of slow counter-trend consolidation
    """
    if len(df) < 25:
        return None

    closes = df["close"].values
    n = len(closes)

    # Check for bull flag — strong up move followed by slight pullback
    pole_end   = n - 10
    pole_start = pole_end - 8
    flag_start = pole_end
    flag_end   = n - 1

    pole_move  = (closes[pole_end] - closes[pole_start]) / closes[pole_start]
    flag_move  = (closes[flag_end] - closes[flag_start]) / closes[flag_start]

    if pole_move > 0.04 and -0.03 < flag_move < 0.01:
        target = round(closes[-1] + (closes[pole_end] - closes[pole_start]), 6)
        return {
            "pattern":  "Bull Flag",
            "signal":   "bullish",
            "strength": "strong",
            "target":   target,
            "reason":   f"Bull Flag — {round(pole_move*100,1)}% pole, tight consolidation. Target: {target}",
        }

    # Bear flag
    if pole_move < -0.04 and -0.01 < flag_move < 0.03:
        target = round(closes[-1] - (closes[pole_start] - closes[pole_end]), 6)
        return {
            "pattern":  "Bear Flag",
            "signal":   "bearish",
            "strength": "strong",
            "target":   target,
            "reason":   f"Bear Flag — {round(pole_move*100,1)}% pole, tight consolidation. Target: {target}",
        }

    return None


def detect_wedge(df: pd.DataFrame) -> dict | None:
    """
    Rising Wedge (bearish): price moves up but highs and lows converging upward — losing momentum.
    Falling Wedge (bullish): price moves down but highs and lows converging downward — buyers stepping in.

    Both are detected via regression slopes on pivot highs and lows:
    Rising wedge:  both slopes positive, high slope < low slope (converging)
    Falling wedge: both slopes negative, high slope > low slope (converging)
    """
    pivot_highs = get_pivot_high_values(df, window=4)
    pivot_lows  = get_pivot_low_values(df,  window=4)

    if len(pivot_highs) < 3 or len(pivot_lows) < 3:
        return None

    ph = pivot_highs[-4:]
    pl = pivot_lows[-4:]
    high_slope = np.polyfit([p["index"] for p in ph], [p["price"] for p in ph], 1)[0]
    low_slope  = np.polyfit([p["index"] for p in pl], [p["price"] for p in pl], 1)[0]

    # Rising wedge: both positive, converging (low slope > high slope)
    if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
        return {
            "pattern":  "Rising Wedge",
            "signal":   "bearish",
            "strength": "strong",
            "reason":   "Rising Wedge — both highs and lows rising but converging. Bearish reversal likely.",
        }

    # Falling wedge: both negative, converging (high slope < low slope, both negative)
    if high_slope < 0 and low_slope < 0 and high_slope < low_slope:
        return {
            "pattern":  "Falling Wedge",
            "signal":   "bullish",
            "strength": "strong",
            "reason":   "Falling Wedge — both highs and lows falling but converging. Bullish reversal likely.",
        }

    return None


def detect_cup_and_handle(df: pd.DataFrame) -> dict | None:
    """
    Cup & Handle (bullish continuation):
      - Cup: rounded U-shaped bottom (smooth curve, not V-shape)
      - Handle: small pullback after cup right shoulder
      - Breakout above the cup lip = entry signal

    Detection uses a simplified approach:
      - Split lookback into thirds
      - Left rim ≈ right rim (within 3%)
      - Middle section (cup bottom) is lower than both rims
      - Handle: last 10% of pattern shows slight pullback
    """
    if len(df) < 60:
        return None

    recent = df.tail(60)
    closes = recent["close"].values
    n = len(closes)

    third = n // 3
    left_rim   = np.mean(closes[:5])
    cup_bottom = np.mean(closes[third: third * 2])
    right_rim  = np.mean(closes[-15:-5])
    handle     = np.mean(closes[-5:])
    current    = closes[-1]

    # Rims at similar level, cup bottom lower
    rims_similar = prices_similar(left_rim, right_rim, 0.04)
    cup_dip      = cup_bottom < left_rim * 0.97
    handle_pull  = handle < right_rim * 0.99   # handle slightly below rim

    if rims_similar and cup_dip and handle_pull:
        target = round(right_rim + (right_rim - cup_bottom), 6)
        return {
            "pattern":  "Cup & Handle",
            "signal":   "bullish",
            "strength": "very strong",
            "rim_level": round(right_rim, 6),
            "target":   target,
            "reason":   f"Cup & Handle forming — breakout above {round(right_rim,4)} targets {target}",
        }

    return None


def detect_rectangle(df: pd.DataFrame) -> dict | None:
    """
    Rectangle / Range: price bouncing between flat support and flat resistance.
    Continuation pattern — breakout direction follows prior trend.

    Detection: last 20-30 candles have multiple touches of both a high and low zone.
    """
    recent = df.tail(30)
    high_zone = recent["high"].quantile(0.9)
    low_zone  = recent["low"].quantile(0.1)
    current   = df["close"].iloc[-1]
    range_pct = (high_zone - low_zone) / low_zone * 100

    # Flat range: between 2% and 15% wide
    if 2 < range_pct < 15:
        high_touches = (recent["high"] >= high_zone * 0.99).sum()
        low_touches  = (recent["low"]  <= low_zone  * 1.01).sum()

        if high_touches >= 2 and low_touches >= 2:
            prior_trend = df["close"].iloc[-40] < df["close"].iloc[-30]
            signal      = "bullish" if prior_trend else "bearish"
            breakout_target_up   = round(high_zone + (high_zone - low_zone), 6)
            breakout_target_down = round(low_zone  - (high_zone - low_zone), 6)
            target = breakout_target_up if signal == "bullish" else breakout_target_down

            return {
                "pattern":    "Rectangle / Range",
                "signal":     signal,
                "strength":   "medium",
                "resistance": round(high_zone, 6),
                "support":    round(low_zone, 6),
                "target":     target,
                "reason":     (f"Rectangle range {round(range_pct,1)}% wide "
                               f"[{round(low_zone,4)} - {round(high_zone,4)}]. "
                               f"Bias {signal} with prior trend. Target: {target}"),
            }

    return None


# ─────────────────────────────────────────────────────────
# PATTERN SCANNER
# ─────────────────────────────────────────────────────────

CHART_DETECTORS = [
    detect_head_and_shoulders,
    detect_double_top_bottom,
    detect_triple_top_bottom,
    detect_triangles,
    detect_flags,
    detect_wedge,
    detect_cup_and_handle,
    detect_rectangle,
]

STRENGTH_SCORE = {"very strong": 3, "strong": 2, "medium": 1, "weak": 0.5}

def scan_chart_patterns(df: pd.DataFrame) -> list[dict]:
    """Runs all chart pattern detectors and returns found patterns."""
    patterns = []
    for detector in CHART_DETECTORS:
        try:
            result = detector(df)
            if result:
                patterns.append(result)
        except Exception as e:
            pass
    return patterns


def aggregate_chart_signals(patterns: list[dict], sd_signal: str) -> dict:
    """
    Combines chart pattern signals + supply/demand signal into final score.
    Supply/demand zone signal gets 1.5x weight — institutional level signal.
    """
    bull_score = 0
    bear_score = 0

    for p in patterns:
        w = STRENGTH_SCORE.get(p.get("strength", "medium"), 1)
        if p["signal"] == "bullish":
            bull_score += w
        elif p["signal"] == "bearish":
            bear_score += w

    # Supply/demand zone signal weighted 1.5x
    sd_weight = 1.5
    if sd_signal == "bullish":
        bull_score += sd_weight
    elif sd_signal == "bearish":
        bear_score += sd_weight

    total = bull_score + bear_score or 1
    bull_pct = round(bull_score / total * 100, 1)
    bear_pct = round(bear_score / total * 100, 1)

    overall = "bullish" if bull_pct > bear_pct + 15 else \
              "bearish" if bear_pct > bull_pct + 15 else "neutral"

    return {
        "bullish_pct":   bull_pct,
        "bearish_pct":   bear_pct,
        "neutral_pct":   round(100 - bull_pct - bear_pct, 1),
        "overall":       overall,
        "pattern_count": len(patterns),
    }


# ─────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def run_chart_analysis(symbol: str) -> dict:
    """
    Full chart pattern + supply/demand zone analysis across 1h, 4h, 1d.
    Call this from main.py.

    Returns:
    {
      "symbol": "ICPUSDT",
      "timeframes": {
        "1h": {
          "source": "Binance",
          "chart_patterns": [ { pattern, signal, strength, reason, target }, ... ],
          "supply_demand":  { demand_zones, supply_zones, nearest_demand,
                              nearest_supply, signal, reason },
          "aggregate":      { bullish_pct, bearish_pct, overall, pattern_count }
        },
        "4h": { ... },
        "1d": { ... }
      }
    }
    """
    normalized = normalize_symbol(symbol)
    print(f"\n📈 Chart pattern analysis for: {normalized}")

    result = {"symbol": normalized, "timeframes": {}}

    for tf_label, tf_config in TIMEFRAMES.items():
        print(f"  [{tf_label}] Fetching data...", end=" ")
        df, source = fetch_ohlcv(normalized, tf_config["interval"], tf_config["limit"])

        if df.empty or len(df) < 60:
            print("insufficient data.")
            result["timeframes"][tf_label] = {"error": "Insufficient data (need 60+ candles)"}
            continue

        print(f"{len(df)} candles from {source}.")

        chart_patterns = scan_chart_patterns(df)
        sd_zones       = detect_supply_demand_zones(df, lookback=100)
        aggregate      = aggregate_chart_signals(chart_patterns, sd_zones["signal"])

        result["timeframes"][tf_label] = {
            "source":         source,
            "chart_patterns": chart_patterns,
            "supply_demand":  sd_zones,
            "aggregate":      aggregate,
            "current_price":  round(df["close"].iloc[-1], 6),
            "last_candle":    str(df.index[-1]),
        }

        # Pretty print
        agg = aggregate
        pattern_names = [p["pattern"] for p in chart_patterns] or ["None detected"]
        print(f"  [{tf_label}] {agg['overall'].upper():8} | "
              f"Bull: {agg['bullish_pct']}% | Bear: {agg['bearish_pct']}% | "
              f"Patterns: {', '.join(pattern_names[:3])}")
        print(f"         S/D: {sd_zones['reason'][:80]}")

    return result


# ─────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sym = input("Enter symbol (e.g. ICP, BTC, NEAR): ").strip()
    run_chart_analysis(sym)