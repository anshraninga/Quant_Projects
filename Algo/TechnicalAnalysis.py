"""
technical_analysis.py
---------------------
Fetches OHLCV data with multi-exchange fallback chain:
  1. Binance        — best quality, most liquid pairs
  2. KuCoin         — 800+ pairs, many small caps
  3. OKX            — strong mid-cap coverage
  4. MEXC           — good for new/recent listings
  5. Gate.io        — wide altcoin coverage
  6. Bitget         — growing exchange
  7. Binance Alpha  — pre-listing / early stage tokens

Note: DexScreener was removed from the fallback chain. The free tier
does not provide OHLCV candle data, so it always returned an empty
DataFrame while still burning a real HTTP request. If/when a paid
DexScreener plan with OHLCV is available, re-add it here.

Computes:
  - RSI (14)
  - MACD (12/26/9)
  - Bollinger Bands (20, 2std)
  - EMA crossovers (9 / 21 / 50 / 200)
  - Volume analysis (vs average)
  - Support & Resistance levels
  - ATR (Average True Range)
  - OBV (On-Balance Volume)
  - Fibonacci Retracement levels

Indicator weights:
  Loaded from indicator_weights.json (produced by indicator_weight_optimizer.py).
  Falls back to hardcoded defaults if file not found.

No API key needed — all public endpoints are free.

Setup:
  pip install requests pandas numpy python-dotenv
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────
# INDICATOR WEIGHTS
# ─────────────────────────────────────────────────────────

_INDICATOR_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "indicator_weights.json"
)

_DEFAULT_INDICATOR_WEIGHTS = {
    "RSI":                1.5,
    "MACD":               1.5,
    "Bollinger Bands":    1.0,
    "EMA Crossovers":     2.0,
    "Volume":             1.0,
    "Support/Resistance": 1.5,
    "OBV":                1.0,
    "Fibonacci":          1.0,
}


def _load_indicator_weights() -> dict:
    if not os.path.exists(_INDICATOR_WEIGHTS_PATH):
        return dict(_DEFAULT_INDICATOR_WEIGHTS)

    try:
        from indicator_weight_optimizer import load_indicator_weights
        normalized = load_indicator_weights(_INDICATOR_WEIGHTS_PATH)
        total_default = sum(_DEFAULT_INDICATOR_WEIGHTS.values())
        rescaled = {k: round(v * total_default, 4) for k, v in normalized.items()}
        if any(v < 0 for v in rescaled.values()):
            raise ValueError("Negative indicator weight after rescaling.")
        print(f"  ✓ Loaded optimized indicator weights from indicator_weights.json")
        return rescaled
    except Exception as e:
        print(f"  ⚠️  Indicator weight load failed: {e} — using defaults.")
        return dict(_DEFAULT_INDICATOR_WEIGHTS)


INDICATOR_WEIGHTS = _load_indicator_weights()


# ─────────────────────────────────────────────────────────
# TIMEFRAME CONFIG
# ─────────────────────────────────────────────────────────

TIMEFRAMES = {
    "1h":  {"interval": "1h",  "limit": 200},
    "4h":  {"interval": "4h",  "limit": 200},
    "1d":  {"interval": "1d",  "limit": 200},
}


# ─────────────────────────────────────────────────────────
# SYMBOL HELPERS
# ─────────────────────────────────────────────────────────

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if symbol.endswith("USD") and not symbol.endswith("USDT"):
        symbol = symbol[:-3] + "USDT"
    if symbol.endswith("USDT") or symbol.endswith("BUSD"):
        return symbol
    KNOWN_BASES = {
        "BTC","ETH","BNB","SOL","XRP","ADA","DOGE","DOT","AVAX","MATIC",
        "LINK","UNI","ATOM","LTC","ETC","BCH","XLM","ALGO","VET","ICP",
        "FIL","THETA","XTZ","EOS","AAVE","COMP","MKR","SNX","NEAR","SAND",
        "MANA","AXS","GALA","ENJ","CHZ","FTM","HBAR","ONE","CELO","ZEC",
        "DASH","XMR","NEO",
    }
    if symbol in KNOWN_BASES:
        return symbol + "USDT"
    if any(symbol.endswith(q) for q in ["BTC","ETH","BNB"]) and len(symbol) > 3:
        return symbol
    return symbol + "USDT"


def base_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()
    for quote in ["USDT","BUSD","USD"]:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[:-len(quote)]
    for quote in ["BTC","ETH","BNB"]:
        if symbol.endswith(quote) and len(symbol) > len(quote) + 1:
            return symbol[:-len(quote)]
    return symbol


# ─────────────────────────────────────────────────────────
# OHLCV PARSERS
# ─────────────────────────────────────────────────────────

def _parse_to_df(rows: list, ts_col: int, o: int, h: int, l: int, c: int, v: int,
                 ts_unit: str = "ms") -> pd.DataFrame:
    try:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df[ts_col], unit=ts_unit)
        df["open"]   = df[o].astype(float)
        df["high"]   = df[h].astype(float)
        df["low"]    = df[l].astype(float)
        df["close"]  = df[c].astype(float)
        df["volume"] = df[v].astype(float)
        df = df[["timestamp","open","high","low","close","volume"]].copy()
        df.set_index("timestamp", inplace=True)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def fetch_binance(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        if not data or isinstance(data, dict): return pd.DataFrame()
        return _parse_to_df(data, 0, 1, 2, 3, 4, 5)
    except Exception as e:
        print(f"    [Binance] {e}"); return pd.DataFrame()


def fetch_kucoin(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    interval_map = {"1h":"1hour","4h":"4hour","1d":"1day",
                    "15m":"15min","5m":"5min","1m":"1min"}
    kc_interval = interval_map.get(interval, interval)
    base = base_symbol(symbol)
    try:
        r = requests.get(
            "https://api.kucoin.com/api/v1/market/candles",
            params={"symbol": f"{base}-USDT", "type": kc_interval},
            timeout=10
        )
        r.raise_for_status()
        data = list(reversed(r.json().get("data",[])))[-limit:]
        if not data: return pd.DataFrame()
        rows = [[int(d[0]),d[1],d[3],d[4],d[2],d[5]] for d in data]
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
        for col in ["open","high","low","close","volume"]: df[col] = df[col].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]].sort_index()
    except Exception as e:
        print(f"    [KuCoin] {e}"); return pd.DataFrame()


def fetch_okx(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    interval_map = {"1h":"1H","4h":"4H","1d":"1D",
                    "15m":"15m","5m":"5m","1m":"1m"}
    base = base_symbol(symbol)
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": f"{base}-USDT",
                    "bar": interval_map.get(interval, interval),
                    "limit": min(limit, 300)},
            timeout=10
        )
        r.raise_for_status()
        data = list(reversed(r.json().get("data",[])))
        if not data: return pd.DataFrame()
        rows = [[int(d[0]),d[1],d[2],d[3],d[4],d[5]] for d in data]
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
        for col in ["open","high","low","close","volume"]: df[col] = df[col].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]].sort_index()
    except Exception as e:
        print(f"    [OKX] {e}"); return pd.DataFrame()


def fetch_mexc(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    try:
        r = requests.get(
            "https://api.mexc.com/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        if not data or isinstance(data, dict): return pd.DataFrame()
        return _parse_to_df(data, 0, 1, 2, 3, 4, 5)
    except Exception as e:
        print(f"    [MEXC] {e}"); return pd.DataFrame()


def fetch_gateio(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    base = base_symbol(symbol)
    try:
        r = requests.get(
            "https://api.gateio.ws/api/v4/spot/candlesticks",
            params={"currency_pair": f"{base}_USDT",
                    "interval": interval, "limit": limit},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        if not data: return pd.DataFrame()
        rows = [[int(d[0]),d[5],d[3],d[4],d[2],d[1]] for d in data]
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
        for col in ["open","high","low","close","volume"]: df[col] = df[col].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]].sort_index()
    except Exception as e:
        print(f"    [Gate.io] {e}"); return pd.DataFrame()


def fetch_bitget(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    interval_map = {"1m":"60","5m":"300","15m":"900",
                    "1h":"3600","4h":"14400","1d":"86400"}
    base = base_symbol(symbol)
    try:
        r = requests.get(
            "https://api.bitget.com/api/spot/v1/market/candles",
            params={"symbol": f"{base}USDT",
                    "period": interval_map.get(interval,"3600"),
                    "limit": str(limit)},
            timeout=10
        )
        r.raise_for_status()
        data = r.json().get("data",[])
        if not data: return pd.DataFrame()
        rows = [[int(d[0]),d[1],d[2],d[3],d[4],d[5]] for d in data]
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
        for col in ["open","high","low","close","volume"]: df[col] = df[col].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]].sort_index()
    except Exception as e:
        print(f"    [Bitget] {e}"); return pd.DataFrame()


def fetch_binance_alpha(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    base = base_symbol(symbol)
    try:
        search_r = requests.get(
            "https://www.binance.com/bapi/alpha/v1/public/alpha/token/list",
            params={"keyword": base, "size": 5}, timeout=10
        )
        search_r.raise_for_status()
        tokens = search_r.json().get("data",{}).get("list",[])
        token  = next((t for t in tokens
                       if t.get("symbol","").upper() == base.upper()), None)
        if not token: return pd.DataFrame()
        token_address = token.get("tokenAddress")
        if not token_address: return pd.DataFrame()
        kline_r = requests.get(
            "https://www.binance.com/bapi/alpha/v1/public/alpha/kline",
            params={"tokenAddress": token_address,
                    "interval": interval, "limit": limit},
            timeout=10
        )
        kline_r.raise_for_status()
        data = kline_r.json().get("data",[])
        if not data: return pd.DataFrame()
        return _parse_to_df(data, 0, 1, 2, 3, 4, 5)
    except Exception as e:
        print(f"    [Binance Alpha] {e}"); return pd.DataFrame()


# ─────────────────────────────────────────────────────────
# EXCHANGE FALLBACK CHAIN
# FIX #7: DexScreener removed — free tier has no OHLCV candle endpoint,
# so it always returned pd.DataFrame() after making a real HTTP call.
# It was burning a network request on every coin that failed all 6 real
# exchanges, contributing nothing. Re-add if a paid plan with OHLCV
# becomes available.
# ─────────────────────────────────────────────────────────

EXCHANGE_CHAIN = [
    ("Binance",       fetch_binance),
    ("KuCoin",        fetch_kucoin),
    ("OKX",           fetch_okx),
    ("MEXC",          fetch_mexc),
    ("Gate.io",       fetch_gateio),
    ("Bitget",        fetch_bitget),
    ("Binance Alpha", fetch_binance_alpha),
]


def fetch_ohlcv(symbol: str, interval: str,
                limit: int = 200) -> tuple[pd.DataFrame, str]:
    normalized = normalize_symbol(symbol)
    for exchange_name, fetch_fn in EXCHANGE_CHAIN:
        df = fetch_fn(normalized, interval, limit)
        if not df.empty and len(df) >= 50:
            return df, exchange_name
    print(f"  ⚠️  All exchanges failed for {symbol} {interval}")
    return pd.DataFrame(), ""


# ─────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────

def calc_rsi(df: pd.DataFrame, period: int = 14) -> dict:
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-10)
    rsi      = 100 - (100 / (1 + rs))
    current  = round(rsi.iloc[-1], 2)
    prev     = rsi.iloc[-2]

    if current < 30:
        signal = "bullish"; reason = f"RSI {current} — oversold, potential bounce"
    elif current > 70:
        signal = "bearish"; reason = f"RSI {current} — overbought, potential pullback"
    elif current < 45 and current < prev:
        signal = "bearish"; reason = f"RSI {current} — falling below midline"
    elif current > 55 and current > prev:
        signal = "bullish"; reason = f"RSI {current} — rising above midline"
    else:
        signal = "neutral"; reason = f"RSI {current} — neutral zone"

    return {"indicator":"RSI","value":current,"signal":signal,"reason":reason}


def calc_macd(df: pd.DataFrame) -> dict:
    ema12      = df["close"].ewm(span=12, adjust=False).mean()
    ema26      = df["close"].ewm(span=26, adjust=False).mean()
    macd_line  = ema12 - ema26
    signal_line= macd_line.ewm(span=9, adjust=False).mean()
    histogram  = macd_line - signal_line

    macd_val   = round(macd_line.iloc[-1], 6)
    signal_val = round(signal_line.iloc[-1], 6)
    hist_val   = round(histogram.iloc[-1], 6)
    prev_hist  = histogram.iloc[-2]

    crossed_above = macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]
    crossed_below = macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]

    if crossed_above:
        signal = "bullish"; reason = "MACD bullish crossover"
    elif crossed_below:
        signal = "bearish"; reason = "MACD bearish crossover"
    elif hist_val > 0 and hist_val > prev_hist:
        signal = "bullish"; reason = "MACD histogram growing positive"
    elif hist_val < 0 and hist_val < prev_hist:
        signal = "bearish"; reason = "MACD histogram growing negative"
    elif macd_val > signal_val:
        signal = "bullish"; reason = "MACD above signal line"
    else:
        signal = "bearish"; reason = "MACD below signal line"

    return {"indicator":"MACD","macd":macd_val,"signal_line":signal_val,
            "histogram":hist_val,"signal":signal,"reason":reason}


def calc_bollinger_bands(df: pd.DataFrame, period: int = 20,
                          std: float = 2.0) -> dict:
    sma       = df["close"].rolling(period).mean()
    std_dev   = df["close"].rolling(period).std()
    upper     = sma + std * std_dev
    lower     = sma - std * std_dev
    bandwidth = (upper - lower) / sma

    price     = df["close"].iloc[-1]
    upper_val = round(upper.iloc[-1], 4)
    lower_val = round(lower.iloc[-1], 4)
    mid_val   = round(sma.iloc[-1], 4)
    bw        = round(bandwidth.iloc[-1], 4)
    bw_avg    = round(bandwidth.rolling(20).mean().iloc[-1], 4)
    pct_b     = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-9)

    if pct_b < 0.1:
        signal = "bullish"; reason = f"Price near lower BB — oversold"
    elif pct_b > 0.9:
        signal = "bearish"; reason = f"Price near upper BB — overbought"
    elif bw < bw_avg * 0.7:
        signal = "neutral"; reason = "BB squeeze — breakout imminent"
    elif price > mid_val:
        signal = "bullish"; reason = f"Price above BB midline ({mid_val})"
    else:
        signal = "bearish"; reason = f"Price below BB midline ({mid_val})"

    return {"indicator":"Bollinger Bands","upper":upper_val,"mid":mid_val,
            "lower":lower_val,"pct_b":round(pct_b,3),"bandwidth":bw,
            "signal":signal,"reason":reason}


def calc_ema_crossovers(df: pd.DataFrame) -> dict:
    ema9   = df["close"].ewm(span=9,   adjust=False).mean()
    ema21  = df["close"].ewm(span=21,  adjust=False).mean()
    ema50  = df["close"].ewm(span=50,  adjust=False).mean()
    ema200 = df["close"].ewm(span=200, adjust=False).mean()

    price = df["close"].iloc[-1]
    e9    = round(ema9.iloc[-1],   4)
    e21   = round(ema21.iloc[-1],  4)
    e50   = round(ema50.iloc[-1],  4)
    e200  = round(ema200.iloc[-1], 4)

    golden_cross = ema50.iloc[-2] < ema200.iloc[-2] and ema50.iloc[-1] > ema200.iloc[-1]
    death_cross  = ema50.iloc[-2] > ema200.iloc[-2] and ema50.iloc[-1] < ema200.iloc[-1]
    bull_count   = sum([price>e9, price>e21, price>e50, price>e200, e9>e21, e21>e50])

    if golden_cross:
        signal = "bullish"; reason = "Golden Cross — EMA50 crossed above EMA200"
    elif death_cross:
        signal = "bearish"; reason = "Death Cross — EMA50 crossed below EMA200"
    elif bull_count >= 5:
        signal = "bullish"; reason = "Strong EMA alignment — price above all EMAs"
    elif bull_count <= 2:
        signal = "bearish"; reason = "Bearish EMA alignment — price below most EMAs"
    elif price > e200:
        signal = "bullish"; reason = "Price above EMA200 — long-term uptrend intact"
    else:
        signal = "neutral"; reason = "Mixed EMA signals — no clear trend"

    return {"indicator":"EMA Crossovers","ema9":e9,"ema21":e21,"ema50":e50,
            "ema200":e200,"price":round(price,4),"signal":signal,"reason":reason}


def calc_volume_analysis(df: pd.DataFrame, period: int = 20) -> dict:
    avg_vol     = df["volume"].rolling(period).mean()
    current_vol = df["volume"].iloc[-1]
    avg         = avg_vol.iloc[-1]
    vol_ratio   = round(current_vol / avg, 2) if avg > 0 else 1.0
    price_up    = df["close"].iloc[-1] > df["close"].iloc[-2]

    if vol_ratio > 1.5 and price_up:
        signal = "bullish"; reason = f"Volume {vol_ratio}x above avg with price increase"
    elif vol_ratio > 1.5 and not price_up:
        signal = "bearish"; reason = f"Volume {vol_ratio}x above avg with price decrease"
    elif vol_ratio < 0.7:
        signal = "neutral"; reason = f"Volume {vol_ratio}x below avg — weak conviction"
    elif price_up:
        signal = "bullish"; reason = "Normal volume with price increase"
    else:
        signal = "bearish"; reason = "Normal volume with price decrease"

    return {"indicator":"Volume","current_volume":round(current_vol,2),
            "avg_volume":round(avg,2),"volume_ratio":vol_ratio,
            "signal":signal,"reason":reason}


def calc_support_resistance(df: pd.DataFrame, lookback: int = 50) -> dict:
    recent = df.tail(lookback)
    price  = df["close"].iloc[-1]
    window = 5
    highs  = recent["high"]
    lows   = recent["low"]

    local_highs = highs[highs == highs.rolling(window, center=True).max()]
    local_lows  = lows[lows == lows.rolling(window, center=True).min()]

    resistance_levels = sorted(local_highs[local_highs > price].values)[:3]
    support_levels    = sorted(local_lows[local_lows < price].values, reverse=True)[:3]

    nearest_resistance = round(resistance_levels[0], 4) if resistance_levels else None
    nearest_support    = round(support_levels[0], 4)    if support_levels    else None
    dist_to_res        = round((nearest_resistance - price) / price * 100, 2) if nearest_resistance else None
    dist_to_sup        = round((price - nearest_support) / price * 100, 2)    if nearest_support    else None

    if nearest_resistance and dist_to_res is not None and dist_to_res < 1.0:
        signal = "bearish"; reason = f"Price within {dist_to_res}% of resistance at {nearest_resistance}"
    elif nearest_support and dist_to_sup is not None and dist_to_sup < 1.0:
        signal = "bullish"; reason = f"Price within {dist_to_sup}% of support at {nearest_support}"
    elif nearest_resistance and nearest_support:
        mid    = (nearest_resistance + nearest_support) / 2
        signal = "bullish" if price > mid else "bearish"
        reason = f"In range — support: {nearest_support}, resistance: {nearest_resistance}"
    else:
        signal = "neutral"; reason = "Insufficient S/R data"

    return {"indicator":"Support/Resistance",
            "nearest_support":nearest_support,"nearest_resistance":nearest_resistance,
            "dist_to_support_pct":dist_to_sup,"dist_to_resistance_pct":dist_to_res,
            "support_levels":[round(x,4) for x in support_levels],
            "resistance_levels":[round(x,4) for x in resistance_levels],
            "signal":signal,"reason":reason}


def calc_atr(df: pd.DataFrame, period: int = 14,
             direction: str = "long") -> dict:
    high       = df["high"]
    low        = df["low"]
    close_prev = df["close"].shift(1)
    tr         = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs()
    ], axis=1).max(axis=1)
    atr         = tr.ewm(span=period, adjust=False).mean()
    current_atr = round(atr.iloc[-1], 6)
    price       = df["close"].iloc[-1]
    atr_pct     = round(current_atr / price * 100, 2)

    if direction == "short":
        stop_loss   = round(price + 1.5 * current_atr, 6)
        take_profit = round(price - 2.5 * current_atr, 6)
    else:
        stop_loss   = round(price - 1.5 * current_atr, 6)
        take_profit = round(price + 2.5 * current_atr, 6)

    return {"indicator":"ATR","atr":current_atr,"atr_pct":atr_pct,
            "suggested_stop_loss":stop_loss,"suggested_take_profit":take_profit,
            "direction":direction,
            "signal":"neutral",
            "reason":f"ATR {current_atr} ({atr_pct}% of price) — SL:{stop_loss} TP:{take_profit}"}


def calc_obv(df: pd.DataFrame) -> dict:
    obv       = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    obv_now   = obv.iloc[-1]
    obv_prev  = obv.iloc[-5]
    price_now = df["close"].iloc[-1]
    price_prev= df["close"].iloc[-5]

    obv_rising   = obv_now > obv_prev
    price_rising = price_now > price_prev

    if obv_rising and price_rising:
        signal = "bullish"; reason = "OBV and price both rising — confirmed uptrend"
    elif not obv_rising and not price_rising:
        signal = "bearish"; reason = "OBV and price both falling — confirmed downtrend"
    elif obv_rising and not price_rising:
        signal = "bullish"; reason = "Bullish OBV divergence — buying despite price drop"
    else:
        signal = "bearish"; reason = "Bearish OBV divergence — selling despite price rise"

    return {"indicator":"OBV","obv":round(obv_now,2),
            "obv_trend":"rising" if obv_rising else "falling",
            "signal":signal,"reason":reason}


def calc_fibonacci(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Fibonacci Retracement Levels.

    FIX #2: The original function NEVER returned "bearish". Every price
    level mapped to "bullish" or "neutral", creating a permanent bullish
    tilt of weight=1.0 in every analysis regardless of market conditions.

    Fix: price in the upper fib zone (>78.6% from low = near the recent high)
    is treated as BEARISH — the asset is extended and approaching resistance.
    Price between 61.8–78.6% is neutral (consolidating in upper range).
    The rest of the zones are unchanged.

    Zone mapping (pct_from_low):
      0–23.6%  → bullish  (deep pullback / strong support)
      23.6–38.2% → bullish (standard retracement — likely bounce)
      38.2–50%   → neutral  (mid-range, no clear bias)
      50–61.8%   → bullish  (held above 50% — momentum intact)
      61.8–78.6% → neutral  (upper range consolidation)
      >78.6%     → bearish  (near recent high — resistance / overextended)
    """
    recent = df.tail(lookback)
    high   = recent["high"].max()
    low    = recent["low"].min()
    price  = df["close"].iloc[-1]
    diff   = high - low

    levels = {
        "0.0":   round(high, 6),
        "23.6":  round(high - 0.236 * diff, 6),
        "38.2":  round(high - 0.382 * diff, 6),
        "50.0":  round(high - 0.500 * diff, 6),
        "61.8":  round(high - 0.618 * diff, 6),
        "78.6":  round(high - 0.786 * diff, 6),
        "100.0": round(low, 6),
    }

    pct_from_low = round((price - low) / diff * 100, 1) if diff > 0 else 50

    if pct_from_low < 23.6:
        signal = "bullish"; reason = f"Price at deep fib ({pct_from_low}%) — strong support zone"
    elif pct_from_low < 38.2:
        signal = "bullish"; reason = f"Price at 23.6–38.2% fib — potential bounce"
    elif pct_from_low < 50:
        signal = "neutral"; reason = f"Price between 38.2–50% fib — mid-range"
    elif pct_from_low < 61.8:
        signal = "bullish"; reason = f"Price reclaimed 50% fib — bullish momentum"
    elif pct_from_low < 78.6:
        # FIX #2: was falling through to neutral — now explicitly neutral
        signal = "neutral"; reason = f"Price at {pct_from_low}% fib — upper range consolidation"
    else:
        # FIX #2: was "bullish" (price near highs = strong uptrend).
        # Now "bearish" — near the recent high means approaching resistance,
        # extended move likely to face selling pressure.
        signal = "bearish"; reason = f"Price near recent high ({pct_from_low}% fib) — overextended, resistance zone"

    fib_values = sorted(levels.values(), reverse=True)
    zone_top = zone_bot = None
    for i in range(len(fib_values) - 1):
        if fib_values[i+1] <= price <= fib_values[i]:
            zone_top, zone_bot = fib_values[i], fib_values[i+1]
            break

    return {"indicator":"Fibonacci","levels":levels,"price":round(price,6),
            "pct_from_low":pct_from_low,"zone_top":zone_top,"zone_bot":zone_bot,
            "signal":signal,"reason":reason}


# ─────────────────────────────────────────────────────────
# SIGNAL AGGREGATOR
# ─────────────────────────────────────────────────────────

def aggregate_signals(indicator_results: list[dict]) -> dict:
    bull_score   = 0.0
    bear_score   = 0.0
    total_weight = 0.0

    for result in indicator_results:
        name   = result["indicator"]
        weight = INDICATOR_WEIGHTS.get(name, 0.0)
        if weight == 0:
            continue
        signal = result.get("signal", "neutral")
        if signal == "bullish":
            bull_score += weight
        elif signal == "bearish":
            bear_score += weight
        total_weight += weight

    if total_weight == 0:
        return {"bullish_pct": 50.0, "bearish_pct": 50.0,
                "neutral_pct": 0.0, "overall": "neutral"}

    bull_pct = round(bull_score / total_weight * 100, 1)
    bear_pct = round(bear_score / total_weight * 100, 1)
    neut_pct = round(max(100 - bull_pct - bear_pct, 0), 1)

    if bull_pct > bear_pct + 15:
        overall = "bullish"
    elif bear_pct > bull_pct + 15:
        overall = "bearish"
    else:
        overall = "neutral"

    return {"bullish_pct": bull_pct, "bearish_pct": bear_pct,
            "neutral_pct": neut_pct, "overall": overall}


# ─────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def run_technical_analysis(symbol: str) -> dict:
    normalized = normalize_symbol(symbol)
    print(f"\n📊 Technical analysis for: {normalized}")

    result = {"symbol": normalized, "timeframes": {}}

    for tf_label, tf_config in TIMEFRAMES.items():
        print(f"  [{tf_label}] Fetching data...", end=" ")
        df, source = fetch_ohlcv(normalized, tf_config["interval"], tf_config["limit"])

        if df.empty or len(df) < 50:
            print("insufficient data from all sources.")
            result["timeframes"][tf_label] = {
                "error": "Insufficient data from all exchanges"
            }
            continue

        print(f"{len(df)} candles from {source}.")

        indicators_no_atr = [
            calc_rsi(df),
            calc_macd(df),
            calc_bollinger_bands(df),
            calc_ema_crossovers(df),
            calc_volume_analysis(df),
            calc_support_resistance(df),
            calc_obv(df),
            calc_fibonacci(df),
        ]

        prelim_agg = aggregate_signals(indicators_no_atr)
        if prelim_agg["bullish_pct"] > prelim_agg["bearish_pct"] + 5:
            atr_direction = "long"
        elif prelim_agg["bearish_pct"] > prelim_agg["bullish_pct"] + 5:
            atr_direction = "short"
        else:
            atr_direction = "long"

        atr_result  = calc_atr(df, direction=atr_direction)
        indicators  = indicators_no_atr + [atr_result]

        aggregate = aggregate_signals(indicators)
        atr_data  = next(i for i in indicators if i["indicator"] == "ATR")
        sr_data   = next(i for i in indicators if i["indicator"] == "Support/Resistance")
        fib_data  = next(i for i in indicators if i["indicator"] == "Fibonacci")

        result["timeframes"][tf_label] = {
            "source":            source,
            "signals":           indicators,
            "aggregate":         aggregate,
            "atr":               atr_data,
            "support_resistance":sr_data,
            "fibonacci":         fib_data,
            "current_price":     round(df["close"].iloc[-1], 6),
            "last_candle":       str(df.index[-1]),
        }

        agg = aggregate
        print(f"  [{tf_label}] {agg['overall'].upper():8} | "
              f"Bull: {agg['bullish_pct']}% | Bear: {agg['bearish_pct']}% | "
              f"SL: {atr_data['suggested_stop_loss']} | "
              f"TP: {atr_data['suggested_take_profit']}")

    return result


# ─────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sym = input("Enter symbol (e.g. ICP, ICPUSDT, BTC): ").strip()
    run_technical_analysis(sym)