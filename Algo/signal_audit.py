"""
signal_audit.py
---------------
Individual signal Sharpe audit — requested by senior quant.

Tests each technical indicator in isolation on the holdout period
to establish a baseline before combining signals. This directly
addresses the concern: "if combined Sharpe >> individual Sharpe,
something is wrong."

For each indicator independently:
  - Generate directional signals (LONG/SHORT/STAY OUT)
  - Simulate trades on holdout data (last 20% of dataset)
  - Calculate annualised Sharpe ratio
  - Show direction accuracy and trade count

A legitimate combined system should produce Sharpe roughly
1.5-2.5x the average individual indicator Sharpe.
If combined Sharpe is 5-10x individual Sharpe, overfitting
is almost certainly present.

Usage:
  python signal_audit.py
  python signal_audit.py --symbols BTC --timeframes 4h 1d
  python signal_audit.py --symbols BTC ETH SOL --timeframes 4h 1d --days 456
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np

try:
    from SignalNormalizer import normalize_technical, combine_signal_vectors
except ImportError:
    print("❌ SignalNormalizer.py not found.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
# Use 5 years for signal audit to cover multiple market regimes
LOOKBACK_DAYS   = 1825   # 5 years of daily data
WARMUP_BARS     = 200
CONVICTION_THR  = 25.0
ATR_SL_MULT     = 1.5
ATR_TP_MULT     = 2.5
ATR_PERIOD      = 14
RISK_PCT        = 0.01


# ─────────────────────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, days: int = LOOKBACK_DAYS, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV candles from Binance with KuCoin fallback."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    end_ms   = int(time.time() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    candles, current = [], start_ms
    # Try Binance
    while current < end_ms:
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                params={"symbol": sym, "interval": interval,
                        "startTime": current, "endTime": end_ms, "limit": 1000},
                timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data or isinstance(data, dict): break
            candles.extend(data)
            current = data[-1][0] + 1
            if len(data) < 1000: break
        except Exception: break

    if not candles:
        # KuCoin fallback
        KUCOIN_TF = {"1d": "1day", "4h": "4hour", "1h": "1hour"}
        kucoin_type = KUCOIN_TF.get(interval, "1day")
        base = sym.replace("USDT","")
        try:
            r = requests.get("https://api.kucoin.com/api/v1/market/candles",
                params={"symbol": f"{base}-USDT", "type": kucoin_type}, timeout=15)
            r.raise_for_status()
            data = list(reversed(r.json().get("data", [])))
            rows = [[int(d[0])*1000, d[1], d[3], d[4], d[2], d[5]] for d in data]
            candles = rows
        except Exception:
            pass

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df[0], unit="ms", utc=True)
    for col, idx in [("open",1),("high",2),("low",3),("close",4),("volume",5)]:
        df[col] = df[idx].astype(float)
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    df.set_index("timestamp", inplace=True)
    return df.sort_index()


# ─────────────────────────────────────────────────────────
# INDIVIDUAL INDICATOR SIGNAL FUNCTIONS
# ─────────────────────────────────────────────────────────

def signal_rsi(df: pd.DataFrame, idx: int) -> str:
    w = df.iloc[max(0,idx-30):idx+1]["close"]
    if len(w) < 15: return "neutral"
    delta = w.diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rsi   = (100 - 100/(1 + gain/(loss+1e-10))).iloc[-1]
    prev  = (100 - 100/(1 + gain/(loss+1e-10))).iloc[-2]
    if rsi < 30: return "bullish"
    if rsi > 70: return "bearish"
    if rsi < 45 and rsi < prev: return "bearish"
    if rsi > 55 and rsi > prev: return "bullish"
    return "neutral"

def signal_macd(df: pd.DataFrame, idx: int) -> str:
    c = df.iloc[max(0,idx-50):idx+1]["close"]
    if len(c) < 30: return "neutral"
    macd = c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean()
    sig  = macd.ewm(span=9,adjust=False).mean()
    hist = macd - sig
    if macd.iloc[-2] < sig.iloc[-2] and macd.iloc[-1] > sig.iloc[-1]: return "bullish"
    if macd.iloc[-2] > sig.iloc[-2] and macd.iloc[-1] < sig.iloc[-1]: return "bearish"
    if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]: return "bullish"
    if hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]: return "bearish"
    return "bullish" if macd.iloc[-1] > sig.iloc[-1] else "bearish"

def signal_bb(df: pd.DataFrame, idx: int) -> str:
    c = df.iloc[max(0,idx-30):idx+1]["close"]
    if len(c) < 21: return "neutral"
    sma   = c.rolling(20).mean()
    std   = c.rolling(20).std()
    upper = sma + 2*std; lower = sma - 2*std
    pct_b = (c.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-9)
    if pct_b < 0.1: return "bullish"
    if pct_b > 0.9: return "bearish"
    return "bullish" if c.iloc[-1] > sma.iloc[-1] else "bearish"

def signal_ema(df: pd.DataFrame, idx: int) -> str:
    c = df.iloc[max(0,idx-210):idx+1]["close"]
    if len(c) < 50: return "neutral"
    e9=c.ewm(span=9,adjust=False).mean(); e21=c.ewm(span=21,adjust=False).mean()
    e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
    price = c.iloc[-1]
    if e50.iloc[-2]<e200.iloc[-2] and e50.iloc[-1]>e200.iloc[-1]: return "bullish"
    if e50.iloc[-2]>e200.iloc[-2] and e50.iloc[-1]<e200.iloc[-1]: return "bearish"
    bc = sum([price>e9.iloc[-1], price>e21.iloc[-1], price>e50.iloc[-1],
              price>e200.iloc[-1], e9.iloc[-1]>e21.iloc[-1], e21.iloc[-1]>e50.iloc[-1]])
    if bc >= 5: return "bullish"
    if bc <= 2: return "bearish"
    return "bullish" if price > e200.iloc[-1] else "neutral"

def signal_sr(df: pd.DataFrame, idx: int) -> str:
    window = df.iloc[max(0,idx-50):idx+1]
    price  = df["close"].iloc[idx]
    lh = window["high"][window["high"]==window["high"].rolling(5,center=True).max()]
    ll = window["low"][window["low"]==window["low"].rolling(5,center=True).min()]
    res = sorted(lh[lh>price].values)
    sup = sorted(ll[ll<price].values, reverse=True)
    nr  = res[0] if res else None
    ns  = sup[0] if sup else None
    if nr and (nr-price)/price*100 < 1.0: return "bearish"
    if ns and (price-ns)/price*100 < 1.0: return "bullish"
    if nr and ns: return "bullish" if price>(nr+ns)/2 else "bearish"
    return "neutral"

def signal_obv(df: pd.DataFrame, idx: int) -> str:
    window = df.iloc[max(0,idx-10):idx+1]
    obv    = (np.sign(window["close"].diff()) * window["volume"]).fillna(0).cumsum()
    if len(obv) < 6: return "neutral"
    obv_up   = obv.iloc[-1] > obv.iloc[-5]
    price_up = window["close"].iloc[-1] > window["close"].iloc[-5]
    if obv_up and price_up: return "bullish"
    if not obv_up and not price_up: return "bearish"
    return "bullish" if obv_up else "bearish"

def signal_fib(df: pd.DataFrame, idx: int) -> str:
    window = df.iloc[max(0,idx-50):idx+1]
    high, low = window["high"].max(), window["low"].min()
    pct = (df["close"].iloc[idx] - low) / (high - low + 1e-9) * 100
    if pct < 38.2: return "bullish"
    if pct < 61.8: return "neutral"
    if pct < 78.6: return "neutral"
    return "bearish"

def signal_volume(df: pd.DataFrame, idx: int) -> str:
    window = df.iloc[max(0,idx-25):idx+1]
    avg    = window["volume"].rolling(20).mean().iloc[-1]
    ratio  = window["volume"].iloc[-1] / (avg + 1e-9)
    up     = df["close"].iloc[idx] > df["close"].iloc[idx-1]
    if ratio > 1.5: return "bullish" if up else "bearish"
    if ratio < 0.7: return "neutral"
    return "bullish" if up else "bearish"

INDICATORS = {
    "RSI":                signal_rsi,
    "MACD":               signal_macd,
    "Bollinger Bands":    signal_bb,
    "EMA Crossovers":     signal_ema,
    "Support/Resistance": signal_sr,
    "OBV":                signal_obv,
    "Fibonacci":          signal_fib,
    "Volume":             signal_volume,
}


# ─────────────────────────────────────────────────────────
# TRADE SIMULATOR (simplified — direction + ATR exit)
# ─────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, idx: int) -> float:
    w = df.iloc[max(0,idx-ATR_PERIOD*2):idx+1]
    if len(w) < 2: return df["close"].iloc[idx] * 0.02
    tr = pd.concat([w["high"]-w["low"],
                    (w["high"]-w["close"].shift(1)).abs(),
                    (w["low"] -w["close"].shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-ATR_PERIOD:].mean())


def simulate_single_signal(df: pd.DataFrame, signal_fn,
                            holdout_start: int) -> list:
    """Run one indicator in isolation and return list of trade P&L pcts."""
    trades = []
    for idx in range(holdout_start, len(df)-1):
        direction = signal_fn(df, idx)
        if direction == "neutral": continue

        entry = float(df["open"].iloc[idx+1])   # enter next bar open
        atr   = compute_atr(df, idx)
        if atr <= 0: continue

        sl_dist = ATR_SL_MULT * atr
        tp_dist = ATR_TP_MULT * atr
        sl = entry - sl_dist if direction=="bullish" else entry + sl_dist
        tp = entry + tp_dist if direction=="bullish" else entry - tp_dist

        # Check exit on next bar only (1d candle = 1 day holding)
        bar = df.iloc[idx+1]
        if direction == "bullish":
            if bar["low"] <= sl:   exit_p = sl
            elif bar["high"] >= tp: exit_p = tp
            else:                   exit_p = float(bar["close"])
        else:
            if bar["high"] >= sl:  exit_p = sl
            elif bar["low"] <= tp: exit_p = tp
            else:                   exit_p = float(bar["close"])

        pct = (exit_p - entry) / entry
        pnl = pct if direction=="bullish" else -pct
        trades.append(pnl)

    return trades


def sharpe_from_trades(trades: list, days_covered: float = 365.0) -> float:
    """
    Annualised Sharpe using actual trade frequency.
    sqrt(252) is correct only if exactly 1 trade/trading-day; using
    trades_per_year keeps the number comparable across timeframes and
    consistent with backtest.py.
    """
    if len(trades) < 10: return float("nan")
    arr            = np.array(trades)
    mean_r         = arr.mean()
    std_r          = arr.std() + 1e-10
    trades_per_year = len(trades) / (days_covered / 365.25)
    ann_factor      = np.sqrt(max(trades_per_year, 1))
    return round(float(mean_r / std_r * ann_factor), 3)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",    nargs="+", default=["BTC"])
    parser.add_argument("--timeframes", nargs="+", default=["4h", "1d"])
    parser.add_argument("--days",       type=int,  default=LOOKBACK_DAYS)
    args = parser.parse_args()

    all_sharpes = []

    for tf in args.timeframes:
        print("\n" + "═"*62)
        print("  INDIVIDUAL SIGNAL SHARPE AUDIT")
        print(f"  Symbols:    {', '.join(args.symbols)}")
        print(f"  Timeframe:  {tf}")
        print(f"  Period:     holdout (last 20% of {args.days} days)")
        print("═"*62)

        indicator_results = {ind: [] for ind in INDICATORS}
        direction_acc     = {ind: [] for ind in INDICATORS}
        total_holdout_days = 0

        for symbol in args.symbols:
            print(f"\n  Fetching {symbol} [{tf}]...", end=" ")
            df = fetch_ohlcv(symbol, days=args.days, interval=tf)
            if df.empty or len(df) < WARMUP_BARS + 30:
                print(f"insufficient data — skipping")
                continue
            print(f"{len(df)} candles")

            holdout_start = max(WARMUP_BARS, int(len(df) * 0.80))
            h_days = len(df) - holdout_start
            total_holdout_days += h_days
            print(f"  Holdout: {df.index[holdout_start].date()} → "
                  f"{df.index[-1].date()} ({h_days} bars)")

            for ind_name, ind_fn in INDICATORS.items():
                trades = simulate_single_signal(df, ind_fn, holdout_start)
                indicator_results[ind_name].extend(trades)
                if trades:
                    dir_acc = sum(1 for t in trades if t > 0) / len(trades) * 100
                    direction_acc[ind_name].append(dir_acc)

        avg_holdout_days = max(total_holdout_days / max(len(args.symbols), 1), 30)

        print("\n" + "═"*62)
        print(f"  {'Indicator':24} {'Sharpe':>8} {'Dir Acc':>8} "
              f"{'Trades':>7} {'Avg Win':>8} {'Avg Loss':>9}")
        print(f"  {'─'*24} {'─'*8} {'─'*8} {'─'*7} {'─'*8} {'─'*9}")

        tf_sharpes = []
        for ind_name, trades in indicator_results.items():
            if not trades:
                print(f"  {ind_name:24} {'N/A':>8}")
                continue
            sh = sharpe_from_trades(trades, days_covered=avg_holdout_days)
            da = round(sum(direction_acc[ind_name]) / len(direction_acc[ind_name]), 1) if direction_acc[ind_name] else 0
            winners = [t for t in trades if t > 0]
            losers  = [t for t in trades if t <= 0]
            avg_win  = round(np.mean(winners)*100, 3) if winners else 0
            avg_loss = round(np.mean(losers)*100,  3) if losers  else 0
            flag = "  OK" if sh > 0.5 else ("  ~" if sh > 0 else "  X")
            print(f"  {ind_name:24} {sh:>8.3f} {da:>7.1f}% "
                  f"{len(trades):>7} {avg_win:>+7.3f}% {avg_loss:>+8.3f}%{flag}")
            if not np.isnan(sh):
                tf_sharpes.append(sh)
                all_sharpes.append(sh)

        if tf_sharpes:
            avg_tf_sharpe = round(np.mean(tf_sharpes), 3)
            print("═"*62)
            print(f"  [{tf}] Avg individual Sharpe: {avg_tf_sharpe:.3f}")
            if avg_tf_sharpe > 0.05:
                print(f"  Healthy combined Sharpe: {avg_tf_sharpe*1.5:.2f}–{avg_tf_sharpe*2.5:.2f}")
                print(f"  Overfitting threshold:   > {avg_tf_sharpe*3:.2f}")
            else:
                print(f"  ⚠️  Avg individual Sharpe is near zero or negative.")
                print(f"  Indicators have no reliable edge on this timeframe.")
                print(f"  Combined system cannot improve on random — skip optimisation.")
            print("═"*62)

    if len(args.timeframes) > 1 and all_sharpes:
        overall_avg = round(np.mean(all_sharpes), 3)
        print(f"\n  Overall avg individual Sharpe ({', '.join(args.timeframes)}): {overall_avg:.3f}")
        if overall_avg > 0.05:
            print(f"  Healthy combined Sharpe: {overall_avg*1.5:.2f}–{overall_avg*2.5:.2f}")
            print(f"  Overfitting threshold:   > {overall_avg*3:.2f}")
        else:
            print(f"  ⚠️  Overall avg Sharpe is near zero — no reliable edge detected across timeframes.")
        print()


if __name__ == "__main__":
    main()