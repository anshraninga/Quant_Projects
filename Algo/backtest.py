"""
backtest.py
-----------
Portfolio backtester for the crypto analysis algo.
Tests technical signals only (Technical + Candlestick + Chart patterns)
since News and Sentiment cannot be reproduced from historical data.

Three independent portfolios — one per timeframe:
  1h  portfolio: 24 windows per day (00:00-00:59, 01:00-01:59, ...)
  4h  portfolio: 6  windows per day (00:00-03:59, 04:00-07:59, ...)
  1d  portfolio: 1  window  per day (00:00-23:59)

Each window:
  - Compute signal at window open using all data up to that point
  - If LONG or SHORT: enter at open price, exit at TP/SL or window close
  - If STAY OUT: skip window, capital stays idle

Position sizing:
  - Risk 1% of current equity per trade
  - Position size = risk_amount / distance_to_SL
  - TP/SL computed via ATR (1.5× ATR for SL, 2.5× ATR for TP)

Metrics:
  - TP/SL win rate (did price hit TP before SL or timeout)
  - Direction accuracy (did price move in predicted direction at all)
  - Monthly/yearly P&L breakdown
  - Equity curve
  - Sharpe, Calmar, max drawdown, profit factor
  - STAY OUT accuracy (was staying out the right call)

Usage:
  python backtest.py
  python backtest.py --symbols BTC ETH --timeframes 1h 4h 1d --capital 10000
"""

import os
import sys
import json
import argparse
import time
import math
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import requests
import pandas as pd
import numpy as np

try:
    from SignalNormalizer import normalize_technical, normalize_technical_soft, combine_signal_vectors
except ImportError:
    print("❌ SignalNormalizer.py not found. Run from the Algo directory.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

# Per-timeframe lookbacks matching the optimizers
LOOKBACK_BY_TF = {
    "1h": 730,    # 2 years
    "4h": 1095,   # 3 years
    "1d": 1825,   # 5 years
}
LOOKBACK_DAYS = 1825  # fallback for fetch
WARMUP_BARS      = 200       # bars needed for EMA200 convergence
CONVICTION_THR   = 40.0      # conviction threshold — 3 price modules only
                             # (news/sentiment fully excluded from backtest; they are
                             # always flat historically and dilute signal without adding edge)
RISK_PCT         = 0.01      # 1% of current equity per trade
ATR_SL_MULT      = 1.5       # SL = entry ± 1.5 × ATR
ATR_TP_MULT      = 2.5       # TP = entry ∓ 2.5 × ATR
ATR_PERIOD       = 14
STAY_OUT_MOVE_THR = 3.0      # % move to flag as "should have traded"

# Post-trade cooldown (bars to skip after a trade completes)
# Prevents the simplified backtest signal from over-trading.
# In live trading you wouldn't enter a new position every single candle.
# Set to match realistic signal frequency observed in main.py:
#   1h: ~2 signals/day → cooldown 11 bars (trade at most every 12h)
#   4h: ~4 signals/day → cooldown 0  (6 windows/day, already fine)
#   1d: ~1 signal/day  → cooldown 0  (1 window/day, already fine)
# Post-trade cooldown per timeframe.
# 1h: max ~8 trades/day → cooldown = 2 bars (trade at most every 3 hours)
#     This prevents over-trading on 1h while keeping genuine signals.
# 4h/1d: no cooldown needed — natural frequency is already low.
COOLDOWN_BARS = {
    "1h": 2,
    "4h": 0,
    "1d": 0,
}
MAX_TRADES_PER_DAY = {
    "1h": 8,    # hard cap: no more than 8 1h trades per calendar day
    "4h": 6,    # natural max (6 windows/day)
    "1d": 1,    # natural max (1 window/day)
}

STARTING_CAPITAL = 10_000.0

WEIGHTS_FILE          = "weights.json"
INDICATOR_WEIGHTS_FILE = "indicator_weights.json"

TIMEFRAME_CONFIG = {
    "1h": {"interval": "1h",  "window_hours": 1,  "windows_per_day": 24},
    "4h": {"interval": "4h",  "window_hours": 4,  "windows_per_day":  6},
    "1d": {"interval": "1d",  "window_hours": 24, "windows_per_day":  1},
}

DEFAULT_WEIGHTS = {
    "technical": 0.50, "candlestick": 0.25, "chart": 0.25,
}


# ─────────────────────────────────────────────────────────
# WEIGHT LOADER
# ─────────────────────────────────────────────────────────

def load_weights() -> tuple[dict, dict]:
    """Load weights from weights.json.

    Returns:
        per_coin : {symbol: {tf: weights}} — coin-specific module weights
        shared   : {tf: weights}           — averaged fallback for unknown coins

    Supports both the old single-level format (weights: {tf: ...}) and the
    new per-coin format produced by weight_optimizer (per_coin + shared keys).
    """
    defaults = {"4h": dict(DEFAULT_WEIGHTS), "1d": dict(DEFAULT_WEIGHTS)}
    try:
        with open(WEIGHTS_FILE) as f:
            data = json.load(f)
        per_coin = data.get("per_coin", {})
        shared   = data.get("shared", data.get("weights", {}))
        for tf in ["4h", "1d"]:
            if tf not in shared:
                shared[tf] = defaults[tf]
        if per_coin:
            print(f"  ✓ Loaded per-coin weights for: {', '.join(per_coin.keys())}")
        else:
            print(f"  ✓ Loaded shared weights from {WEIGHTS_FILE}")
        return per_coin, shared
    except FileNotFoundError:
        print(f"  ⚠️  {WEIGHTS_FILE} not found — using default weights")
        return {}, defaults


# ─────────────────────────────────────────────────────────
# DATA FETCHER  (Binance with fallback)
# ─────────────────────────────────────────────────────────

def _parse_to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df[0], unit="ms", utc=True)
    df["open"]   = df[1].astype(float)
    df["high"]   = df[2].astype(float)
    df["low"]    = df[3].astype(float)
    df["close"]  = df[4].astype(float)
    df["volume"] = df[5].astype(float)
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    df.set_index("timestamp", inplace=True)
    return df.sort_index()


def fetch_ohlcv(symbol: str, interval: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    end_ms   = int(time.time() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    candles, current = [], start_ms
    while current < end_ms:
        try:
            r = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": sym, "interval": interval,
                        "startTime": current, "endTime": end_ms, "limit": 1000},
                timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data or isinstance(data, dict):
                break
            candles.extend(data)
            current = data[-1][0] + 1
            if len(data) < 1000:
                break
        except Exception as e:
            print(f"    [Binance] {e}")
            break
    if not candles:
        return pd.DataFrame()
    df = _parse_to_df(candles)
    print(f"  ✓ {symbol} {interval}: {len(df)} candles")
    return df


# ─────────────────────────────────────────────────────────
# ATR CALCULATOR
# ─────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, idx: int, period: int = ATR_PERIOD) -> float:
    """Compute ATR at index idx using the previous `period` candles."""
    start = max(0, idx - period * 2)
    window = df.iloc[start: idx + 1]
    if len(window) < 2:
        return df["close"].iloc[idx] * 0.02   # fallback: 2% of price
    high  = window["high"]
    low   = window["low"]
    close = window["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return float(tr.iloc[-period:].mean())


# ─────────────────────────────────────────────────────────
# SIGNAL COMPUTERS  (identical to weight_optimizer.py)
# ─────────────────────────────────────────────────────────

def _compute_technical(df: pd.DataFrame, idx: int) -> tuple[float, float]:
    """
    Returns (net_score, agreement) using soft continuous scoring (Fix A+B).
      net_score  ∈ [-1, +1]: positive=bullish, magnitude reflects signal strength
      agreement  ∈ [0,  1]:  unanimity of individual indicator signals
    """
    window = df.iloc[max(0, idx - 200): idx + 1]
    if len(window) < 20:
        return 0.0, 0.5
    close      = window["close"]
    ind_scores = []

    # RSI: tanh((50-rsi)/20)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-10)))
    ind_scores.append(math.tanh((50.0 - float(rsi.iloc[-1])) / 20.0))

    # EMA alignment: 6 binary conditions averaged to [-1, +1]
    e9   = close.ewm(span=9,   adjust=False).mean()
    e21  = close.ewm(span=21,  adjust=False).mean()
    e50  = close.ewm(span=50,  adjust=False).mean()
    e200 = close.ewm(span=200, adjust=False).mean()
    price = float(close.iloc[-1])
    ema_conds = [
        1.0 if price          > e9.iloc[-1]   else -1.0,
        1.0 if price          > e21.iloc[-1]  else -1.0,
        1.0 if price          > e50.iloc[-1]  else -1.0,
        1.0 if price          > e200.iloc[-1] else -1.0,
        1.0 if e9.iloc[-1]    > e21.iloc[-1]  else -1.0,
        1.0 if e21.iloc[-1]   > e50.iloc[-1]  else -1.0,
    ]
    ind_scores.append(sum(ema_conds) / len(ema_conds))

    # Bollinger %B: tanh((0.5 - pct_b) * 4)
    sma   = close.rolling(20).mean()
    std   = close.rolling(20).std()
    pct_b = (float(close.iloc[-1]) - float((sma - 2*std).iloc[-1])) / (float((4*std).iloc[-1]) + 1e-9)
    ind_scores.append(math.tanh((0.5 - pct_b) * 4.0))

    # MACD histogram normalized by price scale
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    close_std = float(close.rolling(20).std().iloc[-1]) + 1e-9
    ind_scores.append(math.tanh(float((macd_line - sig_line).iloc[-1]) / close_std * 10.0))

    net_score = float(sum(ind_scores) / len(ind_scores))
    std_val   = float(np.std(ind_scores)) if len(ind_scores) > 1 else 0.0
    agreement = 1.0 - min(std_val, 1.0)
    return round(net_score, 4), round(agreement, 4)


def _compute_candlestick(df: pd.DataFrame, idx: int):
    if idx < 2:
        return 50.0, 50.0
    c, c1    = df.iloc[idx], df.iloc[idx-1]
    body     = abs(c["close"] - c["open"])
    rng      = c["high"] - c["low"] + 1e-9
    body_pct = body / rng
    bull, bear = 0, 0
    if body_pct < 0.1: bull += 1; bear += 1
    if (c["close"] > c["open"] and c1["close"] < c1["open"] and
            c["close"] > c1["open"] and c["open"] < c1["close"]): bull += 3
    if (c["close"] < c["open"] and c1["close"] > c1["open"] and
            c["close"] < c1["open"] and c["open"] > c1["close"]): bear += 3
    lower = (c["open"] - c["low"])   if c["close"] > c["open"] else (c["close"] - c["low"])
    upper = (c["high"] - c["close"]) if c["close"] > c["open"] else (c["high"] - c["open"])
    if lower > 2 * body and body_pct > 0.1: bull += 2
    if upper > 2 * body and body_pct > 0.1: bear += 2
    if c["close"] > c["open"] and body_pct > 0.6: bull += 1
    if c["close"] < c["open"] and body_pct > 0.6: bear += 1
    total = bull + bear
    return (round(bull/total*100,1), round(bear/total*100,1)) if total else (50.0, 50.0)


def _compute_chart(df: pd.DataFrame, idx: int):
    window = df.iloc[max(0, idx-30): idx+1]
    if len(window) < 10:
        return 50.0, 50.0
    close = window["close"]
    trend = np.polyfit(range(len(close)), close.values, 1)[0]
    price = close.iloc[-1]
    hi, lo = window["high"].max(), window["low"].min()
    bull, bear = 0, 0
    bull += 2 if trend > 0 else 0
    bear += 2 if trend <= 0 else 0
    bull += 1 if price > (hi + lo) / 2 else 0
    bear += 1 if price <= (hi + lo) / 2 else 0
    if (price - lo) / (price + 1e-9) < 0.02: bull += 2
    if (hi - price) / (price + 1e-9) < 0.02: bear += 2
    total = bull + bear
    return (round(bull/total*100,1), round(bear/total*100,1)) if total else (50.0, 50.0)


def get_signal(df: pd.DataFrame, idx: int, weights: dict) -> dict:
    """
    Compute combined signal at bar idx using 3 price modules only.
    News and sentiment are excluded entirely — flat 50/50 historically,
    they contribute zero signal but dilute the conviction gap.
    Price module weights are renormalized to sum=1.0 before use.
    """
    t_net, t_agr = _compute_technical(df, idx)
    cb, cr       = _compute_candlestick(df, idx)
    hb, hr       = _compute_chart(df, idx)

    # Renormalize 3 price weights to 1.0 regardless of what the file stores
    PRICE_KEYS = ["technical", "candlestick", "chart"]
    pw     = {k: weights.get(k, DEFAULT_WEIGHTS.get(k, 1/3)) for k in PRICE_KEYS}
    pw_sum = sum(pw.values()) + 1e-10
    pw     = {k: v / pw_sum for k, v in pw.items()}

    vectors = {
        "technical":   normalize_technical_soft(t_net, t_agr, pw["technical"],   source="technical"),
        "candlestick": normalize_technical(cb, cr,           pw["candlestick"], source="candlestick"),
        "chart":       normalize_technical(hb, hr,           pw["chart"],       source="chart"),
    }
    return combine_signal_vectors(vectors, CONVICTION_THR)


# ─────────────────────────────────────────────────────────
# TRADE SIMULATOR
# ─────────────────────────────────────────────────────────

def simulate_trade(df: pd.DataFrame,
                   entry_idx: int,
                   direction: str,
                   equity: float) -> dict:
    """
    Simulates a single trade from entry_idx to TP/SL or window close.

    Position sizing:
      risk_amount = equity × RISK_PCT
      atr         = ATR at entry
      sl_dist     = ATR_SL_MULT × atr
      tp_dist     = ATR_TP_MULT × atr
      position    = risk_amount / sl_dist  (in units of the asset)

    Exit logic (checked candle by candle after entry):
      - If candle.low  <= SL (LONG)  or candle.high >= SL (SHORT): SL hit
      - If candle.high >= TP (LONG)  or candle.low  <= TP (SHORT): TP hit
      - If next candle after entry_idx: timeout (window closed)

    For intra-candle ordering: we assume TP and SL can both be hit in the
    same candle. We give the adverse move priority (SL checked first) to
    be conservative — this slightly underestimates real performance.
    """
    entry_bar   = df.iloc[entry_idx]
    entry_price = float(entry_bar["open"])   # enter at open of signal candle
    atr         = compute_atr(df, entry_idx)

    sl_dist = ATR_SL_MULT * atr
    tp_dist = ATR_TP_MULT * atr

    if direction == "LONG":
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:   # SHORT
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    risk_amount = equity * RISK_PCT
    if sl_dist <= 0:
        return None   # degenerate ATR

    # Check only the current bar (window = this candle only)
    # The signal fires at the open; we check if high/low of this same candle
    # hits TP or SL. If neither, we close at the candle close (window end).
    bar = df.iloc[entry_idx]
    exit_price  = None
    hit_tp      = False
    hit_sl      = False

    if direction == "LONG":
        if bar["low"] <= sl:
            exit_price = sl;  hit_sl = True
        elif bar["high"] >= tp:
            exit_price = tp;  hit_tp = True
        else:
            exit_price = float(bar["close"])
    else:
        if bar["high"] >= sl:
            exit_price = sl;  hit_sl = True
        elif bar["low"] <= tp:
            exit_price = tp;  hit_tp = True
        else:
            exit_price = float(bar["close"])

    pct_change = (exit_price - entry_price) / entry_price
    pnl_pct    = pct_change if direction == "LONG" else -pct_change

    position_size = risk_amount / sl_dist
    raw_pnl       = pnl_pct * entry_price * position_size
    max_gain      = (ATR_TP_MULT / ATR_SL_MULT) * risk_amount
    max_loss      = -risk_amount
    gross_pnl     = max(min(raw_pnl, max_gain), max_loss)
    pnl_dollar    = gross_pnl

    direction_correct = pnl_pct > 0

    return {
        "entry_price":       round(entry_price, 4),
        "exit_price":        round(exit_price, 4),
        "sl":                round(sl, 4),
        "tp":                round(tp, 4),
        "atr":               round(atr, 4),
        "direction":         direction,
        "hit_tp":            hit_tp,
        "hit_sl":            hit_sl,
        "timeout":           not hit_tp and not hit_sl,
        "pnl_pct":           round(pnl_pct * 100, 4),
        "pnl_dollar":        round(pnl_dollar, 2),
        "direction_correct": direction_correct,
    }


# ─────────────────────────────────────────────────────────
# PORTFOLIO BACKTESTER — ONE TIMEFRAME
# ─────────────────────────────────────────────────────────

def backtest_timeframe(df: pd.DataFrame,
                        symbol: str,
                        timeframe: str,
                        weights: dict,
                        starting_capital: float = STARTING_CAPITAL) -> dict:
    """
    Runs the full portfolio simulation for one symbol + timeframe.

    Walk-forward: at each bar, we only use data UP TO that bar to compute
    signals — no look-ahead. The first WARMUP_BARS bars are skipped.
    """
    print(f"  [{symbol} {timeframe}] Running backtest over {len(df)} bars...")

    equity          = starting_capital
    equity_curve    = [starting_capital]
    trades          = []
    stay_out_log    = []
    cooldown        = COOLDOWN_BARS.get(timeframe, 0)
    bars_since_trade = cooldown   # start ready to trade
    max_daily       = MAX_TRADES_PER_DAY.get(timeframe, 999)
    trades_today    = 0
    current_day     = None

    # Holdout period only: last 20% of data, never seen during optimisation
    holdout_start = max(WARMUP_BARS, int(len(df) * 0.80))
    print(f"    Holdout window: {df.index[holdout_start].date()} to {df.index[-1].date()} ({len(df) - holdout_start} bars)")

    for idx in range(holdout_start, len(df)):
        bar = df.iloc[idx]
        bars_since_trade += 1

        # Reset daily trade counter at start of new calendar day
        bar_day = bar.name.date() if hasattr(bar.name, 'date') else str(bar.name)[:10]
        if bar_day != current_day:
            current_day  = bar_day
            trades_today = 0

        # Compute signal using the PREVIOUS completed bar (idx-1).
        # signal_audit.py pattern: signal fires after bar N closes → enter
        # at open of bar N+1.  Using idx here would read bar idx's close/high/
        # low/volume and then enter at idx's open — look-ahead bias that
        # artificially inflates Sharpe by entering at the bottom of a move
        # you already know closed at the top.
        combined = get_signal(df, idx - 1, weights)
        direction = combined["direction"]

        # Enforce cooldown: skip if we just completed a trade
        if bars_since_trade < cooldown:
            direction = "STAY OUT"

        # Enforce daily trade cap
        if trades_today >= max_daily:
            direction = "STAY OUT"

        if direction == "STAY OUT":
            # Record whether staying out was correct
            if idx + 1 < len(df):
                next_bar   = df.iloc[idx + 1]
                pct_move   = abs(next_bar["close"] - bar["close"]) / bar["close"] * 100
                stay_out_log.append({
                    "timestamp":    bar.name,
                    "price":        float(bar["close"]),
                    "pct_move":     round(pct_move, 3),
                    "was_correct":  pct_move < STAY_OUT_MOVE_THR,
                })
            continue

        # Simulate trade on this bar
        trade = simulate_trade(df, idx, direction, equity)
        if trade is None:
            continue

        trade["timestamp"] = bar.name
        trade["symbol"]    = symbol
        trade["timeframe"] = timeframe
        trade["equity_before"] = round(equity, 2)

        equity += trade["pnl_dollar"]
        equity  = max(equity, 0.01)   # can't go below zero

        trade["equity_after"] = round(equity, 2)
        trades.append(trade)
        equity_curve.append(equity)
        bars_since_trade = 0   # reset cooldown
        trades_today += 1        # increment daily counter

    return {
        "symbol":       symbol,
        "timeframe":    timeframe,
        "trades":       trades,
        "stay_out_log": stay_out_log,
        "equity_curve": equity_curve,
        "final_equity": equity,
    }


# ─────────────────────────────────────────────────────────
# METRICS CALCULATOR
# ─────────────────────────────────────────────────────────

def compute_metrics(result: dict, starting_capital: float) -> dict:
    trades       = result["trades"]
    equity_curve = result["equity_curve"]
    stay_out_log = result["stay_out_log"]
    final_equity = result["final_equity"]

    if not trades:
        return {"error": "No trades generated"}

    # ── Basic trade stats ────────────────────────────────
    n_trades      = len(trades)
    n_tp          = sum(1 for t in trades if t["hit_tp"])
    n_sl          = sum(1 for t in trades if t["hit_sl"])
    n_timeout     = sum(1 for t in trades if t["timeout"])
    n_dir_correct = sum(1 for t in trades if t["direction_correct"])
    n_long        = sum(1 for t in trades if t["direction"] == "LONG")
    n_short       = sum(1 for t in trades if t["direction"] == "SHORT")

    winners = [t for t in trades if t["pnl_pct"] > 0]
    losers  = [t for t in trades if t["pnl_pct"] <= 0]

    win_rate_tpsl = round(n_tp / n_trades * 100, 1)
    win_rate_dir  = round(n_dir_correct / n_trades * 100, 1)

    avg_win  = round(np.mean([t["pnl_pct"] for t in winners]), 3) if winners else 0.0
    avg_loss = round(np.mean([t["pnl_pct"] for t in losers]),  3) if losers  else 0.0

    gross_profit = sum(t["pnl_dollar"] for t in winners) if winners else 0.0
    gross_loss   = abs(sum(t["pnl_dollar"] for t in losers)) if losers else 1e-10
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    # ── Equity curve stats ───────────────────────────────
    eq = np.array(equity_curve)
    total_return   = round((final_equity - starting_capital) / starting_capital * 100, 2)
    peak           = np.maximum.accumulate(eq)
    drawdowns      = (eq - peak) / peak
    max_drawdown   = round(float(drawdowns.min()) * 100, 2)

    # ── Monthly breakdown ────────────────────────────────
    monthly = defaultdict(float)
    for t in trades:
        ts = t["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        key = ts.strftime("%Y-%m")
        monthly[key] += t["pnl_dollar"]

    monthly_pct = {
        k: round(v / starting_capital * 100, 2)
        for k, v in sorted(monthly.items())
    }

    # ── STAY OUT accuracy ────────────────────────────────
    n_stay_out    = len(stay_out_log)
    n_so_correct  = sum(1 for s in stay_out_log if s["was_correct"])
    stay_out_acc  = round(n_so_correct / n_stay_out * 100, 1) if n_stay_out > 0 else 0.0

    # ── Trades per month ─────────────────────────────────
    if trades:
        first_ts = trades[0]["timestamp"]
        last_ts  = trades[-1]["timestamp"]
        if hasattr(first_ts, "to_pydatetime"):
            first_ts = first_ts.to_pydatetime()
            last_ts  = last_ts.to_pydatetime()
        days_covered = max((last_ts - first_ts).days, 1)
        trades_per_month = round(n_trades / (days_covered / 30), 1)
    else:
        days_covered     = 1
        trades_per_month = 0.0

    # Sharpe: annualised by candle-period frequency for the given timeframe so
    # that 1h, 4h and 1d results are on the same consistent scale as the
    # weight_optimizer. Using candle-period frequency (not trade frequency)
    # matches the weight_optimizer convention and avoids the old sqrt(252) error
    # that was correct only for daily-return strategies.
    trade_rets = np.array([t["pnl_pct"] / 100 for t in trades])
    if len(trade_rets) > 1:
        PERIODS_PER_YEAR = {"1h": 8760, "4h": 2190, "1d": 365}
        timeframe  = result.get("timeframe", "1d")
        n_per_year = PERIODS_PER_YEAR.get(timeframe, 365)
        mean_r     = float(np.mean(trade_rets))
        std_r      = float(np.std(trade_rets))
        sharpe     = round(mean_r / (std_r + 1e-10) * np.sqrt(n_per_year), 2)
    else:
        sharpe = 0.0

    calmar = round(total_return / abs(max_drawdown), 2) if max_drawdown != 0 else float("inf")

    return {
        "n_trades":           n_trades,
        "n_long":             n_long,
        "n_short":            n_short,
        "n_tp":               n_tp,
        "n_sl":               n_sl,
        "n_timeout":          n_timeout,
        "win_rate_tpsl":      win_rate_tpsl,
        "win_rate_direction": win_rate_dir,
        "avg_win_pct":        avg_win,
        "avg_loss_pct":       avg_loss,
        "profit_factor":      profit_factor,
        "total_return_pct":   total_return,
        "max_drawdown_pct":   max_drawdown,
        "sharpe_ratio":       sharpe,
        "calmar_ratio":       calmar,
        "trades_per_month":   trades_per_month,
        "final_equity":       round(final_equity, 2),
        "n_stay_out":         n_stay_out,
        "stay_out_accuracy":  stay_out_acc,
        "monthly_pct":        monthly_pct,
    }


# ─────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────

def print_report(symbol: str, timeframe: str,
                 metrics: dict, starting_capital: float):
    if "error" in metrics:
        print(f"  [{symbol} {timeframe}] {metrics['error']}")
        return

    m  = metrics
    tf_label = {"1h": "1 HOUR", "4h": "4 HOUR", "1d": "1 DAY"}.get(timeframe, timeframe)
    total_r   = m["total_return_pct"]
    r_emoji   = "🟢" if total_r > 0 else "🔴"

    print(f"\n{'═'*62}")
    print(f"  📊 BACKTEST — {symbol}  {tf_label}")
    print(f"  Period: holdout — last 20% of dataset (never seen during optimisation)")
    print(f"{'═'*62}")
    print(f"  Starting capital  : ${starting_capital:,.2f}")
    print(f"  Final equity      : ${m['final_equity']:,.2f}  "
          f"{r_emoji} ({total_r:+.2f}%)")
    print(f"  Max drawdown      : {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio      : {m['sharpe_ratio']:.2f}")
    print(f"  Calmar ratio      : {m['calmar_ratio']:.2f}")
    print()
    print(f"  Total trades      : {m['n_trades']}  "
          f"(LONG: {m['n_long']}  SHORT: {m['n_short']})")
    print(f"  Trades / month    : {m['trades_per_month']}")
    print(f"  TP hits           : {m['n_tp']}  "
          f"SL hits: {m['n_sl']}  Timeout: {m['n_timeout']}")
    print()
    print(f"  Win rate (TP/SL)  : {m['win_rate_tpsl']}%  "
          f"← hit TP before SL or timeout")
    print(f"  Direction accuracy: {m['win_rate_direction']}%  "
          f"← price moved in predicted direction")
    print(f"  Avg win           : +{m['avg_win_pct']:.3f}%  "
          f"Avg loss: {m['avg_loss_pct']:.3f}%")
    print(f"  Profit factor     : {m['profit_factor']}x  "
          f"(gross_profit / gross_loss)")
    print()
    print(f"  STAY OUT signals  : {m['n_stay_out']}")
    print(f"  STAY OUT accuracy : {m['stay_out_accuracy']}%  "
          f"(price moved <{STAY_OUT_MOVE_THR}% next candle)")
    print()
    print(f"  Monthly returns:")
    for month, pct in m["monthly_pct"].items():
        bar    = "█" * min(int(abs(pct) * 2), 20)
        sign   = "+" if pct >= 0 else ""
        color  = "▲" if pct >= 0 else "▼"
        pct_display = f"{sign}{pct:8.2f}%"
        print(f"    {month}  {color}  {pct_display}  {bar}")
    print(f"{'═'*62}")


# ─────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────

def save_results(all_results: list, all_metrics: list):
    """Save trade logs and summary to CSV files."""
    # Trade log
    all_trades = []
    for r in all_results:
        all_trades.extend(r["trades"])
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        df_trades.to_csv("backtest_trades.csv", index=False)
        print(f"\n  💾 Trade log saved to backtest_trades.csv ({len(all_trades)} trades)")

    # Summary
    if all_metrics:
        summary_rows = []
        for m, r in zip(all_metrics, all_results):
            if "error" not in m:
                summary_rows.append({
                    "symbol":              r["symbol"],
                    "timeframe":           r["timeframe"],
                    "total_return_pct":    m["total_return_pct"],
                    "final_equity":        m["final_equity"],
                    "win_rate_tpsl":       m["win_rate_tpsl"],
                    "win_rate_direction":  m["win_rate_direction"],
                    "profit_factor":       m["profit_factor"],
                    "sharpe_ratio":        m["sharpe_ratio"],
                    "calmar_ratio":        m["calmar_ratio"],
                    "max_drawdown_pct":    m["max_drawdown_pct"],
                    "n_trades":            m["n_trades"],
                    "stay_out_accuracy":   m["stay_out_accuracy"],
                })
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv("backtest_summary.csv", index=False)
            print(f"  💾 Summary saved to backtest_summary.csv")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio backtester")
    parser.add_argument("--symbols",    nargs="+", default=["BTC"])
    parser.add_argument("--timeframes", nargs="+", default=["4h", "1d"],
                        choices=["1h", "4h", "1d"])
    parser.add_argument("--capital",    type=float, default=STARTING_CAPITAL)
    parser.add_argument("--days",       type=int,   default=LOOKBACK_DAYS)
    args = parser.parse_args()

    print("\n" + "═"*62)
    print("  📈 CRYPTO ALGO BACKTESTER")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"  Capital:    ${args.capital:,.0f} per portfolio")
    print(f"  Risk:       {RISK_PCT*100:.0f}% per trade")
    print(f"  Period:     holdout (last 20%) | 1h={LOOKBACK_BY_TF['1h']}d  4h={LOOKBACK_BY_TF['4h']}d  1d={LOOKBACK_BY_TF['1d']}d")
    print(f"  Signals:    Technical only (news/sentiment excluded)")
    print(f"  Conviction: {CONVICTION_THR}% adjusted gap threshold")
    print("═"*62)

    per_coin_weights, shared_weights = load_weights()

    all_results = []
    all_metrics = []

    for symbol in args.symbols:
        print(f"\n📥 Fetching data for {symbol}...")
        for tf in args.timeframes:
            interval = TIMEFRAME_CONFIG[tf]["interval"]
            df = fetch_ohlcv(symbol, interval, days=LOOKBACK_BY_TF.get(tf, args.days))
            if df.empty or len(df) < WARMUP_BARS + 10:
                print(f"  ⚠️  Insufficient data for {symbol} {tf} — skipping")
                continue

            sym_key         = symbol.upper()
            coin_tf_weights = per_coin_weights.get(sym_key, {}).get(tf)
            if coin_tf_weights:
                weights = coin_tf_weights
                print(f"    Using per-coin weights for {symbol} {tf}")
            else:
                weights = shared_weights.get(tf, DEFAULT_WEIGHTS)
                print(f"    Using shared weights for {symbol} {tf}")
            result  = backtest_timeframe(df, symbol, tf, weights, args.capital)
            metrics = compute_metrics(result, args.capital)

            all_results.append(result)
            all_metrics.append(metrics)
            print_report(symbol, tf, metrics, args.capital)

    save_results(all_results, all_metrics)

    # Final cross-timeframe comparison
    print("\n" + "═"*62)
    print("  📌 SUMMARY COMPARISON")
    print(f"  {'Symbol':6} {'TF':4} {'Return':>8} {'WR(TP/SL)':>10} "
          f"{'Dir Acc':>8} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"  {'─'*6} {'─'*4} {'─'*8} {'─'*10} {'─'*8} {'─'*6} {'─'*7} {'─'*7}")
    for m, r in zip(all_metrics, all_results):
        if "error" in m:
            continue
        sign = "+" if m["total_return_pct"] >= 0 else ""
        print(f"  {r['symbol']:6} {r['timeframe']:4} "
              f"{sign}{m['total_return_pct']:7.2f}% "
              f"{m['win_rate_tpsl']:9.1f}% "
              f"{m['win_rate_direction']:7.1f}% "
              f"{m['profit_factor']:6.2f}x "
              f"{m['sharpe_ratio']:7.2f} "
              f"{m['max_drawdown_pct']:7.2f}%")
    print("═"*62 + "\n")


if __name__ == "__main__":
    main()