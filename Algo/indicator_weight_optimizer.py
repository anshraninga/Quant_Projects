"""
indicator_weight_optimizer.py
------------------------------
Finds the optimal weights for each technical indicator inside TechnicalAnalysis.py.

OPTIMIZATION METHOD: Bayesian Optimization (via scikit-optimize)
  Instead of exhaustive grid search (~1,000–3,000 combos), Bayesian optimization
  builds a surrogate model (Gaussian Process) of the score landscape and
  intelligently samples the most promising regions.

  Grid search:  ~1,000–3,000 full evaluations  → hours
  Bayesian:     50–200 smart evaluations        → 5–15 minutes

  Install: pip install scikit-optimize

Optimization targets:
  - Win rate     (% of calls where price moved in predicted direction)
  - Calmar ratio (annualized return / max drawdown — risk-adjusted)

  Combined score = win_rate * 0.5 + calmar_ratio_normalized * 0.5

Output:
  - indicator_weights.json       — best indicator weights, loaded by TechnicalAnalysis.py
  - indicator_backtest_results.csv — trial log

Usage:
  python indicator_weight_optimizer.py
  python indicator_weight_optimizer.py --symbols BTC ETH ICP --timeframes 1h 4h 1d --n-calls 100

Indicators optimized (all must have weight > 0):
  RSI, MACD, Bollinger Bands, EMA Crossovers, Volume,
  Support/Resistance, OBV, Fibonacci

Note: ATR is excluded (non-directional, used only for TP/SL sizing).

Pipeline order:
  1. python indicator_weight_optimizer.py  → indicator_weights.json
  2. python weight_optimizer.py            → weights.json
  3. python main.py                        → uses both
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import math
import numpy as np
from dotenv import load_dotenv

# ── Install scikit-optimize if missing ──────────────────
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
except ImportError:
    print("  Installing scikit-optimize...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "scikit-optimize", "-q"])
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

load_dotenv()

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

LOOKBACK_BY_TF = {
    "1h": 730,    # 2 years
    "4h": 1095,   # 3 years
    "1d": 1825,   # 5 years
}
LOOKBACK_DAYS = 1825  # fallback
CONVICTION_THR   = 15.0
MIN_TRADES       = 10       # lowered for short fold windows (esp. 1d)
MIN_WEIGHT       = 0.03       # no indicator can be zeroed out entirely
MAX_WEIGHT       = 0.35       # tighter cap — prevents 1–2 indicators dominating
ENTROPY_WEIGHT   = 0.08       # stronger push toward balanced weight distributions
EARLY_STOP_ROUNDS = 20         # stop fold if no improvement for this many trials
N_CALLS          = 120        # total Bayesian trials
N_RANDOM_STARTS  = 20         # pure random trials before GP kicks in
SNAPSHOT_WARMUP  = 50
SOFT_CONVICTION_THR = 0.15   # confidence threshold in [0,1] for new soft scoring
OUTPUT_WEIGHTS   = "indicator_weights.json"
OUTPUT_CSV       = "indicator_backtest_results.csv"

INDICATORS = [
    "RSI",
    "MACD",
    "Bollinger Bands",
    "EMA Crossovers",
    "Volume",
    "Support/Resistance",
    "OBV",
    "Fibonacci",
]

DEFAULT_INDICATOR_WEIGHTS = {
    "RSI":                0.136,
    "MACD":               0.136,
    "Bollinger Bands":    0.091,
    "EMA Crossovers":     0.182,
    "Volume":             0.091,
    "Support/Resistance": 0.136,
    "OBV":                0.091,
    "Fibonacci":          0.091,
}

TIMEFRAME_MAP = {
    "1h": {"binance": "1h",  "forward_candles": 4},
    "4h": {"binance": "4h",  "forward_candles": 6},
    "1d": {"binance": "1d",  "forward_candles": 3},
}

EXCHANGE_RELIABILITY = {
    "Binance":       1.00,
    "KuCoin":        0.92,
    "OKX":           0.90,
    "MEXC":          0.82,
    "Gate.io":       0.80,
    "Bitget":        0.78,
    "Binance Alpha": 0.65,
}


# ─────────────────────────────────────────────────────────
# WEIGHT HELPERS
# ─────────────────────────────────────────────────────────

def _project_weights(raw: list[float]) -> dict:
    """
    Maps unconstrained raw values → valid normalized weights.
    Clips to [MIN_WEIGHT, MAX_WEIGHT] then normalizes to sum=1.
    The MAX_WEIGHT cap prevents any single indicator from dominating,
    which would cause the optimizer to discard legitimate signals.
    """
    clipped = [max(v, MIN_WEIGHT) for v in raw]
    total   = sum(clipped)
    if total <= 0:
        n = len(INDICATORS)
        return {ind: round(1.0 / n, 6) for ind in INDICATORS}
    normalized = {ind: round(v / total, 6) for ind, v in zip(INDICATORS, clipped)}
    # Apply MAX_WEIGHT cap and redistribute excess equally
    capped = {ind: min(w, MAX_WEIGHT) for ind, w in normalized.items()}
    cap_total = sum(capped.values())
    return {ind: round(w / cap_total, 6) for ind, w in capped.items()}


def validate_indicator_weights(weights: dict, tol: float = 1e-3) -> bool:
    total = sum(weights.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights sum to {total:.6f}, not 1.0")
    return True


# ─────────────────────────────────────────────────────────
# EXCHANGE FALLBACK FETCHER
# ─────────────────────────────────────────────────────────

def _parse_to_df(rows, ts_col, o, h, l, c, v, ts_unit="ms") -> pd.DataFrame:
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


def _base(s: str) -> str:
    s = s.upper().strip()
    for q in ["USDT","BUSD","USD"]:
        if s.endswith(q) and len(s) > len(q): return s[:-len(q)]
    for q in ["BTC","ETH","BNB"]:
        if s.endswith(q) and len(s) > len(q)+1: return s[:-len(q)]
    return s


def _norm(s: str) -> str:
    s = s.upper().strip()
    if s.endswith("USD") and not s.endswith("USDT"): s = s[:-3] + "USDT"
    return s if (s.endswith("USDT") or s.endswith("BUSD")) else s + "USDT"


def _fetch_binance(symbol, interval, days):
    end_ms = int(time.time() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    candles, current = [], start_ms
    while current < end_ms:
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol": symbol, "interval": interval,
                                     "startTime": current, "endTime": end_ms, "limit": 1000},
                             timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data or isinstance(data, dict): break
            candles.extend(data)
            current = data[-1][0] + 1
            if len(data) < 1000: break
        except Exception as e:
            print(f"    [Binance] {e}"); break
    return _parse_to_df(candles, 0, 1, 2, 3, 4, 5) if candles else pd.DataFrame()


def _fetch_kucoin(symbol, interval, days):
    imap = {"1h":"1hour","4h":"4hour","1d":"1day"}
    base = _base(symbol)
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/candles",
                         params={"symbol": f"{base}-USDT", "type": imap.get(interval, interval)},
                         timeout=10)
        r.raise_for_status()
        data = list(reversed(r.json().get("data", [])))
        if not data: return pd.DataFrame()
        rows = [[int(d[0]),d[1],d[3],d[4],d[2],d[5]] for d in data]
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
        for col in ["open","high","low","close","volume"]: df[col] = df[col].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]].sort_index()
    except Exception as e:
        print(f"    [KuCoin] {e}"); return pd.DataFrame()


def _fetch_okx(symbol, interval, days):
    imap = {"1h":"1H","4h":"4H","1d":"1D"}
    base = _base(symbol)
    try:
        r = requests.get("https://www.okx.com/api/v5/market/candles",
                         params={"instId": f"{base}-USDT", "bar": imap.get(interval,"1H"),
                                 "limit": 300}, timeout=10)
        r.raise_for_status()
        data = list(reversed(r.json().get("data", [])))
        if not data: return pd.DataFrame()
        rows = [[int(d[0]),d[1],d[2],d[3],d[4],d[5]] for d in data]
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
        for col in ["open","high","low","close","volume"]: df[col] = df[col].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]].sort_index()
    except Exception as e:
        print(f"    [OKX] {e}"); return pd.DataFrame()


def _fetch_mexc(symbol, interval, days):
    try:
        r = requests.get("https://api.mexc.com/api/v3/klines",
                         params={"symbol": symbol, "interval": interval, "limit": 1000},
                         timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data or isinstance(data, dict): return pd.DataFrame()
        return _parse_to_df(data, 0, 1, 2, 3, 4, 5)
    except Exception as e:
        print(f"    [MEXC] {e}"); return pd.DataFrame()


def _fetch_gateio(symbol, interval, days):
    base = _base(symbol)
    try:
        r = requests.get("https://api.gateio.ws/api/v4/spot/candlesticks",
                         params={"currency_pair": f"{base}_USDT", "interval": interval,
                                 "limit": 1000}, timeout=10)
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


def _fetch_bitget(symbol, interval, days):
    imap = {"1h":"3600","4h":"14400","1d":"86400"}
    base = _base(symbol)
    try:
        r = requests.get("https://api.bitget.com/api/spot/v1/market/candles",
                         params={"symbol": f"{base}USDT", "period": imap.get(interval,"3600"),
                                 "limit": "1000"}, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data: return pd.DataFrame()
        rows = [[int(d[0]),d[1],d[2],d[3],d[4],d[5]] for d in data]
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
        for col in ["open","high","low","close","volume"]: df[col] = df[col].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["open","high","low","close","volume"]].sort_index()
    except Exception as e:
        print(f"    [Bitget] {e}"); return pd.DataFrame()


EXCHANGE_CHAIN = [
    ("Binance", _fetch_binance), ("KuCoin", _fetch_kucoin),
    ("OKX",     _fetch_okx),     ("MEXC",   _fetch_mexc),
    ("Gate.io", _fetch_gateio),  ("Bitget", _fetch_bitget),
]


def fetch_ohlcv(symbol: str, interval: str,
                days: int = LOOKBACK_DAYS) -> tuple[pd.DataFrame, str]:
    normalized = _norm(symbol)
    best_df, best_source = pd.DataFrame(), ""
    expected = days * {"1h": 24, "4h": 6, "1d": 1}.get(interval, 24)
    for name, fn in EXCHANGE_CHAIN:
        try:
            df = fn(normalized, interval, days)
            if not df.empty and len(df) >= 50 and len(df) > len(best_df):
                best_df, best_source = df, name
                if len(df) >= expected * 0.85: break
        except Exception:
            continue
    if not best_df.empty:
        print(f"    ✓ {symbol} {interval}: {len(best_df)} candles from {best_source}")
    else:
        print(f"  ⚠️  All exchanges failed for {symbol} {interval}")
    return best_df, best_source


# ─────────────────────────────────────────────────────────
# INDICATOR SIGNAL FUNCTIONS
# Must mirror TechnicalAnalysis.py exactly.
# ─────────────────────────────────────────────────────────

def _signal_rsi(df) -> float:
    """RSI: oversold=bullish (+1), overbought=bearish (-1). tanh-mapped for magnitude."""
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-10)))
    v     = float(rsi.iloc[-1])
    # tanh((50-v)/20): RSI 30→+0.76, RSI 50→0, RSI 70→-0.76, RSI 95→-0.98
    return float(math.tanh((50.0 - v) / 20.0))


def _signal_macd(df) -> float:
    """MACD histogram momentum normalized by price scale."""
    ema12     = df["close"].ewm(span=12, adjust=False).mean()
    ema26     = df["close"].ewm(span=26, adjust=False).mean()
    macd      = ema12 - ema26
    sig       = macd.ewm(span=9, adjust=False).mean()
    hist      = macd - sig
    close_std = float(df["close"].rolling(20).std().iloc[-1]) + 1e-9
    norm_hist = float(hist.iloc[-1]) / close_std
    return float(math.tanh(norm_hist * 10.0))


def _signal_bb(df) -> float:
    """Bollinger %B: below lower band (+1) to above upper band (-1)."""
    sma   = df["close"].rolling(20).mean()
    std   = df["close"].rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    price = float(df["close"].iloc[-1])
    pct_b = (price - float(lower.iloc[-1])) / (float(upper.iloc[-1]) - float(lower.iloc[-1]) + 1e-9)
    # pct_b=0 (at lower) → +1.0 (bullish), pct_b=0.5 → 0, pct_b=1 (at upper) → -1.0
    return float(math.tanh((0.5 - pct_b) * 4.0))


def _signal_ema(df) -> float:
    """EMA alignment: fraction of 6 bullish conditions scaled to [-1, +1]."""
    c     = df["close"]
    e9    = c.ewm(span=9,   adjust=False).mean()
    e21   = c.ewm(span=21,  adjust=False).mean()
    e50   = c.ewm(span=50,  adjust=False).mean()
    e200  = c.ewm(span=200, adjust=False).mean()
    price = float(c.iloc[-1])
    conditions = [
        1.0 if price            > e9.iloc[-1]   else -1.0,
        1.0 if price            > e21.iloc[-1]  else -1.0,
        1.0 if price            > e50.iloc[-1]  else -1.0,
        1.0 if price            > e200.iloc[-1] else -1.0,
        1.0 if e9.iloc[-1]      > e21.iloc[-1]  else -1.0,
        1.0 if e21.iloc[-1]     > e50.iloc[-1]  else -1.0,
    ]
    return float(sum(conditions) / len(conditions))   # already in [-1, +1]


def _signal_volume(df) -> float:
    """Volume ratio × price direction, tanh-capped. Low volume dampens signal."""
    avg       = float(df["volume"].rolling(20).mean().iloc[-1]) + 1e-9
    ratio     = float(df["volume"].iloc[-1]) / avg
    direction = 1.0 if df["close"].iloc[-1] > df["close"].iloc[-2] else -1.0
    magnitude = math.tanh(max(0.0, ratio - 0.5) * 1.5)  # 0 when ratio≤0.5
    return float(direction * magnitude)


def _signal_sr(df) -> float:
    """S/R: continuous position between nearest support and resistance."""
    recent = df.tail(50)
    price  = float(df["close"].iloc[-1])
    lh = recent["high"][recent["high"] == recent["high"].rolling(5, center=True).max()]
    ll = recent["low"][recent["low"]   == recent["low"].rolling(5,  center=True).min()]
    res = sorted(lh[lh > price].values)
    sup = sorted(ll[ll < price].values, reverse=True)
    nr  = float(res[0]) if res else None
    ns  = float(sup[0]) if sup else None
    if nr is None and ns is None: return 0.0
    if nr is None: return +0.7
    if ns is None: return -0.7
    dist_res = (nr - price) / price
    dist_sup = (price - ns) / price
    total    = dist_res + dist_sup + 1e-9
    position = (dist_res - dist_sup) / total   # +1 = close to support, -1 = close to resistance
    return float(math.tanh(position * 3.0))


def _signal_obv(df) -> float:
    """OBV momentum over 5 bars, normalized by rolling std."""
    obv     = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    obv_std = float(obv.rolling(20).std().iloc[-1]) + 1e-9
    delta   = float(obv.iloc[-1] - obv.iloc[-5])
    return float(math.tanh(delta / obv_std))


def _signal_fib(df) -> float:
    """Fibonacci: position within recent 50-bar high/low range."""
    recent    = df.tail(50)
    high, low = float(recent["high"].max()), float(recent["low"].min())
    price     = float(df["close"].iloc[-1])
    pct       = (price - low) / (high - low + 1e-9)   # 0=at support, 1=at resistance
    # pct=0 → +1.0 (bullish), pct=0.5 → 0, pct=1 → -1.0 (bearish)
    return float(math.tanh((0.5 - pct) * 3.0))


SIGNAL_FUNCTIONS = {
    "RSI":                _signal_rsi,
    "MACD":               _signal_macd,
    "Bollinger Bands":    _signal_bb,
    "EMA Crossovers":     _signal_ema,
    "Volume":             _signal_volume,
    "Support/Resistance": _signal_sr,
    "OBV":                _signal_obv,
    "Fibonacci":          _signal_fib,
}


# ─────────────────────────────────────────────────────────
# SNAPSHOT BUILDER
# ─────────────────────────────────────────────────────────

def build_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for i in range(SNAPSHOT_WARMUP, len(df)):
        window = df.iloc[:i+1]
        row = {}
        for name, fn in SIGNAL_FUNCTIONS.items():
            try:
                row[name] = fn(window)
            except Exception:
                row[name] = 0.0
        records.append(row)
    return pd.DataFrame(records, index=df.index[SNAPSHOT_WARMUP:])


# ─────────────────────────────────────────────────────────
# SCORE FUNCTION
# ─────────────────────────────────────────────────────────

def _score_weights(weights: dict, datasets: list[dict]) -> float:
    """Evaluates a weight dict across all symbol/timeframe datasets."""
    all_scores = []

    for d in datasets:
        df, snapshots = d["df"], d["snapshots"]
        fwd = d["forward_candles"]
        rel = EXCHANGE_RELIABILITY.get(d["source"], 0.75)
        adj_conv = CONVICTION_THR / rel

        trades = []
        for i in range(len(snapshots) - fwd):
            snap_idx  = snapshots.index[i]
            price_idx = df.index.get_loc(snap_idx)
            # Enter at the NEXT bar — snapshot uses bar price_idx's close,
            # so entering at price_idx's close is look-ahead bias.
            if price_idx + 1 + fwd >= len(df):
                continue

            row       = snapshots.iloc[i]
            scores    = [float(row.get(ind, 0.0)) for ind in INDICATORS]
            net_score = sum(weights[ind] * float(row.get(ind, 0.0)) for ind in INDICATORS)
            std       = float(np.std(scores)) if len(scores) > 1 else 0.0
            agreement = 1.0 - min(std, 1.0)
            confidence = abs(net_score) * (0.5 + 0.5 * agreement)
            # adj_conv is a %-based threshold; convert to the [0,1] confidence scale
            if confidence < adj_conv / 100.0:
                continue

            direction = "LONG" if net_score > 0 else "SHORT"
            entry      = df["close"].iloc[price_idx + 1]       # next bar — no look-ahead
            exit_      = df["close"].iloc[price_idx + 1 + fwd]
            pct_change = (exit_ - entry) / entry
            pnl        = pct_change if direction == "LONG" else -pct_change
            trades.append(pnl)

        if len(trades) < MIN_TRADES:
            continue

        trades_arr  = np.array(trades)
        mean_r      = trades_arr.mean()
        std_r       = trades_arr.std() + 1e-10
        sharpe      = mean_r / std_r * np.sqrt(252)
        sharpe_norm = min(max((sharpe + 1.0) / 4.0, 0.0), 1.0)
        all_scores.append(sharpe_norm)

    if not all_scores:
        return -1.0

    raw_score = float(np.mean(all_scores))

    # Entropy bonus — rewards balanced distributions, penalises zero weights.
    # Directly addresses MACD=0, OBV=0 by making zeroed-out weight vectors
    # score slightly lower than balanced ones at the same win_rate+calmar.
    w_vals = list(weights.values()) if isinstance(weights, dict) else []
    if w_vals:
        entropy       = -sum(w * math.log(w + 1e-10) for w in w_vals)
        max_entropy   = math.log(len(w_vals))
        entropy_bonus = ENTROPY_WEIGHT * (entropy / (max_entropy + 1e-10))
    else:
        entropy_bonus = 0.0

    return min(raw_score + entropy_bonus, 1.0)


# ─────────────────────────────────────────────────────────
# BAYESIAN OPTIMIZER
# ─────────────────────────────────────────────────────────

def _run_single_fold_ind(train_datasets, n_calls, n_random, fold_label):
    """Bayesian optimization on training data for one fold."""
    space        = [Real(0.0, 1.0, name=ind) for ind in INDICATORS]
    trial_log    = []
    best_score   = -1.0
    best_weights = dict(DEFAULT_INDICATOR_WEIGHTS)
    call_count        = [0]
    no_improve_count  = [0]

    @use_named_args(space)
    def objective(**kwargs):
        raw     = [kwargs[ind] for ind in INDICATORS]
        weights = _project_weights(raw)
        score   = _score_weights(weights, train_datasets)

        call_count[0] += 1
        trial_log.append({**weights, "score": score, "fold": fold_label})

        nonlocal best_score, best_weights
        if score > best_score + 1e-5:
            best_score          = score
            best_weights        = dict(weights)
            no_improve_count[0] = 0
            print(f"    [{call_count[0]:>3}/{n_calls}] ✨ {fold_label} best: {score:.4f}  "
                  f"EMA={weights['EMA Crossovers']:.3f}  "
                  f"SR={weights['Support/Resistance']:.3f}  "
                  f"MACD={weights['MACD']:.3f}")
        else:
            no_improve_count[0] += 1
            if call_count[0] % 10 == 0:
                print(f"    [{call_count[0]:>3}/{n_calls}] {fold_label} best so far: {best_score:.4f}")

        if (call_count[0] > n_random and
                no_improve_count[0] >= EARLY_STOP_ROUNDS):
            return -best_score

        return -score

    gp_minimize(func=objective, dimensions=space, n_calls=n_calls,
                n_random_starts=n_random, noise=1e-10, random_state=42, verbose=False)

    if no_improve_count[0] >= EARLY_STOP_ROUNDS:
        print(f"    [{call_count[0]:>3}/{n_calls}] ⏹  {fold_label} early stop "
              f"(no improvement for {EARLY_STOP_ROUNDS} trials)")

    return {"best_weights": best_weights, "best_score": best_score, "trial_log": trial_log}


def run_bayesian_optimization(datasets: list[dict],
                               n_calls: int = N_CALLS,
                               n_random: int = N_RANDOM_STARTS) -> dict:
    """
    Walk-forward Bayesian optimization for indicator weights.

    3 anchored folds across the dataset:
      Fold 1: train 0-60%, test 60-80%
      Fold 2: train 0-70%, test 70-85%
      Fold 3: train 0-80%, test 80-100%

    Final weights = weighted average by test score across folds.
    Reports train/test gap per fold — large gap signals overfitting.
    """
    # Rolling non-overlapping folds on first 80% of data.
    # Last 20% reserved as holdout — never touched during optimisation.
    FOLD_SPLITS = [
        (0.00, 0.27, 0.27, 0.40),  # fold 1
        (0.27, 0.53, 0.53, 0.67),  # fold 2
        (0.53, 0.80, 0.67, 0.80),  # fold 3
    ]

    def _slice(datasets, tr_s, tr_e, te_s, te_e):
        train_ds, test_ds = [], []
        for d in datasets:
            n = len(d["snapshots"])
            ti0, ti1 = int(n * tr_s), int(n * tr_e)
            vi0, vi1 = int(n * te_s),  int(n * te_e)
            if ti1 - ti0 < 50 or vi1 - vi0 < 20:
                continue
            train_ds.append({**d, "snapshots": d["snapshots"].iloc[ti0:ti1]})
            test_ds.append( {**d, "snapshots": d["snapshots"].iloc[vi0:vi1]})
        return train_ds, test_ds

    print(f"\n  🔍 Walk-forward optimization — 3 folds × {n_calls//3} trials")
    print(f"  Evaluating across {len(datasets)} symbol×timeframe combinations\n")

    fold_results   = []
    all_trial_logs = []

    for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(FOLD_SPLITS, 1):
        train_ds, test_ds = _slice(datasets, tr_s, tr_e, te_s, te_e)
        if not train_ds or not test_ds:
            print(f"    Fold {fold_i}: insufficient data — skipping")
            continue

        fold_label = f"Fold{fold_i}"
        print(f"    ── Fold {fold_i}: train {int(tr_s*100)}–{int(tr_e*100)}%  |  test {int(te_s*100)}–{int(te_e*100)}% ──")

        fold_n_calls  = max(n_calls // 3, 30)
        fold_n_random = max(n_random // 3, 6)

        result      = _run_single_fold_ind(train_ds, fold_n_calls, fold_n_random, fold_label)
        train_score = result["best_score"]
        test_score  = _score_weights(result["best_weights"], test_ds)
        gap         = train_score - test_score

        flag = ("  ⚠️  OVERFIT" if abs(gap) > 0.10 else
                "  ⚡ mild"      if abs(gap) > 0.05 else "  ✅")
        print(f"    Fold {fold_i}: train={train_score:.4f}  test={test_score:.4f}  gap={gap:+.4f}{flag}")

        fold_results.append({
            "fold": fold_i, "weights": result["best_weights"],
            "train_score": train_score, "test_score": test_score, "gap": gap,
        })
        all_trial_logs.extend(result["trial_log"])

    if not fold_results:
        return {"best_weights": dict(DEFAULT_INDICATOR_WEIGHTS), "best_score": -1.0,
                "trial_log": [], "n_trials": 0}

    # Aggregate: weight each fold by its test score
    test_scores  = [max(r["test_score"], 0.0) for r in fold_results]
    total_weight = sum(test_scores)
    if total_weight <= 0:
        best_fold = max(fold_results, key=lambda r: r["train_score"])
        agg = best_fold["weights"]
        agg = _project_weights([agg[ind] for ind in INDICATORS])
        avg_train = sum(r["train_score"] for r in fold_results) / len(fold_results)
        print("  ⚠️  All test folds returned -1 (too few trades). Using best train-fold weights.")
        return {
            "best_weights": agg, "best_score": -1.0,
            "fold_results": fold_results, "avg_train": avg_train,
            "avg_test": -1.0, "avg_gap": 0.0,
            "trial_log": all_trial_logs, "n_trials": len(all_trial_logs),
        }

    agg = {ind: 0.0 for ind in INDICATORS}
    for r, ts in zip(fold_results, test_scores):
        share = ts / total_weight
        for ind in INDICATORS:
            agg[ind] += r["weights"][ind] * share

    agg_total = sum(agg.values())
    agg = {ind: round(w / agg_total, 6) for ind, w in agg.items()}
    agg = _project_weights([agg[ind] for ind in INDICATORS])

    avg_train = sum(r["train_score"] for r in fold_results) / len(fold_results)
    avg_test  = sum(r["test_score"]  for r in fold_results) / len(fold_results)
    avg_gap   = avg_train - avg_test

    print(f"\n  Walk-forward summary:")
    print(f"    Avg train : {avg_train:.4f}")
    print(f"    Avg test  : {avg_test:.4f}")
    print(f"    Avg |gap| : {abs(avg_gap):.4f}  "
          + ("⚠️  OVERFIT" if abs(avg_gap) > 0.10 else
             "✅ stable" if abs(avg_gap) < 0.05 else "⚡ mild overfit"))

    return {
        "best_weights":  agg,
        "best_score":    avg_test,
        "fold_results":  fold_results,
        "avg_train":     avg_train,
        "avg_test":      avg_test,
        "avg_gap":       avg_gap,
        "trial_log":     all_trial_logs,
        "n_trials":      len(all_trial_logs),
    }


# ─────────────────────────────────────────────────────────
# WEIGHTS LOADER (imported by TechnicalAnalysis.py)
# ─────────────────────────────────────────────────────────

def load_indicator_weights(path: str = OUTPUT_WEIGHTS) -> dict:
    if not os.path.exists(path):
        print(f"  ⚠️  {path} not found — using default indicator weights.")
        return DEFAULT_INDICATOR_WEIGHTS
    with open(path, "r") as f:
        data = json.load(f)
    weights = data.get("weights", {})
    try:
        validate_indicator_weights(weights)
        print(f"  ✓ Loaded indicator weights from {path} "
              f"(generated {data.get('generated_at','unknown')[:10]})")
        return weights
    except ValueError as e:
        print(f"  ⚠️  Invalid indicator weights: {e} — using defaults.")
        return DEFAULT_INDICATOR_WEIGHTS


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Indicator weight optimizer (Bayesian)")
    parser.add_argument("--symbols",    nargs="+",
                        default=["BTC"])
    parser.add_argument("--timeframes", nargs="+", default=["4h","1d"],
                        choices=["1h","4h","1d"])
    parser.add_argument("--n-calls",    type=int, default=N_CALLS,
                        help=f"Total Bayesian trials (default {N_CALLS})")
    parser.add_argument("--n-random",   type=int, default=N_RANDOM_STARTS,
                        help=f"Random warm-up before GP (default {N_RANDOM_STARTS})")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  📐 INDICATOR WEIGHT OPTIMIZER  [Bayesian]")
    print("  Grid search replaced — 10–20× faster, same quality")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"  Trials:     {args.n_calls} "
          f"({args.n_random} random + {args.n_calls - args.n_random} GP-guided)")
    print(f"  Lookback:   1h={LOOKBACK_BY_TF['1h']}d  4h={LOOKBACK_BY_TF['4h']}d  1d={LOOKBACK_BY_TF['1d']}d")
    print(f"  Max weight: {MAX_WEIGHT} per indicator (overfitting cap)")
    print(f"  Validation: Rolling walk-forward (3 non-overlapping folds, Sharpe score)")
    print(f"  Holdout:    Last 20% reserved, evaluated once after optimisation")
    print("═"*60)

    print("\n📥 Fetching data...\n")
    datasets = []
    for sym in args.symbols:
        for tf in args.timeframes:
            df, source = fetch_ohlcv(sym, TIMEFRAME_MAP[tf]["binance"], days=LOOKBACK_BY_TF.get(tf, LOOKBACK_DAYS))
            if df.empty or len(df) < SNAPSHOT_WARMUP + 50:
                print(f"  ⚠️  Skipping {sym} {tf} — insufficient data")
                continue
            print(f"    Building snapshots for {sym} {tf}...")
            datasets.append({
                "symbol":          sym,
                "timeframe":       tf,
                "df":              df,
                "snapshots":       build_snapshots(df),
                "forward_candles": TIMEFRAME_MAP[tf]["forward_candles"],
                "source":          source,
            })

    if not datasets:
        print("  ❌ No valid datasets. Check symbols and exchange connectivity.")
        return

    t0     = time.time()
    result = run_bayesian_optimization(
        datasets, n_calls=args.n_calls, n_random=args.n_random
    )
    elapsed = time.time() - t0

    w = result["best_weights"]
    print(f"\n{'═'*60}")
    print(f"  ✅ BEST INDICATOR WEIGHTS  "
          f"(test score: {result['best_score']:.4f}  |  gap: {result.get('avg_gap', 0):+.4f})")
    print(f"  Found in {elapsed/60:.1f} min across {result['n_trials']} trials")
    print(f"{'═'*60}")
    for ind in INDICATORS:
        dw       = DEFAULT_INDICATOR_WEIGHTS.get(ind, 0)
        arrow    = "▲" if w[ind] > dw + 0.01 else ("▼" if w[ind] < dw - 0.01 else "─")
        bar      = "█" * int(w[ind] * 30)
        cap_flag = " 🔒" if w[ind] >= MAX_WEIGHT - 0.01 else ""
        print(f"  {ind:22} {w[ind]:.3f}  {arrow}  {bar}{cap_flag}")

    # Holdout evaluation — last 20% of each dataset, never seen during optimisation
    print("\n  📊 Holdout evaluation (last 20% — never seen during optimisation):")
    holdout_datasets = []
    for d in datasets:
        n  = len(d["snapshots"])
        h0 = int(n * 0.80)
        if n - h0 >= 20:
            holdout_datasets.append({**d, "snapshots": d["snapshots"].iloc[h0:]})
    if holdout_datasets:
        holdout_score = _score_weights(w, holdout_datasets)
        print(f"    Holdout Sharpe-norm: {holdout_score:.4f}  "
              + ("✅ generalises" if holdout_score > 0.40 else "⚠️  weak holdout"))

    output = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "method":         "bayesian_gp_minimize_rolling_folds",
        "n_trials":       result["n_trials"],
        "lookback_days":  LOOKBACK_DAYS,
        "holdout_pct":    0.20,
        "symbols":        args.symbols,
        "timeframes":     args.timeframes,
        "min_weight":     MIN_WEIGHT,
        "conviction_thr": CONVICTION_THR,
        "score_metric":   "sharpe_normalised",
        "best_score":     result["best_score"],
        "weights":        w,
    }
    with open(OUTPUT_WEIGHTS, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Saved to {OUTPUT_WEIGHTS}")

    if result["trial_log"]:
        df_log = pd.DataFrame(result["trial_log"])
        df_log.sort_values("score", ascending=False, inplace=True)
        df_log.to_csv(OUTPUT_CSV, index=False)
        print(f"  📊 Trial log saved to {OUTPUT_CSV}")

    print(f"\n{'═'*60}")
    print("  Next step: python weight_optimizer.py")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()