"""
weight_optimizer.py
-------------------
Finds the optimal module weights for each timeframe (1h, 4h, 1d).

OPTIMIZATION METHOD: Bayesian Optimization (via scikit-optimize)
  Instead of exhaustive grid search (~6,000+ combos), Bayesian optimization
  builds a surrogate model (Gaussian Process) of the score landscape and
  intelligently samples the most promising regions.

  Grid search:  ~6,000 full evaluations  → hours
  Bayesian:     50–150 smart evaluations → 3–10 minutes

  Install: pip install scikit-optimize

Optimization targets:
  - Win rate     (% of calls where price moved in predicted direction)
  - Calmar ratio (annualized return / max drawdown — risk-adjusted)

  Combined score = win_rate * 0.5 + calmar_ratio_normalized * 0.5

Output:
  - weights.json         — best module weights per timeframe, for main.py
  - backtest_results.csv — trial log

Usage:
  python weight_optimizer.py
  python weight_optimizer.py --symbols BTC ETH ICP --timeframes 1h 4h 1d --n-calls 100

Run order:
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

try:
    from SignalNormalizer import normalize_technical, normalize_technical_soft, combine_signal_vectors
except ImportError:
    print("  ❌ SignalNormalizer.py not found. Make sure it is in the same directory.")
    sys.exit(1)

# ── Use the same indicator signal functions as TechnicalAnalysis.py ──────────
# This eliminates the mismatch between the 4-indicator simplified _compute_technical
# and the 8-indicator version that runs at inference time in TechnicalAnalysis.py.
try:
    from indicator_weight_optimizer import (
        SIGNAL_FUNCTIONS as _IND_SIGNAL_FNS,
        load_indicator_weights,
    )
    _IND_WEIGHTS = load_indicator_weights()
    print("  ✓ weight_optimizer: using 8-indicator signal functions from indicator_weight_optimizer")
except Exception as _e:
    print(f"  ⚠️  Could not import indicator signal functions ({_e}) — falling back to simplified 4-indicator version")
    _IND_SIGNAL_FNS = None
    _IND_WEIGHTS    = None

load_dotenv()

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

# Per-timeframe lookbacks — more data for longer timeframes.
# 1h: 2 years covers multiple market cycles, fast enough to fetch
# 4h: 3 years gives meaningful fold sizes
# 1d: 5 years covers bull + bear + sideways regimes properly
LOOKBACK_BY_TF = {
    "1h": 730,    # 2 years  = 17,520 candles
    "4h": 1095,   # 3 years  = 6,570 candles
    "1d": 1825,   # 5 years  = 1,825 candles
}
LOOKBACK_DAYS = 1825  # used as fallback only
CONVICTION_THR   = 15.0
MIN_TRADES       = 10       # minimum trades per fold to score (lowered for 1d short windows)
MIN_WEIGHT       = 0.05       # no module can be zeroed out
MAX_WEIGHT       = 0.40       # no module can dominate (tighter cap reduces overfit)
ENTROPY_WEIGHT   = 0.08       # stronger push toward balanced weight distributions
EARLY_STOP_ROUNDS = 20        # stop fold if no improvement for this many trials
N_CALLS          = 100        # total Bayesian trials per timeframe
N_RANDOM_STARTS  = 15         # pure random trials before GP kicks in
SNAPSHOT_WARMUP  = 200        # bars to skip for EMA200 convergence
OUTPUT_WEIGHTS   = "weights.json"
OUTPUT_CSV       = "backtest_results.csv"

MODULES       = ["technical", "candlestick", "chart", "news", "sentiment"]
# Only price-based modules are searchable — news/sentiment snapshots are always
# flat 50/50 so the optimizer cannot learn useful NLP weights from price history.
# Searching them adds 2 free parameters that fit noise, not signal.
PRICE_MODULES = ["technical", "candlestick", "chart"]
FIXED_NLP_WEIGHT = 0.125   # applied to both news and sentiment after projection

TIMEFRAME_MAP = {
    "1h": {"interval": "1h", "forward_candles": 4},
    "4h": {"interval": "4h", "forward_candles": 6},
    "1d": {"interval": "1d", "forward_candles": 3},
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

# Equal module weights — no optimization needed at module level.
# The signal quality comes from individual indicators (validated via Sharpe audit),
# not from precise module weighting. Equal weights are more robust across regimes
# and eliminate the module-level overfitting problem entirely.
DEFAULT_WEIGHTS = {
    "technical":   0.333,
    "candlestick": 0.333,
    "chart":       0.334,
}


# ─────────────────────────────────────────────────────────
# WEIGHT HELPERS
# ─────────────────────────────────────────────────────────


def _project_price_weights(raw: list[float]) -> dict:
    """
    Projects 3 raw values onto PRICE_MODULES, normalized to sum=1.0.
    News/sentiment are fully excluded — they are always flat historically
    and contribute zero net signal while consuming weight budget.
    """
    clipped = [max(v, MIN_WEIGHT) for v in raw]
    total   = sum(clipped)
    if total <= 0:
        return {m: round(1.0 / len(PRICE_MODULES), 6) for m in PRICE_MODULES}
    normalized = {m: v / total for m, v in zip(PRICE_MODULES, clipped)}
    capped     = {m: min(w, MAX_WEIGHT) for m, w in normalized.items()}
    cap_total  = sum(capped.values())
    return {m: round(w / cap_total, 6) for m, w in capped.items()}


def validate_weights(weights: dict, tol: float = 1e-3) -> bool:
    total = sum(weights.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1.0 — got {total:.6f}")
    if any(v < 0 for v in weights.values()):
        raise ValueError(f"All weights must be >= 0. Got: {weights}")
    return True


# ─────────────────────────────────────────────────────────
# EXCHANGE FALLBACK FETCHER
# ─────────────────────────────────────────────────────────

def _parse_to_df(rows, ts_col, o, h, l, c, v, ts_unit="ms") -> pd.DataFrame:
    try:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df[ts_col], unit=ts_unit)
        for col, idx in [("open",o),("high",h),("low",l),("close",c),("volume",v)]:
            df[col] = df[idx].astype(float)
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


_EXCHANGE_FETCHERS = [
    ("Binance", _fetch_binance), ("KuCoin", _fetch_kucoin),
    ("OKX",     _fetch_okx),     ("MEXC",   _fetch_mexc),
    ("Gate.io", _fetch_gateio),  ("Bitget", _fetch_bitget),
]


def fetch_ohlcv(symbol: str, interval: str,
                days: int = LOOKBACK_DAYS) -> tuple[pd.DataFrame, str]:
    normalized = _norm(symbol)
    best_df, best_source = pd.DataFrame(), ""
    expected = days * {"1h": 24, "4h": 6, "1d": 1}.get(interval, 24)
    for name, fn in _EXCHANGE_FETCHERS:
        try:
            df = fn(normalized, interval, days)
            if not df.empty and len(df) >= 50 and len(df) > len(best_df):
                best_df, best_source = df, name
                if len(df) >= expected * 0.85: break
        except Exception:
            continue
    if not best_df.empty:
        rel = EXCHANGE_RELIABILITY.get(best_source, 0.75)
        print(f"  ✓ {symbol} {interval}: {len(best_df)} candles "
              f"from {best_source} (reliability {rel:.0%})")
    else:
        print(f"  ⚠️  All exchanges failed for {symbol} {interval}")
    return best_df, best_source


# ─────────────────────────────────────────────────────────
# SIGNAL COMPUTERS
# ─────────────────────────────────────────────────────────

def _compute_technical(df: pd.DataFrame, idx: int) -> tuple[float, float]:
    """
    Computes the technical signal at bar `idx`.

    Returns (net_score, agreement) instead of (bull_pct, bear_pct).
      net_score  ∈ [-1, +1]: positive=bullish, magnitude preserved (Fix A)
      agreement  ∈ [0,  1]:  1=unanimous, 0=maximally split (Fix B)

    Uses the same 8 signal functions as TechnicalAnalysis.py when available.
    """
    window = df.iloc[max(0, idx - 200): idx + 1]

    if _IND_SIGNAL_FNS is not None and _IND_WEIGHTS is not None:
        if len(window) < 50:
            return 0.0, 0.5
        ind_scores: dict[str, float] = {}
        n = len(_IND_SIGNAL_FNS)
        for name, fn in _IND_SIGNAL_FNS.items():
            try:
                ind_scores[name] = float(fn(window))
            except Exception:
                ind_scores[name] = 0.0
        net_score = sum(_IND_WEIGHTS.get(name, 1.0 / n) * s
                        for name, s in ind_scores.items())
        score_vals = list(ind_scores.values())
        std        = float(np.std(score_vals)) if len(score_vals) > 1 else 0.0
        agreement  = 1.0 - min(std, 1.0)
        return round(float(net_score), 4), round(agreement, 4)

    # ── Fallback: simplified 4-indicator soft scoring ────────────────────────
    if len(window) < 20:
        return 0.0, 0.5
    close      = window["close"]
    ind_scores = []

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-10)))
    ind_scores.append(math.tanh((50.0 - float(rsi.iloc[-1])) / 20.0))

    # EMA crossover
    e9  = close.ewm(span=9,  adjust=False).mean()
    e21 = close.ewm(span=21, adjust=False).mean()
    ind_scores.append(1.0 if e9.iloc[-1] > e21.iloc[-1] else -1.0)

    # Bollinger %B
    sma   = close.rolling(20).mean()
    std   = close.rolling(20).std()
    pct_b = (float(close.iloc[-1]) - float((sma - 2*std).iloc[-1])) / (float((4*std).iloc[-1]) + 1e-9)
    ind_scores.append(math.tanh((0.5 - pct_b) * 4.0))

    # MACD histogram normalized by price scale
    macd_line  = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig_line   = macd_line.ewm(span=9, adjust=False).mean()
    close_std  = float(close.rolling(20).std().iloc[-1]) + 1e-9
    norm_hist  = float((macd_line - sig_line).iloc[-1]) / close_std
    ind_scores.append(math.tanh(norm_hist * 10.0))

    net_score = float(sum(ind_scores) / len(ind_scores))
    std_val   = float(np.std(ind_scores)) if len(ind_scores) > 1 else 0.0
    agreement = 1.0 - min(std_val, 1.0)
    return round(net_score, 4), round(agreement, 4)


def _compute_candlestick(df: pd.DataFrame, idx: int) -> tuple[float, float]:
    if idx < 2:
        return 50.0, 50.0
    c, c1    = df.iloc[idx], df.iloc[idx - 1]
    body     = abs(c["close"] - c["open"])
    rng      = c["high"] - c["low"] + 1e-9
    body_pct = body / rng
    bull, bear = 0, 0
    if body_pct < 0.1: bull += 1; bear += 1
    if (c["close"] > c["open"] and c1["close"] < c1["open"] and
            c["close"] > c1["open"] and c["open"] < c1["close"]): bull += 3
    if (c["close"] < c["open"] and c1["close"] > c1["open"] and
            c["close"] < c1["open"] and c["open"] > c1["close"]): bear += 3
    lower = (c["open"] - c["low"]) if c["close"] > c["open"] else (c["close"] - c["low"])
    upper = (c["high"] - c["close"]) if c["close"] > c["open"] else (c["high"] - c["open"])
    if lower > 2 * body and body_pct > 0.1: bull += 2
    if upper > 2 * body and body_pct > 0.1: bear += 2
    if c["close"] > c["open"] and body_pct > 0.6: bull += 1
    if c["close"] < c["open"] and body_pct > 0.6: bear += 1
    total = bull + bear
    return (round(bull/total*100,1), round(bear/total*100,1)) if total else (50.0, 50.0)


def _compute_chart(df: pd.DataFrame, idx: int) -> tuple[float, float]:
    window = df.iloc[max(0, idx - 30): idx + 1]
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
    if (price - lo) / (price + 1e-9) < 0.02:      bull += 2
    if (hi - price) / (price + 1e-9) < 0.02:      bear += 2
    total = bull + bear
    return (round(bull/total*100,1), round(bear/total*100,1)) if total else (50.0, 50.0)


# ─────────────────────────────────────────────────────────
# SNAPSHOT BUILDER
# Starts at SNAPSHOT_WARMUP=200 so EMA200 has converged.
# ─────────────────────────────────────────────────────────

def build_signal_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    print(f"    Computing snapshots (warmup={SNAPSHOT_WARMUP} bars)...")
    records = []
    for i in range(SNAPSHOT_WARMUP, len(df)):
        t_net, t_agr = _compute_technical(df, i)
        cb, cr       = _compute_candlestick(df, i)
        hb, hr       = _compute_chart(df, i)
        records.append({
            "tech_net":   t_net, "tech_agr":   t_agr,   # continuous score + agreement
            "cs_bull":    cb,    "cs_bear":    cr,
            "chart_bull": hb,    "chart_bear": hr,
            "news_bull":  50.0,  "news_bear":  50.0,     # flat — no historical NLP
            "sent_bull":  50.0,  "sent_bear":  50.0,
        })
    snap = pd.DataFrame(records, index=df.index[SNAPSHOT_WARMUP:])
    print(f"    ✓ {len(snap)} snapshots")
    return snap


# ─────────────────────────────────────────────────────────
# SCORE FUNCTION
# Uses normalize_technical + combine_signal_vectors — identical
# to what main.py does at runtime.
# ─────────────────────────────────────────────────────────

def _score_weights(weights: dict, datasets: list[dict], timeframe: str = "1d") -> float:
    all_scores = []

    PERIODS_PER_YEAR = {
        "1h":  8760,   # 365 × 24
        "4h":  2190,   # 365 × 6
        "1d":  365,
    }
    annualise = np.sqrt(PERIODS_PER_YEAR.get(timeframe, 365))

    for d in datasets:
        df, snapshots = d["df"], d["snapshots"]
        fwd = d["forward_candles"]
        rel = EXCHANGE_RELIABILITY.get(d["source"], 0.75)
        adj_conv = CONVICTION_THR / rel

        trades = []
        for i in range(len(snapshots) - fwd):
            snap_idx  = snapshots.index[i]
            price_idx = df.index.get_loc(snap_idx)
            # Enter at the NEXT bar after the signal bar to eliminate
            # look-ahead bias (signal uses bar price_idx's close, so the
            # earliest tradeable price is bar price_idx+1).
            if price_idx + 1 + fwd >= len(df):
                continue

            snap = snapshots.iloc[i]
            # 3 price modules only — news/sentiment excluded from scoring.
            # They are always flat (50/50) in historical snapshots and
            # contribute zero net signal while consuming weight budget.
            PRICE_KEYS = ["technical", "candlestick", "chart"]
            pw     = {k: weights.get(k, 1/3) for k in PRICE_KEYS}
            pw_sum = sum(pw.values()) + 1e-10
            pw     = {k: v / pw_sum for k, v in pw.items()}
            vectors = {
                "technical":   normalize_technical_soft(
                    snap["tech_net"], snap["tech_agr"],
                    pw["technical"], source="technical"),
                "candlestick": normalize_technical(
                    snap["cs_bull"], snap["cs_bear"],
                    pw["candlestick"], source="candlestick"),
                "chart":       normalize_technical(
                    snap["chart_bull"], snap["chart_bear"],
                    pw["chart"], source="chart"),
            }

            combined = combine_signal_vectors(vectors, adj_conv)
            if combined["direction"] == "STAY OUT":
                continue

            direction  = combined["direction"]
            entry      = df["close"].iloc[price_idx + 1]       # next bar — no look-ahead
            exit_      = df["close"].iloc[price_idx + 1 + fwd]
            pct_change = (exit_ - entry) / entry
            pnl        = pct_change if direction == "LONG" else -pct_change
            trades.append(pnl)

        if len(trades) < MIN_TRADES:
            continue

        # Sharpe ratio: industry-standard risk-adjusted return metric.
        # More interpretable than win_rate+calmar and directly comparable
        # across strategies and timeframes.
        trades_arr = np.array(trades)
        mean_r  = trades_arr.mean()
        std_r   = trades_arr.std() + 1e-10
        sharpe  = mean_r / std_r * annualise
        # Normalise to 0–1 range: cap at 3.0 (exceptional), floor at -1
        sharpe_norm = min(max((sharpe + 1.0) / 4.0, 0.0), 1.0)
        all_scores.append(sharpe_norm)

    if not all_scores:
        return -1.0

    raw_score = float(np.mean(all_scores))

    # ── Entropy bonus ──────────────────────────────────────────
    # Rewards balanced weight distributions. A vector like
    # [0.50, 0.00, 0.00, 0.45, 0.05] has low entropy and gets a
    # small penalty vs a balanced [0.25, 0.20, 0.20, 0.20, 0.15].
    # This directly discourages zeroing out modules (e.g. MACD=0)
    # and reduces regime-specific overfitting.
    w_vals = list(weights.values()) if isinstance(weights, dict) else []
    if w_vals:
        entropy     = -sum(w * math.log(w + 1e-10) for w in w_vals)
        max_entropy = math.log(len(w_vals))
        entropy_bonus = ENTROPY_WEIGHT * (entropy / (max_entropy + 1e-10))
    else:
        entropy_bonus = 0.0

    return min(raw_score + entropy_bonus, 1.0)


# ─────────────────────────────────────────────────────────
# BAYESIAN OPTIMIZER — PER TIMEFRAME
# ─────────────────────────────────────────────────────────

def _run_single_fold(train_datasets: list[dict],
                      timeframe: str,
                      n_calls: int,
                      n_random: int,
                      fold_label: str) -> dict:
    """
    Runs Bayesian optimization on the training split of one fold.
    Returns best weights and score found on the training data.
    """
    # Only search over price-based modules — NLP weights are fixed at FIXED_NLP_WEIGHT
    # because news/sentiment snapshots are always 50/50 (no historical NLP data).
    # Removing these 2 free parameters cuts the search space from 5D → 3D and
    # prevents the GP from fitting noise into the NLP weight dimensions.
    space        = [Real(MIN_WEIGHT, 1.0, name=mod) for mod in PRICE_MODULES]
    trial_log    = []
    best_score   = -1.0
    best_weights = dict(DEFAULT_WEIGHTS)
    call_count        = [0]
    no_improve_count  = [0]   # early stopping counter

    @use_named_args(space)
    def objective(**kwargs):
        raw     = [kwargs[mod] for mod in PRICE_MODULES]
        weights = _project_price_weights(raw)
        score   = _score_weights(weights, train_datasets, timeframe)

        call_count[0] += 1
        trial_log.append({**{f"w_{m}": weights[m] for m in PRICE_MODULES},
                           "score": score, "timeframe": timeframe, "fold": fold_label})

        nonlocal best_score, best_weights
        if score > best_score + 1e-5:   # meaningful improvement threshold
            best_score          = score
            best_weights        = dict(weights)
            no_improve_count[0] = 0
            print(f"    [{call_count[0]:>3}/{n_calls}] ✨ {fold_label} best: {score:.4f}  "
                  f"tech={weights['technical']:.2f}  cs={weights['candlestick']:.2f}  "
                  f"chart={weights['chart']:.2f}")
        else:
            no_improve_count[0] += 1
            if call_count[0] % 10 == 0:
                print(f"    [{call_count[0]:>3}/{n_calls}] {fold_label} best so far: {best_score:.4f}")

        # Early stopping: if no improvement for EARLY_STOP_ROUNDS trials
        # after the random warm-up phase, signal convergence.
        # gp_minimize doesn't support mid-run stopping natively, so we
        # return a large penalty to steer the GP away from this region
        # while still completing the run (avoids skopt internal errors).
        if (call_count[0] > n_random and
                no_improve_count[0] >= EARLY_STOP_ROUNDS):
            return -best_score   # return best known — GP stops exploring

        return -score

    gp_minimize(
        func=objective, dimensions=space, n_calls=n_calls,
        n_random_starts=n_random, noise=1e-10, random_state=42, verbose=False,
    )

    if no_improve_count[0] >= EARLY_STOP_ROUNDS:
        print(f"    [{call_count[0]:>3}/{n_calls}] ⏹  {fold_label} early stop "
              f"(no improvement for {EARLY_STOP_ROUNDS} trials)")

    return {"best_weights": best_weights, "best_score": best_score, "trial_log": trial_log}


def optimize_timeframe_bayesian(datasets: list[dict],
                                 timeframe: str,
                                 n_calls: int = N_CALLS,
                                 n_random: int = N_RANDOM_STARTS) -> dict:
    """
    Walk-forward Bayesian optimization for a single timeframe.

    Splits each dataset's snapshots into 3 folds:
      Fold 1: train 0–60%,  test 60–80%
      Fold 2: train 0–70%,  test 70–85%   (anchored — always starts from 0)
      Fold 3: train 0–80%,  test 80–100%

    Anchored walk-forward (always train from start) is used rather than
    rolling windows because our datasets are only ~1 year — rolling would
    leave too few training samples per fold.

    Each fold independently optimizes on its train slice, then scores on
    its test slice. Final weights = weighted average across folds where
    each fold's contribution scales with its test score (better-generalizing
    folds contribute more).

    Overfitting signal: large gap between train_score and test_score.
    Healthy: test ≈ train ± 0.05. Overfit: test < train - 0.10.
    """
    # Rolling non-overlapping folds on the training portion (first 80% of data).
    # The last 20% is reserved as a true holdout and never touched here.
    # Each fold trains on a completely different slice — no shared training data
    # between folds, which eliminates the shared-data inflation artifact.
    #
    #  Fold 1: train 0–27%,  test 27–40%
    #  Fold 2: train 27–53%, test 53–67%
    #  Fold 3: train 53–80%, test 67–80%  (test overlaps slightly to use all data)
    FOLD_SPLITS = [
        (0.00, 0.27, 0.27, 0.40),  # fold 1
        (0.27, 0.53, 0.53, 0.67),  # fold 2
        (0.53, 0.80, 0.67, 0.80),  # fold 3
    ]

    def _slice_datasets(datasets, train_start, train_end, test_start, test_end):
        train_ds, test_ds = [], []
        for d in datasets:
            n = len(d["snapshots"])
            ti0, ti1 = int(n * train_start), int(n * train_end)
            vi0, vi1 = int(n * test_start),  int(n * test_end)
            if ti1 - ti0 < 50 or vi1 - vi0 < 20:
                continue
            train_snap = d["snapshots"].iloc[ti0:ti1]
            test_snap  = d["snapshots"].iloc[vi0:vi1]
            train_ds.append({**d, "snapshots": train_snap})
            test_ds.append({**d,  "snapshots": test_snap})
        return train_ds, test_ds

    print(f"\n  [{timeframe}] Walk-forward optimization — 3 folds × {n_calls} trials "
          f"across {len(datasets)} symbols...")

    fold_results  = []
    all_trial_logs = []

    for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(FOLD_SPLITS, 1):
        train_ds, test_ds = _slice_datasets(datasets, tr_s, tr_e, te_s, te_e)
        if not train_ds or not test_ds:
            print(f"    Fold {fold_i}: insufficient data — skipping")
            continue

        fold_label  = f"Fold{fold_i}"
        train_pct_s = int(tr_s * 100)
        train_pct_e = int(tr_e * 100)
        test_pct_s  = int(te_s * 100)
        test_pct_e  = int(te_e * 100)
        print(f"\n    ── Fold {fold_i}: train {train_pct_s}–{train_pct_e}%  |  test {test_pct_s}–{test_pct_e}% ──")

        fold_n_calls  = max(n_calls // 3, 30)
        fold_n_random = max(n_random // 3, 8)

        result     = _run_single_fold(train_ds, timeframe, fold_n_calls, fold_n_random, fold_label)
        train_score = result["best_score"]
        test_score  = _score_weights(result["best_weights"], test_ds, timeframe)
        gap         = train_score - test_score

        overfit_warn = ""
        if abs(gap) > 0.10:
            overfit_warn = "  ⚠️  HIGH VARIANCE (|gap| > 0.10)"
        elif abs(gap) > 0.05:
            overfit_warn = "  ⚡ mild instability"

        print(f"    Fold {fold_i} result: "
              f"train={train_score:.4f}  test={test_score:.4f}  "
              f"gap={gap:+.4f}{overfit_warn}")

        fold_results.append({
            "fold":          fold_i,
            "weights":       result["best_weights"],
            "train_score":   train_score,
            "test_score":    test_score,
            "gap":           gap,
        })
        all_trial_logs.extend(result["trial_log"])

    if not fold_results:
        return {"best_weights": dict(DEFAULT_WEIGHTS), "best_score": -1.0, "trial_log": []}

    # ── Aggregate: weight each fold by its test score ──
    # Folds that generalize better contribute more to the final weights.
    # Use max(test_score, 0) so negative-scoring folds don't subtract.
    # If all test scores are <= 0 (e.g. 1d with too few test candles),
    # fall back to equal weighting across folds so we still get an answer.
    test_scores = [max(r["test_score"], 0.0) for r in fold_results]
    total_weight = sum(test_scores)
    if total_weight <= 0:
        # All folds returned -1 (insufficient test trades) — use equal weights
        # and take the best train-score fold's weights as the answer
        best_fold = max(fold_results, key=lambda r: r["train_score"])
        agg_weights = best_fold["weights"]
        agg_weights = _project_price_weights([agg_weights[mod] for mod in PRICE_MODULES])
        avg_train = sum(r["train_score"] for r in fold_results) / len(fold_results)
        print(f"  [{timeframe}] ⚠️  All test folds returned -1 (too few trades in test window)")
        print(f"    Using best train-fold weights. Consider adding more symbols or reducing MIN_TRADES.")
        return {
            "best_weights": agg_weights,
            "best_score":   -1.0,
            "fold_results": fold_results,
            "avg_train":    avg_train,
            "avg_test":     -1.0,
            "avg_gap":      0.0,
            "trial_log":    all_trial_logs,
        }

    agg_weights = {mod: 0.0 for mod in PRICE_MODULES}
    for r, ts in zip(fold_results, test_scores):
        fold_share = ts / total_weight
        for mod in PRICE_MODULES:
            agg_weights[mod] += r["weights"][mod] * fold_share

    # Re-normalize and apply cap
    agg_weights = _project_price_weights([agg_weights[mod] for mod in PRICE_MODULES])

    avg_train = sum(r["train_score"] for r in fold_results) / len(fold_results)
    avg_test  = sum(r["test_score"]  for r in fold_results) / len(fold_results)
    avg_gap   = avg_train - avg_test

    print(f"\n  [{timeframe}] Walk-forward summary:")
    print(f"    Avg train score : {avg_train:.4f}")
    print(f"    Avg test score  : {avg_test:.4f}")
    print(f"    Avg |gap|       : {abs(avg_gap):.4f}  "
          + ("⚠️  HIGH VARIANCE" if abs(avg_gap) > 0.10 else "✅ stable" if abs(avg_gap) < 0.05 else "⚡ mild instability"))

    return {
        "best_weights":  agg_weights,
        "best_score":    avg_test,    # report test score, not train
        "fold_results":  fold_results,
        "avg_train":     avg_train,
        "avg_test":      avg_test,
        "avg_gap":       avg_gap,
        "trial_log":     all_trial_logs,
    }


# ─────────────────────────────────────────────────────────
# WEIGHTS LOADER — imported by main.py
# ─────────────────────────────────────────────────────────

def load_weights(path: str = OUTPUT_WEIGHTS) -> dict:
    """Load shared (averaged) module weights from weights.json.

    Returns {tf: weights} — the shared fallback used by main.py.
    Supports both the old format (weights: {tf: ...}) and the new
    per-coin format (shared: {tf: ...}).
    For per-coin weights, use load_per_coin_weights().
    """
    defaults = {tf: dict(DEFAULT_WEIGHTS) for tf in ["1h", "4h", "1d"]}
    if not os.path.exists(path):
        print(f"  ⚠️  {path} not found — using default weights.")
        return defaults
    with open(path, "r") as f:
        data = json.load(f)
    # Support both old format (weights: {tf: ...}) and new (shared: {tf: ...})
    weights_by_tf = data.get("shared", data.get("weights", {}))
    for tf, w in list(weights_by_tf.items()):
        try:
            validate_weights(w)
        except ValueError as e:
            print(f"  ⚠️  Invalid weights for {tf}: {e} — using defaults.")
            weights_by_tf[tf] = defaults[tf]
    for tf in ["1h", "4h", "1d"]:
        if tf not in weights_by_tf:
            weights_by_tf[tf] = defaults[tf]
    print(f"  ✓ Loaded shared weights from {path} "
          f"(generated {data.get('generated_at', 'unknown')[:10]})")
    return weights_by_tf


def load_per_coin_weights(path: str = OUTPUT_WEIGHTS) -> dict:
    """Load per-coin module weights from weights.json.

    Returns {symbol: {tf: weights}}.
    Empty dict if file not found or uses old shared-only format.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("per_coin", {})


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Module weight optimizer (Bayesian)")
    parser.add_argument("--symbols",    nargs="+",
                        default=["BTC"])
    parser.add_argument("--timeframes", nargs="+", default=["4h","1d"],
                        choices=["1h","4h","1d"])
    parser.add_argument("--n-calls",    type=int, default=N_CALLS,
                        help=f"Total Bayesian trials per timeframe (default {N_CALLS})")
    parser.add_argument("--n-random",   type=int, default=N_RANDOM_STARTS,
                        help=f"Random warm-up before GP (default {N_RANDOM_STARTS})")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  ⚙️  MODULE WEIGHT OPTIMIZER  [Bayesian]")
    print("  Grid search replaced — 10–20× faster, same quality")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"  Trials:     {args.n_calls} per timeframe × 3 folds "
          f"({args.n_random} random + {args.n_calls - args.n_random} GP-guided)")
    print(f"  Lookback:   1h={LOOKBACK_BY_TF['1h']}d  4h={LOOKBACK_BY_TF['4h']}d  1d={LOOKBACK_BY_TF['1d']}d  |  Warmup: {SNAPSHOT_WARMUP} bars")
    print(f"  Max weight: {MAX_WEIGHT} per module (overfitting cap)")
    print(f"  Validation: Rolling walk-forward (3 non-overlapping folds, Sharpe score)")
    print(f"  Holdout:    Last 20% of data reserved, evaluated once after optimisation")
    print("═"*60)

    # ── Fetch data ───────────────────────────────────────
    print("\n📥 Fetching data...\n")
    data_store = {}   # {tf: {sym: {df, snapshots, source}}}

    for tf in args.timeframes:
        data_store[tf] = {}
        for sym in args.symbols:
            try:
                df, source = fetch_ohlcv(sym, TIMEFRAME_MAP[tf]["interval"], days=LOOKBACK_BY_TF.get(tf, LOOKBACK_DAYS))
                if df.empty or len(df) < SNAPSHOT_WARMUP + 50:
                    print(f"  ⚠️  Skipping {sym} {tf} — insufficient data")
                    continue

                n_total    = len(df)
                n_train    = int(n_total * 0.80)
                df_train   = df.iloc[:n_train]
                df_holdout = df.iloc[n_train:]

                print(f"    Train: {df_train.index[0].date()} → {df_train.index[-1].date()} "
                      f"({n_train} bars)  |  Holdout: {df_holdout.index[0].date()} → "
                      f"{df_holdout.index[-1].date()} ({len(df_holdout)} bars)")

                snap_train = build_signal_snapshots(df_train)

                # Build holdout snapshots with a 200-bar warmup buffer so
                # rolling indicators (EMA200 etc.) have converged before the
                # first holdout bar. Slice the buffer off afterwards so only
                # true holdout bars remain — no future data leaks back.
                df_holdout_with_buffer = df.iloc[max(0, n_train - SNAPSHOT_WARMUP):]
                snap_holdout_raw = build_signal_snapshots(df_holdout_with_buffer)
                snap_holdout = snap_holdout_raw.iloc[
                    len(snap_holdout_raw) - len(df_holdout):
                ]

                data_store[tf][sym] = {
                    "df":           df_train,
                    "snapshots":    snap_train,
                    "df_holdout":   df_holdout,
                    "snap_holdout": snap_holdout,
                    "source":       source,
                }
            except Exception as e:
                print(f"  ⚠️  Skipping {sym} {tf}: {e}")

    # ── Per-coin module weight optimisation ──────────────
    # Indicator weights (8 params) are kept shared across coins — they are
    # already loaded from indicator_weights.json and mathematically universal.
    # Module weights (3 price params) are optimised per coin because each coin
    # has different regime dynamics (SOL momentum-driven, BTC mean-reverting, etc.).
    # With only 3 parameters and Bayesian GP, even single-coin datasets provide
    # sufficient signal to avoid overfitting.
    print("\n🔍 Running per-coin Bayesian optimisation...\n")
    per_coin_weights = {}   # {symbol: {tf: weights}}
    all_trial_logs   = []
    t0 = time.time()

    for sym in args.symbols:
        sym_optimal        = {}
        sym_header_printed = False

        for tf in args.timeframes:
            sym_data = data_store.get(tf, {})
            if sym not in sym_data:
                print(f"  ⚠️  No data for {sym} {tf} — skipping.")
                continue

            if not sym_header_printed:
                print(f"\n  {'═'*56}")
                print(f"  🪙  {sym}  — per-coin module weight optimisation")
                print(f"  {'═'*56}")
                sym_header_printed = True

            coin_datasets = [{
                **sym_data[sym],
                "forward_candles": TIMEFRAME_MAP[tf]["forward_candles"],
            }]

            result = optimize_timeframe_bayesian(
                coin_datasets, tf, n_calls=args.n_calls, n_random=args.n_random
            )

            if result["best_weights"]:
                sym_optimal[tf] = result["best_weights"]
                w = result["best_weights"]
                print(f"\n  ✅ Best weights for {sym} {tf}  "
                      f"(test score: {result['best_score']:.4f}  |  "
                      f"gap: {result.get('avg_gap', 0):+.4f})")
                for mod in PRICE_MODULES:
                    dw       = DEFAULT_WEIGHTS.get(mod, 0)
                    arrow    = "▲" if w[mod] > dw + 0.01 else ("▼" if w[mod] < dw - 0.01 else "─")
                    cap_flag = " 🔒" if w[mod] >= MAX_WEIGHT - 0.01 else ""
                    print(f"     {mod:12}: {w[mod]:.3f}  {arrow}  (default: {dw:.3f}){cap_flag}")

            all_trial_logs.extend(result.get("trial_log", []))

        if sym_optimal:
            per_coin_weights[sym] = sym_optimal

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")

    # ── Compute shared fallback as average of per-coin ────
    # Used by main.py for coins not in per_coin_weights.
    shared_weights = {}
    for tf in args.timeframes:
        tf_ws = [per_coin_weights[s][tf]
                 for s in per_coin_weights if tf in per_coin_weights[s]]
        if not tf_ws:
            shared_weights[tf] = dict(DEFAULT_WEIGHTS)
            continue
        avg = {mod: float(np.mean([w[mod] for w in tf_ws])) for mod in PRICE_MODULES}
        avg_total = sum(avg.values())
        shared_weights[tf] = {mod: round(v / avg_total, 6) for mod, v in avg.items()}

    # ── Holdout evaluation per coin ────────────────────────────
    # Evaluate each coin's weights on the last 20% of that coin's data —
    # never touched during optimisation or cross-validation.
    print("\n  📊 Holdout evaluation (last 20% per coin — never seen during optimisation):")
    for sym in args.symbols:
        for tf in args.timeframes:
            sym_data = data_store.get(tf, {})
            if sym not in sym_data or tf not in per_coin_weights.get(sym, {}):
                continue
            d            = sym_data[sym]
            holdout_snap = d.get("snap_holdout")
            df_holdout   = d.get("df_holdout")
            if holdout_snap is None or df_holdout is None or len(holdout_snap) < 20:
                continue
            # Use df_holdout as the price reference so snapshot timestamps
            # (which index into holdout bars) resolve correctly.
            holdout_ds = [{
                "df":              df_holdout,
                "snapshots":       holdout_snap,
                "forward_candles": TIMEFRAME_MAP[tf]["forward_candles"],
                "source":          d["source"],
            }]
            holdout_score = _score_weights(per_coin_weights[sym][tf], holdout_ds, timeframe=tf)
            print(f"    {sym} {tf}: holdout_sharpe_norm={holdout_score:.4f}  "
                  + ("✅ generalises" if holdout_score > 0.40 else "⚠️  weak holdout"))

    # ── Save ─────────────────────────────────────────────
    if per_coin_weights:
        output = {
            "generated_at":         datetime.now(timezone.utc).isoformat(),
            "method":               "bayesian_gp_minimize_per_coin_rolling_folds",
            "lookback_days":        LOOKBACK_DAYS,
            "holdout_pct":          0.20,
            "snapshot_warmup":      SNAPSHOT_WARMUP,
            "symbols":              args.symbols,
            "conviction_threshold": CONVICTION_THR,
            "score_metric":         "sharpe_normalised",
            "normalizer":           "signal_normalizer.combine_signal_vectors",
            "per_coin":             per_coin_weights,
            "shared":               shared_weights,   # averaged fallback for unknown coins
        }
        with open(OUTPUT_WEIGHTS, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  💾 Weights saved to {OUTPUT_WEIGHTS}")

    if all_trial_logs:
        df_log = pd.DataFrame(all_trial_logs)
        df_log.sort_values("score", ascending=False, inplace=True)
        df_log.to_csv(OUTPUT_CSV, index=False)
        print(f"  📊 Trial log saved to {OUTPUT_CSV}")

    print(f"\n{'═'*60}")
    print("  Next step: python main.py")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()