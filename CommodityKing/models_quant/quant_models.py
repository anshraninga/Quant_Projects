"""
Five pure mathematical models for CommodityKing.

Rules:
  - No API calls, no LLM calls, no side effects
  - Every function takes data in, returns a dict out
  - All inputs must be validated before heavy computation
"""

from __future__ import annotations

from collections import Counter
import re

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: KALMAN FILTER
# Purpose: separate true price trend from noise.
#          Determines if a price move is signal or noise.
# ══════════════════════════════════════════════════════════════════════════════

def kalman_filter(prices: np.ndarray) -> dict:
    """
    1D Kalman filter for a price series.

    State:       true price level
    Observation: noisy market price

    Returns:
      filtered:      np.ndarray — smoothed price series
      uncertainty:   np.ndarray — posterior variance at each step
      latest_signal: float      — current deviation in std devs
      is_signal:     bool       — True if |latest_signal| > 1.5σ
      trend:         str        — "up" or "down" vs 5 steps ago
      signal_sigma:  float      — |latest_signal|
    """
    prices = np.asarray(prices, dtype=float)

    if len(prices) < 2:
        return {
            "filtered":      prices.copy(),
            "uncertainty":   np.array([1.0] * len(prices)),
            "latest_signal": 0.0,
            "is_signal":     False,
            "trend":         "up",
            "signal_sigma":  0.0,
        }

    n = len(prices)
    diffs = np.diff(prices)

    # State estimate and error covariance
    x = prices[0]
    P = 1.0

    # Process noise: how much true price can change per step
    Q = float(np.var(diffs)) * 0.1
    # Measurement noise: how noisy are observed market prices
    R = float(np.var(diffs))

    # Guard against zero variance (flat series)
    if R < 1e-12:
        R = 1e-12
    if Q < 1e-14:
        Q = 1e-14

    filtered    = np.zeros(n)
    uncertainty = np.zeros(n)

    for i, z in enumerate(prices):
        # Predict
        P = P + Q
        # Update
        K = P / (P + R)          # Kalman gain
        x = x + K * (z - x)
        P = (1 - K) * P
        filtered[i]    = x
        uncertainty[i] = P

    # Signal strength: how many σ is current price from Kalman trend?
    residuals     = prices - filtered
    sigma         = float(np.std(residuals)) + 1e-10
    latest_signal = float(residuals[-1]) / sigma

    lookback      = min(5, n - 1)
    trend         = "up" if filtered[-1] > filtered[-1 - lookback] else "down"

    return {
        "filtered":      filtered,
        "uncertainty":   uncertainty,
        "latest_signal": round(latest_signal, 3),
        "is_signal":     abs(latest_signal) > 1.5,
        "trend":         trend,
        "signal_sigma":  round(abs(latest_signal), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: HMM REGIME CLASSIFIER
# Purpose: classify commodity into current market regime.
#          Same news event has different implications in different regimes.
# ══════════════════════════════════════════════════════════════════════════════

def hmm_regime(weekly_returns: np.ndarray) -> dict:
    """
    2-state Hidden Markov Model on weekly returns.
    States: low volatility vs high volatility.

    Fit on full history, classify current regime.
    Requires at least 52 data points (one year of weekly returns).

    Returns:
      current_regime:     int   — 0 or 1 (raw HMM state index)
      regime_label:       str   — "low_volatility" | "high_volatility"
      confidence:         float — probability of being in current state
      is_high_volatility: bool
      regime_history:     np.ndarray — full state sequence
    """
    from hmmlearn.hmm import GaussianHMM

    weekly_returns = np.asarray(weekly_returns, dtype=float)

    if len(weekly_returns) < 52:
        return {
            "current_regime":     0,
            "regime_label":       "insufficient_data",
            "confidence":         0.5,
            "is_high_volatility": False,
            "regime_history":     np.array([]),
        }

    X = weekly_returns.reshape(-1, 1)

    try:
        model = GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(X)
        states = model.predict(X)
        probs  = model.predict_proba(X)

        # Identify which HMM state corresponds to high volatility
        # (the state whose assigned returns have higher variance)
        state_vars    = [
            np.var(weekly_returns[states == s]) if np.any(states == s) else 0.0
            for s in [0, 1]
        ]
        high_vol_state = int(np.argmax(state_vars))

        current_state = int(states[-1])
        current_conf  = float(probs[-1][current_state])
        is_high_vol   = current_state == high_vol_state
        label         = "high_volatility" if is_high_vol else "low_volatility"

        return {
            "current_regime":     current_state,
            "regime_label":       label,
            "confidence":         round(current_conf, 3),
            "is_high_volatility": is_high_vol,
            "regime_history":     states,
        }

    except Exception as exc:
        return {
            "current_regime":     0,
            "regime_label":       "model_error",
            "confidence":         0.5,
            "is_high_volatility": False,
            "regime_history":     np.array([]),
            "error":              str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: SUPPLY SHOCK INDEX (SSI)
# Purpose: single composite score quantifying multi-factor supply stress.
#          Calibrates agent conviction.
# ══════════════════════════════════════════════════════════════════════════════

def supply_shock_index(
    weather_zscores: list[float],
    geo_event_rate:  float,
    inventory_dev:   float,
    weights:         dict,
) -> dict:
    """
    Weighted combination of three supply stress signals.
    All inputs are normalised to z-score scale before combining.

    weather_zscores: list of z-scores from weather agent regions
                     (uses the single max-|z| region — worst case drives signal)
    geo_event_rate:  Bayesian surprise score (0–1), scaled by ×3
    inventory_dev:   (current − 5yr_avg) / 5yr_avg
                     negative = below average → bullish supply signal
                     positive = above average → bearish supply signal
    weights:         {"weather": float, "geopolitical": float, "inventory": float}
                     must sum to 1.0

    SSI interpretation (standard deviations):
      |SSI| < 1.0 : normal
      |SSI| 1–2   : elevated — watch
      |SSI| 2–3   : high — material signal
      |SSI| > 3   : extreme — comparable to 2022 wheat shock
    """
    # Weather: worst affected region drives the signal
    if weather_zscores:
        weather_component = float(max(weather_zscores, key=abs))
    else:
        weather_component = 0.0

    # Geo: scale surprise score (0–1) to z-score range
    geo_component = float(geo_event_rate) * 3.0

    # Inventory: below-average stocks are bullish (positive SSI)
    inv_component = -float(inventory_dev) * 3.0

    w = weights
    ssi = (
        w["weather"]      * weather_component +
        w["geopolitical"] * geo_component +
        w["inventory"]    * inv_component
    )

    if   abs(ssi) < 1.0: level = "normal"
    elif abs(ssi) < 2.0: level = "elevated"
    elif abs(ssi) < 3.0: level = "high"
    else:                level = "extreme"

    return {
        "ssi":                 round(float(ssi), 3),
        "level":               level,
        "direction":           "bullish" if ssi > 0 else "bearish",
        "weather_component":   round(float(weather_component * w["weather"]), 3),
        "geo_component":       round(float(geo_component * w["geopolitical"]), 3),
        "inventory_component": round(float(inv_component * w["inventory"]), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4: BAYESIAN SURPRISE SCORE
# Purpose: quantify how unexpected a news event is.
#          Expected events are already priced in.
#          Unexpected events create trading opportunities.
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_surprise(
    articles_today: list[str],
    articles_30d:   list[str],
    commodity_name: str,       # unused in computation, kept for prompt parity
) -> dict:
    """
    Estimates how surprising today's news is relative to the
    recent 30-day base rate of similar language.

    Method:
      1. Count keyword frequency in 30-day history
      2. Compare today's keyword profile to that base rate
      3. Surprise = KL divergence (today || history), normalised to [0, 1]

    Returns:
      surprise_score:      float — 0 (routine) to 1 (highly unexpected)
      top_surprise_terms:  list  — words most overrepresented today
      interpretation:      str
      kl_divergence:       float — raw KL value (nats)
    """
    def _tokenise(texts: list[str]) -> Counter:
        words: list[str] = []
        for t in texts:
            words.extend(re.findall(r'\b[a-z]{4,}\b', t.lower()))
        return Counter(words)

    if not articles_30d:
        return {
            "surprise_score":     0.5,
            "top_surprise_terms": [],
            "interpretation":     "insufficient_history",
            "kl_divergence":      0.0,
        }

    hist_freq  = _tokenise(articles_30d)
    today_freq = _tokenise(articles_today) if articles_today else Counter()

    if not today_freq:
        return {
            "surprise_score":     0.0,
            "top_surprise_terms": [],
            "interpretation":     "no_news_today",
            "kl_divergence":      0.0,
        }

    hist_total  = sum(hist_freq.values()) + 1e-10
    today_total = sum(today_freq.values()) + 1e-10
    vocab       = set(hist_freq.keys()) | set(today_freq.keys())

    kl             = 0.0
    surprise_terms: list[str] = []

    for word in vocab:
        p = today_freq.get(word, 0) / today_total
        q = (hist_freq.get(word, 0) + 1) / (hist_total + len(vocab))
        if p > 0:
            kl += p * np.log(p / q + 1e-10)
            if p / q > 3.0:   # word appears 3× more often today than baseline
                surprise_terms.append(word)

    # Cap at 2.0 nats → maps to score of 1.0
    surprise_score = min(float(kl) / 2.0, 1.0)

    if   surprise_score < 0.20: interp = "routine_news"
    elif surprise_score < 0.50: interp = "moderately_unusual"
    elif surprise_score < 0.75: interp = "notably_surprising"
    else:                       interp = "highly_unexpected"

    return {
        "surprise_score":     round(surprise_score, 3),
        "top_surprise_terms": surprise_terms[:5],
        "interpretation":     interp,
        "kl_divergence":      round(float(kl), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5: COINTEGRATION MONITOR
# Purpose: detect when related commodities diverge from historical relationship.
#          Mean reversion signal.
# ══════════════════════════════════════════════════════════════════════════════

def cointegration_spread(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    symbol_a: str,
    symbol_b: str,
) -> dict:
    """
    Tests cointegration (Engle-Granger) and computes current spread z-score.

    High spread z-score → symbol_a expensive relative to symbol_b
    Low spread z-score  → symbol_a cheap relative to symbol_b

    Requires at least 52 observations (one year of weekly data).
    Returns not-cointegrated if p-value ≥ 0.05.
    """
    from statsmodels.tsa.stattools import coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    prices_a = np.asarray(prices_a, dtype=float)
    prices_b = np.asarray(prices_b, dtype=float)

    if len(prices_a) < 52 or len(prices_b) < 52:
        return {
            "cointegrated": False,
            "reason":       "insufficient_data",
            "pair":         f"{symbol_a}/{symbol_b}",
        }

    min_len = min(len(prices_a), len(prices_b))
    a = prices_a[-min_len:]
    b = prices_b[-min_len:]

    try:
        _, pvalue, _ = coint(a, b)
        is_coint     = bool(pvalue < 0.05)

        if not is_coint:
            return {
                "cointegrated":   False,
                "pvalue":         round(float(pvalue), 4),
                "interpretation": "not_cointegrated",
                "pair":           f"{symbol_a}/{symbol_b}",
            }

        # Compute hedge ratio via OLS: a = α + β·b + ε
        model       = OLS(a, add_constant(b)).fit()
        hedge_ratio = float(model.params[1])
        spread      = a - hedge_ratio * b

        spread_mean = float(np.mean(spread))
        spread_std  = float(np.std(spread)) + 1e-10
        current_z   = float((spread[-1] - spread_mean) / spread_std)

        if abs(current_z) > 2.0:
            signal = (
                f"{symbol_a} expensive vs {symbol_b}"
                if current_z > 0
                else f"{symbol_a} cheap vs {symbol_b}"
            )
        else:
            signal = "within_normal_range"

        return {
            "cointegrated":   True,
            "pvalue":         round(float(pvalue), 4),
            "hedge_ratio":    round(hedge_ratio, 4),
            "spread_zscore":  round(current_z, 3),
            "signal":         signal,
            "mean_reverting": abs(current_z) > 2.0,
            "pair":           f"{symbol_a}/{symbol_b}",
        }

    except Exception as exc:
        return {
            "cointegrated": False,
            "reason":       str(exc),
            "pair":         f"{symbol_a}/{symbol_b}",
        }
