"""
signal_normalizer.py
--------------------
Normalizes outputs from all 5 modules into a unified SignalVector
before weights are applied in main.py.

Problems this solves:
  1. BERT models (FinBERT, CryptoBERT) are overconfident — raw softmax
     outputs like [0.91, 0.03, 0.06] overstate certainty. Temperature
     scaling recalibrates them toward their true confidence.

  2. Neutral mass was being discarded before reaching combine_signals(),
     meaning a 70%-neutral sentiment signal was injecting noise at full
     weight. Now neutral reduces effective_weight automatically.

  3. All 5 modules previously emitted different shaped outputs — some
     had timeframes, some didn't, some had 3-class probabilities, some
     had binary bull/bear. Everything now passes through a unified
     SignalVector so the combiner treats them identically.

Architecture:
  Raw model output
      → temperature_scale()       # recalibrate softmax overconfidence
      → to_signal_vector()        # standardize to [bull, bear, neutral]
      → apply_confidence_weight() # neutral mass attenuates effective weight
      → SignalVector              # what combine_signals() consumes

Temperature values (T):
  T = 1.0  → no change (trust the model fully)
  T > 1.0  → soften distribution (more uncertainty)
  T < 1.0  → sharpen distribution (more certainty)

  CryptoBERT: T = 1.3  (crypto-native, earns more trust → less softening)
  FinBERT:    T = 1.8  (general finance, less domain fit → more softening)
  Technical:  T = 1.0  (deterministic indicators, no recalibration needed)
  Candlestick:T = 1.0
  Chart:      T = 1.0

Usage:
  from signal_normalizer import normalize_sentiment, normalize_news,
                                 normalize_technical, SignalVector

  # In SentimentAnalysis.py — wrap CryptoBERT output before returning
  vec = normalize_sentiment(raw_cryptobert_scores, n_posts=30)

  # In NewsAnalysis.py — wrap FinBERT output before returning
  vec = normalize_news(raw_finbert_scores, n_articles=8)

  # In main.py combine_signals() — consume SignalVector uniformly
  bull_score += vec.bull * vec.effective_weight
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────
# TEMPERATURE CONFIG
# ─────────────────────────────────────────────────────────

TEMPERATURES = {
    "cryptobert": 1.3,
    "finbert":    1.8,
    "technical":  1.0,
    "candlestick":1.0,
    "chart":      1.0,
}

MODEL_TRUST = {
    "cryptobert": 1.00,
    "finbert":    0.82,
    "technical":  1.00,
    "candlestick":1.00,
    "chart":      1.00,
}

MIN_EFFECTIVE_WEIGHT_RATIO = 0.15


# ─────────────────────────────────────────────────────────
# SIGNAL VECTOR
# ─────────────────────────────────────────────────────────

@dataclass
class SignalVector:
    """
    Unified signal representation for any module.

    Fields:
      bull             : calibrated bullish probability  (0.0 → 1.0)
      bear             : calibrated bearish probability  (0.0 → 1.0)
      neutral          : calibrated neutral probability  (0.0 → 1.0)
      raw_bull         : original bull score before calibration
      raw_bear         : original bear score before calibration
      confidence       : 1 - neutral  (how decisive the signal is)
      effective_weight : base_weight × trust × confidence_factor
                         used by combine_signals() instead of raw weight
      source           : module name for logging
      n_samples        : number of posts/articles/candles contributing
      temperature      : T value used for scaling
      metadata         : optional extras. Always includes "base_weight" so
                         main.py can rescale effective_weight correctly when
                         optimized module weights differ from defaults.
    """
    bull:             float
    bear:             float
    neutral:          float
    raw_bull:         float
    raw_bear:         float
    confidence:       float
    effective_weight: float
    source:           str
    n_samples:        int   = 0
    temperature:      float = 1.0
    metadata:         dict  = field(default_factory=dict)

    def bullish_pct(self) -> float:
        total = self.bull + self.bear
        return round(self.bull / total * 100, 1) if total > 0 else 50.0

    def bearish_pct(self) -> float:
        total = self.bull + self.bear
        return round(self.bear / total * 100, 1) if total > 0 else 50.0

    def to_dict(self) -> dict:
        return {
            "bull":             round(self.bull, 4),
            "bear":             round(self.bear, 4),
            "neutral":          round(self.neutral, 4),
            "confidence":       round(self.confidence, 4),
            "effective_weight": round(self.effective_weight, 4),
            "bullish_pct":      self.bullish_pct(),
            "bearish_pct":      self.bearish_pct(),
            "source":           self.source,
            "n_samples":        self.n_samples,
            "temperature":      self.temperature,
            **self.metadata,
        }

    def __repr__(self):
        return (
            f"SignalVector({self.source} | "
            f"bull={self.bull:.3f} bear={self.bear:.3f} neut={self.neutral:.3f} | "
            f"conf={self.confidence:.3f} eff_w={self.effective_weight:.3f})"
        )


# ─────────────────────────────────────────────────────────
# CORE MATH
# ─────────────────────────────────────────────────────────

def temperature_scale(logits_or_probs: np.ndarray, T: float) -> np.ndarray:
    """
    Applies temperature scaling to a probability vector.

    If input is already a probability simplex (sums to ~1.0), we first
    convert back to log-space, apply T, then re-normalize via softmax.
    This is the standard post-hoc calibration technique from
    Guo et al. (2017) "On Calibration of Modern Neural Networks".

    Args:
        logits_or_probs: array of shape (n_classes,) — probabilities or logits
        T: temperature value. T=1 → no change. T>1 → softer. T<1 → sharper.

    Returns:
        Calibrated probability array, sums to 1.0.
    """
    if T <= 0:
        raise ValueError(f"Temperature must be > 0, got {T}")
    if T == 1.0:
        arr = np.array(logits_or_probs, dtype=np.float64)
        return arr / (arr.sum() + 1e-10)

    arr = np.array(logits_or_probs, dtype=np.float64)

    if np.all(arr >= 0) and abs(arr.sum() - 1.0) < 0.05:
        arr = np.clip(arr, 1e-10, 1.0)
        log_probs = np.log(arr)
    else:
        log_probs = arr

    scaled  = log_probs / T
    shifted = scaled - scaled.max()
    exp_s   = np.exp(shifted)
    return exp_s / exp_s.sum()


def compute_effective_weight(
    neutral_mass: float,
    base_weight: float,
    model_trust: float,
    min_ratio: float = MIN_EFFECTIVE_WEIGHT_RATIO,
) -> float:
    """
    Computes the effective weight for a module after confidence adjustment.

    Formula:
        confidence_factor = 1 - neutral_mass
        confidence_factor = clip(confidence_factor, min_ratio, 1.0)
        effective_weight  = base_weight × model_trust × confidence_factor
    """
    confidence_factor = max(1.0 - neutral_mass, min_ratio)
    return round(base_weight * model_trust * confidence_factor, 6)


def aggregate_bert_scores(scores_list: list[dict],
                          bull_key: str = "bullish_score",
                          bear_key: str = "bearish_score",
                          neut_key: str = "neutral_score",
                          weights: Optional[list[float]] = None) -> tuple[float, float, float]:
    """
    Weighted average of raw BERT probability scores across multiple items
    (posts or articles) BEFORE temperature scaling.

    weights: optional per-item weights (e.g. recency decay). Falls back to
             uniform weights when None or mismatched length.
    """
    if not scores_list:
        return 0.0, 0.0, 1.0

    n = len(scores_list)
    if weights is None or len(weights) != n:
        weights = [1.0] * n

    total_w  = sum(weights) + 1e-10
    avg_bull = sum(s.get(bull_key, 0.0) * w for s, w in zip(scores_list, weights)) / total_w
    avg_bear = sum(s.get(bear_key, 0.0) * w for s, w in zip(scores_list, weights)) / total_w
    avg_neut = sum(s.get(neut_key, 1.0) * w for s, w in zip(scores_list, weights)) / total_w

    total = avg_bull + avg_bear + avg_neut + 1e-10
    return avg_bull / total, avg_bear / total, avg_neut / total


# ─────────────────────────────────────────────────────────
# MODULE-SPECIFIC NORMALIZERS
# ─────────────────────────────────────────────────────────

def normalize_sentiment(
    raw_scores: list[dict],
    base_weight: float = 0.125,
    user_bull_pct: Optional[float] = None,
    user_bear_pct: Optional[float] = None,
    n_posts: int = 0,
) -> SignalVector:
    """
    Normalizes CryptoBERT output from SentimentAnalysis.py.

    Applies:
      1. Average raw softmax scores across all posts
      2. Temperature scaling (T=1.3)
      3. Optional blending with StockTwits user tags (20% weight)
      4. Confidence weighting via neutral mass
    """
    T     = TEMPERATURES["cryptobert"]
    trust = MODEL_TRUST["cryptobert"]

    avg_bull, avg_bear, avg_neut = aggregate_bert_scores(raw_scores)

    calibrated = temperature_scale(
        np.array([avg_bull, avg_bear, avg_neut]), T
    )
    cal_bull, cal_bear, cal_neut = calibrated

    if user_bull_pct is not None and user_bear_pct is not None:
        total_tagged = user_bull_pct + user_bear_pct
        if total_tagged > 0:
            u_bull = user_bull_pct / total_tagged * (1 - cal_neut)
            u_bear = user_bear_pct / total_tagged * (1 - cal_neut)
            u_neut = cal_neut
            BLEND  = 0.20
            cal_bull = (1 - BLEND) * cal_bull + BLEND * u_bull
            cal_bear = (1 - BLEND) * cal_bear + BLEND * u_bear
            cal_neut = (1 - BLEND) * cal_neut + BLEND * u_neut
            s = cal_bull + cal_bear + cal_neut + 1e-10
            cal_bull, cal_bear, cal_neut = cal_bull/s, cal_bear/s, cal_neut/s

    eff_w = compute_effective_weight(cal_neut, base_weight, trust)

    return SignalVector(
        bull=round(cal_bull, 4),
        bear=round(cal_bear, 4),
        neutral=round(cal_neut, 4),
        raw_bull=round(avg_bull, 4),
        raw_bear=round(avg_bear, 4),
        confidence=round(1.0 - cal_neut, 4),
        effective_weight=eff_w,
        source="CryptoBERT/StockTwits",
        n_samples=n_posts,
        temperature=T,
        # FIX #5: store base_weight so main.py can rescale correctly
        # without hardcoding 0.125.
        metadata={
            "base_weight":   base_weight,
            "user_bull_pct": user_bull_pct,
            "user_bear_pct": user_bear_pct,
            "model_trust":   trust,
        },
    )


def normalize_news(
    raw_scores: list[dict],
    base_weight: float = 0.125,
    n_articles: int = 0,
    article_weights: Optional[list[float]] = None,
) -> SignalVector:
    """
    Normalizes FinBERT output from NewsAnalysis.py.

    Applies:
      1. Weighted average of raw softmax scores (recency-decayed if provided)
      2. Temperature scaling (T=1.8)
      3. Confidence weighting via neutral mass + model trust (0.82)
    """
    T     = TEMPERATURES["finbert"]
    trust = MODEL_TRUST["finbert"]

    avg_bull, avg_bear, avg_neut = aggregate_bert_scores(raw_scores, weights=article_weights)

    calibrated = temperature_scale(
        np.array([avg_bull, avg_bear, avg_neut]), T
    )
    cal_bull, cal_bear, cal_neut = calibrated

    eff_w = compute_effective_weight(cal_neut, base_weight, trust)

    return SignalVector(
        bull=round(cal_bull, 4),
        bear=round(cal_bear, 4),
        neutral=round(cal_neut, 4),
        raw_bull=round(avg_bull, 4),
        raw_bear=round(avg_bear, 4),
        confidence=round(1.0 - cal_neut, 4),
        effective_weight=eff_w,
        source="FinBERT/News",
        n_samples=n_articles,
        temperature=T,
        # FIX #5: store base_weight so main.py can rescale correctly.
        metadata={
            "base_weight": base_weight,
            "model_trust": trust,
        },
    )


def normalize_technical(
    bullish_pct: float,
    bearish_pct: float,
    base_weight: float,
    source: str = "Technical",
    n_samples: int = 0,
) -> SignalVector:
    """
    Wraps technical/candlestick/chart module outputs into a SignalVector.
    No temperature scaling needed — these are deterministic indicator votes.

    FIX #4: The original formula used a LINEAR decisiveness curve:
        decisiveness = |bull - bear|
    This was too aggressive — a genuine 72/28 bull split got decisiveness=0.44,
    neutral_mass=0.56, and effective_weight dropped to 44% of its base value.
    A 60/40 split got down to 20% — strong signals were being heavily penalized.

    Fix: use a SQUARE ROOT curve so moderate conviction (60/40, 65/35)
    retains most of its effective weight, while truly ambiguous signals
    (50/50, 52/48) are still properly attenuated:
        decisiveness = sqrt(|bull - bear|)
        60/40 → |0.6-0.4|=0.20 → sqrt=0.45  (was 0.20)
        65/35 → |0.65-0.35|=0.30 → sqrt=0.55 (was 0.30)
        72/28 → |0.72-0.28|=0.44 → sqrt=0.66 (was 0.44)
        80/20 → |0.80-0.20|=0.60 → sqrt=0.77 (was 0.60)
        50/50 → 0.0 → 0.0 (unchanged — still fully neutral)

    This preserves the correct behavior at extremes while treating
    moderate directional conviction more fairly.
    """
    total = bullish_pct + bearish_pct + 1e-10
    bull  = bullish_pct / total
    bear  = bearish_pct / total

    # FIX #4: sqrt curve — softer attenuation for moderate conviction
    raw_decisiveness = abs(bull - bear)          # linear: 0.0 → 1.0
    decisiveness     = raw_decisiveness ** 0.5   # sqrt:   0.0 → 1.0, gentler slope

    # Signal-density penalty: fewer signals firing → less reliable aggregate.
    # With 8 indicators expected, a signal from only 2 carries less information
    # than one from 7. We shrink decisiveness toward 0 proportionally.
    # n_samples=0 means "unknown" (e.g. called from weight optimizer) → no penalty.
    if n_samples > 0:
        EXPECTED_SIGNALS = 8   # matches TechnicalAnalysis.py indicator count
        density      = min(n_samples / EXPECTED_SIGNALS, 1.0)
        # Blend: full decisiveness at density=1.0; halved at density=0.0
        decisiveness = decisiveness * (0.5 + 0.5 * density)

    neutral_mass     = round(1.0 - decisiveness, 4)

    # Re-express as 3-component vector preserving relative bull/bear ratio
    cal_bull = round(bull * decisiveness, 4)
    cal_bear = round(bear * decisiveness, 4)
    cal_neut = round(neutral_mass, 4)
    s = cal_bull + cal_bear + cal_neut + 1e-10
    cal_bull, cal_bear, cal_neut = cal_bull/s, cal_bear/s, cal_neut/s

    eff_w = compute_effective_weight(cal_neut, base_weight, MODEL_TRUST["technical"])

    return SignalVector(
        bull=round(cal_bull, 4),
        bear=round(cal_bear, 4),
        neutral=round(cal_neut, 4),
        raw_bull=round(bull, 4),
        raw_bear=round(bear, 4),
        confidence=round(decisiveness, 4),
        effective_weight=eff_w,
        source=source,
        n_samples=n_samples,
        temperature=1.0,
        # base_weight stored for consistency; technical rescaling in
        # main.py uses _safe_technical_vec (not sv rescaling), but
        # keeping it here makes the interface uniform.
        metadata={"base_weight": base_weight},
    )


def normalize_technical_soft(
    net_score: float,
    agreement: float,
    base_weight: float,
    source: str = "Technical",
) -> SignalVector:
    """
    Normalizes technical signals using continuous net_score and agreement.

    Fix A+B implementation — replaces the old (bullish_pct, bearish_pct) interface
    for the technical module in the optimization and backtesting pipeline.

    Args:
        net_score  : weighted mean of individual indicator scores, ∈ [-1, +1]
                     positive = net bullish, negative = net bearish
        agreement  : 1 - normalized_std of individual scores, ∈ [0, 1]
                     1.0 = all indicators point same way, 0.0 = maximally split
        base_weight: module weight used by combine_signal_vectors
        source     : module name for logging

    Confidence formula:
        confidence = |net_score| × (0.5 + 0.5 × agreement)
        The 0.5 floor means magnitude carries 50% weight even with zero agreement.
        Full agreement doubles the effective confidence for a given magnitude.
        This directly addresses:
          Problem 1 — RSI 67 vs RSI 95 now produce different net_score magnitudes
          Problem 2 — unanimous agreement amplifies confidence; mixed signals dampen it
    """
    net_score = float(np.clip(net_score, -1.0, 1.0))
    agreement = float(np.clip(agreement,  0.0, 1.0))

    direction_strength = abs(net_score)
    confidence         = direction_strength * (0.5 + 0.5 * agreement)

    # Decompose direction
    raw_bull = float(max(0.0, net_score))
    raw_bear = float(max(0.0, -net_score))

    # Build calibrated 3-component vector
    cal_bull = raw_bull * confidence
    cal_bear = raw_bear * confidence
    cal_neut = max(0.0, 1.0 - confidence)

    s = cal_bull + cal_bear + cal_neut + 1e-10
    cal_bull /= s
    cal_bear /= s
    cal_neut /= s

    eff_w = compute_effective_weight(cal_neut, base_weight, MODEL_TRUST["technical"])

    return SignalVector(
        bull             = round(cal_bull, 4),
        bear             = round(cal_bear, 4),
        neutral          = round(cal_neut, 4),
        raw_bull         = round(raw_bull, 4),
        raw_bear         = round(raw_bear, 4),
        confidence       = round(confidence, 4),
        effective_weight = eff_w,
        source           = source,
        n_samples        = 0,
        temperature      = 1.0,
        metadata         = {
            "base_weight": base_weight,
            "net_score":   round(net_score, 4),
            "agreement":   round(agreement, 4),
        },
    )


# ─────────────────────────────────────────────────────────
# COMBINE SIGNALS
# ─────────────────────────────────────────────────────────

def combine_signal_vectors(
    vectors: dict[str, SignalVector],
    conviction_threshold: float = 15.0,
) -> dict:
    """
    Combines SignalVectors from all modules into a final trading signal.

    Uses effective_weight instead of raw weight — uncertain modules
    automatically contribute less. Conviction gap is checked AFTER
    neutral is accounted for via overall_confidence scaling.
    """
    total_eff_weight = sum(v.effective_weight for v in vectors.values())

    if total_eff_weight < 1e-6:
        return _neutral_result(vectors, conviction_threshold)

    bull_score = sum(v.bull    * v.effective_weight for v in vectors.values())
    bear_score = sum(v.bear    * v.effective_weight for v in vectors.values())
    neut_score = sum(v.neutral * v.effective_weight for v in vectors.values())

    bull = bull_score / total_eff_weight
    bear = bear_score / total_eff_weight
    neut = neut_score / total_eff_weight

    total    = bull + bear + neut + 1e-10
    bull_pct = round(bull / total * 100, 1)
    bear_pct = round(bear / total * 100, 1)
    neut_pct = round(neut / total * 100, 1)

    directional_total = bull + bear + 1e-10
    bull_dir = round(bull / directional_total * 100, 1)
    bear_dir = round(bear / directional_total * 100, 1)
    gap      = bull_dir - bear_dir

    overall_confidence = round(1.0 - (neut / total), 4)
    adjusted_gap       = round(gap * overall_confidence, 1)

    if adjusted_gap > conviction_threshold:
        direction = "LONG"
        overall   = "bullish"
    elif adjusted_gap < -conviction_threshold:
        direction = "SHORT"
        overall   = "bearish"
    else:
        direction = "STAY OUT"
        overall   = "neutral"

    breakdown = {}
    for name, v in vectors.items():
        breakdown[name] = {
            **v.to_dict(),
            "weight_used": round(v.effective_weight / total_eff_weight * 100, 1),
        }

    return {
        "bullish_pct":        bull_pct,
        "bearish_pct":        bear_pct,
        "neutral_pct":        neut_pct,
        "bull_directional":   bull_dir,
        "bear_directional":   bear_dir,
        "gap":                round(gap, 1),
        "adjusted_gap":       adjusted_gap,
        "overall_confidence": overall_confidence,
        "direction":          direction,
        "overall":            overall,
        "total_eff_weight":   round(total_eff_weight, 4),
        "module_breakdown":   breakdown,
    }


def _neutral_result(vectors: dict, threshold: float) -> dict:
    return {
        "bullish_pct": 33.3, "bearish_pct": 33.3, "neutral_pct": 33.4,
        "bull_directional": 50.0, "bear_directional": 50.0,
        "gap": 0.0, "adjusted_gap": 0.0, "overall_confidence": 0.0,
        "direction": "STAY OUT", "overall": "neutral",
        "total_eff_weight": 0.0,
        "module_breakdown": {k: v.to_dict() for k, v in vectors.items()},
    }


# ─────────────────────────────────────────────────────────
# DIAGNOSTIC PRINTER
# ─────────────────────────────────────────────────────────

def print_normalization_report(vectors: dict[str, SignalVector], combined: dict):
    print(f"\n{'─'*65}")
    print(f"  🔬 SIGNAL NORMALIZATION REPORT")
    print(f"{'─'*65}")
    print(f"  {'Module':<16} {'Raw Bull':>8} {'Cal Bull':>8} "
          f"{'Neutral':>8} {'Conf':>6} {'EffW':>7} {'Share':>6}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*7} {'─'*6}")

    total_eff = combined["total_eff_weight"]
    for name, v in vectors.items():
        share = v.effective_weight / total_eff * 100 if total_eff > 0 else 0
        print(
            f"  {name:<16} "
            f"{v.raw_bull*100:>7.1f}% "
            f"{v.bull*100:>7.1f}% "
            f"{v.neutral*100:>7.1f}% "
            f"{v.confidence:>5.2f}  "
            f"{v.effective_weight:>6.4f} "
            f"{share:>5.1f}%"
        )

    print(f"{'─'*65}")
    print(f"  Final: Bull {combined['bullish_pct']}% | "
          f"Bear {combined['bearish_pct']}% | "
          f"Neutral {combined['neutral_pct']}%")
    print(f"  Directional gap: {combined['gap']}% → "
          f"Adjusted (×conf): {combined['adjusted_gap']}%")
    print(f"  Overall confidence: {combined['overall_confidence']:.2f} | "
          f"Signal: {combined['direction']}")
    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Temperature Scaling Demo ===")
    overconfident = np.array([0.91, 0.03, 0.06])
    print(f"Raw FinBERT output:              {overconfident}")
    print(f"After T=1.8 (FinBERT):           {temperature_scale(overconfident, 1.8).round(4)}")
    print(f"After T=1.3 (CryptoBERT):        {temperature_scale(overconfident, 1.3).round(4)}")
    print(f"After T=1.0 (no change):         {temperature_scale(overconfident, 1.0).round(4)}")

    print("\n=== normalize_technical — FIX #4 sqrt curve demo ===")
    for b, bear in [(72, 28), (65, 35), (60, 40), (55, 45), (80, 20), (50, 50)]:
        v = normalize_technical(b, bear, base_weight=0.35, source="test")
        print(f"  {b}/{bear} → conf={v.confidence:.3f}  eff_w={v.effective_weight:.4f}  "
              f"(base=0.35)")

    print("\n=== Confidence Weighting Demo ===")
    uncertain_scores = [
        {"bullish_score": 0.12, "bearish_score": 0.10, "neutral_score": 0.78},
    ] * 10
    vec_uncertain = normalize_news(uncertain_scores, base_weight=0.125, n_articles=10)
    print(f"Uncertain FinBERT: {vec_uncertain}")

    confident_scores = [
        {"bullish_score": 0.72, "bearish_score": 0.13, "neutral_score": 0.15},
    ] * 20
    vec_confident = normalize_sentiment(confident_scores, base_weight=0.125, n_posts=20)
    print(f"Confident CryptoBERT: {vec_confident}")

    print("\n=== Full Combine Demo ===")
    tech_vec  = normalize_technical(72.0, 28.0, base_weight=0.35, source="technical")
    cs_vec    = normalize_technical(60.0, 40.0, base_weight=0.20, source="candlestick")
    chart_vec = normalize_technical(55.0, 45.0, base_weight=0.20, source="chart")

    vectors = {
        "technical":   tech_vec,
        "candlestick": cs_vec,
        "chart":       chart_vec,
        "news":        vec_uncertain,
        "sentiment":   vec_confident,
    }

    combined = combine_signal_vectors(vectors, conviction_threshold=15.0)
    print_normalization_report(vectors, combined)