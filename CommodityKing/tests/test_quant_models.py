"""
Unit tests for all five quant models in quant_models.py.
Each test uses deterministic synthetic data — no API calls, no randomness
(seeds are fixed where needed).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np

from models_quant.quant_models import (
    kalman_filter,
    hmm_regime,
    supply_shock_index,
    bayesian_surprise,
    cointegration_spread,
)

RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: KALMAN FILTER
# ══════════════════════════════════════════════════════════════════════════════

class TestKalmanFilter(unittest.TestCase):

    def _noisy_trend(self, n=200, slope=0.5, noise_std=5.0):
        t = np.arange(n, dtype=float)
        return 500.0 + slope * t + RNG.normal(0, noise_std, n)

    def test_output_keys(self):
        prices = self._noisy_trend()
        result = kalman_filter(prices)
        for key in ("filtered", "uncertainty", "latest_signal",
                    "is_signal", "trend", "signal_sigma"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_filtered_length_matches_input(self):
        prices = self._noisy_trend(100)
        result = kalman_filter(prices)
        self.assertEqual(len(result["filtered"]), 100)
        self.assertEqual(len(result["uncertainty"]), 100)

    def test_filtered_smoother_than_input(self):
        """Kalman output must have lower variance than raw noisy input."""
        prices = self._noisy_trend(200, noise_std=10.0)
        result = kalman_filter(prices)
        self.assertLess(np.var(result["filtered"]), np.var(prices))

    def test_spike_triggers_is_signal(self):
        """A single large spike at the end must be detected as a signal."""
        prices = np.ones(100) * 500.0
        prices[-1] = 550.0          # +50 spike (>>1.5σ for a flat series)
        result = kalman_filter(prices)
        self.assertTrue(result["is_signal"],
                        f"Expected is_signal=True, got signal={result['latest_signal']}")

    def test_flat_series_no_signal(self):
        """A perfectly flat series has zero residual — no signal."""
        prices = np.ones(50) * 300.0
        result = kalman_filter(prices)
        self.assertFalse(result["is_signal"])
        self.assertAlmostEqual(result["signal_sigma"], 0.0, places=3)

    def test_trend_direction_up(self):
        prices = np.linspace(100, 200, 60)
        result = kalman_filter(prices)
        self.assertEqual(result["trend"], "up")

    def test_trend_direction_down(self):
        prices = np.linspace(200, 100, 60)
        result = kalman_filter(prices)
        self.assertEqual(result["trend"], "down")

    def test_single_point_does_not_crash(self):
        result = kalman_filter(np.array([500.0]))
        self.assertIn("filtered", result)

    def test_two_points_does_not_crash(self):
        result = kalman_filter(np.array([500.0, 502.0]))
        self.assertIn("is_signal", result)

    def test_signal_sigma_is_abs_of_latest(self):
        prices = self._noisy_trend()
        result = kalman_filter(prices)
        self.assertAlmostEqual(
            result["signal_sigma"],
            abs(result["latest_signal"]),
            places=2,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: HMM REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class TestHMMRegime(unittest.TestCase):

    def _mixed_returns(self, n_low=150, n_high=100):
        """Two-regime return series: low-vol followed by high-vol."""
        low_vol  = RNG.normal(0.001, 0.01, n_low)
        high_vol = RNG.normal(0.000, 0.04, n_high)
        return np.concatenate([low_vol, high_vol])

    def test_output_keys(self):
        returns = self._mixed_returns()
        result  = hmm_regime(returns)
        for key in ("current_regime", "regime_label", "confidence",
                    "is_high_volatility", "regime_history"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_insufficient_data_returns_gracefully(self):
        short = RNG.normal(0, 0.01, 30)
        result = hmm_regime(short)
        self.assertEqual(result["regime_label"], "insufficient_data")
        self.assertEqual(result["confidence"], 0.5)

    def test_high_vol_tail_classified_correctly(self):
        """When we feed a series ending in high-vol, regime should be high_volatility."""
        returns = self._mixed_returns(n_low=100, n_high=150)
        result  = hmm_regime(returns)
        self.assertEqual(result["regime_label"], "high_volatility",
                         "High-vol tail should be classified as high_volatility")

    def test_confidence_between_0_and_1(self):
        returns = self._mixed_returns()
        result  = hmm_regime(returns)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_regime_history_length(self):
        returns = self._mixed_returns()
        result  = hmm_regime(returns)
        self.assertEqual(len(result["regime_history"]), len(returns))

    def test_regime_label_is_valid(self):
        returns = self._mixed_returns()
        result  = hmm_regime(returns)
        self.assertIn(result["regime_label"],
                      ("low_volatility", "high_volatility",
                       "insufficient_data", "model_error"))

    def test_exactly_52_weeks(self):
        """Boundary: exactly 52 weeks should not return insufficient_data."""
        returns = RNG.normal(0, 0.02, 52)
        result  = hmm_regime(returns)
        self.assertNotEqual(result["regime_label"], "insufficient_data")

    def test_51_weeks_returns_insufficient(self):
        returns = RNG.normal(0, 0.02, 51)
        result  = hmm_regime(returns)
        self.assertEqual(result["regime_label"], "insufficient_data")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: SUPPLY SHOCK INDEX
# ══════════════════════════════════════════════════════════════════════════════

class TestSupplyShockIndex(unittest.TestCase):

    WHEAT_WEIGHTS = {"weather": 0.45, "geopolitical": 0.35, "inventory": 0.20}
    GOLD_WEIGHTS  = {"weather": 0.00, "geopolitical": 0.50, "inventory": 0.50}

    def test_output_keys(self):
        result = supply_shock_index([0.0], 0.0, 0.0, self.WHEAT_WEIGHTS)
        for key in ("ssi", "level", "direction",
                    "weather_component", "geo_component", "inventory_component"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_zero_inputs_returns_normal(self):
        result = supply_shock_index([], 0.0, 0.0, self.WHEAT_WEIGHTS)
        self.assertEqual(result["ssi"], 0.0)
        self.assertEqual(result["level"], "normal")

    def test_extreme_weather_zscore(self):
        """z-score of +4 in a 0.45-weighted commodity should push SSI high."""
        result = supply_shock_index([4.0], 0.0, 0.0, self.WHEAT_WEIGHTS)
        self.assertGreater(result["ssi"], 1.0,
                           "High weather z-score should produce elevated SSI")

    def test_extreme_geo_bullish(self):
        """Max surprise score (1.0) * geo weight * 3 = 0.35*3 = 1.05 for wheat."""
        result = supply_shock_index([], 1.0, 0.0, self.WHEAT_WEIGHTS)
        self.assertAlmostEqual(result["ssi"], 0.35 * 3.0, places=3)
        self.assertEqual(result["direction"], "bullish")

    def test_inventory_surplus_bearish(self):
        """Positive inventory_dev (surplus) → negative inv_component → bearish SSI."""
        result = supply_shock_index([], 0.0, 0.5, self.WHEAT_WEIGHTS)
        self.assertLess(result["ssi"], 0.0)
        self.assertEqual(result["direction"], "bearish")

    def test_inventory_deficit_bullish(self):
        """Negative inventory_dev (deficit) → positive inv_component → bullish SSI."""
        result = supply_shock_index([], 0.0, -0.3, self.WHEAT_WEIGHTS)
        self.assertGreater(result["ssi"], 0.0)
        self.assertEqual(result["direction"], "bullish")

    def test_weather_zero_weight_for_gold(self):
        """Gold has 0 weather weight — weather z-scores should not affect SSI."""
        result_no_weather = supply_shock_index([], 0.5, 0.0, self.GOLD_WEIGHTS)
        result_weather    = supply_shock_index([5.0], 0.5, 0.0, self.GOLD_WEIGHTS)
        self.assertAlmostEqual(result_no_weather["ssi"], result_weather["ssi"], places=6)

    def test_level_thresholds(self):
        cases = [
            ([0.5], 0.0, 0.0,  "normal"),
            ([3.0], 0.0, 0.0,  "elevated"),
            ([4.5], 0.3, -0.2, "high"),
        ]
        for zscores, geo, inv, expected_level in cases:
            result = supply_shock_index(zscores, geo, inv, self.WHEAT_WEIGHTS)
            self.assertEqual(
                result["level"], expected_level,
                f"Expected '{expected_level}' for z={zscores} geo={geo} inv={inv}, "
                f"got '{result['level']}' (ssi={result['ssi']})"
            )

    def test_components_sum_to_ssi(self):
        """weather + geo + inventory components must sum to the total SSI."""
        result = supply_shock_index([2.0], 0.4, -0.1, self.WHEAT_WEIGHTS)
        component_sum = round(
            result["weather_component"] +
            result["geo_component"] +
            result["inventory_component"], 3
        )
        self.assertAlmostEqual(result["ssi"], component_sum, places=2)

    def test_max_zscore_used_not_average(self):
        """SSI uses max |z| across regions, not average."""
        result_one  = supply_shock_index([3.0], 0.0, 0.0, self.WHEAT_WEIGHTS)
        result_many = supply_shock_index([3.0, 0.1, 0.1], 0.0, 0.0, self.WHEAT_WEIGHTS)
        self.assertAlmostEqual(result_one["ssi"], result_many["ssi"], places=6)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4: BAYESIAN SURPRISE SCORE
# ══════════════════════════════════════════════════════════════════════════════

class TestBayesianSurprise(unittest.TestCase):

    ROUTINE_HISTORY = [
        "wheat prices stable amid moderate export demand",
        "USDA releases weekly crop progress report",
        "Kansas winter wheat conditions improve slightly",
        "grain markets mixed ahead of WASDE report",
        "wheat exports from Ukraine port continue normally",
    ] * 10  # 50 articles representing normal baseline

    def test_output_keys(self):
        result = bayesian_surprise(["wheat prices rise"], self.ROUTINE_HISTORY, "Wheat")
        for key in ("surprise_score", "top_surprise_terms",
                    "interpretation", "kl_divergence"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_no_history_returns_half(self):
        result = bayesian_surprise(["some news"], [], "Wheat")
        self.assertEqual(result["surprise_score"], 0.5)
        self.assertEqual(result["interpretation"], "insufficient_history")

    def test_no_today_returns_zero(self):
        result = bayesian_surprise([], self.ROUTINE_HISTORY, "Wheat")
        self.assertEqual(result["surprise_score"], 0.0)
        self.assertEqual(result["interpretation"], "no_news_today")

    def test_routine_news_scores_low(self):
        """Same language AND vocabulary breadth as history should score low.

        A single short sentence has a concentrated word distribution and will
        score high against a diverse corpus even with routine vocabulary — that
        is correct model behaviour. To test 'routine', today must have the same
        breadth as history (multiple diverse sentences), so p ≈ q and KL ≈ 0.
        """
        today = [
            "wheat prices stable amid moderate export demand",
            "USDA releases weekly crop progress report",
            "Kansas winter wheat conditions improve slightly",
            "grain markets mixed ahead of WASDE report",
            "wheat exports from Ukraine port continue normally",
        ]
        result = bayesian_surprise(today, self.ROUTINE_HISTORY, "Wheat")
        self.assertLess(result["surprise_score"], 0.3,
                        f"Routine news scored too high: {result['surprise_score']}")

    def test_shocking_news_scores_high(self):
        """Entirely new vocabulary should score high."""
        today = [
            "russia bans exports nuclear escalation embargo sanctions",
            "catastrophic drought destroys entire harvest unprecedented famine",
            "invasion blockade military conflict destroys ports",
        ]
        result = bayesian_surprise(today, self.ROUTINE_HISTORY, "Wheat")
        self.assertGreater(result["surprise_score"], 0.3,
                           f"Shocking news scored too low: {result['surprise_score']}")

    def test_surprise_score_bounded_0_to_1(self):
        extremes = [
            "zxqvb qxzpw jklmn unique bizarre unprecedented catastrophic"
        ] * 20
        result = bayesian_surprise(extremes, self.ROUTINE_HISTORY, "Wheat")
        self.assertGreaterEqual(result["surprise_score"], 0.0)
        self.assertLessEqual(result["surprise_score"], 1.0)

    def test_kl_divergence_non_negative(self):
        result = bayesian_surprise(
            ["some wheat news today"], self.ROUTINE_HISTORY, "Wheat"
        )
        self.assertGreaterEqual(result["kl_divergence"], 0.0)

    def test_interpretation_labels(self):
        valid_labels = {
            "routine_news", "moderately_unusual",
            "notably_surprising", "highly_unexpected",
            "insufficient_history", "no_news_today",
        }
        result = bayesian_surprise(
            ["wheat rises sharply"], self.ROUTINE_HISTORY, "Wheat"
        )
        self.assertIn(result["interpretation"], valid_labels)

    def test_top_surprise_terms_max_five(self):
        today = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]
        result = bayesian_surprise(today, self.ROUTINE_HISTORY, "Wheat")
        self.assertLessEqual(len(result["top_surprise_terms"]), 5)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5: COINTEGRATION MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class TestCointegrationSpread(unittest.TestCase):

    def _cointegrated_pair(self, n=260):
        """Two genuinely cointegrated series: b is a random walk, a tracks b."""
        b = np.cumsum(RNG.normal(0, 1, n)) + 1000
        a = 1.5 * b + RNG.normal(0, 5, n)   # a = 1.5b + noise → cointegrated
        return a, b

    def _independent_pair(self, n=260):
        """Two independent random walks — not cointegrated."""
        a = np.cumsum(RNG.normal(0, 1, n)) + 500
        b = np.cumsum(RNG.normal(0, 1, n)) + 500
        return a, b

    def test_output_keys_cointegrated(self):
        a, b = self._cointegrated_pair()
        result = cointegration_spread(a, b, "gold", "silver")
        for key in ("cointegrated", "pair"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_cointegrated_pair_detected(self):
        """Genuinely cointegrated series should be flagged as cointegrated."""
        a, b = self._cointegrated_pair(n=260)
        result = cointegration_spread(a, b, "gold", "silver")
        self.assertTrue(result["cointegrated"],
                        f"Expected cointegrated=True, got pvalue={result.get('pvalue')}")

    def test_cointegrated_result_has_spread_zscore(self):
        a, b = self._cointegrated_pair()
        result = cointegration_spread(a, b, "gold", "silver")
        if result["cointegrated"]:
            self.assertIn("spread_zscore", result)
            self.assertIn("hedge_ratio", result)
            self.assertIn("signal", result)

    def test_insufficient_data(self):
        a = RNG.normal(0, 1, 40)
        b = RNG.normal(0, 1, 40)
        result = cointegration_spread(a, b, "wheat", "corn")
        self.assertFalse(result["cointegrated"])
        self.assertEqual(result.get("reason"), "insufficient_data")

    def test_pair_label_correct(self):
        a, b = self._cointegrated_pair()
        result = cointegration_spread(a, b, "gold", "silver")
        self.assertEqual(result["pair"], "gold/silver")

    def test_exactly_52_observations(self):
        a, b = self._cointegrated_pair(n=52)
        result = cointegration_spread(a, b, "oil_brent", "natural_gas")
        # Should not crash and should not return insufficient_data
        self.assertNotEqual(result.get("reason"), "insufficient_data")

    def test_51_observations_insufficient(self):
        a, b = self._cointegrated_pair(n=51)
        result = cointegration_spread(a, b, "oil_brent", "natural_gas")
        self.assertFalse(result["cointegrated"])
        self.assertEqual(result.get("reason"), "insufficient_data")

    def test_mean_reverting_flag_on_extreme_spread(self):
        """Force an extreme spread by injecting a spike in series a."""
        a, b = self._cointegrated_pair(n=260)
        a[-1] += 200.0    # large spike → current spread far from mean
        result = cointegration_spread(a, b, "gold", "silver")
        if result["cointegrated"]:
            self.assertTrue(result["mean_reverting"],
                            f"Expected mean_reverting=True, z={result.get('spread_zscore')}")

    def test_spread_zscore_near_zero_for_normal_spread(self):
        """Without any spike, spread z-score should be small."""
        a, b = self._cointegrated_pair(n=260)
        result = cointegration_spread(a, b, "corn", "soybeans")
        if result["cointegrated"]:
            self.assertLess(abs(result["spread_zscore"]), 3.0,
                            "Z-score should be small for a well-behaved pair")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
