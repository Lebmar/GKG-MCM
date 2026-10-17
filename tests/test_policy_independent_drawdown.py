"""Synthetic tests of the interval-product MDD lower bound; no market I/O."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("synthetic_drawdown_bound", ROOT / "scripts/analyze_policy_independent_drawdown.py")
BOUND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUND)
from safeinvest.core import execute_target


def mdd(wealth):
    wealth = np.asarray(wealth)
    return np.max(1 - wealth / np.maximum.accumulate(wealth, axis=-1), axis=-1)


class EnvelopeTests(unittest.TestCase):
    def test_all_interval_product_enumeration_matches_log_peak_algorithm(self):
        rng = np.random.default_rng(783)
        gross = rng.uniform(.55, 1.35, (20, 7, 3)); inflation_gross = rng.uniform(.93, 1.2, (20, 7))
        result = BOUND.drawdown_envelope(gross - 1, inflation_gross - 1)
        ideal_gross = np.max(gross, axis=-1) / inflation_gross
        expected = []
        for path in ideal_gross:
            declines = [0.0]
            for start in range(len(path)):
                for end in range(start + 1, len(path) + 1):
                    declines.append(1 - np.prod(path[start:end]))
            expected.append(max(declines))
        np.testing.assert_allclose(result["mdd_lower_bound"], expected, atol=5e-15, rtol=0)
        np.testing.assert_allclose(result["mdd_lower_bound"], mdd(result["ideal_real_wealth"]), atol=5e-15, rtol=0)

    def test_always_rising_envelope_has_zero_drawdown_including_initial_point(self):
        result = BOUND.drawdown_envelope(np.full((2, 4, 3), .05), np.zeros((2, 4)))
        np.testing.assert_array_equal(result["mdd_lower_bound"], 0)
        np.testing.assert_array_equal(result["witness_peak_time"], 0)
        np.testing.assert_array_equal(result["witness_trough_time"], 0)
        # An initial loss is observed even when there is only one year.
        loss = BOUND.drawdown_envelope(np.full((1, 1, 3), -.1), np.zeros((1, 1)))
        self.assertAlmostEqual(loss["mdd_lower_bound"][0], .1)

    def test_peak_to_trough_witness_is_not_loss_from_initial_wealth(self):
        result = BOUND.drawdown_envelope(np.array([[[.2] * 3, [-.1] * 3]]), np.zeros((1, 2)))
        self.assertAlmostEqual(result["mdd_lower_bound"][0], .1)
        self.assertEqual(result["witness_peak_time"][0], 1)
        self.assertEqual(result["witness_trough_time"][0], 2)
        self.assertAlmostEqual(result["ideal_real_wealth"][0, 1], 120)
        self.assertAlmostEqual(result["ideal_real_wealth"][0, 2], 108)

    def test_common_inflation_can_force_loss_even_with_nonnegative_nominal_returns(self):
        result = BOUND.drawdown_envelope(np.zeros((1, 2, 3)), np.full((1, 2), .2))
        self.assertAlmostEqual(result["mdd_lower_bound"][0], 1 - 1 / 1.2 ** 2)
        self.assertGreater(result["mdd_lower_bound"][0], .3)

    def test_clairvoyant_asset_switching_is_an_optimistic_relaxation(self):
        gross = np.array([[[.5, 1.1, 1], [1.1, .5, 1]]])
        result = BOUND.drawdown_envelope(gross - 1, np.zeros((1, 2)))
        np.testing.assert_array_equal(result["best_asset_index"], [[1, 0]])
        np.testing.assert_allclose(result["annual_real_envelope"], 1.1, atol=1e-15, rtol=0)
        self.assertEqual(result["mdd_lower_bound"][0], 0)
        self.assertGreater(mdd([100, 50, 55]), .4)
        self.assertIn("not a causal", BOUND.drawdown_envelope.__doc__)

    def test_arbitrary_fee_bearing_causal_hold_restore_paths_obey_interval_bound(self):
        rng = np.random.default_rng(1038)
        n, years = 25, 8
        gross = rng.uniform(.72, 1.25, (n, years, 3)); inflation_gross = rng.uniform(.96, 1.12, (n, years))
        bound = BOUND.drawdown_envelope(gross - 1, inflation_gross - 1)
        target_bank = rng.dirichlet(np.ones(3), size=(years, n))
        for cost in (0, .0025, .05):
            x = np.zeros((n, 3)); x[:, 2] = 100
            real = np.full((n, years + 1), 100.0); cpi = np.ones(n)
            for year in range(years):
                # Uses only earlier observed equity returns and an independent
                # target bank, never that year's actual outcome.
                restore = np.ones(n, bool) if year == 0 else gross[:, year - 1, 0] < 1
                post, fee, _ = execute_target(x, target_bank[year], cost)
                self.assertTrue(np.all(fee >= 0))
                x = np.where(restore[:, None], post, x) * gross[:, year]
                cpi *= inflation_gross[:, year]
                real[:, year + 1] = x.sum(axis=1) / cpi
            actual_mdd = mdd(real)
            self.assertTrue(np.all(actual_mdd + 5e-14 >= bound["mdd_lower_bound"]))
            actual_factors = real[:, 1:] / real[:, :-1]
            self.assertTrue(np.all(actual_factors <= bound["annual_real_envelope"] + 5e-14))
            for start in range(years):
                for end in range(start + 1, years + 1):
                    self.assertTrue(np.all(real[:, end] / real[:, start]
                                           <= np.prod(bound["annual_real_envelope"][:, start:end], axis=1) + 5e-14))

    def test_fee_cannot_improve_bound_and_no_terminal_liquidation_is_added(self):
        result = BOUND.drawdown_envelope(np.full((1, 1, 3), -.1), np.zeros((1, 1)))
        post, fee, _ = execute_target([0, 0, 100], [1, 0, 0], .0025)
        actual = [100, float(np.sum(post * .9))]
        self.assertAlmostEqual(fee, 100 * .0025 / 1.0025, places=13)
        self.assertAlmostEqual(actual[-1], 90 / 1.0025, places=12)
        self.assertGreater(mdd(actual), result["mdd_lower_bound"][0])

    def test_pointwise_wealth_dominance_alone_does_not_order_mdd(self):
        upper, lower = np.array([100, 200, 120]), np.array([100, 110, 105])
        self.assertTrue(np.all(upper >= lower))
        self.assertGreater(mdd(upper), mdd(lower))

    def test_intrayear_switching_is_outside_the_theorem(self):
        result = BOUND.drawdown_envelope(np.full((1, 1, 3), -.2), np.zeros((1, 1)))
        self.assertAlmostEqual(result["mdd_lower_bound"][0], .2)
        # Assets A:(2,.4), B:(.4,2) both have annualgross.8.  A then B
        # inside the year returns4 and invalidates this annual-trading bound.
        self.assertEqual(mdd([100, 400]), 0)
        self.assertGreater(result["mdd_lower_bound"][0], mdd([100, 400]))

    def test_each_horizon_uses_only_its_own_prefix(self):
        gross = np.array([[[1.2] * 3, [.8] * 3, [.7] * 3, [1.1] * 3]])
        short = BOUND.drawdown_envelope(gross - 1, np.zeros((1, 4)), 2)
        long = BOUND.drawdown_envelope(gross - 1, np.zeros((1, 4)), 3)
        self.assertAlmostEqual(short["mdd_lower_bound"][0], .2)
        self.assertAlmostEqual(long["mdd_lower_bound"][0], .44)
        self.assertEqual(short["ideal_real_wealth"].shape, (1, 3))
        self.assertEqual(short["witness_trough_time"][0], 2)

    def test_invalid_shapes_gross_and_horizon_are_rejected(self):
        for market, infl in ((np.empty((0, 2, 3)), np.empty((0, 2))), (np.zeros((2, 3)), np.zeros((2, 1))),
                             (np.zeros((1, 2, 3)), np.zeros((1, 3))), (np.full((1, 1, 3), -1), np.zeros((1, 1))),
                             (np.zeros((1, 1, 3)), np.full((1, 1), -1)), (np.full((1, 1, 3), np.nan), np.zeros((1, 1)))):
            with self.subTest(shape=np.shape(market)), self.assertRaises(ValueError):
                BOUND.drawdown_envelope(market, infl)
        for horizon in (0, -1, 3, 1.0, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                BOUND.drawdown_envelope(np.zeros((1, 2, 3)), np.zeros((1, 2)), horizon)


if __name__ == "__main__":
    unittest.main()
