"""Independent accounting and tail-loss examples for development research."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from safeinvest_research.metrics import discrete_cvar, empirical_risk, path_metrics, summarize_paths
from safeinvest_research.static import simplex_grid, simulate_annual_targets


def scalar_reference(returns, inflation, targets, rate):
    """Independent scalar bisection fee equation; no production helper calls."""
    n, horizon, _ = returns.shape
    nominal = np.full((n, len(targets), horizon + 1), 100.0)
    real = nominal.copy()
    fees = np.zeros((n, len(targets), horizon))
    moved = fees.copy()
    for scenario in range(n):
        for candidate, target in enumerate(targets):
            holdings = [0.0, 0.0, 100.0]
            price = 1.0
            for year in range(horizon):
                wealth = sum(holdings)
                lower, upper = 0.0, wealth
                for _ in range(100):
                    cost = (lower + upper) / 2
                    transfers = sum(max((wealth - cost) * target[a] - holdings[a], 0.0) for a in range(3))
                    if cost < rate * transfers:
                        lower = cost
                    else:
                        upper = cost
                cost = (lower + upper) / 2
                post = [(wealth - cost) * target[a] for a in range(3)]
                fees[scenario, candidate, year] = cost
                moved[scenario, candidate, year] = sum(max(post[a] - holdings[a], 0.0) for a in range(3))
                holdings = [post[a] * (1 + returns[scenario, year, a]) for a in range(3)]
                price *= 1 + inflation[scenario, year]
                nominal[scenario, candidate, year + 1] = sum(holdings)
                real[scenario, candidate, year + 1] = sum(holdings) / price
    return nominal, real, fees, moved


class GridTests(unittest.TestCase):
    def test_full_simplex_contains_vertices_and_exact_equal_once(self):
        for step, count in ((0.05, 232), (0.025, 862)):
            with self.subTest(step=step):
                weights = simplex_grid(step)
                self.assertEqual(weights.shape, (count, 3))
                assert_allclose(weights.sum(axis=1), 1.0, atol=1e-15)
                self.assertTrue(np.all(weights >= 0))
                for vertex in np.eye(3):
                    self.assertEqual(np.count_nonzero(np.all(weights == vertex, axis=1)), 1)
                self.assertEqual(np.count_nonzero(np.all(weights == 1 / 3, axis=1)), 1)
                self.assertEqual(len(np.unique(weights, axis=0)), count)

    def test_equal_weight_deduplicates_when_already_on_lattice(self):
        weights = simplex_grid(1 / 3)
        self.assertEqual(weights.shape, (10, 3))
        self.assertEqual(np.count_nonzero(np.all(weights == 1 / 3, axis=1)), 1)
        self.assertEqual(simplex_grid(0.05, include_equal=False).shape, (231, 3))

    def test_bad_steps_are_rejected(self):
        for step in (0, -0.05, 1.1, 0.3, np.nan, np.inf, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                simplex_grid(step)


class AnnualAccountingTests(unittest.TestCase):
    def test_initial_fee_is_funded_and_based_on_actual_transfer(self):
        result = simulate_annual_targets(np.zeros((1, 1, 3)), np.zeros((1, 1)), [1, 0, 0])
        cost = 100 * 0.0025 / 1.0025
        self.assertAlmostEqual(result["fees"][0, 0, 0], cost)
        self.assertAlmostEqual(result["moved"][0, 0, 0], 100 - cost)
        self.assertAlmostEqual(result["nominal_wealth"][0, 0, 1] + cost, 100)
        assert_allclose(result["fees"], 0.0025 * result["moved"])
        metrics = path_metrics(result["real_wealth"])
        self.assertAlmostEqual(metrics["mdd"][0, 0], cost / 100)

    def test_actual_twenty_transfer_costs_point_zero_five(self):
        target = np.array([20, 0, 79.95]) / 99.95
        result = simulate_annual_targets(np.zeros((1, 1, 3)), np.zeros((1, 1)), target)
        assert_allclose(result["moved"], 20)
        assert_allclose(result["fees"], 0.05)
        assert_allclose(result["nominal_wealth"], [[[100, 99.95]]])

    def test_all_bills_never_transfers_or_pays_fees(self):
        market = np.array([[[0.3, -0.1, 0.02], [-0.5, 0.2, 0.03]]])
        result = simulate_annual_targets(market, [[0.02, 0.03]], [[0, 0, 1]])
        assert_array_equal(result["fees"], np.zeros((1, 1, 2)))
        assert_array_equal(result["moved"], np.zeros((1, 1, 2)))
        assert_array_equal(result["trades"], np.zeros((1, 1, 2), dtype=bool))
        assert_allclose(result["real_wealth"], [[[100, 100, 100]]])

    def test_vectorized_paths_match_independent_scalar_reference(self):
        market = np.array([
            [[0.2, -0.1, 0.01], [-0.3, 0.15, 0.02], [0.07, -0.05, 0.01]],
            [[-0.4, 0.2, 0.02], [0.3, -0.15, 0.03], [0.1, 0.1, -0.01]],
        ])
        inflation = np.array([[0.03, 0.02, -0.01], [0.06, -0.02, 0.04]])
        targets = np.array([[1 / 3] * 3, [0, 0, 1], [0.6, 0.4, 0], [1, 0, 0]])
        for rate in (0.0, 0.0025, 0.25):
            with self.subTest(rate=rate):
                expected = scalar_reference(market, inflation, targets, rate)
                result = simulate_annual_targets(market, inflation, targets, rate)
                for key, value in zip(("nominal_wealth", "real_wealth", "fees", "moved"), expected):
                    assert_allclose(result[key], value, atol=2e-12, rtol=2e-13)
                self.assertEqual(result["real_wealth"].shape, (2, 4, 4))

    def test_rebalancing_restores_target_after_drift(self):
        market = np.array([[[1.0, 0, 0], [0, 0, 0]]])
        result = simulate_annual_targets(market, [[0, 0]], [0.5, 0, 0.5], cost_rate=0)
        assert_allclose(result["nominal_wealth"], [[[100, 150, 150]]])
        assert_allclose(result["moved"], [[[50, 25]]])

    def test_bad_shapes_and_gross_factors_are_rejected(self):
        bad = [
            (np.zeros((2, 3)), np.zeros((2, 1)), [1, 0, 0], .0025),
            (np.zeros((1, 2, 3)), np.zeros((1, 1)), [1, 0, 0], .0025),
            (np.array([[[-1, 0, 0]]]), np.zeros((1, 1)), [1, 0, 0], .0025),
            (np.zeros((1, 1, 3)), [[np.nan]], [1, 0, 0], .0025),
            (np.zeros((1, 1, 3)), [[0]], [.5, .5, .5], .0025),
            (np.zeros((1, 1, 3)), [[0]], [-.1, .1, 1], .0025),
            (np.zeros((1, 1, 3)), [[0]], [1, 0, 0], True),
        ]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                simulate_annual_targets(*args)


class ResearchMetricTests(unittest.TestCase):
    def test_hand_computed_real_growth_drawdown_and_volatility(self):
        # Zero-cost all-equity: nominal 100 -> 132 -> 108.9,
        # prices 1 -> 1.1 -> 1.21, real 100 -> 120 -> 90.
        market = np.array([[[.32, 0, 0], [-.175, 0, 0]]])
        result = simulate_annual_targets(market, [[.1, .1]], [1, 0, 0], cost_rate=0)
        assert_allclose(result["real_wealth"], [[[100, 120, 90]]])
        metrics = path_metrics(result["real_wealth"])
        assert_allclose(metrics["log_growth"], [[np.log(.9) / 2]])
        assert_allclose(metrics["cagr"], [[np.sqrt(.9) - 1]])
        assert_allclose(metrics["mdd"], [[.25]])
        assert_allclose(metrics["volatility"], [[np.std([.2, -.25], ddof=1)]])

    def test_cvar_splits_probability_atom_at_tail_boundary(self):
        self.assertAlmostEqual(float(discrete_cvar([.1, .2, .9], [.89, .10, .01])), .27)
        self.assertAlmostEqual(float(discrete_cvar([.1, .21], [.9, .1])), .21)
        self.assertAlmostEqual(float(discrete_cvar([.1, .1, .1])), .1)

    def test_cvar_equal_samples_include_fractional_observation(self):
        # Worst 10% of 12 observations is one full maximum plus 0.2
        # observations of the second largest, divided by 1.2 observations.
        values = np.arange(12) / 20
        expected = (.55 + .2 * .5) / 1.2
        self.assertAlmostEqual(float(discrete_cvar(values)), expected)
        matrix = np.column_stack((values, values / 2))
        assert_allclose(discrete_cvar(matrix), [expected, expected / 2])
        assert_allclose(discrete_cvar(matrix.T, axis=1), [expected, expected / 2])

    def test_cvar_matches_independent_variational_formula(self):
        losses = np.array([[-.2, .9], [.1, .3], [.4, .2], [.8, -.1]])
        probabilities = np.array([.15, .50, .35, 0])
        for level in (.1, .5, .9, .99):
            expected = [
                min(eta + sum(probabilities * np.maximum(losses[:, column] - eta, 0)) / (1 - level)
                    for eta in losses[:, column])
                for column in range(2)
            ]
            assert_allclose(discrete_cvar(losses, probabilities, level), expected, atol=1e-14)

    def test_breach_equal_boundary_uses_declared_numerical_tolerance(self):
        risk = empirical_risk([.30000000000000004, .3 + 5e-13, .3 + 2e-12], .3)
        self.assertAlmostEqual(float(risk["breach_probability"]), 1 / 3)
        self.assertAlmostEqual(float(risk["expected_excess"]), 2e-12 / 3, delta=1e-16)

    def test_summary_distinguishes_log_growth_from_median_cagr(self):
        metrics = path_metrics([[100, 200], [100, 80], [100, 100]])
        summary = summarize_paths(metrics, loss_tolerance=.1)
        self.assertAlmostEqual(float(summary["mean_log_growth"]), np.log(1.6) / 3)
        self.assertAlmostEqual(float(summary["median_cagr"]), 0)
        self.assertAlmostEqual(float(summary["p10_cagr"]), -.2)
        self.assertAlmostEqual(float(summary["mean_mdd"]), .2 / 3)
        self.assertAlmostEqual(float(summary["cvar90_mdd"]), .2)
        self.assertAlmostEqual(float(summary["breach_probability"]), 1 / 3)
        self.assertAlmostEqual(float(summary["expected_excess"]), .1 / 3)

    def test_inverted_cdf_p90_and_direct_breach_count_agree(self):
        # Linear P90 interpolates to .189 and would pass L=.20, despite two
        # breaches among 12 paths. Inverse empirical CDF gives .21 and fails.
        losses = np.array([0] * 10 + [.21, .8])
        wealth = np.column_stack((np.full(12, 100.0), 100 * (1 - losses)))
        summary = summarize_paths(path_metrics(wealth), loss_tolerance=.2)
        self.assertAlmostEqual(float(summary["p90_mdd"]), .21)
        self.assertAlmostEqual(float(summary["breach_probability"]), 2 / 12)
        self.assertLess(float(np.quantile(losses, .9, method="linear")), .2)
        for tolerance in (0, .19, .20, .21, .5, .8, 1):
            summary = summarize_paths(path_metrics(wealth), loss_tolerance=tolerance)
            quantile_pass = float(summary["p90_mdd"]) <= tolerance + 1e-12
            count_pass = float(summary["breach_probability"]) <= .1 + 1e-12
            self.assertEqual(quantile_pass, count_pass)

    def test_standard_even_median_is_distinct_from_inverse_cdf(self):
        summary = summarize_paths(path_metrics([[100, 100], [100, 121]]))
        self.assertAlmostEqual(float(summary["median_cagr"]), .105)
        self.assertAlmostEqual(float(summary["median_terminal_real"]), 110.5)

    def test_nonfinite_and_bad_probability_inputs_are_rejected(self):
        for values in ([np.nan, .1], [np.inf, .1], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                discrete_cvar(values)
        for probabilities in ([.5, .500001], [.5, -.5], [1], [np.nan, .5]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                discrete_cvar([.1, .2], probabilities)
        for wealth in ([[100, np.nan]], [[100, 0]], [[99, 100]], [[100]]):
            with self.subTest(wealth=wealth), self.assertRaises(ValueError):
                path_metrics(wealth)


if __name__ == "__main__":
    unittest.main()
