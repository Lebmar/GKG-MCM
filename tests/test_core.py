"""Economic invariants for execution, purchasing-power metrics, and data gates."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from safeinvest.core import actual_wealth, execute_target, metric_arrays
from safeinvest.data import REQUIRED_COLUMNS, load_paths


class ExecutionTests(unittest.TestCase):
    def test_actual_transfer_twenty_costs_five_hundredths(self):
        target = np.array([20.0, 0.0, 79.95]) / 99.95
        post, cost, moved = execute_target([0, 0, 100], target)
        assert_allclose(post, [20, 0, 79.95], atol=1e-12)
        self.assertAlmostEqual(float(moved), 20.0)
        self.assertAlmostEqual(float(cost), 0.05)

    def test_allocation_fee_is_self_financing(self):
        post, cost, moved = execute_target([0, 0, 100], [1, 0, 0])
        self.assertAlmostEqual(float(cost), 0.0025 * 100 / 1.0025)
        self.assertAlmostEqual(float(moved), float(post[0]))
        self.assertAlmostEqual(float(cost), 0.0025 * float(moved))
        self.assertAlmostEqual(float(post.sum() + cost), 100.0)

    def test_vector_execution_and_cost_equation(self):
        rng = np.random.default_rng(20261017)
        holdings = rng.dirichlet([0.4, 0.4, 0.4], 2000) * rng.uniform(1, 10000, (2000, 1))
        targets = rng.dirichlet([0.3, 0.3, 0.3], 2000)
        # The large fee also exercises changes of destination active set.
        for rate in (0.0, 0.0025, 0.25, 0.99):
            post, costs, moved = execute_target(holdings, targets, rate)
            assert_allclose(post.sum(axis=1) + costs, holdings.sum(axis=1), rtol=1e-14)
            assert_allclose(costs, rate * np.maximum(post - holdings, 0).sum(axis=1), rtol=2e-13, atol=1e-12)
            assert_allclose(moved, np.maximum(post - holdings, 0).sum(axis=1))
            assert_allclose(post / post.sum(axis=1, keepdims=True), targets, atol=1e-14)
            self.assertTrue(np.all(post >= 0))

    def test_no_trade_has_no_cost_and_preserves_holdings(self):
        holdings = np.array([[20, 30, 50], [1.23, 4.56, 7.89]])
        post, cost, moved = execute_target(holdings, holdings / holdings.sum(axis=1, keepdims=True))
        assert_array_equal(post, holdings)
        assert_array_equal(cost, [0, 0])
        assert_array_equal(moved, [0, 0])

    def test_zero_cost_and_broadcast(self):
        post, cost, moved = execute_target([[0, 0, 100], [50, 0, 150]], [0.5, 0.25, 0.25], 0)
        assert_allclose(post, [[50, 25, 25], [100, 50, 50]])
        assert_array_equal(cost, [0, 0])
        assert_allclose(moved, [75, 100])

    def test_zero_wealth_and_input_validation(self):
        post, cost, moved = execute_target([0, 0, 0], [0.3, 0.3, 0.4])
        assert_array_equal(post, [0, 0, 0])
        self.assertEqual(float(cost), 0)
        self.assertEqual(float(moved), 0)
        for holdings, target, rate in [
            ([-1, 1, 100], [0, 0, 1], 0.0025),
            ([0, 0, 100], [0.4, 0.4, 0.4], 0.0025),
            ([0, 0, 100], [0, 0, 1], -0.001),
            ([0, 0, 100], [0, 0, 1], 1),
            ([0, 0, np.nan], [0, 0, 1], 0.0025),
        ]:
            with self.subTest(holdings=holdings, target=target, rate=rate), self.assertRaises(ValueError):
                execute_target(holdings, target, rate)


class MetricTests(unittest.TestCase):
    def test_inflation_deflation_preserves_purchasing_power(self):
        wealth = [[100, 110, 121]]
        inflation = [[0.1, 0.1]]
        assert_allclose(actual_wealth(wealth, inflation), [[100, 100, 100]])
        metrics = metric_arrays(wealth, inflation)
        assert_allclose(metrics["cagr"], [0], atol=1e-14)
        assert_allclose(metrics["mdd"], [0], atol=1e-14)
        assert_allclose(metrics["volatility"], [0], atol=1e-14)
        assert_allclose(metrics["terminal_real"], [100])

    def test_drawdown_is_from_running_peak_not_initial_wealth(self):
        metrics = metric_arrays([[100, 120, 90]], [[0, 0]])
        assert_allclose(metrics["mdd"], [0.25])
        assert_allclose(metrics["cagr"], [np.sqrt(0.9) - 1])
        assert_allclose(metrics["volatility"], [np.std([0.2, -0.25], ddof=1)])

    def test_initial_cost_is_in_first_return_and_maximum_drawdown(self):
        post, cost, _ = execute_target([0, 0, 100], [1, 0, 0])
        wealth = [[100, post.sum(), post.sum()]]
        metrics = metric_arrays(wealth, [[0, 0]])
        first_loss = float(cost) / 100
        assert_allclose(metrics["mdd"], [first_loss])
        assert_allclose(metrics["cagr"], [np.sqrt(1 - first_loss) - 1])
        assert_allclose(metrics["volatility"], [np.std([-first_loss, 0], ddof=1)])

    def test_real_drawdown_can_occur_with_rising_nominal_wealth(self):
        metrics = metric_arrays([[100, 105]], [[0.1]])
        assert_allclose(metrics["mdd"], [1 - 1.05 / 1.1])
        assert_allclose(metrics["cagr"], [1.05 / 1.1 - 1])
        assert_array_equal(metrics["volatility"], [0])

    def test_deflation_and_path_validation(self):
        assert_allclose(actual_wealth([[100, 99]], [[-0.01]]), [[100, 100]])
        for wealth, inflation in [([[100, 110]], [[-1]]), ([[100]], [[]]), ([[100, 110]], [[0, 0]]), ([[100, 0]], [[0]])]:
            with self.subTest(wealth=wealth, inflation=inflation), self.assertRaises(ValueError):
                metric_arrays(wealth, inflation)


class DataTests(unittest.TestCase):
    def write_csv(self, directory, rows):
        path = Path(directory) / "market.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(REQUIRED_COLUMNS)
            writer.writerows(rows)
        return path

    def complete_rows(self, sid):
        return [[sid, period, sid / 1000, 0.02, 0.01, 0.03] for period in range(1, 31)]

    def test_preserves_requested_order_and_sorts_periods(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, list(reversed(self.complete_rows(1) + self.complete_rows(2))))
            returns, inflation, ids = load_paths(path, [2, 1])
        self.assertEqual(returns.shape, (2, 30, 3))
        self.assertEqual(inflation.shape, (2, 30))
        assert_array_equal(ids, [2, 1])
        assert_allclose(returns[:, 0, 0], [0.002, 0.001])
        assert_allclose(inflation, 0.03)

    def test_final_gate_runs_before_opening_file(self):
        with self.assertRaisesRegex(ValueError, "allow_final"):
            load_paths("file-must-not-be-opened.csv", [121])
        with self.assertRaisesRegex(ValueError, "allow_final"):
            load_paths("file-must-not-be-opened.csv", [121], allow_final="False")
        # Opt-in behavior is checked on synthetic data, never final contest data.
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, self.complete_rows(121))
            returns, _, ids = load_paths(path, [121], allow_final=True)
        self.assertEqual(returns.shape, (1, 30, 3))
        assert_array_equal(ids, [121])

    def test_unrequested_market_values_are_not_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, self.complete_rows(1) + [[121, 1, "do not inspect", "", "", ""]])
            returns, _, ids = load_paths(path, [1])
        self.assertEqual(returns.shape, (1, 30, 3))
        assert_array_equal(ids, [1])

    def test_incomplete_duplicate_and_invalid_gross_are_rejected(self):
        valid = self.complete_rows(1)
        invalid_gross = [list(row) for row in valid]
        invalid_gross[0][2] = -1
        nonfinite = [list(row) for row in valid]
        nonfinite[0][5] = "nan"
        for rows in (valid[:-1], valid + [valid[0]], invalid_gross, nonfinite):
            with self.subTest(rows=rows[:1]), tempfile.TemporaryDirectory() as directory:
                path = self.write_csv(directory, rows)
                with self.assertRaises(ValueError):
                    load_paths(path, [1])

    def test_invalid_requested_ids_are_rejected(self):
        for ids in ([], [1, 1], [0], [181], [1.5], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                load_paths("file-must-not-be-opened.csv", ids)


if __name__ == "__main__":
    unittest.main()
