"""Synthetic joint-order/accounting invariants; no competition data is read."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_temporal_runner", ROOT / "scripts/run_temporal_order_diagnostic.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class JointOrderingTests(unittest.TestCase):
    def test_seeded_orders_use_one_continuous_rng_and_each_path_has_a_permutation(self):
        n, years, seed = 6, 8, 87020261019
        orders = RUNNER.make_orders(n, years, seed)
        self.assertEqual(set(orders), {"original", "reverse", "permuted"})
        rng = np.random.default_rng(seed)
        expected = np.stack([rng.permutation(years) for _ in range(n)])
        np.testing.assert_array_equal(orders["permuted"], expected)
        np.testing.assert_array_equal(orders["original"], np.tile(np.arange(years), (n, 1)))
        np.testing.assert_array_equal(orders["reverse"], np.tile(np.arange(years - 1, -1, -1), (n, 1)))
        for order in orders.values():
            np.testing.assert_array_equal(np.sort(order, axis=1), np.tile(np.arange(years), (n, 1)))
        self.assertGreater(len({tuple(row) for row in orders["permuted"]}), 1)
        again = RUNNER.make_orders(n, years, seed)
        np.testing.assert_array_equal(again["permuted"], orders["permuted"])

    def test_assets_and_inflation_move_as_the_same_complete_joint_year(self):
        returns = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3) / 100
        inflation = np.arange(10, dtype=float).reshape(2, 5) / 1000
        before_returns, before_inflation = returns.copy(), inflation.copy()
        order = np.array([[4, 1, 3, 0, 2], [2, 0, 4, 1, 3]])
        r, i = RUNNER.apply_order(returns, inflation, order)
        for path in range(2):
            joint = np.column_stack((returns[path], inflation[path]))
            np.testing.assert_array_equal(np.column_stack((r[path], i[path])), joint[order[path]])
        np.testing.assert_array_equal(returns, before_returns)
        np.testing.assert_array_equal(inflation, before_inflation)

    def test_duplicate_or_out_of_range_indices_are_rejected(self):
        returns = np.zeros((2, 4, 3)); inflation = np.zeros((2, 4))
        bad = (np.array([[0, 1, 1, 3], [0, 1, 2, 3]]),
               np.array([[0, 1, 2, 4], [0, 1, 2, 3]]),
               np.array([[-1, 1, 2, 3], [0, 1, 2, 3]]))
        for indices in bad:
            with self.subTest(indices=indices.tolist()), self.assertRaises(ValueError):
                RUNNER.apply_order(returns, inflation, indices)


class FinancialOrderTests(unittest.TestCase):
    def market(self):
        returns = np.array([[[.2, -.03, .01], [-.15, .1, .015], [.3, -.02, -.005],
                             [.04, .08, .02], [-.12, -.04, .005], [.1, .02, .01]],
                            [[-.1, .07, .02], [.05, -.08, .01], [.3, .02, .005],
                             [-.2, .06, -.01], [.15, -.03, .02], [.02, .04, .01]]])
        inflation = np.array([[.02, .03, -.01, .025, .015, .01], [.01, .02, .04, -.01, .03, .02]])
        targets = np.array([[.5, .3, .2], [.17, .11, .72], [1, 0, 0], [0, 0, 1]])
        return returns, inflation, targets

    def test_buy_and_hold_complete_terminal_wealth_is_order_invariant(self):
        market, inflation, targets = self.market()
        original = RUNNER.fixed.simulate_execution(market, inflation, targets, "initial_then_hold")
        for order in RUNNER.make_orders(2, 6, 2006).values():
            r, i = RUNNER.apply_order(market, inflation, order)
            changed = RUNNER.fixed.simulate_execution(r, i, targets, "initial_then_hold")
            np.testing.assert_allclose(changed["nominal_wealth"][..., -1], original["nominal_wealth"][..., -1], atol=1e-11, rtol=1e-14)
            np.testing.assert_allclose(changed["real_wealth"][..., -1], original["real_wealth"][..., -1], atol=1e-11, rtol=1e-14)
            np.testing.assert_array_equal(changed["fees"][..., 1:], 0)

    def test_same_terminal_buy_and_hold_path_can_have_different_mdd(self):
        market = np.array([[[-.2, 0, 0], [.25, 0, 0], [-.2, 0, 0], [.25, 0, 0]]])
        inflation = np.zeros((1, 4)); target = [1, 0, 0]
        original = RUNNER.fixed.simulate_execution(market, inflation, target, "initial_then_hold")
        r, i = RUNNER.apply_order(market, inflation, np.array([[0, 2, 1, 3]]))
        clustered = RUNNER.fixed.simulate_execution(r, i, target, "initial_then_hold")
        original_values = RUNNER.fixed.horizon_values(original, 4)
        clustered_values = RUNNER.fixed.horizon_values(clustered, 4)
        np.testing.assert_allclose(original["real_wealth"][..., -1], clustered["real_wealth"][..., -1], atol=1e-12, rtol=0)
        self.assertGreater(float(clustered_values["mdd"][0, 0] - original_values["mdd"][0, 0]), .15)

    def test_zero_cost_annual_restoration_terminal_is_order_invariant(self):
        market, inflation, targets = self.market()
        original = RUNNER.fixed.simulate_execution(market, inflation, targets, "annual_restore", cost_rate=0)
        for order in RUNNER.make_orders(2, 6, 2026).values():
            r, i = RUNNER.apply_order(market, inflation, order)
            changed = RUNNER.fixed.simulate_execution(r, i, targets, "annual_restore", cost_rate=0)
            np.testing.assert_allclose(changed["real_wealth"][..., -1], original["real_wealth"][..., -1], atol=1e-11, rtol=1e-14)
            np.testing.assert_array_equal(changed["fees"], 0)

    def test_proportional_fee_annual_terminal_invariant_when_last_joint_year_fixed(self):
        market, inflation, targets = self.market()
        original = RUNNER.fixed.simulate_execution(market, inflation, targets, "annual_restore", cost_rate=.0025)
        # With restoration every year the fee after each drift is a factor
        # determined by that year's vector.  The last year has no later fee;
        # fixing it while permuting the interior preserves the full product.
        indices = np.array([[4, 1, 3, 0, 2, 5], [2, 0, 4, 1, 3, 5]])
        r, i = RUNNER.apply_order(market, inflation, indices)
        changed = RUNNER.fixed.simulate_execution(r, i, targets, "annual_restore", cost_rate=.0025)
        np.testing.assert_allclose(changed["real_wealth"][..., -1], original["real_wealth"][..., -1], atol=1e-11, rtol=1e-14)
        self.assertTrue(np.any(original["fees"] > 0))
        # Cumulative nominal fee amounts need not be invariant: their wealth
        # bases depend on the order even when terminal net wealth is equal.
        self.assertFalse(np.allclose(changed["fees"].sum(axis=-1), original["fees"].sum(axis=-1), atol=1e-10, rtol=0))

    def test_c_review_keeps_all_paths_and_has_no_financial_fallback(self):
        market = np.zeros((2, 30, 3)); inflation = np.zeros((2, 30))
        simulation = RUNNER.fixed.simulate_execution(market, inflation, [[.5, .3, .2]], "annual_restore")
        fields = RUNNER.fixed.horizon_values(simulation, 30)
        fit = {"menu": [{"name": "Balanced", "candidate_index": 3, "weights": [.5, .3, .2]}],
               "assignments": {"C": "Review"}}
        rows = RUNNER.client_c_rows(fields, fit, [17, 23], context="toy", scope="full", fold=None,
                                    order="original", policy="annual_restore")
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["scenario_id"] for r in rows}, {17, 23})
        for row in rows:
            self.assertEqual(row["client"], "C")
            self.assertEqual(row["horizon"], 30)
            self.assertEqual(row["loss_tolerance"], .4)
            self.assertFalse(row["assigned"])
            self.assertIsNone(row["candidate_index"])
            for key in (*RUNNER.fixed.VALUE_FIELDS, "breach", "excess"):
                self.assertIsNone(row[key])


if __name__ == "__main__":
    unittest.main()
