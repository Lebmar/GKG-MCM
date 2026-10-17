"""Causal runner and accounting tests on small synthetic empirical forecasts."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_finite_runner", ROOT / "scripts/run_finite_horizon_execution_prototype.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class CausalExecutionTests(unittest.TestCase):
    def solution(self, step=.05):
        gross = np.array([[[1.15, .97, 1.01], [.85, 1.06, 1.01]]])
        return RUNNER.solve_finite_horizon(gross, [[.5, .25, .25], [.8, .1, .1]], horizon=3, step=step)

    def test_exact_duplicate_mapping_keeps_distinct_offmesh_states_unrounded(self):
        solution = self.solution()
        u = np.array([[.499, .501, 0], [.499, .501, 0], [.499 + 1e-13, .501 - 1e-13, 0]])
        actual = RUNNER.evaluate_action
        seen = []
        def spy(model, current, h, k, **kwargs):
            seen.append(current.copy())
            return actual(model, current, h, k, **kwargs)
        with patch.object(RUNNER, "evaluate_action", side_effect=spy):
            grouped = RUNNER.evaluate_unique(solution, u, 2, 0)
        self.assertEqual(seen[0].shape, (2, 3))
        direct = actual(solution, u, 2, 0)
        for key in direct:
            assert_allclose(grouped[key], direct[key], atol=1e-14, rtol=0)
        self.assertTrue(np.array_equal(seen[0][0], u[0]) or np.array_equal(seen[0][1], u[0]))
        self.assertFalse(np.array_equal(seen[0][0], seen[0][1]))

    def test_h1_offmesh_query_cannot_use_concave_grid_q_interpolation(self):
        gross = np.array([[[2., .5, 1.], [.5, 2., 1.]]])
        solution = RUNNER.solve_finite_horizon(gross, [.5, .5, 0], horizon=1)
        u = np.array([.499, .501, 0])
        decision = RUNNER.evaluate_unique(solution, u, 1, 0)
        fee = RUNNER.execute_target(u, [.5, .5, 0], .0025)[1]
        expected = np.log1p(-fee) + np.log(gross.reshape(-1, 3) @ np.array([.5, .5, 0])).mean() - np.log(gross.reshape(-1, 3) @ u).mean()
        self.assertLess(expected, 0)
        self.assertAlmostEqual(float(decision["gain"]), float(expected), places=13)
        self.assertFalse(bool(decision["trade"]))

    def test_initial_gate_is_optional_and_true_h2_can_amortize_one_fee(self):
        gross = np.array([[[1.0015, 1., 1.]]])
        solution = RUNNER.solve_finite_horizon(gross, [1., 0, 0], horizon=2)
        market = np.tile(gross - 1, (2, 2, 1)); inflation = np.zeros((2, 2))
        short, short_trace = RUNNER.simulate_gate(market, inflation, solution, 1, "finite_horizon_gate")
        long, long_trace = RUNNER.simulate_gate(market, inflation, solution, 2, "finite_horizon_gate")
        myopic, myopic_trace = RUNNER.simulate_gate(market, inflation, solution, 2, "myopic_gate")
        self.assertFalse(short_trace["trade"].any())
        self.assertFalse(myopic_trace["trade"].any())
        self.assertTrue(long_trace["trade"][..., 0].all())
        self.assertFalse(long_trace["trade"][..., 1].any())
        assert_allclose(long["nominal_wealth"][..., -1], 100 * 1.0015 ** 2 / 1.0025, atol=1e-12, rtol=0)
        self.assertTrue(np.all(myopic["nominal_wealth"] == 100))
        self.assertTrue(np.all(short["fees"] == 0))
        # A long-policy prefix must not be substituted for a fresh H=1 policy.
        self.assertFalse(np.array_equal(short["nominal_wealth"], long["nominal_wealth"][..., :2]))
        assert_array_equal(long_trace["remaining_horizon"][0, 0], [2, 1])
        assert_array_equal(myopic_trace["decision_h"][0, 0], [1, 1])

    def test_realized_gain_is_after_action_and_includes_one_exact_fee(self):
        solution = RUNNER.solve_finite_horizon(np.array([[[1.1, 1., 1.]]]), [1., 0, 0], horizon=1)
        # A profitable training forecast is followed by an adverse evaluation return.
        market = np.array([[[-.5, 0, 0]]]); inflation = np.array([[.03]])
        sim, trace = RUNNER.simulate_gate(market, inflation, solution, 1, "finite_horizon_gate")
        self.assertTrue(trace["trade"].all())
        self.assertGreater(trace["one_period_gain"][0, 0, 0], 0)
        expected = np.log(.5 / 1.0025)
        self.assertAlmostEqual(trace["restore_realized_paired_gain"][0, 0, 0], expected, places=14)
        self.assertAlmostEqual(trace["selected_action_realized_gain"][0, 0, 0], expected, places=14)
        self.assertAlmostEqual(sim["real_wealth"][0, 0, -1], 100 * .5 / (1.0025 * 1.03), places=12)
        self.assertAlmostEqual(sim["fees"][0, 0, 0], 100 * .0025 / 1.0025, places=13)

    def test_future_evaluation_suffix_does_not_change_earlier_actions(self):
        solution = self.solution()
        market = np.array([[[.02, -.01, .01], [.03, .02, .01], [-.2, .01, 0]],
                           [[-.03, .01, .01], [.01, -.02, .01], [.2, -.04, .01]]])
        inflation = np.full((2, 3), .02)
        changed = market.copy(); changed[:, 1:] = [[.9, -.4, .02], [-.8, .5, .01]]
        changed_inflation = inflation.copy(); changed_inflation[:, 1:] = .5
        for policy in RUNNER.DYNAMIC:
            before, trace_before = RUNNER.simulate_gate(market, inflation, solution, 3, policy)
            after, trace_after = RUNNER.simulate_gate(changed, changed_inflation, solution, 3, policy)
            for key in ["trade", "gain", "one_period_gain", "pretrade_weights", "remaining_horizon", "decision_h"]:
                if key == "pretrade_weights":
                    assert_array_equal(trace_before[key][..., :2, :], trace_after[key][..., :2, :])
                else:
                    assert_array_equal(trace_before[key][..., :2], trace_after[key][..., :2])
            assert_array_equal(before["real_wealth"][..., :2], after["real_wealth"][..., :2])
            assert_array_equal(before["fees"][..., :2], after["fees"][..., :2])

    def test_myopic_execution_is_independent_of_future_value_mesh(self):
        market = np.array([[[.01, .02, 0], [.2, -.1, 0], [-.15, .03, .01]]])
        inflation = np.zeros((1, 3))
        left, tl = RUNNER.simulate_gate(market, inflation, self.solution(.05), 3, "myopic_gate")
        right, tr = RUNNER.simulate_gate(market, inflation, self.solution(.025), 3, "myopic_gate")
        assert_array_equal(tl["trade"], tr["trade"])
        assert_allclose(left["real_wealth"], right["real_wealth"], atol=0, rtol=0)


class RecordAndPairingTests(unittest.TestCase):
    def test_frozen_review_null_and_reference_has_no_customer_assignment(self):
        solution = RUNNER.solve_finite_horizon(np.array([[[1.02, 1.01, 1.]]]), [[.5, .25, .25]], horizon=2)
        market = np.zeros((2, 2, 3)); inflation = np.zeros((2, 2))
        sim, _ = RUNNER.simulate_gate(market, inflation, solution, 2, "finite_horizon_gate")
        fit = {"menu": [{"name": "Conservative", "candidate_index": 7, "weights": [.5, .25, .25]}],
               "assignments": {"A": "Review", "B": "Conservative"}}
        profiles = [{"name": name, "horizon": 2, "loss_tolerance": .10} for name in ["A", "B"]]
        targets, clients = [], []
        RUNNER.add_rows(targets, clients, sim, fit, profiles, [1, 2], context="full", scope="full", fold=None,
                        policy="finite_horizon_gate", mesh=.05, horizon=2)
        data = pd.DataFrame(clients)
        self.assertEqual(len(data), 4)
        self.assertTrue(data.loc[data.client == "A", [*RUNNER.fixed.VALUE_FIELDS, "breach", "initial_trade"]].isna().all().all())
        summary = RUNNER.summarize_with_initial(data, ["client", "policy"], clients=True)
        self.assertTrue(pd.isna(summary.loc[summary.client == "A", "initial_trade_fraction"].iloc[0]))
        self.assertEqual(summary.loc[summary.client == "A", "review_paths"].iloc[0], 2)
        reference_clients = []
        RUNNER.add_rows([], reference_clients, sim, {"menu": fit["menu"]}, profiles, [1, 2], context="full", scope="full",
                        fold=None, policy="equal_weight_annual", mesh=None, horizon=2)
        self.assertEqual(reference_clients, [])

    def test_multiple_alternatives_pair_to_same_actual_annual_paths(self):
        rows = []
        policies = [("annual_restore", None, .01), ("initial_then_hold", None, .02),
                    ("myopic_gate", .05, .03), ("finite_horizon_gate", .05, .04),
                    ("finite_horizon_gate", .025, .05)]
        for policy, mesh, value in policies:
            for scenario in [1, 2]:
                rows.append({"scenario_id": scenario, "policy": policy, "mesh_step": mesh,
                             **{field: value for field in RUNNER.fixed.VALUE_FIELDS}})
        paired = RUNNER.comparisons_to_annual(pd.DataFrame(rows), ["scenario_id"])
        self.assertEqual(len(paired), 8)
        for policy, mesh, value in policies[1:]:
            mask = paired.comparison_policy == policy
            mask &= paired.mesh_step.isna() if mesh is None else paired.mesh_step == mesh
            assert_allclose(paired.loc[mask, "delta_log_growth"], value - .01, atol=1e-14, rtol=0)

    def test_action_trace_remaining_horizon_and_current_state_are_passed_without_market(self):
        solution = RUNNER.solve_finite_horizon(np.array([[[1.02, 1.01, 1.]]]), [1., 0, 0], horizon=2)
        market = np.zeros((1, 2, 3)); inflation = np.zeros((1, 2))
        actual = RUNNER.evaluate_unique; seen = []
        def audit(model, u, remaining, target_ids, **kwargs):
            self.assertIs(model, solution)
            self.assertEqual(set(kwargs), {"chunk_size"})
            self.assertEqual(u.shape[-1], 3)
            seen.append(remaining)
            return actual(model, u, remaining, target_ids, **kwargs)
        with patch.object(RUNNER, "evaluate_unique", side_effect=audit):
            RUNNER.simulate_gate(market, inflation, solution, 2, "finite_horizon_gate")
        self.assertEqual(seen, [2, 1])


if __name__ == "__main__":
    unittest.main()
