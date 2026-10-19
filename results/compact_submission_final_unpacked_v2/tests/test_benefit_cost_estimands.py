"""Closed synthetic benefit/cost identities and necessary counterexamples."""
import math
import unittest

import numpy as np

from diagnostics.benefit_cost_estimands import (
    continuation_decomposition, hold_to_horizon_parts, one_step_parts,
    two_period_annual_witness, zero_preference_local_gate,
)


class BenefitCostEstimandTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., 0., 100.]); self.w = np.array([.5, .25, .25])

    def test_single_fee_decomposition_and_common_inflation_cancellation(self):
        a = one_step_parts(self.x, self.w, [1.2, 1.03, 1.01])
        b = one_step_parts(self.x, self.w, [1.2, 1.03, 1.01], inflation_gross=1.25)
        self.assertAlmostEqual(a["fee"], 100 * .0025 * .75 / (1 + .0025 * .75), places=13)
        self.assertAlmostEqual(a["net_log_advantage"], a["direct_net_log_advantage"], places=14)
        self.assertAlmostEqual(a["net_log_advantage"], b["direct_net_log_advantage"], places=14)
        self.assertNotAlmostEqual(a["restore_real"], b["restore_real"])

    def test_direct_net_ratio_already_contains_fee_and_double_count_can_reverse_sign(self):
        parts = one_step_parts(self.x, [1., 0., 0.], [1.004, 1., 1.])
        self.assertGreater(parts["direct_net_log_advantage"], 0.)
        self.assertLess(parts["direct_net_log_advantage"] - parts["current_log_cost"], 0.)

    def test_no_further_trade_horizon_identity_uses_quantities_that_drift(self):
        gross = np.array([[1.2, .97, 1.01], [.8, 1.1, 1.02], [1.3, 1.04, 1.01]])
        result = hold_to_horizon_parts(self.x, self.w, gross, inflation_gross_path=[1.02, 1.1, .99])
        self.assertAlmostEqual(result["net_log_advantage"], result["direct_net_log_advantage"], places=14)
        # Constant-quantity cumulative factors are not products of annual
        # constant-weight portfolio factors, even without future fees.
        self.assertGreater(abs(float(self.w @ gross.prod(axis=0)) - float(np.prod(gross @ self.w))), .001)

    def test_common_annual_continuation_without_different_action_response_is_homogeneous(self):
        gross = np.array([[.8, 1.1, 1.], [1.2, 1.03, 1.01]])
        hold = two_period_annual_witness(self.x, self.w, gross, initial_action="hold")
        paid = two_period_annual_witness(self.x, self.w, gross, initial_action="restore")
        free = two_period_annual_witness(self.x, self.w, gross, initial_action="restore", charge_current_fee=False)
        rho = one_step_parts(self.x, self.w, gross[0])["fee_fraction"]
        d = continuation_decomposition(hold, paid, free, rho)
        self.assertEqual(paid["actions"], ["restore", "restore"])
        self.assertEqual(free["actions"], ["restore", "restore"])
        self.assertAlmostEqual(paid["fees"][1], (1 - rho) * free["fees"][1], places=13)
        self.assertAlmostEqual(d["continuation_response_residual"], 0., places=14)
        self.assertAlmostEqual(d["direct_net_log_advantage"], d["current_fee_free_action_benefit"] - d["current_log_cost"], places=14)

    def test_same_historical_peak_pending_response_breaks_naive_fee_offset(self):
        gross = np.array([[.2, 1., 1.], [2., 1., 1.]])
        hold = two_period_annual_witness(self.x, self.w, gross, initial_action="hold")
        paid = two_period_annual_witness(self.x, self.w, gross, initial_action="restore")
        free = two_period_annual_witness(self.x, self.w, gross, initial_action="restore", charge_current_fee=False)
        rho = one_step_parts(self.x, self.w, gross[0])["fee_fraction"]
        d = continuation_decomposition(hold, paid, free, rho)
        self.assertEqual(paid["actions"], ["restore", "hold"])
        self.assertEqual(free["actions"], ["restore", "restore"])
        self.assertTrue(paid["ever_breached"][1]); self.assertFalse(free["ever_breached"][1])
        self.assertLess(d["continuation_response_residual"], -.2)
        self.assertAlmostEqual(d["direct_net_log_advantage"], d["decomposed_net_log_advantage"], places=14)
        self.assertNotAlmostEqual(d["direct_net_log_advantage"], d["current_fee_free_action_benefit"] - d["current_log_cost"], places=5)

    def test_positive_expected_log_gain_is_not_full_horizon_risk_feasibility(self):
        high = one_step_parts(self.x, self.w, [1.5, 1., 1.])
        low = one_step_parts(self.x, self.w, [.1, 1., 1.])
        expected = .85 * high["net_log_advantage"] + .15 * low["net_log_advantage"]
        self.assertGreater(expected, 0.)
        self.assertGreater(1 - low["restore_real"] / 100, .4)
        self.assertLess(1 - low["hold_real"] / 100, .4)
        self.assertGreater(.15, .1)  # Explicit probability, not an estimated market count.

    def test_one_period_gate_can_reject_profitable_fee_amortization_forever(self):
        gross = np.array([1.003, 1., 1.])
        short = zero_preference_local_gate(gross[None, :], self.x, self.w)
        long = hold_to_horizon_parts(self.x, self.w, np.tile(gross, (2, 1)))
        self.assertEqual(short["action"], "hold")
        self.assertLess(short["estimated_net_log_advantage"], 0.)
        self.assertGreater(long["net_log_advantage"], 0.)
        self.assertEqual(zero_preference_local_gate(gross[None, :], self.x, self.w)["action"], "hold")

    def test_expected_log_is_not_log_of_expected_wealth(self):
        gross = np.array([[3., 1., 1.], [.2, .2, 1.]])
        parts = [one_step_parts(self.x, self.w, g) for g in gross]
        self.assertGreater(np.mean([p["restore_real"] for p in parts]), 100.)
        self.assertLess(np.mean([p["net_log_advantage"] for p in parts]), 0.)
        self.assertEqual(zero_preference_local_gate(gross, self.x, self.w)["action"], "hold")

    def test_zero_preference_gate_holds_ties_and_does_not_recharge_cost(self):
        result = zero_preference_local_gate(np.ones((3, 3)), [20., 30., 50.], [.2, .3, .5])
        self.assertEqual(result["action"], "hold")
        self.assertIsNone(result["risk_preference_parameter"])
        favorable = zero_preference_local_gate(np.array([[1.004, 1., 1.]]), self.x, [1., 0., 0.])
        self.assertEqual(favorable["action"], "restore")
        self.assertAlmostEqual(favorable["estimated_net_log_advantage"], one_step_parts(self.x, [1., 0., 0.], [1.004, 1., 1.])["direct_net_log_advantage"], places=14)
        self.assertGreater(favorable["estimated_net_log_advantage"] / 30, 0.)


if __name__ == "__main__": unittest.main()
