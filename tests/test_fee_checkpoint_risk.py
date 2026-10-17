"""Synthetic observation-definition sensitivities, without market data."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_fee_checkpoint", ROOT / "scripts/check_fee_checkpoint_risk.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class FeeObservationTests(unittest.TestCase):
    def test_initial_fee_is_observed_even_when_net_annual_return_is_positive(self):
        wealth = np.array([[[100., 109.]]]); fee = np.array([[[.25]]]); inflation = np.zeros((1, 1))
        before = wealth.copy()
        result = RUNNER.fee_observation_mdd(wealth, fee, inflation)
        self.assertEqual(result["annual_mdd"][0, 0], 0.)
        self.assertAlmostEqual(result["feeobs_mdd"][0, 0], .0025, places=14)
        assert_array_equal(wealth, before)
        assert_allclose(result["real_wealth"], wealth, atol=0, rtol=0)

    def test_known_second_fee_can_flip_a_loss_tolerance_event(self):
        result = RUNNER.fee_observation_mdd(np.array([[[100., 90.01, 110.]]]), np.array([[[0., .02]]]), np.zeros((1, 2)))
        self.assertAlmostEqual(result["annual_mdd"][0, 0], .0999, places=14)
        self.assertAlmostEqual(result["feeobs_mdd"][0, 0], .1001, places=14)
        summary = RUNNER.summarize_pair(result["annual_mdd"][0], result["feeobs_mdd"][0], .10)
        self.assertEqual((summary["annual_breach_count"], summary["feeobs_breach_count"], summary["newly_flagged_paths"]), (0, 1, 1))

    def test_transaction_price_is_previous_year_price_not_current_year_end(self):
        wealth = np.array([[[100., 121., 150.]]]); fees = np.array([[[0., 11.]]])
        inflation = np.array([[.10, .10]])
        result = RUNNER.fee_observation_mdd(wealth, fees, inflation)
        self.assertAlmostEqual(result["post_fee_real"][0, 0, 1], (121 - 11) / 1.1, places=12)
        self.assertAlmostEqual(result["feeobs_mdd"][0, 0], 1 - 100 / 110, places=14)
        wrong = 1 - (110 - 11 / 1.21) / 110
        self.assertNotAlmostEqual(result["feeobs_mdd"][0, 0], wrong, places=5)
        altered = RUNNER.fee_observation_mdd(wealth, fees, np.array([[.10, .50]]))
        # Year-2 inflation occurs after the second fee observation.
        assert_array_equal(result["post_fee_real"], altered["post_fee_real"])

    def test_zero_fee_reproduces_annual_metric_without_changing_peaks(self):
        wealth = np.array([[[100., 103., 95., 107.], [100., 90., 89., 100.]]])
        result = RUNNER.fee_observation_mdd(wealth, np.zeros((1, 2, 3)), np.array([[.02, -.01, .03]]))
        assert_array_equal(result["annual_mdd"], result["feeobs_mdd"])
        self.assertTrue(np.all(result["post_fee_real"] <= result["annual_peak"][..., :-1]))

    def test_monotonic_risk_severity_and_fee_increment_bound(self):
        wealth = np.array([[[100., 110., 96., 120.], [100., 98., 99., 101.], [100., 80., 105., 99.]]])
        fees = np.array([[[.2, .1, .15], [.1, .2, .1], [.25, .2, .1]]])
        result = RUNNER.fee_observation_mdd(wealth, fees, np.array([[.01, .03, .02]]))
        self.assertTrue(np.all(result["feeobs_mdd"] >= result["annual_mdd"]))
        self.assertTrue(np.all(result["mdd_increment"] <= result["fee_to_prior_peak"].max(axis=-1) + 1e-14))
        summary = RUNNER.summarize_pair(result["annual_mdd"].ravel(), result["feeobs_mdd"].ravel(), .10)
        self.assertGreaterEqual(summary["feeobs_expected_excess"], summary["annual_expected_excess"])
        self.assertGreaterEqual(summary["feeobs_cvar90"], summary["annual_cvar90"])

    def test_negative_unaffordable_or_mismatched_fees_rejected(self):
        wealth = np.array([[[100., 102.]]]); inflation = np.zeros((1, 1))
        for fees in (np.array([[[-1.]]]), np.array([[[100.]]]), np.zeros((1, 1, 2))):
            with self.assertRaises(ValueError):
                RUNNER.fee_observation_mdd(wealth, fees, inflation)


if __name__ == "__main__":
    unittest.main()
