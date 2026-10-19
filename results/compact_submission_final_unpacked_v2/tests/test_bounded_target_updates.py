"""Incremental learner selection must preserve the original family and cohort."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bounded_runner", ROOT / "scripts/run_bounded_target_updates.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def summaries(n=96, risk=7, growth=.02):
    return [{**runner.controller_meta(i), "n_training": n, "breach_count": risk, "mean_log_growth": growth} for i in range(18)]


class IncrementalLearningTests(unittest.TestCase):
    def test_simple_mode_wins_equal_growth_without_forcing_updates(self):
        selection = runner.select_expanded(summaries(), 7, 96)
        self.assertEqual(selection["selected_controller_order"], 0)
        self.assertFalse(selection["selected_controller"]["updates_enabled"])

    def test_larger_family_can_improve_training_objective_under_same_cap(self):
        rows = summaries(); rows[10]["mean_log_growth"] = .025
        selection = runner.select_expanded(rows, 7, 96)
        self.assertEqual(selection["selected_controller_order"], 10)
        self.assertGreater(selection["maximum_eligible_mean_log_growth"], selection["fixed_family_maximum_eligible_mean_log_growth"])

    def test_high_growth_ineligible_update_does_not_win(self):
        rows = summaries(); rows[9].update(mean_log_growth=.5, breach_count=8)
        selection = runner.select_expanded(rows, 7, 96)
        self.assertEqual(selection["selected_controller_order"], 0)
        self.assertEqual(selection["training_breach_cap"], 7)

    def test_absolute_alpha_cap_is_integer_and_can_bind(self):
        rows = summaries(risk=10)
        selection = runner.select_expanded(rows, 12, 96)
        self.assertEqual(selection["training_breach_cap"], 9)
        self.assertIsNone(selection["selected_controller_order"])
        self.assertEqual(selection["execution_status"], "execution_review")

    def test_global_growth_band_is_not_a_rolling_chain_of_near_ties(self):
        rows = summaries()
        rows[0]["mean_log_growth"] = .02
        rows[1]["mean_log_growth"] = .02 + .8e-12
        rows[9]["mean_log_growth"] = .02 + 1.6e-12
        selection = runner.select_expanded(rows, 7, 96)
        self.assertEqual(selection["selected_controller_order"], 1)

    def test_missing_old_candidate_is_rejected(self):
        with self.assertRaises(ValueError): runner.select_expanded(summaries()[1:], 7, 96)

    def test_fractional_or_boolean_risk_count_is_rejected(self):
        for bad in (1.5, True, np.bool_(True)):
            rows = summaries(); rows[4]["breach_count"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError): runner.select_expanded(rows, 7, 96)

    def test_price_and_mode_axes_never_remove_cpi_or_target_matrix(self):
        # Deliberately n=p=H+1=3; shape coincidence is not an axis identifier.
        family = {"real_wealth": np.zeros((3, 3, 3)), "cpi": np.ones((3, 3)),
                  "holdings": np.ones((3, 3, 3, 3)), "target_matrix": np.eye(3), "annual_prices": np.arange(3)}
        arm = runner.extract_controller(family, 1)
        self.assertEqual(arm["real_wealth"].shape, (3, 3))
        self.assertEqual(arm["holdings"].shape, (3, 3, 3))
        np.testing.assert_array_equal(arm["cpi"], family["cpi"])
        np.testing.assert_array_equal(arm["target_matrix"], family["target_matrix"])

    def test_parent_crossing_identity_preserves_nulls_and_dates(self):
        runner.verify_crossings([None, 4, 30], [np.nan, 4., 30.])
        with self.assertRaises(AssertionError): runner.verify_crossings([0, 4, 30], [np.nan, 4., 30.])
        with self.assertRaises(AssertionError): runner.verify_crossings([None, 5, 30], [np.nan, 4., 30.])


if __name__ == "__main__": unittest.main()
