"""Synthetic comparator and diagnostic examples; no market-data access."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from safeinvest_research.diagnostics import (
    empirical_screen_flags, label_order_diagnostics, policy_flags,
    reference_menu, sampling_flags,
)


def fit(assignment="Conservative", count=3, n=120, weights=None):
    automatic = assignment != "Review"
    return {"alpha": .1, "assignments": {"A": assignment},
            "client_product_details": ([{"client": "A", "chosen": True,
                                         "breach_count": count, "n": n}] if automatic else []),
            "menu": ([{"name": assignment, "weights": weights or [.1, .1, .8]}]
                     if automatic else [])}


class ReferenceMenuTests(unittest.TestCase):
    def test_frontier_endpoints_and_midpoint_tie_prefer_growth(self):
        result = reference_menu([.01, .02, .03, .04, .009], [.1, .2, .3, .4, .15])
        self.assertEqual(result["indices"], [0, 2, 3])
        self.assertEqual(result["reference_frontier"], [0, 1, 2, 3])
        self.assertFalse(result["degenerate"])
        self.assertAlmostEqual(result["risk_midpoint"], .25)

    def test_duplicate_endpoints_do_not_create_a_third_risk_level(self):
        result = reference_menu([.01, .01, .03, .03], [.1, .1, .3, .3])
        self.assertEqual(result["indices"], [0, 2])
        self.assertTrue(result["degenerate"])
        self.assertFalse(result["complete"])

    def test_dominating_endpoint_can_leave_only_one_economic_product(self):
        result = reference_menu([.01, .02, .02], [.3, .1, .1])
        self.assertEqual(result["indices"], [1])
        self.assertTrue(result["degenerate"])

    def test_nonfinite_or_mismatched_frontier_inputs_rejected(self):
        for growth, risk in (([], []), ([0, 1], [0]), ([np.nan], [.1])):
            with self.subTest(growth=growth), self.assertRaises(ValueError):
                reference_menu(growth, risk)


class OrderTests(unittest.TestCase):
    def metrics(self):
        return {h: {"mdd": np.tile(loss, (4, 1)),
                    "log_growth": np.tile(growth, (4, 1))}
                for h, loss, growth in [
                    (5, [.3, .2, .1], [.03, .02, .01]),
                    (15, [.1, .2, .3], [.01, .02, .03]),
                    (30, [.2, .2, .4], [.02, .02, .04])]}

    def test_reference_sort_cross_horizon_reversals_and_ties(self):
        result = label_order_diagnostics(self.metrics(), [2, 0, 1], [0, 1])
        self.assertEqual(result["reference_label_candidate_indices"], [0, 1, 2])
        self.assertTrue(result["any_risk_label_reversal"])
        self.assertTrue(result["any_cross_horizon_risk_reversal"])
        self.assertEqual(len(result["by_horizon"]["5"]["cvar90_mdd"]["against_reference_label_order"]), 3)
        self.assertEqual(result["by_horizon"]["30"]["cvar90_mdd"]["tied_pairs"], [[0, 1]])

    def test_heldout_rows_cannot_change_training_order_diagnostics(self):
        metrics = self.metrics()
        before = label_order_diagnostics(metrics, [0, 1, 2], [0, 1])
        for fields in metrics.values():
            fields["mdd"][2:] = np.nan
            fields["log_growth"][2:] = 500
        self.assertEqual(before, label_order_diagnostics(metrics, [0, 1, 2], [0, 1]))

    def test_p90_and_cvar_can_disagree_at_the_reference_horizon(self):
        losses = np.column_stack((np.full(12, .2), np.r_[np.zeros(11), 1.]))
        metrics = {15: {"mdd": losses, "log_growth": np.zeros_like(losses)}}
        result = label_order_diagnostics(metrics, [0, 1], np.arange(12))
        self.assertEqual(result["reference_label_candidate_indices"], [0, 1])
        self.assertEqual(result["by_horizon"]["15"]["p90_mdd"]["against_reference_label_order"], [[0, 1]])
        self.assertFalse(result["any_cross_horizon_risk_reversal"])


class FlagTests(unittest.TestCase):
    def test_noninteger_count_budget_and_one_path_boundary(self):
        passed = empirical_screen_flags(9, 96, .1)
        failed = empirical_screen_flags(10, 96, .1)
        self.assertTrue(passed["empirical_pass"])
        self.assertTrue(passed["one_path_boundary_pass"])
        self.assertFalse(failed["empirical_pass"])
        self.assertTrue(failed["one_path_boundary_fail"])
        self.assertFalse(empirical_screen_flags(96, 96, 1.)["one_path_boundary_pass"])

    def test_development_consistency_is_explicitly_not_certification(self):
        result = sampling_flags(fit(), [fit(n=96)]*5, {"A": {"breach_count": 6, "n": 120}})["A"]
        self.assertEqual(result["grade"], "development_consistent_auto")
        self.assertFalse(result["population_certified"])

    def test_internal_calibration_warning_uses_assigned_oof_denominator(self):
        result = sampling_flags(fit(), [fit(n=96)]*5, {"A": {"breach_count": 13, "n": 120}})["A"]
        self.assertEqual(result["grade"], "internal_calibration_warning")
        self.assertAlmostEqual(result["oof_breach_probability"], 13/120)

    def test_referrals_are_not_imputed_as_zero_risk_investments(self):
        result = sampling_flags(fit("Review"), [fit("Review")]*5,
                                {"A": {"breach_count": 0, "n": 0}})["A"]
        self.assertEqual(result["grade"], "consistent_referral")
        self.assertIsNone(result["oof_breach_probability"])

    def test_fold_coverage_disagreement_and_boundary_are_separate_warnings(self):
        folds = [fit(n=96)]*4 + [fit("Review")]
        mixed = sampling_flags(fit(), folds, {"A": {"breach_count": 5, "n": 96}})["A"]
        self.assertEqual(mixed["grade"], "sampling_sensitive")
        self.assertEqual(mixed["oof_n_automatically_assigned"], 96)
        boundary = sampling_flags(fit(count=12), [fit(n=96)]*5,
                                   {"A": {"breach_count": 6, "n": 120}})["A"]
        self.assertEqual(boundary["grade"], "sampling_sensitive")
        self.assertTrue(boundary["full_screen"]["one_path_boundary_pass"])

    def test_partial_fold_results_cannot_receive_a_consistent_grade(self):
        result = sampling_flags(fit(), [fit(n=96)]*4,
                                {"A": {"breach_count": 5, "n": 96}})["A"]
        self.assertEqual(result["grade"], "incomplete_internal_diagnostics")

    def test_policy_changes_are_not_called_sampling_failure(self):
        cases = {"main": {"A": "Conservative"}, "relaxed": {"A": "Balanced"},
                 "strict": {"A": "Review"}}
        result = policy_flags(cases, "main")["A"]
        self.assertEqual(result["grade"], "P2")
        self.assertFalse(result["target_comparison_available"])

    def test_same_role_can_hide_a_changed_target(self):
        result = policy_flags({"main": fit(weights=[.1, .1, .8]),
                               "reserve": fit(weights=[.05, .05, .9])}, "main")["A"]
        self.assertEqual(result["grade"], "P1")
        self.assertTrue(result["target_policy_sensitive"])
        self.assertFalse(result["role_policy_sensitive"])

    def test_bad_count_inputs_are_rejected(self):
        for k, n, alpha in ((1, 0, .1), (2, 1, .1), (True, 5, .1), (0, 5, np.nan)):
            with self.subTest(k=k, n=n), self.assertRaises(ValueError):
                empirical_screen_flags(k, n, alpha)


if __name__ == "__main__":
    unittest.main()
