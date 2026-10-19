"""Synthetic complete-path calibration tests; no competition returns loaded."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "research"), str(ROOT / "scripts"), str(ROOT / "src")]
from experimental_execution.nested_dynamic_selection import (BETAS, classify_assignment, price_selection,
    final_status, denominator_counts, beta_summary, choose_beta)


def candidates(n=72, counts=None, growth=None):
    return [{"execution_candidate_order": i, "n_training": n,
             "breach_count": (counts or [5]*9)[i], "mean_log_growth": (growth or [.03]*9)[i]} for i in range(9)]


def row(sid, beta=.1, status="automated", assignment="Balanced", bad=False, growth=.03):
    auto = status == "automated"
    return {"scenario_id": sid, "training_beta": beta, "dynamic_inner_fold": 1+(sid % 4),
            "q1_effective_assignment": assignment, "execution_status": status, "assigned": auto,
            "log_growth": growth if auto else None, "mdd": (.5 if bad else .1) if auto else None,
            "breach": bool(bad) if auto else None}


class PriceTests(unittest.TestCase):
    def test_integer_beta_and_relative_reference_caps(self):
        result = price_selection(candidates(), 6, 72, .075)
        self.assertEqual(result["training_breach_cap"], 5)
        self.assertEqual(result["service_alpha"], .1)
        self.assertEqual(result["training_beta"], .075)
        self.assertEqual(price_selection(candidates(), 4, 72, .1)["execution_status"], "execution_review")

    def test_zero_observed_events_is_only_a_training_screen(self):
        selected = price_selection(candidates(counts=[0]*9), 5, 72, 0.)
        self.assertEqual(selected["selected_candidate_order"], 0)
        self.assertEqual(selected["training_breach_cap"], 0)

    def test_one_global_growth_band_does_not_chain_ties(self):
        g = [1-1.5e-12, 1-.75e-12, 1.] + [0.]*6
        self.assertEqual(price_selection(candidates(growth=g), 6, 72, .1)["selected_candidate_order"], 1)

    def test_no_eligible_does_not_use_annual_fallback(self):
        selected = price_selection(candidates(counts=[8]*9), 7, 72, .1)
        self.assertIsNone(selected["selected_candidate_order"])
        self.assertEqual(selected["execution_review_reason"], "no_admissible_training_price")

    def test_malformed_family_or_counts_rejected(self):
        for wrong in [candidates()[:-1], candidates(counts=[True]*9), candidates(n=70), candidates(growth=[np.nan]*9)]:
            with self.assertRaises(ValueError): price_selection(wrong, 6, 72, .1)
        for beta in [True, -.1, .06]:
            with self.assertRaises(ValueError): price_selection(candidates(), 6, 72, beta)


class CaseAndNullTests(unittest.TestCase):
    def test_outside_balanced_case_is_not_risk_review(self):
        self.assertEqual(classify_assignment("Growth"), "not_balanced_case")
        self.assertEqual(classify_assignment("Review"), "risk_review")
        with self.assertRaises(ValueError): classify_assignment("balanced")

    def test_final_reason_priority_does_not_change_static_case(self):
        self.assertEqual(final_status("not_balanced_case", None, None)[0], "not_balanced_case")
        self.assertEqual(final_status("risk_review", .1, 4)[0], "risk_review")
        self.assertEqual(final_status("balanced_case", None, 4)[1], "no_eligible_inner_beta")
        self.assertEqual(final_status("balanced_case", .05, None)[1], "no_admissible_training_price")
        self.assertEqual(final_status("balanced_case", .05, 0), ("automated", None))

    def test_all_original_case_denominators_preserved(self):
        rows = [row(1), row(2,status="execution_review"), row(3,status="risk_review",assignment="Review"),
                row(4,status="not_balanced_case",assignment="Conservative")]
        d=denominator_counts(rows)
        self.assertEqual(d,{"original_n":4,"balanced_case_n":2,"risk_review_n":1,"not_balanced_case_n":1,"case_auto_n":1,"case_execReview_n":1})

    def test_referral_or_other_product_cannot_be_filled_with_zero_loss(self):
        bad=row(1,status="not_balanced_case",assignment="Growth"); bad["mdd"]=0.
        with self.assertRaises(ValueError): denominator_counts([bad])
        with self.assertRaises(ValueError): denominator_counts([row(1,assignment="Growth")])
        with self.assertRaises(ValueError): denominator_counts([row(1),row(1)])


class BetaTests(unittest.TestCase):
    def test_risk_denominator_is_auto_not_original120(self):
        rows=[row(1,bad=True)]+[row(i,status="execution_review") for i in range(2,11)]
        s=beta_summary(rows,.1,list(range(1,11)))
        self.assertEqual(s["conditional_breach_probability"],1.)
        self.assertFalse(s["eligible"])

    def test_zero_coverage_is_null_not_passing_zero_risk(self):
        s=beta_summary([row(1,status="risk_review",assignment="Review")],.1,[1])
        self.assertIsNone(s["breach_count"]); self.assertFalse(s["eligible"])

    def test_all_paths_once_and_matching_whole_h_event(self):
        with self.assertRaises(ValueError): beta_summary([row(1)],.1,[1,2])
        bad=row(1); bad["mdd"]=.5
        with self.assertRaises(ValueError): beta_summary([bad],.1,[1])

    def test_pool_counts_instead_of_mean_fold_rates(self):
        rows=[row(i,bad=(i==1)) for i in range(1,11)]
        rows[0]["dynamic_inner_fold"]=1
        for x in rows[1:]: x["dynamic_inner_fold"]=2
        s=beta_summary(rows,.1,list(range(1,11)))
        self.assertTrue(s["eligible"]); self.assertAlmostEqual(s["conditional_breach_probability"],.1)

    def test_priority_is_risk_eligibility_then_coverage_then_growth_then_smallest_beta(self):
        summaries=[]
        for beta,n,g,bad in zip(BETAS,[10,9,10,10,10],[.9,.5,.03,.03+5e-13,.8],[True,False,False,False,True]):
            rows=[row(i,beta=beta,growth=g,bad=bad) for i in range(1,n+1)]
            rows += [row(i,beta=beta,status="execution_review") for i in range(n+1,11)]
            summaries.append(beta_summary(rows,beta,list(range(1,11))))
        self.assertEqual(choose_beta(summaries)["selected_beta"],.05)

    def test_no_supported_beta_preserves_null_selection(self):
        summaries=[beta_summary([row(1,beta=b,status="execution_review")],b,[1]) for b in BETAS]
        self.assertIsNone(choose_beta(summaries)["selected_beta"])
        with self.assertRaises(ValueError): choose_beta(summaries[:-1])


if __name__ == "__main__": unittest.main()
