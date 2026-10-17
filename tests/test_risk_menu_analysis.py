"""Synthetic checks for predeclared comparator and sensitivity analysis.

Importing the analysis script and calling its array helpers never loads market
data. All candidate metrics below are explicitly constructed test examples.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
SPEC = importlib.util.spec_from_file_location("synthetic_risk_menu_analysis", ROOT / "scripts/analyze_risk_menu_diagnostics.py")
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)

from safeinvest_research.assessment import assess_fixed_menu
from safeinvest_research.selection import fit_menu


def fields(growth, losses, horizon=15, n=120):
    growth = np.broadcast_to(np.asarray(growth, float), (n, len(growth))).copy()
    losses = np.broadcast_to(np.asarray(losses, float), growth.shape).copy()
    return {"log_growth": growth, "cagr": np.expm1(growth), "mdd": losses,
            "volatility": np.zeros_like(growth), "terminal_real": 100 * np.exp(horizon * growth)}


def fixture():
    weights = np.array([[0, 0, 1], [.25, .25, .5], [.5, .25, .25], [.75, .25, 0], [1, 0, 0]])
    growth = [.01, .015, .02, .03, .04]
    risk = np.array([.01, .05, .1, .2, .5])
    metrics = {5: fields(growth, risk / 2, 5), 15: fields(growth, risk), 30: fields(growth, risk * 1.2, 30)}
    profiles = [
        {"name": "A", "horizon": 5, "loss_tolerance": .015, "liquidity": "high"},
        {"name": "B", "horizon": 15, "loss_tolerance": .15, "liquidity": "low"},
        {"name": "C", "horizon": 30, "loss_tolerance": .001, "liquidity": "high"},
    ]
    universe = set(range(1, 121))
    folds = []
    for i in range(5):
        test = list(range(24 * i + 1, 24 * (i + 1) + 1))
        folds.append({"fold": i + 1, "test_ids": test, "train_ids": sorted(universe - set(test))})
    floors = {"high": 0, "medium": 0, "low": 0}
    protocol = {"clients": profiles, "alpha_reference": .1, "alpha_cases": [.1],
                "liquidity_cases": {"risk_only": floors}, "validation": {"outer_folds": 5, "folds": folds},
                "predeclared_sensitivity": {"grid_step": .025, "cost_rates": [0, .0025, .005]}}
    arrays = {"scenario_ids": np.arange(1, 121), "weights": weights}
    return protocol, arrays, metrics


class FixedMenuAssessmentTests(unittest.TestCase):
    def test_assessment_preserves_names_and_ignores_nontraining_rows(self):
        protocol, arrays, metrics = fixture()
        menu = [{"name": "Existing growth name", "candidate_index": 0},
                {"name": "Existing safety name", "candidate_index": 2}]
        before = assess_fixed_menu(metrics, arrays["weights"], menu, protocol["clients"], [0, 1])
        changed = copy.deepcopy(metrics)
        for item in changed.values():
            item["log_growth"][2:] = 100
            item["mdd"][2:] = .99
        after = assess_fixed_menu(changed, arrays["weights"], menu, protocol["clients"], [0, 1])
        self.assertEqual(before, after)
        self.assertEqual([p["name"] for p in before["menu"]], [p["name"] for p in menu])
        self.assertEqual(before["assignments"]["B"], "Existing safety name")
        self.assertEqual(before["assignments"]["C"], "Review")

    def test_assessment_on_selected_menu_matches_selection_assignments(self):
        protocol, arrays, metrics = fixture()
        fitted = fit_menu(metrics, arrays["weights"], protocol["clients"], np.arange(96))
        assessed = assess_fixed_menu(metrics, arrays["weights"], fitted["menu"], protocol["clients"], np.arange(96))
        self.assertEqual(fitted["assignments"], assessed["assignments"])
        chosen_fit = {(d["client"], d["candidate_index"]) for d in fitted["client_product_details"] if d["chosen"]}
        chosen_assessed = {(d["client"], d["candidate_index"]) for d in assessed["client_product_details"] if d["chosen"]}
        self.assertEqual(chosen_fit, chosen_assessed)


class ComparatorAnalysisTests(unittest.TestCase):
    def test_h15_comparator_construction_never_uses_other_scenario_rows(self):
        _, arrays, metrics = fixture()
        before = ANALYSIS.comparator_catalogue(metrics, arrays["weights"], [0, 1])
        changed = copy.deepcopy(metrics)
        changed[15]["log_growth"][2:] = [10, -10, 20, -20, 30]
        changed[15]["mdd"][2:] = [.9, .1, .8, .2, 0]
        after = ANALYSIS.comparator_catalogue(changed, arrays["weights"], [0, 1])
        self.assertEqual(before, after)
        self.assertEqual([p["candidate_index"] for p in before["menu"]], [0, 3, 4])

    def test_fold_frozen_comparator_keeps_test_losses_and_referral_denominators(self):
        protocol, arrays, metrics = fixture()
        metrics[5]["mdd"][:8, 0] = .2
        result = ANALYSIS.comparator_analysis(protocol, arrays, metrics)
        case = ANALYSIS.RUNNER.case_id(.1, "risk_only")
        fold = result["fold_assessments"][case][0]
        chosen = next(row for row in fold["client_product_details"] if row["client"] == "A" and row["chosen"])
        self.assertEqual(chosen["n"], 96)
        self.assertEqual(chosen["breach_count"], 0)
        paths = [row for row in result["oof_paths"] if row["client"] == "A" and row["fold"] == 1]
        self.assertEqual(len(paths), 24)
        self.assertEqual(sum(row["breach"] for row in paths), 8)
        self.assertTrue(all(row["assignment"] == "Conservative" for row in paths))
        summary = {row["client"]: row for row in result["oof_summary"]}
        self.assertEqual(summary["A"]["assigned_paths"], 120)
        self.assertAlmostEqual(summary["A"]["breach_probability"], 8 / 120)
        self.assertEqual((summary["C"]["total_paths"], summary["C"]["review_paths"]), (120, 120))
        self.assertIsNone(summary["C"]["breach_probability"])
        self.assertIsNone(summary["C"]["mean_log_growth"])
        full_objective = result["full_assessments"][case]["objective_diagnostics"]
        self.assertEqual(full_objective["covered_clients"], 2)
        self.assertAlmostEqual(full_objective["maximum_growth_regret"], .01)


class SensitivityAnalysisTests(unittest.TestCase):
    def test_cost_frozen_customers_and_new_screen_are_separate(self):
        profiles = [{"name": "A", "horizon": 15, "loss_tolerance": .1, "liquidity": "high"},
                    {"name": "B", "horizon": 15, "loss_tolerance": .01, "liquidity": "high"}]
        protocol = {"clients": profiles}
        weights = np.array([[0, 0, 1], [.5, 0, .5], [1, 0, 0]])
        base = {"alpha": .1, "liquidity_floors": {"high": 0, "medium": 0, "low": 0},
                "menu": [{"name": "Conservative", "candidate_index": 0, "weights": [0, 0, 1]}],
                "assignments": {"A": "Conservative", "B": "Review"}}
        original = copy.deepcopy(base)
        by_cost = {0: {15: fields([.02] * 3, [.005] * 3)},
                   .0025: {15: fields([.015] * 3, [.05] * 3)},
                   .005: {15: fields([.01] * 3, [.2] * 3)}}
        result = ANALYSIS.cost_analysis(protocol, base, by_cost, {cost: weights for cost in by_cost})
        self.assertEqual(base, original)
        self.assertEqual(result["fixed_menu_rescreened"]["0"]["assignments"]["B"], "Conservative")
        self.assertEqual(result["fixed_menu_rescreened"]["0.005"]["assignments"]["A"], "Review")
        summaries = {(row["cost_rate"], row["client"]): row for row in result["frozen_assignment_summary"]}
        self.assertEqual(summaries[(.005, "A")]["assigned_paths"], 120)
        self.assertEqual(summaries[(.005, "A")]["breach_count"], 120)
        for cost in by_cost:
            self.assertEqual(summaries[(cost, "B")]["assigned_paths"], 0)
            self.assertIsNone(summaries[(cost, "B")]["mean_log_growth"])

    def test_local_resolution_cannot_select_an_attractive_outside_target(self):
        weights = np.array([[0, 0, 1], [.025, 0, .975], [.05, 0, .95], [.25, .25, .5], [1, 0, 0]])
        profiles = [{"name": "A", "horizon": 15, "loss_tolerance": .1, "liquidity": "high"},
                    {"name": "B", "horizon": 15, "loss_tolerance": .01, "liquidity": "high"}]
        protocol = {"clients": profiles, "predeclared_sensitivity": {"grid_step": .025}}
        coarse = {"alpha": .1, "liquidity_floors": {"high": 0, "medium": 0, "low": 0},
                  "menu": [{"name": "Conservative", "candidate_index": 0, "weights": [0, 0, 1]}],
                  "assignments": {"A": "Conservative", "B": "Review"}}
        metrics = {15: fields([.01, .02, .03, .1, .2], [.05, .06, .07, .001, .001])}
        arrays = {"weights": weights, "scenario_ids": np.arange(1, 121)}
        original = copy.deepcopy(coarse)
        result = ANALYSIS.local_resolution_analysis(protocol, coarse, arrays, metrics)
        self.assertEqual(coarse, original)
        self.assertEqual(result["local_to_global_fine_indices"], [0, 1, 2])
        self.assertEqual(result["local_candidate_count"], 3)
        self.assertTrue(result["coarse_reassessment_matches_original_assignments"])
        self.assertTrue(all(item["global_fine_candidate_index"] < 3 for item in result["fine_local_fit"]["menu"]))
        self.assertAlmostEqual(result["coarse_common_universe_objective"]["maximum_growth_regret"], .02)
        self.assertAlmostEqual(result["fine_common_universe_objective"]["maximum_growth_regret"], 0)

    def test_flags_use_automatic_oof_denominator_and_keep_policy_separate(self):
        protocol, arrays, metrics = fixture()
        catalogue = ANALYSIS.comparator_catalogue(metrics, arrays["weights"], np.arange(120))
        full = assess_fixed_menu(metrics, arrays["weights"], catalogue["menu"], protocol["clients"], np.arange(120))
        case = ANALYSIS.RUNNER.case_id(.1, "risk_only")
        folds = [copy.deepcopy(full) for _ in range(5)]
        folds[0]["assignments"]["A"] = "Review"
        counts = [{"case": case, "client": "A", "assigned_test_paths": 96, "breach_count": 4},
                  {"case": case, "client": "B", "assigned_test_paths": 120, "breach_count": 20},
                  {"case": case, "client": "C", "assigned_test_paths": 0, "breach_count": None}]
        result = ANALYSIS.flags_analysis(protocol, metrics, {case: full}, {case: folds}, counts)
        sampling = result["by_case"][case]["sampling"]
        self.assertEqual(sampling["A"]["grade"], "sampling_sensitive")
        self.assertEqual(sampling["A"]["oof_n_automatically_assigned"], 96)
        self.assertEqual(sampling["B"]["grade"], "internal_calibration_warning")
        self.assertEqual(sampling["C"]["grade"], "consistent_referral")
        self.assertIsNone(sampling["C"]["oof_breach_probability"])
        self.assertTrue(all(item["grade"] == "P0" for item in result["policy_dependence"].values()))


if __name__ == "__main__":
    unittest.main()
