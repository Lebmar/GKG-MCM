"""Synthetic nested screen selection, admission and leakage checks."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_nested_static", ROOT / "scripts/run_nested_static_calibration.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def fixture():
    protocol = json.loads(RUNNER.PROTOCOL.read_text())
    profiles = protocol["clients"]
    weights = np.array([[0, 0, 1.], [.3, .3, .4], [.6, .3, .1], [1., 0, 0]])
    growth = np.tile([.01, .02, .03, .04], (120, 1))
    losses = np.tile([.02, .10, .20, .50], (120, 1))
    metrics = {h: {"log_growth": growth.copy(), "cagr": np.expm1(growth), "mdd": losses.copy(),
                    "volatility": np.zeros_like(growth), "terminal_real": 100 * np.exp(h * growth)}
               for h in [5, 6, 15, 25, 30]}
    return protocol, profiles, weights, metrics


def conditional_rows(all_review=False):
    rows = []
    for fold in range(1, 5):
        for profile in ["A", "B", "C"]:
            for scene in range(24):
                auto = not all_review and (profile == "B" or profile == "A" and fold == 1)
                rows.append({"client": profile, "inner_fold": fold, "assigned": auto,
                             "breach": (scene < 3 if profile == "A" else False) if auto else None,
                             "log_growth": .02 if auto else None})
    return rows


class SplitAndCausalityTests(unittest.TestCase):
    def test_saved_inner_ids_are_complements_and_each_validation_path_is_unique(self):
        protocol, _, _, _ = fixture()
        for context in protocol["contexts"]:
            RUNNER.validate_context(context)
            tests = [i for fold in context["inner_folds"] for i in fold["test_ids"]]
            self.assertEqual(sorted(tests), sorted(context["training_ids"]))
        invalid = []
        for change in ["duplicate", "overlap", "float", "wrong_seed"]:
            context = copy.deepcopy(protocol["contexts"][1])
            if change == "duplicate": context["inner_folds"][0]["test_ids"][0] = context["inner_folds"][0]["test_ids"][1]
            if change == "overlap": context["inner_folds"][0]["train_ids"][0] = context["inner_folds"][0]["test_ids"][0]
            if change == "float": context["inner_folds"][0]["train_ids"][0] = float(context["inner_folds"][0]["train_ids"][0])
            if change == "wrong_seed": context["inner_seed"] += 1
            invalid.append(context)
        for context in invalid:
            with self.assertRaises(ValueError): RUNNER.validate_context(context)

    def test_outer_evaluation_perturbation_cannot_change_gamma_menu_or_admission(self):
        protocol, profiles, weights, metrics = fixture()
        context = protocol["contexts"][1]
        before = RUNNER.calibrate_context(metrics, weights, profiles, context)
        changed = copy.deepcopy(metrics)
        heldout = np.array(context["outer_test_ids"]) - 1
        for fields in changed.values():
            for value in fields.values(): value[heldout] = np.nan
        after = RUNNER.calibrate_context(changed, weights, profiles, context)
        self.assertEqual(before, after)
        self.assertEqual(before[0]["selection"]["selected_gamma"], 0.)

    def test_entire_menu_is_reselected_inside_every_inner_train(self):
        protocol, profiles, weights, metrics = fixture()
        context = protocol["contexts"][1]
        actual = RUNNER.fit_menu; seen = []
        def spy(fields, candidates, clients, indices, **kwargs):
            seen.append((indices.copy(), kwargs["alpha"]))
            return actual(fields, candidates, clients, indices, **kwargs)
        with patch.object(RUNNER, "fit_menu", side_effect=spy):
            RUNNER.calibrate_context(metrics, weights, profiles, context)
        self.assertEqual(len(seen), 21)
        for number, gamma in enumerate(RUNNER.GAMMAS):
            for fold, (indices, used_gamma) in zip(context["inner_folds"], seen[number * 4:number * 4 + 4], strict=True):
                self.assertEqual(used_gamma, gamma)
                self.assertEqual(indices.tolist(), (np.asarray(fold["train_ids"]) - 1).tolist())
                self.assertFalse(np.intersect1d(indices, np.asarray(fold["test_ids"]) - 1).size)
        self.assertEqual(seen[-1][0].tolist(), (np.asarray(context["training_ids"]) - 1).tolist())


class GammaAndAdmissionTests(unittest.TestCase):
    def test_service_alpha_and_solver_gamma_are_distinct_with_no_legacy_alpha(self):
        protocol, profiles, weights, metrics = fixture()
        fitted = RUNNER.fit_gamma(metrics, weights, profiles, np.arange(90), .075)
        self.assertEqual(fitted["service_alpha"], .10)
        self.assertEqual(fitted["solver_gamma"], .075)
        self.assertEqual(fitted["training_screen_gamma"], .075)
        self.assertNotIn("alpha", fitted)
        self.assertTrue(all("training_screen_feasible" in d and "sample_feasible" not in d for d in fitted["client_product_details"]))

    def test_conditional_risk_uses_auto_n_and_zero_auto_is_null(self):
        summary = RUNNER.inner_summary(conditional_rows(), .025)
        a, b, c = [summary["profiles"][name] for name in ["A", "B", "C"]]
        self.assertEqual((a["auto_n"], a["auto_folds"], a["breach_count"]), (24, 1, 3))
        self.assertEqual(a["conditional_breach_probability"], 3 / 24)
        self.assertFalse(summary["eligible"])
        self.assertLess(3 / a["total_paths"], .10)
        self.assertEqual((b["auto_n"], b["auto_folds"]), (96, 4))
        self.assertEqual(c["auto_n"], 0)
        self.assertIsNone(c["breach_count"])
        self.assertIsNone(c["conditional_breach_probability"])
        self.assertIsNone(c["empirical_service_risk_pass"])

    def test_all_zero_coverage_is_ineligible_and_none_selected(self):
        summary = RUNNER.inner_summary(conditional_rows(all_review=True), 0.)
        self.assertFalse(summary["eligible"])
        self.assertIsNone(summary["raw_auto_unit_mean_log_growth"])
        selected = RUNNER.select_gamma([summary])
        self.assertIsNone(selected["selected_gamma"])
        self.assertEqual(selected["status"], "calibration_unsupported")

    def test_selection_priorities_and_single_global_growth_tolerance(self):
        rows = [{"training_screen_gamma": gamma, "raw_auto_path_profile_units": coverage,
                 "raw_auto_unit_mean_log_growth": growth, "eligible": eligible}
                for gamma, coverage, growth, eligible in [(0., 120, .03, True), (.025, 120, .03 + .5e-12, True),
                                                           (.05, 96, .1, True), (.075, 120, .04, False)]]
        self.assertEqual(RUNNER.select_gamma(rows)["selected_gamma"], 0.)
        rows[1]["raw_auto_unit_mean_log_growth"] = .03 + 2e-12
        self.assertEqual(RUNNER.select_gamma(rows)["selected_gamma"], .025)

    def test_admission_preserves_raw_menu_but_does_not_upgrade_zero_evidence_profile(self):
        profiles = [{"name": name} for name in ["A", "B", "C"]]
        raw = {"menu": [{"name": "Conservative"}, {"name": "Balanced"}],
               "assignments": {"A": "Conservative", "B": "Balanced", "C": "Review"}}
        before = copy.deepcopy(raw)
        evidence = {"profiles": {name: {"auto_n": n, "auto_folds": int(n > 0), "breach_count": 0 if n else None,
                                        "conditional_breach_probability": 0. if n else None}
                                  for name, n in [("A", 0), ("B", 1), ("C", 0)]}}
        admitted = RUNNER.apply_admission(raw, profiles, evidence)
        self.assertEqual(raw, before)
        self.assertEqual(admitted["raw_assignments"], raw["assignments"])
        self.assertEqual(admitted["effective_assignments"], {"A": "Review", "B": "Balanced", "C": "Review"})
        self.assertEqual(admitted["admission_details"][0]["reason"], "no_inner_auto_evidence")
        self.assertEqual((admitted["raw_covered_profiles"], admitted["effective_covered_profiles"]), (2, 1))

    def test_unsupported_context_never_calls_default_gamma_refit(self):
        protocol, profiles, weights, metrics = fixture()
        fake = {"menu": [], "assignments": {p["name"]: "Review" for p in profiles}}
        with patch.object(RUNNER, "fit_gamma", side_effect=lambda *a, **k: copy.deepcopy(fake)) as fitter:
            result = RUNNER.calibrate_context(metrics, weights, profiles, protocol["contexts"][1])[0]
        self.assertEqual(fitter.call_count, 20)
        self.assertIsNone(result["selection"]["selected_gamma"])
        self.assertIsNone(result["raw_fit"])
        self.assertEqual(set(result["effective_assignments"].values()), {"Review"})
        self.assertTrue(all(d["reason"] == "calibration_unsupported" for d in result["admission_details"]))


class ReferralAndComparisonTests(unittest.TestCase):
    def test_review_paths_and_conditional_summary_are_null_not_zero_risk(self):
        _, profiles, _, metrics = fixture()
        assignments = {profile["name"]: "Review" for profile in profiles}
        rows = RUNNER.path_records(metrics, profiles, None, [1, 2, 3], context="unsupported", gamma=None,
                                    view="effective", assignments=assignments)
        data = pd.DataFrame(rows)
        self.assertTrue(data[["candidate_index", "log_growth", "cagr", "mdd", "breach", "excess"]].isna().all().all())
        summary = RUNNER.performance_summary(rows, ["client"])
        self.assertTrue(all(s["auto_n"] == 0 and s["review_n"] == 3 and s["breach_count"] is None and s["mean_log_growth"] is None for s in summary))

    def test_common_auto_comparison_retains_coverage_and_conditional_denominator(self):
        new, old = [], []
        for scene in [1, 2, 3]:
            assigned = scene == 1
            new.append({"outer_fold": 1, "scenario_id": scene, "client": "A", "view": "effective", "assigned": assigned,
                        "breach": True if assigned else None, "log_growth": .02 if assigned else None})
            old.append({"fold": 1, "scenario_id": scene, "client": "A", "assigned": True, "breach": False, "log_growth": .03})
        compared = RUNNER.compare_common_auto(new, pd.DataFrame(old))[0]
        self.assertEqual((compared["nested_auto_n"], compared["primary_auto_n"], compared["common_auto_n"]), (1, 3, 1))
        self.assertEqual((compared["nested_common_breaches"], compared["primary_common_breaches"]), (1, 0))
        self.assertAlmostEqual(compared["paired_common_mean_log_growth_delta"], -.01)


if __name__ == "__main__":
    unittest.main()
