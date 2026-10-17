"""Synthetic finite-family runner checks; never load market or old final data."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_simultaneous_runner", ROOT / "scripts/run_simultaneous_risk_diagnostic.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def fixture():
    universe = set(range(1, 121))
    folds = []
    for number in range(5):
        test = list(range(24 * number + 1, 24 * (number + 1) + 1))
        folds.append({"fold": number + 1, "train_ids": sorted(universe - set(test)), "test_ids": test})
    profiles = [{"name": name, "horizon": horizon, "loss_tolerance": .10, "liquidity": liquidity}
                for name, horizon, liquidity in [("A", 5, "high"), ("B", 15, "medium"),
                                                 ("C", 30, "low"), ("D", 25, "medium"), ("E", 6, "high")]]
    protocol = {"clients": profiles, "horizons": [5, 6, 15, 25, 30], "alpha_cases": [.05, .10, .20],
                "liquidity_cases": {"risk_only": {"high": 0., "medium": 0., "low": 0.},
                                    "reference_reserve": {"high": .30, "medium": .10, "low": 0.},
                                    "larger_reserve": {"high": .50, "medium": .20, "low": 0.}},
                "validation": {"folds": folds}}
    evidence_protocol = {"family_events": 1160, "delta": .05}
    weights = np.array([[0., 0., 1.], [.3, .3, .4], [.6, .3, .1], [1., 0., 0.]])
    arrays = {"scenario_ids": np.arange(1, 121), "weights": weights}
    growth = np.tile([.01, .02, .03, .04], (120, 1))
    losses = np.full((120, 4), .05)
    losses[0, 1] = .20
    losses[:8, 2] = .20
    losses[:, 3] = .20
    metrics = {horizon: {"log_growth": growth.copy(), "cagr": np.expm1(growth), "mdd": losses.copy(),
                         "volatility": np.zeros_like(growth), "terminal_real": 100 * np.exp(horizon * growth)}
               for horizon in protocol["horizons"]}
    return protocol, evidence_protocol, arrays, metrics


class EvidenceScreenTests(unittest.TestCase):
    def test_solver_threshold_is_not_a_new_requested_preference(self):
        protocol, _, arrays, metrics = fixture()
        evidence = RUNNER.family_evidence(metrics, arrays["weights"], protocol["clients"], np.arange(120))
        fitted = RUNNER.fit_evidence_case(metrics, arrays["weights"], protocol["clients"], evidence,
                                          requested_alpha=.10, liquidity_floors=protocol["liquidity_cases"]["risk_only"])
        self.assertEqual(fitted["requested_alpha"], .10)
        self.assertEqual(fitted["solver_empirical_alpha"], 0.)
        self.assertNotIn("alpha", fitted)
        self.assertEqual(fitted["supported_max_breach_count"], 0)
        self.assertEqual(fitted["family_bound"]["family_events"], 1160)
        self.assertAlmostEqual(fitted["family_bound"]["per_event_delta"], .05 / 1160)
        chosen = [row for row in fitted["family_pair_details"] if row["chosen"]]
        self.assertEqual(len(chosen), 5)
        self.assertTrue(all(row["candidate_index"] == 0 and row["simultaneous_risk_upper"] <= .1 for row in chosen))
        # One observed breach passes the old .10 point-frequency screen but not this bound.
        one = next(row for row in fitted["family_pair_details"] if row["client"] == "A" and row["candidate_index"] == 1)
        self.assertLess(one["breach_probability"], .10)
        self.assertGreater(one["simultaneous_risk_upper"], .10)
        self.assertEqual(one["evidence_status"], "risk_budget_not_supported")
        json.dumps(fitted, allow_nan=False)

    def test_zero_count_insufficient_means_review_without_calling_solver(self):
        protocol, _, arrays, metrics = fixture()
        evidence = RUNNER.family_evidence(metrics, arrays["weights"], protocol["clients"], np.arange(120))
        with patch.object(RUNNER, "fit_menu", side_effect=AssertionError("Do not call a negative-alpha solver")) as solver:
            fitted = RUNNER.fit_evidence_case(metrics, arrays["weights"], protocol["clients"], evidence,
                                              requested_alpha=.05, liquidity_floors=protocol["liquidity_cases"]["risk_only"])
        solver.assert_not_called()
        self.assertEqual(set(fitted["assignments"].values()), {"Review"})
        self.assertEqual(fitted["menu"], [])
        self.assertIsNone(fitted["selection"])
        self.assertIsNone(fitted["completion"])
        self.assertIsNone(fitted["solver_empirical_alpha"])
        self.assertEqual(len(fitted["family_pair_details"]), 20)
        self.assertTrue(all(row["evidence_status"] == "evidence_insufficient" for row in fitted["family_pair_details"]))
        self.assertFalse(any(row["chosen"] or row["evidence_eligible"] for row in fitted["family_pair_details"]))
        self.assertNotIn("unsafe", json.dumps(fitted))

    def test_nine_cases_reuse_one_family_bound_and_three_mathematical_cutoffs(self):
        protocol, evidence_protocol, arrays, metrics = fixture()
        with patch.object(RUNNER, "supported_count_limit", wraps=RUNNER.supported_count_limit) as limits, \
                patch.object(RUNNER, "family_evidence", wraps=RUNNER.family_evidence) as bounds:
            fits, pairs = RUNNER.fit_context(protocol, evidence_protocol, arrays, metrics, np.arange(120), context="synthetic")
        self.assertEqual(bounds.call_count, 1)
        self.assertEqual(limits.call_count, 3)
        self.assertEqual(len(fits), 9)
        self.assertEqual(len(pairs), 9 * 5 * 4)
        self.assertTrue(all(row["family_events"] == 1160 and row["delta"] == .05 for row in pairs))
        by_pair = {}
        for row in pairs:
            by_pair.setdefault((row["client"], row["candidate_index"]), set()).add(row["simultaneous_risk_upper"])
        self.assertTrue(all(len(values) == 1 for values in by_pair.values()))

    def test_liquidity_can_preclude_supported_risk_without_changing_its_upper(self):
        protocol, _, arrays, metrics = fixture()
        # The only zero-breach candidate now has a bill weight of .10.
        weights = arrays["weights"][[2, 1, 0, 3]]
        evidence = RUNNER.family_evidence(metrics, weights, protocol["clients"], np.arange(120))
        fitted = RUNNER.fit_evidence_case(metrics, weights, protocol["clients"], evidence,
                                          requested_alpha=.10, liquidity_floors=protocol["liquidity_cases"]["reference_reserve"])
        a = next(r for r in fitted["family_pair_details"] if r["client"] == "A" and r["candidate_index"] == 0)
        self.assertTrue(a["risk_evidence_supported"])
        self.assertFalse(a["liquidity_pass"])
        self.assertFalse(a["evidence_eligible"])
        self.assertEqual(fitted["assignments"]["A"], "Review")
        self.assertNotEqual(fitted["assignments"]["C"], "Review")

    def test_train_indices_are_unique_and_heldout_outcomes_cannot_change_fit(self):
        protocol, _, arrays, metrics = fixture()
        indices = np.arange(24, 120)
        def fit(fields):
            evidence = RUNNER.family_evidence(fields, arrays["weights"], protocol["clients"], indices)
            return RUNNER.fit_evidence_case(fields, arrays["weights"], protocol["clients"], evidence,
                                             requested_alpha=.10, liquidity_floors=protocol["liquidity_cases"]["risk_only"])
        before = fit(metrics)
        changed = copy.deepcopy(metrics)
        for fields in changed.values():
            fields["mdd"][:24] = .999
            fields["log_growth"][:24] = 100
        self.assertEqual(before, fit(changed))
        for bad in ([0, 0, 1], [-1], [120], [0.0], []):
            with self.subTest(indices=bad), self.assertRaises(ValueError):
                RUNNER.family_evidence(metrics, arrays["weights"], protocol["clients"], bad)

    def test_wrong_count_cutoff_and_too_small_family_are_rejected(self):
        protocol, _, arrays, metrics = fixture()
        with self.assertRaises(ValueError):
            RUNNER.family_evidence(metrics, arrays["weights"], protocol["clients"], np.arange(120), family_events=19)
        evidence = RUNNER.family_evidence(metrics, arrays["weights"], protocol["clients"], np.arange(120))
        with self.assertRaises(ValueError):
            RUNNER.fit_evidence_case(metrics, arrays["weights"], protocol["clients"], evidence,
                                      requested_alpha=.10, count_limit=1, liquidity_floors=protocol["liquidity_cases"]["risk_only"])

    def test_solver_cannot_silently_assign_a_pair_with_insufficient_evidence(self):
        protocol, _, arrays, metrics = fixture()
        evidence = RUNNER.family_evidence(metrics, arrays["weights"], protocol["clients"], np.arange(120))
        actual_fit = RUNNER.fit_menu
        def corrupt(*args, **kwargs):
            fitted = actual_fit(*args, **kwargs)
            # Mimic an incompatible helper accidentally assigning a one-breach candidate.
            chosen = next(row for row in fitted["client_product_details"] if row["chosen"])
            chosen["candidate_index"] = 1
            return fitted
        with patch.object(RUNNER, "fit_menu", side_effect=corrupt), self.assertRaises(RuntimeError):
            RUNNER.fit_evidence_case(metrics, arrays["weights"], protocol["clients"], evidence,
                                      requested_alpha=.10, liquidity_floors=protocol["liquidity_cases"]["risk_only"])


class ValidationAndOutputTests(unittest.TestCase):
    def test_whole_pipeline_refits_each_fold_without_heldout_indices(self):
        protocol, evidence_protocol, arrays, metrics = fixture()
        # Only the first test block contains breaches for the high-growth first target.
        for fields in metrics.values():
            fields["log_growth"][:, 0] = .05
            fields["mdd"][:, 0] = .05
            fields["mdd"][:24, 0] = .20
            fields["mdd"][:, 1] = .05
            fields["mdd"][:, 2:] = .20
        seen = []
        actual_fit = RUNNER.fit_menu
        def fitted(fields, weights, profiles, indices, **kwargs):
            seen.append(np.asarray(indices).copy())
            return actual_fit(fields, weights, profiles, indices, **kwargs)
        with patch.object(RUNNER, "fit_menu", side_effect=fitted), patch("builtins.print"):
            folds, pairs, records, summaries = RUNNER.run_validation(protocol, evidence_protocol, arrays, metrics)
        # Three .05 cases/fold short-circuit; other six redo the common-menu fit.
        self.assertEqual(len(seen), 5 * 6)
        for fold_number, fold in enumerate(protocol["validation"]["folds"]):
            expected = np.array(fold["train_ids"]) - 1
            for indices in seen[fold_number * 6:(fold_number + 1) * 6]:
                assert_array_equal(indices, expected)
                self.assertFalse(np.intersect1d(indices, np.array(fold["test_ids"]) - 1).size)
        case = RUNNER.primary.case_id(.10, "risk_only")
        chosen = [next(d["candidate_index"] for d in fit["client_product_details"] if d["client"] == "A" and d["chosen"])
                  for fit in folds[case]]
        self.assertEqual(chosen, [0, 1, 1, 1, 1])
        self.assertEqual(len(records), 9 * 5 * 120)
        self.assertEqual(len(pairs), 5 * 9 * 5 * 4)
        a = next(r for r in summaries if r["case"] == case and r["client"] == "A")
        self.assertEqual(a["assigned_test_paths"], 120)
        self.assertEqual(a["breach_count"], 24)
        self.assertTrue(a["internal_calibration_warning"])

    def test_referral_outcomes_stay_null_and_keep_all_denominators(self):
        protocol, evidence_protocol, arrays, metrics = fixture()
        with patch("builtins.print"):
            _, _, records, summaries = RUNNER.run_validation(protocol, evidence_protocol, arrays, metrics)
        strict = RUNNER.primary.case_id(.05, "risk_only")
        paths = [row for row in records if row["case"] == strict]
        self.assertEqual(len(paths), 5 * 120)
        for row in paths:
            self.assertFalse(row["assigned"])
            self.assertEqual(row["assignment"], "Review")
            for key in ["candidate_index", "training_simultaneous_risk_upper", "cagr", "log_growth", "mdd", "breach", "excess"]:
                self.assertIsNone(row[key])
        summary = next(r for r in summaries if r["case"] == strict and r["client"] == "A")
        self.assertEqual((summary["total_test_paths"], summary["assigned_test_paths"], summary["review_test_paths"]), (120, 0, 120))
        self.assertEqual((summary["auto_folds"], summary["review_folds"]), (0, 5))
        self.assertIsNone(summary["breach_count"])
        self.assertIsNone(summary["breach_probability"])
        self.assertIsNone(summary["mean_log_growth"])

    def test_partial_automatic_assignments_use_assigned_risk_denominator(self):
        records = []
        for fold in range(1, 6):
            for scenario in range(24):
                assigned = fold == 1
                records.append({"case": "partial", "client": "A", "requested_alpha": .1,
                                "fold": fold, "assigned": assigned,
                                "breach": scenario < 3 if assigned else None,
                                "log_growth": .02 if assigned else None, "cagr": .02 if assigned else None,
                                "mdd": .2 if assigned else None, "excess": .1 if assigned else None})
        summary = RUNNER.summarize_oof(records)[0]
        self.assertEqual((summary["total_test_paths"], summary["assigned_test_paths"], summary["review_test_paths"]), (120, 24, 96))
        self.assertEqual(summary["breach_count"], 3)
        self.assertEqual(summary["breach_probability"], 3 / 24)
        self.assertTrue(summary["internal_calibration_warning"])

    def test_scope_guard_rejects_final_ids_new_grid_cost_or_events(self):
        protocol = json.loads(RUNNER.primary.PROTOCOL.read_text())
        evidence = json.loads(RUNNER.PROTOCOL.read_text())
        arrays = {"scenario_ids": np.arange(1, 121), "weights": RUNNER.primary.simplex_grid(.05)}
        RUNNER.validate_scope(protocol, evidence, arrays)
        variants = []
        changed_ids = copy.deepcopy(arrays)
        changed_ids["scenario_ids"][-1] = 121
        variants.append((protocol, evidence, changed_ids))
        for key, value in [("cost_rate", .005), ("weight_grid_step", .025)]:
            changed = copy.deepcopy(protocol)
            changed["candidate_family"][key] = value
            variants.append((changed, evidence, arrays))
        changed = copy.deepcopy(protocol)
        changed["clients"][0]["loss_tolerance"] = .11
        variants.append((changed, evidence, arrays))
        for args in variants:
            with self.assertRaises(ValueError):
                RUNNER.validate_scope(*args)

    def test_main_writes_reproducible_scope_metadata_using_mocked_cache_only(self):
        protocol = json.loads(RUNNER.primary.PROTOCOL.read_text())
        weights = RUNNER.primary.simplex_grid(.05)
        arrays = {"scenario_ids": np.arange(1, 121), "weights": weights}
        # Lowest candidate ID strictly dominates, keeping synthetic enumeration small.
        growth = np.tile(np.linspace(.04, .01, len(weights)), (120, 1))
        metrics = {h: {"log_growth": growth.copy(), "cagr": np.expm1(growth), "mdd": np.full_like(growth, .05),
                        "volatility": np.zeros_like(growth), "terminal_real": 100 * np.exp(h * growth)}
                   for h in protocol["horizons"]}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(RUNNER.primary, "get_cache", return_value=(arrays, metrics)) as cache, \
                patch.object(RUNNER, "source_manifest", return_value={"synthetic": "fixed"}), \
                patch.object(RUNNER, "development_input_manifest", return_value={"scope": "constructed arrays only"}), \
                patch.object(sys, "argv", ["run_simultaneous_risk_diagnostic.py", "all", "--out", directory]), \
                patch("builtins.print"):
            RUNNER.main()
            cache.assert_called_once()
            self.assertEqual(cache.call_args.kwargs, {"step": .05, "cost": .0025})
            manifest = json.loads((Path(directory) / "all_manifest.json").read_text())
            self.assertIs(manifest["final_scenarios_loaded"], False)
            self.assertEqual(manifest["sources"], {"synthetic": "fixed"})
            pairs = pd.read_csv(Path(directory) / "all_family_pair_screening.csv")
            self.assertEqual(len(pairs), 6 * 9 * 1160)
            strict = pairs.loc[pairs.requested_alpha == .05]
            self.assertEqual(set(strict.evidence_status), {"evidence_insufficient"})
            self.assertFalse(strict.chosen.any())
            fits = json.loads((Path(directory) / "full_development_menus.json").read_text())
            self.assertEqual(next(iter(fits.values()))["family_pair_screening"]["file"], "all_family_pair_screening.csv")
            for name, digest in manifest["outputs_sha256"].items():
                self.assertEqual(RUNNER.primary.sha(Path(directory) / name), digest)

    def test_source_change_aborts_before_writing_outputs(self):
        arrays = {"scenario_ids": np.arange(1, 121), "weights": RUNNER.primary.simplex_grid(.05)}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(RUNNER.primary, "get_cache", return_value=(arrays, {})), \
                patch.object(RUNNER, "source_manifest", side_effect=[{"code": "before"}, {"code": "after"}]), \
                patch.object(RUNNER, "development_input_manifest", return_value={"scope": "synthetic"}), \
                patch.object(RUNNER, "fit_context", return_value=({}, [])), \
                patch.object(sys, "argv", ["run_simultaneous_risk_diagnostic.py", "fit", "--out", directory]):
            with self.assertRaises(RuntimeError):
                RUNNER.main()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
