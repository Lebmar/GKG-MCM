"""Synthetic runner checks: folds, referral denominators, reselection and cache.

All output is isolated in temporary directories. Market loading is always
mocked; these tests never read the market CSV or archived V1 results.
"""

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
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
SPEC = importlib.util.spec_from_file_location("synthetic_research_runner", ROOT / "scripts/run_risk_menu_research.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def protocol_fixture():
    universe = set(range(1, 121))
    folds = []
    for fold in range(5):
        test = list(range(24 * fold + 1, 24 * (fold + 1) + 1))
        folds.append({"fold": fold + 1, "train_ids": sorted(universe - set(test)), "test_ids": test})
    return {
        "allowed_scenarios": list(range(1, 121)),
        "horizons": [1],
        "clients": [
            {"name": "A", "horizon": 1, "loss_tolerance": .1, "liquidity": "high"},
            {"name": "B", "horizon": 1, "loss_tolerance": .1, "liquidity": "high"},
        ],
        "alpha_cases": [.1],
        "liquidity_cases": {"risk_only": {"high": 0, "medium": 0, "low": 0}},
        "validation": {"outer_folds": 5, "folds": folds},
        "stability": {"bootstrap_replicates": 3, "bootstrap_seed": 9182},
    }


def synthetic_arrays():
    arrays = {"scenario_ids": np.arange(1, 121), "weights": np.eye(3)}
    growth = np.tile([.01, .02, .03], (120, 1))
    losses = np.full((120, 3), .05)
    losses[:8, 0] = .2
    metrics = {1: {"log_growth": growth, "cagr": np.expm1(growth), "mdd": losses,
                   "volatility": np.zeros((120, 3)), "terminal_real": 100 * np.exp(growth)}}
    return arrays, metrics


class FoldAndOOFTests(unittest.TestCase):
    def test_dependent_stage_requires_matching_fit_sources_and_menu_hash(self):
        fit = {"synthetic_case": {"menu": []}}
        with tempfile.TemporaryDirectory() as directory, patch.object(RUNNER, "OUT", Path(directory)), \
                patch.object(RUNNER, "source_manifest", return_value={"synthetic": "fixed"}):
            menu_path = Path(directory) / "full_development_menus.json"
            RUNNER.write_json(menu_path, fit)
            manifest_path = Path(directory) / "fit_manifest.json"
            manifest = {"sources": {"synthetic": "fixed"}, "menus_sha256": RUNNER.sha(menu_path)}
            RUNNER.write_json(manifest_path, manifest)
            self.assertEqual(RUNNER.compatible_full_fit(), fit)
            RUNNER.write_json(menu_path, {"altered": True})
            with self.assertRaises(RuntimeError):
                RUNNER.compatible_full_fit()
            RUNNER.write_json(menu_path, fit)
            manifest["sources"] = {"synthetic": "outdated"}
            RUNNER.write_json(manifest_path, manifest)
            with self.assertRaises(RuntimeError):
                RUNNER.compatible_full_fit()

    def test_fold_fit_uses_only_training_and_referrals_keep_coverage_denominator(self):
        protocol = protocol_fixture()
        arrays, metrics = synthetic_arrays()
        seen = []

        def fit(fields, weights, profiles, indices, **kwargs):
            fold = len(seen)
            expected_train = np.asarray(protocol["validation"]["folds"][fold]["train_ids"]) - 1
            test = np.asarray(protocol["validation"]["folds"][fold]["test_ids"]) - 1
            assert_array_equal(indices, expected_train)
            self.assertEqual(len(np.intersect1d(indices, test)), 0)
            seen.append(np.array(indices))
            return {"menu": [{"name": "Conservative", "candidate_index": 0, "weights": [1, 0, 0]}],
                    "assignments": {"A": "Conservative" if fold == 0 else "Review", "B": "Review"}}

        with tempfile.TemporaryDirectory() as directory, patch.object(RUNNER, "OUT", Path(directory)), \
                patch.object(RUNNER, "fit_menu", side_effect=fit), \
                patch.object(RUNNER, "source_manifest", return_value={"synthetic": "fixed"}):
            _, summaries = RUNNER.run_validation(protocol, arrays, metrics)
            paths = pd.read_csv(Path(directory) / "validation_oof_paths.csv")
        self.assertEqual(len(seen), 5)
        by_client = {row["client"]: row for row in summaries}
        a, b = by_client["A"], by_client["B"]
        self.assertEqual((a["total_test_paths"], a["assigned_test_paths"], a["review_test_paths"]), (120, 24, 96))
        self.assertEqual((a["auto_folds"], a["review_folds"], a["breach_count"]), (1, 4, 8))
        self.assertAlmostEqual(a["breach_probability"], 8 / 24)
        self.assertTrue(a["internal_calibration_warning"])
        self.assertEqual((b["total_test_paths"], b["assigned_test_paths"], b["review_test_paths"]), (120, 0, 120))
        self.assertIsNone(b["breach_probability"])
        self.assertIsNone(b["mean_log_growth"])
        referred = paths.loc[~paths.assigned]
        self.assertTrue(referred[["cagr", "log_growth", "mdd", "breach", "excess"]].isna().all().all())

    def test_invalid_fold_universes_duplicates_and_test_reuse_are_rejected(self):
        variants = []
        zero = protocol_fixture()
        zero["validation"]["folds"][0]["test_ids"][0] = 0
        variants.append(zero)
        duplicate = protocol_fixture()
        duplicate["validation"]["folds"][0]["test_ids"][0] = 2
        variants.append(duplicate)
        reused = protocol_fixture()
        reused["validation"]["folds"][1] = copy.deepcopy(reused["validation"]["folds"][0])
        reused["validation"]["folds"][1]["fold"] = 2
        variants.append(reused)
        for protocol in variants:
            with self.subTest(protocol=protocol["validation"]["folds"][:2]), self.assertRaises((RuntimeError, ValueError)):
                RUNNER.validate_folds(protocol)

    def test_source_change_during_stage_invalidates_result(self):
        protocol = protocol_fixture()
        arrays, metrics = synthetic_arrays()
        calls = []

        def sources():
            calls.append(1)
            return {"synthetic": "before" if len(calls) == 1 else "after"}

        fit = {"menu": [], "assignments": {"A": "Review", "B": "Review"}}
        with tempfile.TemporaryDirectory() as directory, patch.object(RUNNER, "OUT", Path(directory)), \
                patch.object(RUNNER, "fit_menu", side_effect=lambda *args, **kwargs: copy.deepcopy(fit)), \
                patch.object(RUNNER, "source_manifest", side_effect=sources), self.assertRaises(RuntimeError):
            RUNNER.run_validation(protocol, arrays, metrics)


class BootstrapRunnerTests(unittest.TestCase):
    def test_bootstrap_reselects_whole_paths_and_aligns_only_equal_cardinalities(self):
        protocol = protocol_fixture()
        arrays, metrics = synthetic_arrays()
        case = RUNNER.case_id(.1, "risk_only")
        names = ["Conservative", "Balanced", "Growth"]
        targets = np.eye(3)
        base = {"alpha": .1, "liquidity_floors": protocol["liquidity_cases"]["risk_only"],
                "menu": [{"name": names[i], "weights": targets[i].tolist(), "candidate_index": i} for i in range(3)]}
        samples = np.random.default_rng(protocol["stability"]["bootstrap_seed"]).integers(0, 120, size=(3, 120))
        permutations = ([2, 0, 1], [0, 2], [1, 2, 0])
        observed = []

        def fit(fields, weights, profiles, indices, **kwargs):
            replicate = len(observed)
            assert_array_equal(indices, samples[replicate])
            self.assertLess(len(set(indices.tolist())), 120)
            observed.append(indices.copy())
            order = permutations[replicate]
            menu = [{"name": names[i], "weights": targets[candidate].tolist(), "candidate_index": candidate}
                    for i, candidate in enumerate(order)]
            assigned = {"A": menu[0]["name"], "B": "Review" if replicate == 1 else menu[1]["name"]}
            details = [{"client": "A", "candidate_index": order[0], "chosen": True}]
            if replicate != 1:
                details.append({"client": "B", "candidate_index": order[1], "chosen": True})
            return {"selection": {"objective": [1 if replicate == 1 else 2, 0, 0], "identified_indices": order[:2]},
                    "completion": {"complete": len(menu) == 3, "three_distinct_reference_risks": len(menu) == 3},
                    "menu": menu, "assignments": assigned, "client_product_details": details}

        with tempfile.TemporaryDirectory() as directory, patch.object(RUNNER, "OUT", Path(directory)), \
                patch.object(RUNNER, "fit_menu", side_effect=fit), \
                patch.object(RUNNER, "source_manifest", return_value={"synthetic": "fixed"}):
            result = RUNNER.run_bootstrap(protocol, arrays, metrics, {case: base}, replicates=3)
            assignments = pd.read_csv(Path(directory) / "bootstrap" / case / "assignments.csv")
        self.assertEqual(len(observed), 3)
        self.assertEqual(result["aligned_replicates"], 2)
        self.assertEqual(result["automatic_frequency"], {"A": 1.0, "B": 2 / 3})
        assert_allclose(result["aligned_weight_quantiles_025_50_975"], np.broadcast_to(targets, (3, 3, 3)))
        self.assertTrue(np.isnan(assignments.loc[1, "B_candidate_index"]))
        self.assertEqual(assignments.loc[0, "A_candidate_index"], 2)


class CacheRunnerTests(unittest.TestCase):
    def synthetic_market(self):
        return np.zeros((120, 1, 3)), np.zeros((120, 1)), np.arange(1, 121)

    def cache_patches(self, directory, market):
        return (patch.object(RUNNER, "OUT", Path(directory)),
                patch.object(RUNNER, "load_paths", return_value=market),
                patch.object(RUNNER, "simplex_grid", return_value=np.eye(3)))

    def test_cache_reads_only_development_and_rejects_changed_development_arrays(self):
        market = self.synthetic_market()
        protocol = protocol_fixture()
        with tempfile.TemporaryDirectory() as directory:
            out, loader, grid = self.cache_patches(directory, market)
            with out, loader as mocked_loader, grid:
                RUNNER.get_cache(protocol)
                args, kwargs = mocked_loader.call_args
                self.assertEqual(list(args[1]), list(range(1, 121)))
                self.assertIs(kwargs["allow_final"], False)
                changed = self.synthetic_market()
                changed[0][0, 0, 0] = .01
                mocked_loader.return_value = changed
                with self.assertRaises(RuntimeError):
                    RUNNER.get_cache(protocol)

    def test_cache_parameter_rounding_collision_cannot_return_wrong_fee_results(self):
        market = self.synthetic_market()
        with tempfile.TemporaryDirectory() as directory:
            out, loader, grid = self.cache_patches(directory, market)
            with out, loader, grid:
                RUNNER.get_cache(protocol_fixture(), cost=.0025)
                self.assertEqual(RUNNER.cache_key(.05, .0025), RUNNER.cache_key(.05, .0025001))
                with self.assertRaises(RuntimeError):
                    RUNNER.get_cache(protocol_fixture(), cost=.0025001)

    def test_cache_corrupted_bytes_are_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            out, loader, grid = self.cache_patches(directory, self.synthetic_market())
            with out, loader, grid:
                RUNNER.get_cache(protocol_fixture())
                file = next((Path(directory) / "cache").glob("*.npz"))
                file.write_bytes(file.read_bytes() + b"corrupted")
                with self.assertRaises(RuntimeError):
                    RUNNER.get_cache(protocol_fixture())

    def test_cache_cannot_expose_non_development_scenario_ids(self):
        market = self.synthetic_market()
        market[2][-1] = 121
        with tempfile.TemporaryDirectory() as directory:
            out, loader, grid = self.cache_patches(directory, market)
            with out, loader, grid, self.assertRaises(RuntimeError):
                RUNNER.get_cache(protocol_fixture())


if __name__ == "__main__":
    unittest.main()
