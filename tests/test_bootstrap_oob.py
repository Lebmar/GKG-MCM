"""Synthetic-only tests of frozen-assignment bootstrap OOB arithmetic.

No market file, saved bootstrap result, or existing OOB output is read.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
SPEC = importlib.util.spec_from_file_location("synthetic_bootstrap_oob", ROOT / "scripts/analyze_bootstrap_oob.py")
OOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OOB)

METRICS = ("mean_log_growth", "breach_count", "breach_probability", "expected_excess", "cvar90_mdd")
PREFIXES = ("inbag", "oob", "original120_including_inbag")


def fixture():
    samples = np.array([[0, 0, 1, 2, 2, 2], [1, 1, 3, 4, 4, 4]])
    assignments = [
        {"replicate": 0, "A": "Conservative", "A_candidate_index": 0.0,
         "B": "Review", "B_candidate_index": None},
        {"replicate": 1, "A": "Balanced", "A_candidate_index": 1.0,
         "B": "Conservative", "B_candidate_index": 0.0},
    ]
    menus = [{"replicate": replicate, "candidate_indices_by_risk_label": [0, 1, 2],
              "weights_by_risk_label": [[0, 0, 1], [0, 1, 0], [1, 0, 0]]} for replicate in range(2)]
    profiles = [{"name": client, "horizon": 5, "loss_tolerance": .2} for client in ("A", "B")]
    growth = np.column_stack(([.01, .02, .03, -.01, .04, .10], np.full(6, .05), np.full(6, .08)))
    drawdown = np.column_stack(([.1, .15, .3, .4, .5, .05], np.full(6, .05), np.zeros(6)))
    metrics = {5: {"log_growth": growth, "mdd": drawdown}}
    return samples, assignments, menus, profiles, metrics


class SampleReconstructionTests(unittest.TestCase):
    def test_reconstructs_one_continuous_rng_stream_without_resetting_seed(self):
        expected = np.random.default_rng(18473).integers(0, 120, size=(4, 120))
        actual = OOB.reconstruct_samples(18473, 4, 120)
        assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (4, 120))
        self.assertTrue(np.all(actual >= 0) and np.all(actual < 120))
        self.assertTrue(all(len(set(row.tolist())) < 120 for row in actual))
        self.assertFalse(np.array_equal(actual[0], actual[1]))

    def test_multiplicity_and_empty_oob_sample_are_mathematically_distinct(self):
        growth = np.array([.01, .02, .03, -.01, .04, .10])
        losses = np.array([.1, .15, .3, .4, .5, .05])
        sample = OOB.sample_metrics(growth, losses, np.array([0, 0, 1, 2, 2, 2]), .2)
        self.assertAlmostEqual(sample["mean_log_growth"], (.01 * 2 + .02 + .03 * 3) / 6)
        self.assertEqual(sample["breach_count"], 3)
        self.assertAlmostEqual(sample["breach_probability"], .5)
        self.assertAlmostEqual(sample["expected_excess"], .05)
        self.assertAlmostEqual(sample["cvar90_mdd"], .3)
        empty = OOB.sample_metrics(growth, losses, np.array([], dtype=int), .2)
        self.assertTrue(all(empty[field] is None for field in METRICS))

    def test_fractional_tail_atom_and_declared_risk_boundary(self):
        losses = np.arange(12) / 20
        sample = OOB.sample_metrics(np.zeros(12), losses, np.arange(12), .2)
        self.assertAlmostEqual(sample["cvar90_mdd"], (.55 + .2 * .5) / 1.2)
        self.assertEqual(sample["breach_count"], 7)
        boundary = OOB.sample_metrics(np.zeros(3), np.array([.2 + 5e-17, .2 + 5e-13, .2 + 2e-12]), np.arange(3), .2)
        self.assertEqual(boundary["breach_count"], 1)
        self.assertAlmostEqual(boundary["breach_probability"], 1 / 3)
        self.assertAlmostEqual(boundary["expected_excess"], 2e-12 / 3, delta=1e-16)


class FrozenOOBRowsTests(unittest.TestCase):
    def test_partition_preserves_inbag_multiplicity_and_original_data_overlap(self):
        samples, assignments, menus, profiles, metrics = fixture()
        rows, memberships = OOB.build_rows(samples, assignments, menus, profiles, metrics, n_scenarios=6)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(memberships), 2)
        by_rep = {record["replicate"]: record for record in memberships}
        for replicate, sample in enumerate(samples):
            record = by_rep[replicate]
            self.assertEqual(record["inbag_indices_with_multiplicity"], sample.tolist())
            self.assertEqual(record["inbag_unique_indices"], sorted(set(sample.tolist())))
            self.assertEqual(set(record["inbag_unique_indices"]) | set(record["oob_indices"]), set(range(6)))
            self.assertFalse(set(record["inbag_unique_indices"]) & set(record["oob_indices"]))
        a = next(row for row in rows if row["replicate"] == 0 and row["client"] == "A")
        self.assertEqual((a["inbag_n"], a["inbag_unique_n"], a["oob_n"]), (6, 3, 3))
        self.assertEqual(a["inbag_breach_count"], 3)
        self.assertAlmostEqual(a["inbag_breach_probability"], .5)
        self.assertEqual(a["oob_breach_count"], 2)
        self.assertAlmostEqual(a["oob_breach_probability"], 2 / 3)
        self.assertAlmostEqual(a["original120_including_inbag_mean_log_growth"], (.01 + .02 + .03 - .01 + .04 + .10) / 6)

    def test_review_has_no_bill_fallback_and_frozen_target_is_not_reoptimized(self):
        samples, assignments, menus, profiles, metrics = fixture()
        rows, _ = OOB.build_rows(samples, assignments, menus, profiles, metrics, n_scenarios=6)
        reviewed = next(row for row in rows if row["replicate"] == 0 and row["client"] == "B")
        self.assertFalse(reviewed["assigned"])
        self.assertIsNone(reviewed["candidate_index"])
        for prefix in PREFIXES:
            self.assertTrue(all(reviewed[f"{prefix}_{field}"] is None for field in METRICS))
        frozen = next(row for row in rows if row["replicate"] == 0 and row["client"] == "A")
        self.assertTrue(frozen["assigned"])
        self.assertEqual(frozen["candidate_index"], 0)
        # Candidate2 has higher growth and zero loss on every OOB path, but
        # the saved assignment chooses candidate0 and must retain its losses.
        self.assertAlmostEqual(frozen["oob_mean_log_growth"], (-.01 + .04 + .10) / 3)
        self.assertLess(frozen["oob_mean_log_growth"], .08)
        self.assertGreater(frozen["oob_breach_probability"], 0)

    def test_no_oob_paths_produce_null_oob_metrics_not_zero_risk(self):
        _, assignments, menus, profiles, metrics = fixture()
        metrics = {5: {field: value[:4] for field, value in metrics[5].items()}}
        rows, memberships = OOB.build_rows(np.array([[0, 1, 2, 3]]), assignments[:1], menus[:1], profiles, metrics, n_scenarios=4)
        automatic = next(row for row in rows if row["client"] == "A")
        self.assertEqual(automatic["oob_n"], 0)
        self.assertEqual(memberships[0]["oob_indices"], [])
        self.assertTrue(all(automatic[f"oob_{field}"] is None for field in METRICS))
        self.assertIsNotNone(automatic["inbag_mean_log_growth"])
        self.assertIsNotNone(automatic["original120_including_inbag_mean_log_growth"])

    def test_assignment_cannot_reference_a_target_absent_from_frozen_menu(self):
        samples, assignments, menus, profiles, metrics = fixture()
        assignments[0]["A_candidate_index"] = 2.0
        menus[0]["candidate_indices_by_risk_label"] = [0, 1]
        menus[0]["weights_by_risk_label"] = menus[0]["weights_by_risk_label"][:2]
        with self.assertRaises((ValueError, RuntimeError)):
            OOB.build_rows(samples, assignments, menus, profiles, metrics, n_scenarios=6)


if __name__ == "__main__":
    unittest.main()
