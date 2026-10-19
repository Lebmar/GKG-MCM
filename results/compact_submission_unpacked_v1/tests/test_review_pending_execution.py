"""Runner-level financial reporting, crossing/censoring and provenance tests.

The execution kernel has its own independent causal/accounting test suite. These
tests load no market data and never refit a product or customer assignment.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "research"), str(ROOT / "src")]
import run_review_pending_execution as runner


class ReviewPendingRunnerTests(unittest.TestCase):
    def test_recovery_freezes_trigger_peak_and_reports_observed_time(self):
        value = runner.recovery_from_path([100, 80, 90, 100, 95], .1)
        self.assertEqual(value["first_crossing"], 1)
        self.assertEqual(value["peak_at_trigger"], 100)
        self.assertEqual(value["recovery_endpoint"], 3)
        self.assertEqual(value["recovery_time"], 2)
        self.assertEqual(value["available_followup"], 3)
        self.assertEqual(value["observed_followup"], 2)
        self.assertFalse(value["right_censored"])

    def test_equality_is_not_a_breach_and_later_nonrecovery_is_censored(self):
        value = runner.recovery_from_path([100, 90, 80, 85], .1)
        self.assertEqual(value["first_crossing"], 2)
        self.assertTrue(value["right_censored"])
        self.assertIsNone(value["recovery_time"])
        self.assertEqual(value["available_followup"], 1)
        self.assertEqual(value["observed_followup"], 1)

    def test_crossing_at_horizon_retains_breach_with_zero_followup(self):
        value = runner.recovery_from_path([100, 101, 80], .1)
        self.assertEqual(value["first_crossing"], 2)
        self.assertEqual(value["available_followup"], 0)
        self.assertEqual(value["observed_followup"], 0)
        self.assertTrue(value["right_censored"])

    def test_no_crossing_recovery_fields_are_null(self):
        value = runner.recovery_from_path([100, 90, 110], .1)
        self.assertFalse(value["triggered"])
        self.assertTrue(all(item is None for key, item in value.items() if key != "triggered"))

    def test_recovery_does_not_reset_historical_severity(self):
        value = runner.recovery_from_path([100, 80, 100, 70], .1)
        self.assertEqual(value["first_crossing"], 1)
        self.assertEqual(value["recovery_endpoint"], 2)
        self.assertAlmostEqual(value["added_severity"], .1)

    def test_review_remains_null_in_each_arm_and_pair(self):
        rows = runner.review_rows(context="fold1", scope="oof", fold=1, client="A", scenario_ids=[3, 8], horizon=5, loss_tolerance=.1)
        pairs = runner.pair_records(rows, deepcopy(rows))
        self.assertEqual(len(pairs), 2)
        for row in rows:
            self.assertFalse(row["assigned"])
            for field in ["log_growth", "mdd", "breach", "first_crossing", "recovered", "right_censored", "actual_trade_count"]:
                self.assertIsNone(row[field])
        self.assertIsNone(pairs[0]["delta_mdd"])
        self.assertIsNone(pairs[0]["delta_breach"])

    def test_pairing_requires_unchanged_first_crossing_and_breach(self):
        row = runner.review_rows(context="x", scope="full", fold=None, client="B", scenario_ids=[1], horizon=15, loss_tolerance=.25)[0]
        changed = deepcopy(row); changed["first_crossing"] = 2
        with self.assertRaises(RuntimeError): runner.pair_records([row], [changed])

    def test_baseline_looks_up_global_cache_through_explicit_local_mapping(self):
        row = {"context": "x", "scenario_id": 2, "client": "C", "assigned": True, "assignment": "Balanced",
               "candidate_local_index": 0, "candidate_original232_index": 2, "horizon": 30,
               **{key: j + .1 for j, key in enumerate(runner.METRIC_FIELDS)},
               "nominal_fees": .4, "nominal_moved": 160., "actual_trade_fraction": 1.}
        old = {**row, "candidate_index": 0}
        cache = {}
        for key in runner.METRIC_FIELDS:
            value = np.full((3, 4), -99.); value[1, 2] = row[key]; cache["H30__" + key] = value
        for key, field in [("fees", "nominal_fees"), ("moved", "nominal_moved"), ("trade_fraction", "actual_trade_fraction")]:
            value = np.full((3, 4), -99.); value[1, 2] = row[field]; cache["H30__" + key] = value
        self.assertEqual(runner.verify_baseline([row], pd.DataFrame([old]), cache), 0.)
        wrong = {**row, "candidate_original232_index": 0}
        with self.assertRaises(RuntimeError): runner.verify_baseline([wrong], pd.DataFrame([old]), cache)

    def test_unfrozen_kernel_is_rejected_before_market_replay(self):
        protocol = {"frozen_dependencies_sha256": {}, "execution_kernel": {"ready": False, "path": "missing", "sha256": None}}
        with self.assertRaises(RuntimeError): runner.frozen_dependencies(protocol)

    def test_null_review_groups_remain_null_in_summaries(self):
        rows = runner.review_rows(context="fold1", scope="oof", fold=1, client="A", scenario_ids=[3, 8], horizon=5, loss_tolerance=.1)
        financial = runner.summaries([{**row, "arm": "annual"} for row in rows], ["scope", "arm", "client"])
        paired = runner.paired_summaries(runner.pair_records(rows, deepcopy(rows)), ["scope", "client"])
        self.assertEqual(len(financial), 2)
        for value in financial:
            self.assertEqual(value["review_n"], 2)
            self.assertEqual(value["summary_n"], 0)
            self.assertIsNone(value["breach_probability"])
            self.assertIsNone(value["cvar90_mdd"])
        self.assertTrue(all(row["sum_delta_breach"] is None for row in paired))

    def test_triggered_record_recomputes_censoring_and_preserves_fee_units(self):
        recovery = runner.recovery_from_path([100, 80, 90], .1)
        arm = {"real_wealth": np.array([[[100., 80., 90.]]]), "fees": np.array([[[.1, 0.]]]),
               "moved": np.array([[[40., 0.]]]), "fee_fraction": np.array([[[.001, 0.]]]),
               "moved_fraction": np.array([[[.4, 0.]]]), "trades": np.array([[[True, False]]]),
               "end_year_tv_drift": np.array([[[.1, .2]]]), "end_weights": np.array([[[[.2, .3, .5], [.3, .3, .4]]]])}
        for key in ["first_crossing", "recovery_endpoint", "recovery_time", "available_followup", "observed_followup"]:
            arm[key] = np.array([[recovery[key] if recovery[key] is not None else -1]])
        for key in ["triggered", "recovered", "right_censored"]: arm[key] = np.array([[bool(recovery[key])]])
        for key in ["drawdown_at_trigger", "peak_at_trigger", "real_wealth_at_trigger", "added_severity"]:
            arm[key] = np.array([[recovery[key] if recovery[key] is not None else np.nan]])
        options = {"context": "x", "scope": "oof", "fold": 1, "client": "C", "assignment": "Balanced", "candidate_local": 1,
                   "candidate_global": 7, "scenario_ids": [3], "horizon": 2, "loss_tolerance": .1}
        value = runner.records_for_arm(arm, **options)[0]
        self.assertEqual(value["nominal_fees"], .1)
        self.assertEqual(value["actual_trade_fraction"], .5)
        self.assertTrue(value["right_censored"])
        self.assertIsNone(value["recovery_time"])
        self.assertEqual(value["observed_followup"], 1)
        arm["first_crossing"] = np.array([[2]])
        with self.assertRaises(RuntimeError): runner.records_for_arm(arm, **options)


if __name__ == "__main__": unittest.main()
