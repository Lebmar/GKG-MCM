"""Synthetic admission, known-cost and denominator checks; no market reads."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cost_runner", ROOT / "scripts/run_pathwise_cost_stress.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def toy_decisions():
    gates, paths = [], []
    for label, rate in runner.RATES:
        for client, assigned in (("B", True), ("A", False)):
            common = {"cost_label": label, "cost_rate": rate, "scope": "outer_oof", "context": "toy",
                      "client": client, "scenario_id": 1, "policy": "pathwise_hold_rollout", "horizon": 3, "assigned": assigned}
            paths.append(common)
            if not assigned: continue
            acts = ["restore", "hold", "hold"] if label == "zero" else ["restore", "restore", "hold"]
            for t, action in enumerate(acts):
                gates.append({**common, "year": t + 1, "action": action,
                    "actual_trade": action == "restore", "pending_at_decision": t == 2 and label != "zero"})
    return pd.DataFrame(gates), pd.DataFrame(paths)


class CostStressTests(unittest.TestCase):
    def test_prospective_declared_rates_scope_and_ready_guard(self):
        protocol = json.loads(runner.PROTOCOL.read_text())
        runner.validate_protocol(protocol, require_ready=False)
        not_ready = deepcopy(protocol); not_ready["execution_ready"] = False
        with self.assertRaises(RuntimeError): runner.validate_protocol(not_ready)
        for mutate in (lambda p: p["rates"].reverse(),
                       lambda p: p["rates"][0].update(cost_rate=False),
                       lambda p: p["rates"][1].update(cost_rate=.002),
                       lambda p: p["data"].update(allowed_scenario_ids=[1, 180])):
            broken = deepcopy(protocol); mutate(broken)
            with self.assertRaises(ValueError): runner.validate_protocol(broken, require_ready=False)

    def test_changed_frozen_input_or_execution_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ["fit.json", "scripts/run_pathwise_cost_stress.py", "tests/test_pathwise_cost_stress.py", "data.csv"]
            for name in names:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(name)
            protocol = {"frozen_input_sha256": {"fit.json": runner.sha(root / "fit.json")},
                        "execution_source_sha256": {name: runner.sha(root / name) for name in names[1:3]}}
            with patch.object(runner, "ROOT", root), patch.object(runner.frozen, "DATA", root / "data.csv"):
                result = runner.verify_inputs(protocol)
                self.assertIn("data.csv", result)
                (root / "fit.json").write_text("changed")
                with self.assertRaises(RuntimeError): runner.verify_inputs(protocol)
                (root / "fit.json").write_text("fit.json")
                (root / names[1]).write_text("changed code")
                with self.assertRaises(RuntimeError): runner.verify_inputs(protocol)

    def test_dataframe_reproduction_checks_keys_nulls_flags_and_finance(self):
        expected = pd.DataFrame([{"id": 1, "assigned": True, "wealth": 103., "breach": False},
                                 {"id": 2, "assigned": False, "wealth": None, "breach": None}])
        actual = expected.copy()
        receipt = runner.compare_frames(actual, expected, ["id"])
        self.assertTrue(receipt["identities_and_null_masks_match"])
        self.assertEqual(receipt["rows"], 2)
        actual.loc[1, "wealth"] = 0
        with self.assertRaises(RuntimeError): runner.compare_frames(actual, expected, ["id"])
        actual = expected.copy(); actual.loc[0, "wealth"] = 102
        with self.assertRaises(AssertionError): runner.compare_frames(actual, expected, ["id"])
        actual = expected.copy(); actual.loc[0, "breach"] = True
        with self.assertRaises(AssertionError): runner.compare_frames(actual, expected, ["id"])
        with self.assertRaises(RuntimeError): runner.compare_frames(expected.iloc[[0, 0]], expected, ["id"])

    def test_trace_reproduction_includes_strings_booleans_and_nan(self):
        arrays = {"wealth": np.array([[100., 99.95]]), "q": np.array([[np.nan, 0.]]),
                  "actions": np.array([["restore", "hold"]]), "ever": np.array([[False, True]])}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.npz"; np.savez(path, **arrays)
            receipt = runner.compare_trace(arrays, path)
            self.assertEqual(receipt["array_fields"], 4)
            bad = arrays | {"ever": np.array([[False, False]])}
            with self.assertRaises(AssertionError): runner.compare_trace(bad, path)
            with self.assertRaises(RuntimeError): runner.compare_trace({"wealth": arrays["wealth"]}, path)

    def test_known_cost_enters_actual_initial_fee_and_forecast_restore_fee(self):
        actual = np.array([[[.5, 0, 0], [0, 0, 0]]])
        forecast = np.zeros((2, 2, 3)); inflation = np.zeros((1, 2)); forecast_pi = np.zeros((2, 2))
        for label, rate in runner.RATES:
            out = runner.simulate_policy(actual, inflation, forecast, forecast_pi, [.5, 0, .5], L=.5, cost_rate=rate)
            self.assertAlmostEqual(out["fees"][0, 0], 50 * rate / (1 + .5 * rate), places=13)
            expected_qr = np.log1p(-.1 * rate / (1 + .5 * rate))
            self.assertAlmostEqual(out["qg_restore"][0, 1], expected_qr, places=14)
            self.assertAlmostEqual(out["qg_hold"][0, 1], 0, places=14)
            self.assertEqual(out["actions"][0, 1], "hold")
            np.testing.assert_allclose(out["fees"], rate * out["moved"], atol=2e-15, rtol=0)

    def test_every_declared_rate_has_its_own_supported_comparator(self):
        market = np.array([[[.5, 0, 0], [.02, .02, .02], [.03, .03, .03]],
                           [[0, .4, 0], [.02, .02, .02], [.03, .03, .03]]])
        pi, w = np.zeros((2, 3)), [.4, .3, .3]
        item = {"name": "Balanced", "candidate_local_index": 3, "candidate_original232_index": 8}
        for label, rate in runner.RATES:
            reference = runner.base.squeeze_pending(runner.simulate_paired(market, pi, w, .4, cost_rate=rate)["pending_hold"])
            proposed = runner.simulate_policy(market, pi, market, pi, w, L=.4, cost_rate=rate)
            rows, summary = runner.frozen.verify_training_property(reference, proposed, context="toy", client="B",
                ids=[1, 2], horizon=3, tolerance=.4, item=item)
            self.assertEqual(summary["breach_order_violations"], 0)
            self.assertEqual(len(rows), 2)
            self.assertGreaterEqual(summary["minimum_total_log_growth_delta"], -1e-14)

    def test_whole_trajectory_decision_change_denominators_and_review_null(self):
        gates, paths = toy_decisions()
        changes, summary = runner.decision_changes(gates, paths)
        zero = changes.loc[(changes.cost_label == "zero") & changes.assigned].iloc[0]
        self.assertEqual(zero.all_decisions, 3)
        self.assertAlmostEqual(zero.action_changed_fraction, 1 / 3)
        self.assertAlmostEqual(zero.trade_flag_changed_fraction, 1 / 3)
        self.assertAlmostEqual(zero.pending_flag_changed_fraction, 1 / 3)
        self.assertEqual(zero.both_active_postinitial_decisions, 1)
        self.assertEqual(zero.both_active_action_changed_count, 1)
        report = summary.loc[(summary.cost_label == "zero") & (summary.client == "B")].iloc[0]
        self.assertEqual(report.both_active_action_changed_fraction, 1)
        self.assertTrue(changes.loc[~changes.assigned, "action_changed_fraction"].isna().all())
        self.assertTrue(summary.loc[summary.client == "A", "action_changed_fraction"].isna().all())
        self.assertTrue((summary.loc[summary.cost_label == "base", "action_changed_fraction"].dropna() == 0).all())

    def test_missing_decision_and_review_imputation_are_rejected(self):
        gates, paths = toy_decisions()
        with self.assertRaises(ValueError): runner.decision_changes(gates.iloc[1:], paths)
        duplicate = pd.concat([gates, gates.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError): runner.decision_changes(duplicate, paths)
        mismatched = paths.copy(); mismatched.loc[mismatched.client == "B", "horizon"] = 4
        with self.assertRaises(ValueError): runner.decision_changes(gates, mismatched)

    def test_review_training_fields_are_null_not_safe_outcomes(self):
        common = {"context": "toy", "client": "A", "horizon": 5, "loss_tolerance": .1, "cost_label": "base", "cost_rate": .0025}
        rows = runner.review_training(common, [1, 2])
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertFalse(row["assigned"])
            for field in ("reference_total_log_growth", "pathwise_total_log_growth", "total_log_growth_delta", "reference_breach", "pathwise_breach"):
                self.assertIsNone(row[field])


if __name__ == "__main__": unittest.main()
