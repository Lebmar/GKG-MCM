"""Synthetic fixed-selection cost sensitivity and parent-audit admission."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("selected_cost", ROOT / "scripts/run_selected_controller_cost_stress.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def toy_inputs(review=False):
    context = "full_development"; ids = [1, 2]
    fits = {context: {"training_ids": ids, "outer_test_ids": [], "context_id": 0,
        "effective_assignments": {"C": "Balanced"}, "raw_fit": {"menu": [{"name": r} for r in runner.bounded.ROLES]}}}
    price = {context: {"profiles": {"C": {"selected_candidate_order": 8}}}}
    update = {context: {"selected_controller_order": None if review else 10, "parent_fixed_selected_candidate_order": 8}}
    old, new = [], []
    for sid in ids:
        common = {"context": context, "client": "C", "scenario_id": sid, "horizon": 30, "assigned": True,
                  "terminal_real": 100 + sid, "breach": False, "risk_assignment": "Balanced"}
        old.append({**common, "policy": runner.priced.LEARNER, "execution_candidate_order": 8})
        old.append({**common, "policy": runner.POLICIES[2], "execution_candidate_order": None})
        new.append({**common, "policy": runner.POLICIES[0], "execution_candidate_order": 8, "controller_order": 8, "updates_enabled": False})
        new.append({**common, "policy": runner.POLICIES[1], "assigned": not review,
            "execution_candidate_order": None if review else 1, "controller_order": None if review else 10,
            "updates_enabled": None if review else True, "terminal_real": None if review else 100 + sid,
            "breach": None if review else False})
    return fits, price, update, pd.DataFrame(old), pd.DataFrame(new)


def toy_pairs_and_gates():
    paths, gates = [], []
    for label, rate in runner.RATES:
        for policy in runner.POLICIES:
            assigned = policy != runner.POLICIES[1]
            row = {"context": "toy", "scope": "outer_oof", "client": "C", "scenario_id": 1,
                "cost_label": label, "cost_rate": rate, "policy": policy, "assigned": assigned, "horizon": 3,
                **{f: 1. for f in runner.FINANCIAL}}
            if not assigned:
                row.update(dict.fromkeys(runner.FINANCIAL))
            paths.append(row)
            if not assigned: continue
            for t in range(3):
                gates.append({**row, "year": t + 1, "action": "hold" if label == "zero" and t == 1 else "restore",
                    "actual_trade": not (label == "zero" and t == 1), "pending_at_decision": False,
                    "ever_breached": t == 2, "target_after": 2 if label == "zero" and t > 0 else 1,
                    "target_updated": label == "zero" and t == 1})
    return pd.DataFrame(paths), pd.DataFrame(gates)


class SelectedControllerStressTests(unittest.TestCase):
    def test_protocol_is_blocked_until_audited_parents_and_freeze(self):
        protocol = json.loads(runner.PROTOCOL.read_text()); runner.validate_protocol(protocol, ready=False)
        protocol["execution_ready"] = False; protocol["parents_ready"] = False
        with self.assertRaises(RuntimeError): runner.validate_protocol(protocol)
        for mutate in (lambda p: p["rates"].reverse(), lambda p: p["rates"][0].update(cost_rate=False),
            lambda p: p["client"].update(horizon=15), lambda p: p["data"].update(allowed_scenario_ids=[1, 180])):
            broken = deepcopy(protocol); mutate(broken)
            with self.assertRaises(ValueError): runner.validate_protocol(broken, ready=False)

    def test_frozen_plan_reads_parent_selected_mode_price_and_review(self):
        args = toy_inputs(); plan = runner.build_controller_plan(*args)["full_development"]
        fixed, expanded = [plan["learners"][p] for p in runner.POLICIES[:2]]
        self.assertEqual(fixed["price_kind"], "risk_first"); self.assertFalse(fixed["updates_enabled"])
        self.assertEqual(expanded["price_order"], 1); self.assertTrue(expanded["updates_enabled"])
        self.assertEqual(expanded["annual_price"], .001)
        self.assertNotIn("terminal_real", expanded["parent_row_metadata"])
        self.assertNotIn("breach", expanded["parent_row_metadata"])
        self.assertEqual(plan["evaluation_ids"], [1, 2])
        reviewed = runner.build_controller_plan(*toy_inputs(True))["full_development"]["learners"][runner.POLICIES[1]]
        self.assertIsNone(reviewed["selected_controller_order"]); self.assertIsNone(reviewed["updates_enabled"])
        bad = list(toy_inputs()); bad[2]["full_development"]["parent_fixed_selected_candidate_order"] = 0
        with self.assertRaises(RuntimeError): runner.build_controller_plan(*bad)

    def test_parent_manifest_and_audit_hashes_must_pass_and_be_current(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = root / "parent"; directory.mkdir()
            (root / "core.py").write_text("core"); (directory / "paths.csv").write_text("data")
            source = {"core.py": runner.sha(root / "core.py")}
            artifact = {"parent/paths.csv": runner.sha(directory / "paths.csv")}
            manifest = {"sources": source, "sources_at_end": source, "input_hashes": {}, "input_hashes_at_end": {},
                "outputs_sha256": {"paths.csv": artifact["parent/paths.csv"]}, "final_scenarios_loaded": False, "scenario_ids_loaded": list(range(1, 121))}
            audit = {"status": "passed", "source_and_artifact_sha256_start": source | artifact, "source_and_artifact_sha256_end": source | artifact}
            runner.write_json(directory / "manifest.json", manifest); runner.write_json(directory / "independent_audit.json", audit)
            with patch.object(runner, "ROOT", root):
                runner.checked_parent(directory)
                (directory / "paths.csv").write_text("changed")
                with self.assertRaises(RuntimeError): runner.checked_parent(directory)
                (directory / "paths.csv").write_text("data"); audit["status"] = "failed"
                runner.write_json(directory / "independent_audit.json", audit)
                with self.assertRaises(RuntimeError): runner.checked_parent(directory)
                audit["status"] = "passed"; runner.write_json(directory / "independent_audit.json", audit)
                manifest["sources"] = manifest["sources_at_end"] = {"core.py": "contradictory-source-claim"}
                manifest["input_hashes"] = manifest["input_hashes_at_end"] = source
                runner.write_json(directory / "manifest.json", manifest)
                with self.assertRaises(RuntimeError): runner.checked_parent(directory)

    def test_one_frozen_controller_only_and_known_cost_enters_actual_and_Q(self):
        market = np.zeros((1, 3, 3)); market[0, 0, 0] = .5
        forecast = np.zeros((2, 3, 3)); pi, fpi = np.zeros((1, 3)), np.zeros((2, 3))
        targets = [[.2, .2, .6], [.5, 0, .5], [.7, .1, .2]]
        chosen = {"selected_controller_order": 4, "price_order": 4, "price_kind": "finite", "annual_price": .03, "updates_enabled": False}
        with patch.object(runner, "H", 3):
            for label, rate in runner.RATES:
                with patch.object(runner, "simulate_controllers", wraps=runner.simulate_controllers) as call:
                    arm = runner.evaluate_frozen_controller(market, pi, forecast, fpi, targets, chosen, rate)
                    self.assertEqual(len(call.call_args.args[5]), 1)
                    np.testing.assert_array_equal(call.call_args.args[5], [.03])
                    self.assertFalse(call.call_args.kwargs["updates_enabled"])
                    self.assertEqual(call.call_args.kwargs["cost_rate"], rate)
                    self.assertEqual(call.call_args.kwargs["horizon"], 3)
                self.assertAlmostEqual(arm["fees"][0, 0], 50 * rate / (1 + .5 * rate), places=13)
                self.assertAlmostEqual(arm["qg_restore"][0, -1], np.log1p(-.1 * rate / (1 + .5 * rate)), places=14)
                np.testing.assert_array_equal(arm["annual_prices"], [.03])
            with patch.object(runner, "simulate_controllers", side_effect=AssertionError("Review must not simulate")):
                self.assertIsNone(runner.evaluate_frozen_controller(market, pi, forecast, fpi, targets, {"selected_controller_order": None}, .01))
        changed = chosen | {"annual_price": .1}
        with self.assertRaises(RuntimeError): runner.evaluate_frozen_controller(market, pi, forecast, fpi, targets, changed, .01)

    def test_catalog_metadata_is_excluded_honestly_while_all_behavior_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "parent.npz"; prefix = "ctx__C__learner"
            np.savez(path, **{prefix + "__scenario_ids": np.array([1]), prefix + "__annual_prices": np.array([0., .03, np.inf]),
                prefix + "__real_wealth": np.array([[100., 105.]]), prefix + "__actions": np.array([["restore"]])})
            arm = {"annual_prices": np.array([.03]), "real_wealth": np.array([[100., 105.]]), "actions": np.array([["restore"]])}
            with np.load(path, allow_pickle=False) as archive:
                receipt = runner.compare_parent_trace(arm, [1], archive, prefix, .03)
                self.assertEqual(receipt["excluded_metadata_fields"], ["annual_prices"])
                self.assertEqual(receipt["new_actual_price_catalog_size"], 1)
                self.assertEqual(receipt["parent_catalog_metadata"]["risk_first_indices"], [2])
                self.assertEqual(receipt["behavior_state_Q_fields"], 3)
                changed = arm | {"real_wealth": np.array([[100., 104.]])}
                with self.assertRaises(AssertionError): runner.compare_parent_trace(changed, [1], archive, prefix, .03)

    def test_equal_thirds_remains_annual_after_breach_reference_B_holds(self):
        market = np.array([[[-.9, 0, 0], [.2, 0, 0], [0, 0, 0]]]); pi = np.zeros((1, 3))
        equal = runner.simulate_paired(market, pi, [1 / 3] * 3, .1, cost_rate=.0025)
        annual = runner.augment_reference(runner.reference.squeeze_pending(equal["annual"]), -1)
        pending = runner.augment_reference(runner.reference.squeeze_pending(equal["pending_hold"]), 1)
        self.assertEqual(annual["first_crossing"][0], 1)
        np.testing.assert_array_equal(annual["actions"], "restore")
        np.testing.assert_array_equal(annual["pending_at_decision"], False)
        np.testing.assert_array_equal(annual["target_updated"], False)
        np.testing.assert_array_equal(annual["current_target_index"], -1)
        np.testing.assert_array_equal(pending["actions"][0, 1:], "hold")
        np.testing.assert_array_equal(pending["fees"][0, 1:], 0)
        self.assertGreater(annual["fees"][0, 1], 0)

    def test_paired_defaults_and_contrasts_preserve_review_null(self):
        frame, _ = toy_pairs_and_gates()
        pairs, summary = runner.paired_table(frame)
        self.assertEqual(len(pairs), 20)
        review = pairs.loc[pairs.policy == runner.POLICIES[1]]
        self.assertFalse(review.common_auto.any())
        self.assertTrue(review.filter(like="delta_").isna().all().all())
        self.assertTrue(summary.loc[summary.policy == runner.POLICIES[1], "mean_delta_log_growth"].isna().all())
        contrasts, _ = runner.paired_table(frame, contrasts=True)
        self.assertEqual(len(contrasts), 20)
        self.assertFalse(contrasts.loc[contrasts.policy == runner.POLICIES[1], "common_auto"].any())
        self.assertNotIn("delta_terminal_target_index", contrasts)

    def test_whole_trajectory_target_action_change_denominators(self):
        frame, gates = toy_pairs_and_gates()
        with patch.object(runner, "H", 3):
            paths, summary = runner.decision_changes(gates, frame)
            zero = paths.loc[(paths.cost_label == "zero") & (paths.policy == runner.POLICIES[0])].iloc[0]
            self.assertEqual(zero.all_decisions, 3); self.assertEqual(zero.both_prebreach_postinitial_decisions, 1)
            self.assertAlmostEqual(zero.action_changed_fraction, 1 / 3)
            self.assertAlmostEqual(zero.target_after_changed_fraction, 2 / 3)
            self.assertAlmostEqual(zero.target_updated_changed_fraction, 1 / 3)
            self.assertTrue(paths.loc[~paths.assigned, "all_decisions"].isna().all())
            self.assertTrue(summary.loc[summary.policy == runner.POLICIES[1], "action_changed_fraction"].isna().all())
            with self.assertRaises(RuntimeError): runner.decision_changes(gates.iloc[1:], frame)

    def test_base_CSV_compares_finance_flags_identity_and_null_masks(self):
        expected = pd.DataFrame([{"sid": 1, "wealth": 103., "assigned": True}, {"sid": 2, "wealth": None, "assigned": False}])
        runner.compare_frame(expected, expected, list(expected), ["sid"])
        bad = expected.copy(); bad.loc[1, "wealth"] = 0
        with self.assertRaises(RuntimeError): runner.compare_frame(bad, expected, list(expected), ["sid"])
        bad = expected.copy(); bad.loc[0, "wealth"] = 102
        with self.assertRaises(AssertionError): runner.compare_frame(bad, expected, list(expected), ["sid"])


if __name__ == "__main__": unittest.main()
