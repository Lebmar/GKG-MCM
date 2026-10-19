import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fixed_diagnostic_runner_test", ROOT / "scripts/run_fixed_state_forecast_diagnostic.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FixedStateDiagnosticRunnerTests(unittest.TestCase):
    def test_full_inventory_uses_only_guarded_development_and_keeps_skips_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); parent = root / "parent"; parent.mkdir()
            (root / "docs").mkdir(); protocol_path = root / "docs/new.json"; out = root / "new"
            groups = {"actual": {}, "validation": {}}; arrays = {k: {} for k in groups}
            identities, jobs = [], []
            for g in range(27):
                scope = "actual" if g < 3 else "validation"; key = f"group{g}"
                context = "full_development" if g == 0 else "outer_fold_1" if g == 1 else "outer_fold_5" if g == 2 else f"parent{g}"
                jid = f"job{g}"; train = [1] if g == 0 else [2]
                jobs.append({"job_id": jid, "training_ids": train, "parent_context": context})
                groups[scope][key] = {"scenario_ids": [1], "q_subset_scenario_id": 1,
                    "q_subset_years": [2, 16, 30], "selected_candidate_order": 4,
                    "metadata": {"job_id": jid, "context": context}}
                arm = {"holdings": np.tile([30., 30., 40.], (1, 31, 1)),
                    "cpi": np.ones((1, 31)), "real_peak": np.full((1, 31), 100.),
                    "running_mdd": np.zeros((1, 31)), "ever_breached": np.zeros((1, 31), bool),
                    "pending_at_decision": np.zeros((1, 30), bool), "actions": np.full((1, 30), "hold"),
                    "forecast_n": np.ones((1, 30), int), "targets": np.array([.3, .3, .4]),
                    "annual_prices": np.array([.03]), "predicted_score": np.zeros((1, 30))}
                for action in ("hold", "restore"):
                    arm["qg_" + action] = np.zeros((1, 30)); arm["qbad_" + action + "_count"] = np.zeros((1, 30), int)
                for year in (2, 16, 30):
                    pending = g >= 25 and year == 30
                    if pending: arm["pending_at_decision"][0, year - 1] = True
                    identities.append({"scope": scope, "group": key, "scenario_id": 1, "year": year,
                        "training_n": 1, "status": "pending_skip_no_replacement" if pending else "audited"})
                for name, values in arm.items(): arrays[scope][key + "__" + name] = values
            noauto = []
            for j in range(3):
                jid = f"missing{j}"; jobs.append({"job_id": jid, "training_ids": [2], "parent_context": f"outer_missing{j}"})
                noauto.append({"job_id": jid, "scope": "actual", "scenario_id": 1,
                               "years": [2, 16, 30], "reason": "no_automated_selected_price_no_replacement"})
            (parent / "trace_groups.json").write_text(json.dumps(groups))
            (parent / "independent_audit.json").write_text(json.dumps({"q_subset_results": identities, "q_nonautomated_job_skips": noauto}))
            for scope, file in (("actual", "actual_traces.npz"), ("validation", "validation_traces.npz")):
                np.savez_compressed(parent / file, **arrays[scope])
            (root / "docs/nested_dynamic_calibration_protocol.json").write_text(json.dumps({"jobs": jobs}))
            p = {"execution_ready": True, "execution_sources_sha256": {"stub": "a" * 64},
                 "data": {"scenario_ids": list(range(1, 121)), "allow_final": False},
                 "identities": identities, "nonautomated_jobs": noauto, "budget_minutes": 1}
            protocol_path.write_text(json.dumps(p)); calls = []

            def load(path, ids, *, allow_final):
                calls.append((list(ids), allow_final))
                return np.zeros((120, 30, 3)), np.zeros((120, 30)), np.arange(1, 121)

            with patch.multiple(runner, ROOT=root, PARENT=parent, PROTOCOL=protocol_path, OUT=out), \
                    patch.object(runner, "sources", return_value={"stub": "a" * 64}), \
                    patch.object(runner, "inputs", return_value={}), patch.object(runner, "load_paths", side_effect=load):
                runner.main()
            self.assertEqual(calls, [(list(range(1, 121)), False)])
            inventory = pd.read_csv(out / "state_inventory.csv"); branches = pd.read_csv(out / "branch_paths.csv")
            comp = pd.read_csv(out / "state_comparisons.csv"); manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(inventory.status.value_counts().to_dict(), {"active_evaluated": 79,
                "nonautomated_skip_no_replacement": 9, "pending_skip_no_replacement": 2})
            self.assertEqual((len(branches), len(comp)), (158, 79))
            self.assertTrue(set(branches.state_id).isdisjoint(set(inventory.loc[inventory.status.ne("active_evaluated"), "state_id"])))
            self.assertEqual(manifest["analysis_scope_counts"], {"inner_validation": 70, "outer_oof": 6, "full_development_descriptive": 3})
            self.assertTrue(branches.history_mdd.eq(0).all())
            self.assertTrue(branches.nominal_fees.eq(0).all())
            self.assertTrue(comp.score_action_agreement.all())
            self.assertFalse(manifest["final_scenarios_loaded"])
            self.assertFalse(manifest["new_policy_fit_or_selection"])


if __name__ == "__main__": unittest.main()
