"""Artifact admission checks protect the frozen empirical comparison."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fan_runner", ROOT / "scripts/run_fan_rollout_execution.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FrozenAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contexts = json.loads((runner.STRICT / "nested_context_refits.json").read_text())
        cls.mapping = json.loads((runner.STRICT / "candidate_mapping.json").read_text())

    def test_known_whole_path_partition_is_admitted(self):
        runner.validate_contexts(self.contexts)

    def test_a_training_test_overlap_is_rejected(self):
        contexts = deepcopy(self.contexts)
        contexts["outer_fold_1"]["training_ids"][0] = contexts["outer_fold_1"]["outer_test_ids"][0]
        with self.assertRaises(ValueError): runner.validate_contexts(contexts)

    def test_test_path_reused_across_folds_is_rejected(self):
        contexts = deepcopy(self.contexts)
        contexts["outer_fold_2"]["outer_test_ids"] = contexts["outer_fold_1"]["outer_test_ids"][:]
        contexts["outer_fold_2"]["training_ids"] = contexts["outer_fold_1"]["training_ids"][:]
        with self.assertRaises(ValueError): runner.validate_contexts(contexts)

    def test_final_identity_or_boolean_cannot_enter_context(self):
        for bad in (121, True):
            contexts = deepcopy(self.contexts)
            contexts["outer_fold_1"]["training_ids"][0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError): runner.validate_contexts(contexts)

    def test_local_global_candidate_confusion_is_rejected(self):
        item = deepcopy(self.contexts["full_development"]["raw_fit"]["menu"][0])
        runner.validate_product(item, self.mapping)
        item["candidate_local_index"] = item["candidate_original232_index"]
        with self.assertRaises(ValueError): runner.validate_product(item, self.mapping)

    def test_changed_target_or_zero_asset_is_rejected(self):
        for weights in ([.25, .31, .44], [.25, 0, .75]):
            item = deepcopy(self.contexts["full_development"]["raw_fit"]["menu"][0]); item["weights"] = weights
            with self.subTest(weights=weights), self.assertRaises(ValueError): runner.validate_product(item, self.mapping)

    def test_stale_frozen_artifact_fails_before_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); artifact = root / "fit.json"; artifact.write_text("original")
            protocol = {"frozen_input_sha256": {"fit.json": hashlib.sha256(artifact.read_bytes()).hexdigest()}}
            with patch.object(runner, "ROOT", root):
                runner.frozen_inputs(protocol)
                artifact.write_text("changed")
                with self.assertRaises(RuntimeError): runner.frozen_inputs(protocol)

    def test_review_zero_imputation_is_rejected(self):
        row = {"context": "full_development", "client": "A", "view": "effective", "scenario_id": 1,
               "assignment": "Review", "assigned": False,
               **dict.fromkeys([*runner.METRICS, "breach", "excess", "candidate_local_index"])}
        runner.verify_review(pd.DataFrame([row]), "full_development", "A", [1])
        row["mdd"] = 0.
        with self.assertRaises(RuntimeError): runner.verify_review(pd.DataFrame([row]), "full_development", "A", [1])

    def test_protocol_scope_cannot_silently_change(self):
        protocol = json.loads(runner.PROTOCOL.read_text()); runner.validate_protocol(protocol)
        protocol["state_and_accounting"]["cost_rate"] = 0.
        with self.assertRaises(ValueError): runner.validate_protocol(protocol)


if __name__ == "__main__": unittest.main()
