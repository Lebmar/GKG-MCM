"""Safety and exactness checks for the disposable-copy replay adapter."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("replay_execution", Path(__file__).resolve().parents[1] / "scripts/reproduce_execution_stage.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ExactArtifactTests(unittest.TestCase):
    def test_reject_original_root_before_deletion(self):
        with self.assertRaises(AssertionError):
            replay.replay_stage(replay.ROOT, "toy", "toy.py")

    def test_bytes_and_different_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = [Path(folder) / name for name in ["left.csv", "right.csv"]]
            left.write_text("value\n1\n"); right.write_bytes(left.read_bytes())
            self.assertEqual(replay.identical_artifact(left, right)["comparison"], "byte_exact")
            right.write_text("value\n1.000000000000001\n")
            with self.assertRaises(AssertionError): replay.identical_artifact(left, right)

    def test_npz_exact_numeric_and_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = [Path(folder) / name for name in ["left.npz", "right.npz"]]
            np.savez(left, x=np.array([1., np.nan]), label=np.array(["hold"]))
            np.savez_compressed(right, x=np.array([1., np.nan]), label=np.array(["hold"]))
            self.assertEqual(replay.identical_artifact(left, right)["arrays"], 2)
            np.savez_compressed(right, x=np.array([1. + 1e-15, np.nan]), label=np.array(["hold"]))
            with self.assertRaises(AssertionError): replay.identical_artifact(left, right)

    def test_npz_dtype_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = [Path(folder) / name for name in ["left.npz", "right.npz"]]
            np.savez(left, x=np.array([1], dtype="int64"))
            np.savez(right, x=np.array([1], dtype="int32"))
            with self.assertRaises(AssertionError): replay.identical_artifact(left, right)

    def test_copy_stage_restores_frozen_parent_metadata_and_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root, copy = [Path(folder) / name for name in ["original", "copy"]]
            name = "toy_stage"; original = root / "results/model_review" / name
            target = copy / "results/model_review" / name
            for path in [original, target]: path.mkdir(parents=True)
            artifact = b"value\n1\n"
            (original / "values.csv").write_bytes(artifact)
            frozen = {"outputs_sha256": {"values.csv": replay.sha(original / "values.csv")}, "final_scenarios_loaded": False, "time": "frozen"}
            (original / "manifest.json").write_text(json.dumps(frozen))
            (target / "stale.csv").write_text("must disappear before run")

            def fake_run(command, **kwargs):
                self.assertFalse(target.exists())
                self.assertEqual(kwargs["cwd"], copy.resolve())
                self.assertTrue(str(copy.resolve() / "scripts") in command[1])
                target.mkdir(parents=True)
                (target / "values.csv").write_bytes(artifact)
                (target / "manifest.json").write_text(json.dumps({**frozen, "time": "replayed"}))
                return type("Result", (), {"returncode": 0, "stdout": str(copy.resolve()), "stderr": ""})()

            before = (original / "manifest.json").read_bytes()
            with patch.object(replay, "ROOT", root), patch.object(replay.subprocess, "run", fake_run):
                receipt, log = replay.replay_stage(copy, name, "toy.py")
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual((original / "manifest.json").read_bytes(), before)
            self.assertEqual((target / "manifest.json").read_bytes(), before)
            self.assertEqual(log, "<isolated-copy>")
            self.assertFalse((target / "stale.csv").exists())


if __name__ == "__main__": unittest.main()
