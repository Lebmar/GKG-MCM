"""Do not disguise a support-property failure with a numerical tolerance."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pathwise_runner", ROOT / "scripts/run_pathwise_fan_execution.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class SupportPropertyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.base = {"real_wealth": np.array([[100., 110.]]), "ever_breached": np.array([[False, False]])}
        self.item = {"name": "Balanced", "candidate_local_index": 117, "candidate_original232_index": 155}

    def check(self, proposed):
        return runner.verify_training_property(self.base, proposed, context="toy", client="C", ids=[1],
            horizon=1, tolerance=.4, item=self.item)

    def test_small_computed_loss_is_explicit_even_within_audit_guard(self):
        proposed = {"real_wealth": np.array([[100., 110. * np.exp(-1e-12)]]),
                    "ever_breached": np.array([[False, False]])}
        rows, summary = self.check(proposed)
        self.assertLess(rows[0]["total_log_growth_delta"], 0)
        self.assertTrue(rows[0]["computed_growth_delta_strictly_negative"])
        self.assertEqual(summary["strictly_negative_computed_growth_delta_count"], 1)
        self.assertEqual(summary["negative_beyond_audit_guard_count"], 0)
        self.assertEqual(summary["gate_growth_allowance"], 0)

    def test_material_growth_property_failure_stops_experiment(self):
        proposed = {"real_wealth": np.array([[100., 109.]]), "ever_breached": np.array([[False, False]])}
        with self.assertRaises(RuntimeError): self.check(proposed)

    def test_new_breach_is_never_accepted_under_numerical_guard(self):
        proposed = {"real_wealth": np.array([[100., 111.]]), "ever_breached": np.array([[False, True]])}
        with self.assertRaises(RuntimeError): self.check(proposed)

    def test_terminal_gain_does_not_erase_old_absorbing_breach(self):
        self.base["ever_breached"][0, -1] = True
        proposed = {"real_wealth": np.array([[100., 112.]]), "ever_breached": np.array([[False, True]])}
        rows, summary = self.check(proposed)
        self.assertTrue(rows[0]["reference_breach"])
        self.assertTrue(rows[0]["pathwise_breach"])
        self.assertEqual(summary["breach_order_violations"], 0)


if __name__ == "__main__": unittest.main()
