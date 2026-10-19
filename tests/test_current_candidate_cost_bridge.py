"""Pure synthetic guards for exact reuse of historical cost evidence."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/bridge_current_candidate_cost_evidence.py"
SPEC = importlib.util.spec_from_file_location("current_candidate_cost_bridge", SOURCE)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class ExactBridgeTests(unittest.TestCase):
    def test_paired_nan_is_allowed_but_not_missing_value(self):
        a = np.array([1., np.nan])
        self.assertTrue(bridge.exact_array(a, a.copy(), "q")["exact_values_and_NULL_mask"])
        with self.assertRaises(ValueError):
            bridge.exact_array(a, np.array([1., 0.]), "q")

    def test_one_ulp_difference_is_rejected(self):
        with self.assertRaises(ValueError):
            bridge.exact_array(np.array([1.]), np.array([np.nextafter(1., 2.)]), "wealth")

    def test_shape_and_dtype_are_part_of_contract(self):
        with self.assertRaises(ValueError):
            bridge.exact_array(np.array([1.]), np.array([[1.]]), "wealth")
        with self.assertRaises(ValueError):
            bridge.exact_array(np.array([1], dtype=np.int64), np.array([1.]), "actions")

    def test_action_string_difference_is_rejected(self):
        with self.assertRaises(ValueError):
            bridge.exact_array(np.array(["hold"], dtype="U7"), np.array(["restore"], dtype="U7"), "actions")

    def test_new_or_missing_field_stops_bridge(self):
        a = {"wealth": np.array([100.])}
        with self.assertRaises(ValueError):
            bridge.pair_arrays(a, a, expected_fields=["wealth", "fees"])
        with self.assertRaises(ValueError):
            bridge.pair_arrays(a, {}, expected_fields=["wealth"])

    def test_only_declared_metadata_can_supply_absent_field(self):
        a = {"targets": np.array([.25,.25,.5])}
        result = bridge.pair_arrays(a, {}, expected_fields=["targets"], explicit_metadata=a)
        self.assertIn("explicitly declared", result[0]["historical_provenance"])

    def test_duplicate_and_incomplete_scenario_sets_fail_before_financials(self):
        with self.assertRaises(ValueError):
            bridge.compare_csv_paths([{"scenario_id": "1"}] * 120, [])
        with self.assertRaises(ValueError):
            bridge.summarize_paths([{"scenario_id": str(i)} for i in range(1,120)])

    def test_csv_bool_null_and_nonfinite_are_distinct(self):
        self.assertIsNone(bridge.finite_or_none(""))
        self.assertIs(bridge.finite_or_none("False"), False)
        self.assertEqual(bridge.finite_or_none("0"), 0.)
        with self.assertRaises(ValueError):
            bridge.finite_or_none("nan")


if __name__ == "__main__":
    unittest.main()
