"""Mathematical evidence boundaries, using no market data."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from safeinvest_research.evidence import simultaneous_upper, supported_count_limit, minimum_zero_count_sample


class EvidenceTests(unittest.TestCase):
    def test_zero_count_closed_form_and_binomial_inversion(self):
        per_event = .05 / 1160
        values = simultaneous_upper(np.array([0, 1, 12, 120]), 120, 1160)
        self.assertAlmostEqual(values[0], 1 - per_event ** (1 / 120), places=13)
        for k, upper in zip([0, 1, 12], values[:3]):
            self.assertAlmostEqual(binom.cdf(k, 120, upper), per_event, places=13)
        self.assertEqual(values[-1], 1)

    def test_family_wide_10pct_supports_only_zero_not_one(self):
        self.assertEqual(supported_count_limit(120, .10, 1160), 0)
        self.assertLess(simultaneous_upper(np.array(0), 120, 1160), .10)
        self.assertGreater(simultaneous_upper(np.array(1), 120, 1160), .10)
        self.assertEqual(supported_count_limit(96, .10, 1160), 0)

    def test_no_support_is_not_a_negative_risk_estimate(self):
        self.assertEqual(supported_count_limit(120, .05, 1160), -1)
        required = minimum_zero_count_sample(.05, 1160)
        self.assertEqual(required, 196)
        self.assertGreater(simultaneous_upper(np.array(0), required - 1, 1160), .05)
        self.assertLessEqual(simultaneous_upper(np.array(0), required, 1160), .05)

    def test_bounds_monotone_with_count_family_size_and_sample_evidence(self):
        counts = np.arange(11)
        bounds = simultaneous_upper(counts, 120, 1160)
        self.assertTrue(np.all(np.diff(bounds) > 0))
        self.assertTrue(np.all(bounds > simultaneous_upper(counts, 120, 1)))
        self.assertLess(simultaneous_upper(np.array(0), 120, 1160),
                        simultaneous_upper(np.array(0), 96, 1160))

    def test_invalid_counts_or_confidence_fail(self):
        for args in [([.5], 120, 1160), ([121], 120, 1160), ([0], 0, 1160),
                     ([0], 120, True), ([0], 120, 1160, 0)]:
            with self.assertRaises(ValueError):
                simultaneous_upper(*args)


if __name__ == "__main__":
    unittest.main()
