"""Independent synthetic checks, with no supplied market rows or policy tuning."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from experimental_execution.forecast_ensemble import (
    build_forecast_ensemble, build_forecast_weights,
)


def scalar_conditional(training, actual):
    """Own scalar four-channel Gaussian and fixed uniform-shrinkage oracle."""
    n = len(training)
    means = [math.fsum(float(row[c]) for row in training) / n for c in range(4)]
    scales = [max(math.sqrt(math.fsum((float(row[c]) - means[c]) ** 2
        for row in training) / n), 1e-8) for c in range(4)]
    kernels = [math.exp(-.5 * math.fsum(((float(actual[c]) - float(row[c])) / scales[c]) ** 2
        for c in range(4)) / 4) for row in training]
    total = math.fsum(kernels)
    return [(.5 / n + .5 * kernel / total) if total else 1 / n for kernel in kernels]


class ForecastEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.ids = [9, 1, 7, 4, 2]
        self.training = np.array([
            [.12, -.01, .005, .03], [-.10, .04, -.01, .07],
            [.03, .02, .006, .02], [.25, -.02, .003, .01],
            [-.04, .06, .001, .04],
        ])
        self.actual = np.array([[.05, .025, .004, .025], [-.08, .05, .001, .05]])

    def ensemble(self, **kwargs):
        return build_forecast_ensemble(self.ids, period_index=4, n_actual=2,
            training_previous_state=self.training, actual_previous_state=self.actual, **kwargs)

    def test_inception_only_uniform_and_no_future_or_previous_state(self):
        result = build_forecast_ensemble([2, 1], period_index=0, n_actual=3)
        self.assertEqual(result.labels, ("uniform",))
        np.testing.assert_array_equal(result.weights, np.full((3, 1, 2), .5))
        self.assertFalse(result.metadata["conditional_available"])
        self.assertEqual(result.metadata["block_training_ids"], [])
        with self.assertRaises(ValueError):
            build_forecast_ensemble(self.ids, period_index=0, n_actual=2,
                training_previous_state=self.training, actual_previous_state=self.actual)

    def test_conditional_matches_independent_scalar_calculation(self):
        result = self.ensemble()
        self.assertEqual(result.weights.shape, (2, 5, 5))
        for a in range(2):
            np.testing.assert_allclose(result.weights[a, 1],
                scalar_conditional(self.training, self.actual[a]), atol=2e-16, rtol=0)
        np.testing.assert_allclose(result.weights.sum(axis=2), 1, atol=2e-16, rtol=0)
        self.assertTrue(np.all(result.weights[:, 1] >= .5 / 5))

    def test_sorted_balanced_blocks_map_back_to_original_order(self):
        result = self.ensemble()
        self.assertEqual(result.metadata["block_training_ids"], [[1, 2], [4, 7], [9]])
        expected = np.array([[0, .5, 0, 0, .5], [0, 0, .5, .5, 0], [1, 0, 0, 0, 0]])
        for a in range(2):
            np.testing.assert_array_equal(result.weights[a, 2:], expected)
        self.assertFalse(result.metadata["independent_ensemble_samples"])
        self.assertFalse(result.metadata["confidence_interval"])
        self.assertFalse(result.metadata["safe_support_probabilities"])

    def test_training_id_order_is_only_an_alignment_not_a_predictor(self):
        order = np.array([2, 0, 4, 1, 3])
        result = self.ensemble()
        permuted = build_forecast_ensemble(np.asarray(self.ids)[order], period_index=4, n_actual=2,
            training_previous_state=self.training[order], actual_previous_state=self.actual)
        np.testing.assert_allclose(permuted.weights, result.weights[:, :, order], atol=3e-16, rtol=0)
        self.assertEqual(permuted.metadata["block_training_ids"], result.metadata["block_training_ids"])

    def test_training_moments_do_not_use_actual_batch(self):
        together = self.ensemble()
        separate = build_forecast_ensemble(self.ids, period_index=4, n_actual=1,
            training_previous_state=self.training, actual_previous_state=self.actual[:1])
        np.testing.assert_array_equal(together.weights[:1], separate.weights)
        np.testing.assert_allclose(together.metadata["training_mean"],
            [math.fsum(self.training[:, c]) / 5 for c in range(4)], atol=1e-17, rtol=0)
        self.assertEqual(together.metadata["training_mean"], separate.metadata["training_mean"])
        self.assertEqual(together.metadata["training_scale"], separate.metadata["training_scale"])

    def test_channel_translation_and_positive_scaling_invariance(self):
        result = self.ensemble()
        scale = np.array([2., 3., .8, 1.5]); shift = np.array([.02, -.01, .1, .05])
        transformed = build_forecast_ensemble(self.ids, period_index=4, n_actual=2,
            training_previous_state=self.training * scale + shift,
            actual_previous_state=self.actual * scale + shift)
        np.testing.assert_allclose(transformed.weights, result.weights, atol=3e-16, rtol=0)

    def test_constant_channels_use_floor_and_stay_finite(self):
        states = np.tile([.02, .01, .005, .03], (5, 1))
        result = build_forecast_ensemble(self.ids, period_index=1, n_actual=2,
            training_previous_state=states, actual_previous_state=states[:2])
        np.testing.assert_array_equal(result.weights[:, :2], np.full((2, 2, 5), .2))
        self.assertEqual(result.metadata["training_scale"], [1e-8] * 4)

    def test_all_far_away_underflow_falls_back_to_uniform(self):
        actual = np.array([[1e100] * 4, self.actual[0]])
        result = build_forecast_ensemble(self.ids, period_index=1, n_actual=2,
            training_previous_state=self.training, actual_previous_state=actual)
        np.testing.assert_array_equal(result.weights[0, :2], np.full((2, 5), .2))
        self.assertEqual(result.metadata["kernel_uniform_fallback_rows"], [0])
        np.testing.assert_allclose(result.weights[1, 1], scalar_conditional(self.training, self.actual[0]),
            atol=2e-16, rtol=0)

    def test_inputs_not_mutated_and_output_weights_read_only(self):
        before_train, before_actual = self.training.copy(), self.actual.copy()
        result = self.ensemble()
        np.testing.assert_array_equal(self.training, before_train)
        np.testing.assert_array_equal(self.actual, before_actual)
        with self.assertRaises(ValueError):
            result.weights[0, 0, 0] = 0

    def test_development_only_unique_integer_training_ids(self):
        for ids in ([1, 121, 2], [0, 2, 3], [1, 1, 2], [1., 2., 3.], [True, False, True], [], [[1, 2, 3]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_forecast_ensemble(ids, period_index=0, n_actual=1)

    def test_invalid_periods_or_row_counts_rejected(self):
        for period in (True, -1, 30, 1.5):
            with self.subTest(period=period), self.assertRaises(ValueError):
                build_forecast_ensemble(self.ids, period_index=period, n_actual=1)
        for n in (0, -1, 1., True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                build_forecast_ensemble(self.ids, period_index=0, n_actual=n)

    def test_invalid_state_or_missing_state_rejected(self):
        for bad in (np.zeros((5, 3)), np.full((5, 4), np.nan), np.full((5, 4), -1.),
                    np.full((5, 4), np.inf), np.full((5, 4), True)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                build_forecast_ensemble(self.ids, period_index=1, n_actual=2,
                    training_previous_state=bad, actual_previous_state=self.actual)
        with self.assertRaises(ValueError):
            build_forecast_ensemble(self.ids, period_index=1, n_actual=2)
        with self.assertRaises(ValueError):
            build_forecast_ensemble([1, 2], period_index=1, n_actual=1,
                training_previous_state=np.zeros((2, 4)), actual_previous_state=np.zeros((1, 4)))

    def test_dict_adapter_infers_actual_count_and_preserves_calendar_honesty(self):
        direct = self.ensemble()
        adapted = build_forecast_weights(self.training, self.actual, self.ids)
        np.testing.assert_array_equal(adapted["weights"], direct.weights)
        self.assertEqual(adapted["labels"], list(direct.labels))
        self.assertIsNone(adapted["metadata"]["period_index"])
        self.assertTrue(adapted["metadata"]["calendar_alignment_supplied_by_caller"])

    def test_dict_adapter_inception_and_mismatches(self):
        result = build_forecast_weights(None, None, self.ids, actual_n=2)
        self.assertEqual(result["weights"].shape, (2, 1, 5))
        self.assertEqual(result["metadata"]["period_index"], 0)
        for fn in (
            lambda: build_forecast_weights(None, None, self.ids),
            lambda: build_forecast_weights(None, self.actual, self.ids),
            lambda: build_forecast_weights(self.training, None, self.ids),
            lambda: build_forecast_weights(self.training, self.actual, self.ids, actual_n=1),
        ):
            with self.assertRaises(ValueError):
                fn()


if __name__ == "__main__":
    unittest.main()
