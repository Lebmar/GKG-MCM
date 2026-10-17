"""Global-tolerance regressions; all inputs are synthetic score matrices."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from safeinvest_research.metrics import NUMERIC_TOL
from safeinvest_research.selection import objective_within_thresholds, optimize_menu


def independent_global_rule(scores, max_products=3, only_size=None):
    """Enumerate globally, without production comparison/pruning helpers."""
    scores = np.asarray(scores, float)
    oracle = [max([x for x in row if np.isfinite(x)], default=0.) for row in scores]
    menus = []
    sizes = [only_size] if only_size is not None else range(1, min(max_products, scores.shape[1])+1)
    for size in sizes:
        for ids in combinations(range(scores.shape[1]), size):
            regrets = []
            for j, row in enumerate(scores):
                achieved = max([row[i] for i in ids if np.isfinite(row[i])], default=None)
                if achieved is not None:
                    regrets.append(oracle[j]-achieved)
            menus.append((ids, len(regrets), max(regrets, default=0.),
                          sum(regrets)/len(regrets) if regrets else 0.))
    coverage = max(item[1] for item in menus)
    if coverage == 0:
        return [], [0, 0., 0.], (0, 0., 0.)
    rstar = min(item[2] for item in menus if item[1] == coverage)
    first = [item for item in menus if item[1] == coverage and item[2] <= rstar+NUMERIC_TOL]
    astar = min(item[3] for item in first)
    final = [item for item in first if item[3] <= astar+NUMERIC_TOL]
    chosen = min(final, key=lambda item: (len(item[0]), item[0]))
    return list(chosen[0]), list(chosen[1:]), (coverage, rstar, astar)


class GlobalToleranceTests(unittest.TestCase):
    def test_two_layer_tolerance_counterexample(self):
        t = 1e-12
        scores = [[3, 1+2.1*t, 3+.7*t, 3+2.8*t],
                  [3+.7*t, 2, 3+2.1*t, 1],
                  [2+.7*t, 2+2.8*t, 2+1.4*t, 2+.7*t],
                  [3+2.8*t, .7*t, 1+1.4*t, 3]]
        expected_menu, expected_objective, extrema = independent_global_rule(scores)
        for prune in (False, True):
            for batch_size in (1, 3, 50_000):
                with self.subTest(prune=prune, batch_size=batch_size), patch(
                        "safeinvest_research.selection._COMBINATION_BATCH_SIZE", batch_size):
                    actual = optimize_menu(scores, prune=prune)
                    self.assertEqual(actual["identified_indices"], expected_menu)
                    assert_allclose(actual["objective"], expected_objective, atol=1e-25, rtol=0)
                    self.assertLessEqual(actual["objective"][1], extrema[1]+NUMERIC_TOL)
                    self.assertTrue(objective_within_thresholds(actual["objective"], actual["global_thresholds"]))

    def test_pairwise_chain_cannot_expand_global_band(self):
        scores = [[0., .9e-12, 1.8e-12, 2.7e-12]]
        for batch_size in (1, 2, 50_000):
            with self.subTest(batch_size=batch_size), patch(
                    "safeinvest_research.selection._COMBINATION_BATCH_SIZE", batch_size):
                result = optimize_menu(scores, max_products=1, prune=False)
                self.assertEqual(result["identified_indices"], [2])
                self.assertEqual(result["global_thresholds"]["exact_min_maximum_regret"], 0.)
                self.assertLessEqual(result["objective"][1], NUMERIC_TOL)

    def test_near_tolerance_random_matrices_match_global_enumeration(self):
        rng = np.random.default_rng(2026101703)
        for case in range(80):
            scores = rng.integers(-3, 7, size=(4, 6)).astype(float)
            scores += rng.integers(0, 7, size=scores.shape)*.35e-12
            scores[rng.random(scores.shape) < .15] = -np.inf
            expected_menu, expected_objective, _ = independent_global_rule(scores)
            for prune in (True, False):
                with self.subTest(case=case, prune=prune), patch(
                        "safeinvest_research.selection._COMBINATION_BATCH_SIZE", 4):
                    result = optimize_menu(scores, prune=prune)
                    self.assertEqual(result["identified_indices"], expected_menu)
                    assert_allclose(result["objective"], expected_objective, atol=1e-24, rtol=0)

    def test_per_size_optima_have_independent_single_bands(self):
        scores = np.eye(3)
        result = optimize_menu(scores, prune=False)
        self.assertEqual(result["identified_indices"], [0, 1, 2])
        for record in result["optima_by_size"]:
            menu, objective, extrema = independent_global_rule(scores, only_size=record["size"])
            self.assertEqual(record["indices"], menu)
            assert_allclose(record["objective"], objective, atol=1e-15, rtol=0)
            self.assertAlmostEqual(record["thresholds"]["exact_min_maximum_regret"], extrema[1])
        self.assertEqual(result["enumerated_menu_count"], 7)
        self.assertGreater(result["score_cache_bytes"], 0)

    def test_completion_may_improve_score_but_cannot_receive_extra_tolerance(self):
        thresholds = {"coverage": 5, "maximum_regret_upper": 1e-12,
                      "mean_regret_upper": 1e-12}
        self.assertTrue(objective_within_thresholds([5, .9e-12, .8e-12], thresholds))
        self.assertTrue(objective_within_thresholds([5, 0., 0.], thresholds))
        self.assertFalse(objective_within_thresholds([5, 1.1e-12, .8e-12], thresholds))
        self.assertFalse(objective_within_thresholds([5, .9e-12, 1.1e-12], thresholds))
        self.assertFalse(objective_within_thresholds([4, 0., 0.], thresholds))

    def test_empty_feasibility_has_explicit_finite_thresholds(self):
        result = optimize_menu(np.full((3, 4), -np.inf))
        self.assertEqual(result["identified_indices"], [])
        self.assertEqual(result["objective"], [0, 0., 0.])
        self.assertEqual(result["enumerated_menu_count"], 0)
        self.assertTrue(objective_within_thresholds(result["objective"], result["global_thresholds"]))


if __name__ == "__main__":
    unittest.main()
