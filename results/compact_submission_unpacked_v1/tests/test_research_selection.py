"""Independent exhaustive synthetic checks for common-menu selection.

These tests never load market files. The brute-force oracle uses Python loops,
all original targets, and an exact lexicographic objective on integer scores.
"""

from __future__ import annotations

import copy
from itertools import combinations
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from safeinvest_research.selection import complete_menu, fit_menu, optimize_menu


def brute_force_menu(scores, max_products=3):
    """Independent enumeration with no dominance pruning or production helpers."""
    clients = len(scores)
    candidates = len(scores[0])
    oracle = [max((score for score in row if np.isfinite(score)), default=0.0) for row in scores]
    options = []
    for size in range(1, min(max_products, candidates) + 1):
        for menu in combinations(range(candidates), size):
            achieved = []
            regrets = []
            for client in range(clients):
                available = [scores[client][i] for i in menu if np.isfinite(scores[client][i])]
                if available:
                    achieved.append(client)
                    regrets.append(oracle[client] - max(available))
            count = len(achieved)
            objective = (count, max(regrets, default=0.0), sum(regrets) / count if count else 0.0)
            options.append(((-count, objective[1], objective[2], size, menu), objective, menu))
    if not any(np.isfinite(score) for row in scores for score in row):
        return (0, 0.0, 0.0), ()
    _, objective, menu = min(options, key=lambda item: item[0])
    return objective, menu


class MenuOptimizerTests(unittest.TestCase):
    def test_profile_dominance_pruning_matches_independent_bruteforce(self):
        rng = np.random.default_rng(872026)
        # Integer scores keep objective separations far from the 1e-12 guard.
        for case in range(160):
            scores = rng.integers(-4, 11, size=(int(rng.integers(1, 6)), int(rng.integers(3, 8)))).astype(float)
            scores[rng.random(scores.shape) < .30] = -np.inf
            limit = int(rng.integers(1, 4))
            expected_objective, expected_menu = brute_force_menu(scores.tolist(), limit)
            for prune in (True, False):
                with self.subTest(case=case, prune=prune, limit=limit):
                    actual = optimize_menu(scores, max_products=limit, prune=prune)
                    assert_allclose(actual["objective"], expected_objective, atol=1e-12, rtol=0)
                    self.assertEqual(len(actual["identified_indices"]), len(expected_menu))
                    self.assertEqual(actual["identified_indices"], list(expected_menu))

    def test_identifies_smallest_one_two_or_three_product_menu(self):
        cases = [
            ([[3, 2, 1], [2, 1, 0]], 1),
            ([[3, -np.inf, 2], [-np.inf, 3, 2]], 2),
            ([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3),
        ]
        for scores, size in cases:
            with self.subTest(size=size):
                result = optimize_menu(scores)
                self.assertEqual(len(result["identified_indices"]), size)
                assert_allclose(result["objective"], [len(scores), 0, 0])

    def test_coverage_precedes_regret_and_impossible_clients_are_explicit(self):
        result = optimize_menu([[10, -np.inf, 0], [-np.inf, 10, 0], [-np.inf] * 3], max_products=1)
        self.assertEqual(result["identified_indices"], [2])
        assert_allclose(result["objective"], [2, 10, 10])
        self.assertEqual(result["oracle_feasible"], [True, True, False])

    def test_maximum_regret_precedes_mean_then_mean_breaks_ties(self):
        minimax = optimize_menu([[8, 7, 10], [8, 10, 0], [8, 10, 0]], max_products=1)
        self.assertEqual(minimax["identified_indices"], [0])
        assert_allclose(minimax["objective"], [3, 2, 2])
        mean_tie = optimize_menu([[8, 8, 10], [8, 10, 0], [10, 10, 0]], max_products=1)
        self.assertEqual(mean_tie["identified_indices"], [1])
        assert_allclose(mean_tie["objective"], [3, 2, 2 / 3])

    def test_equal_targets_have_no_artificial_demand_for_extra_products(self):
        result = optimize_menu([[3, 3, 3], [2, 2, 2]])
        self.assertEqual(result["identified_indices"], [0])
        self.assertEqual(result["useful_candidate_count"], 1)

    def test_higher_id_dominator_cannot_erase_global_menu_id_tie(self):
        scores = [[1, 2, 3], [3, 3, 0]]
        for prune in (True, False):
            with self.subTest(prune=prune):
                result = optimize_menu(scores, prune=prune)
                self.assertEqual(result["identified_indices"], [0, 2])
                assert_allclose(result["objective"], [2, 0, 0])

    def test_tolerance_dominance_cannot_delete_a_cycle_of_all_candidates(self):
        scores = [[2e-12, 1e-12, 0], [1e-12, 0, 2e-12], [0, 2e-12, 1e-12]]
        result = optimize_menu(scores)
        self.assertEqual(result["objective"][0], 3)
        self.assertGreater(result["useful_candidate_count"], 0)
        self.assertTrue(result["identified_indices"])

    def test_every_client_infeasible_returns_empty_identified_menu(self):
        result = optimize_menu(np.full((5, 4), -np.inf))
        self.assertEqual(result["identified_indices"], [])
        self.assertEqual(result["objective"], [0, 0, 0])
        self.assertEqual(result["oracle_feasible"], [False] * 5)


class CompletionTests(unittest.TestCase):
    def test_identical_reference_points_do_not_fabricate_three_risk_roles(self):
        for identified in ([], [0]):
            with self.subTest(identified=identified):
                result = complete_menu(identified, [.01, .01, .01], [.1, .1, .1])
                self.assertFalse(result["complete"])
                self.assertLess(len(result["indices"]), 3)
                self.assertEqual(result["distinct_reference_risk_count"], 1)
                self.assertFalse(result["three_distinct_reference_risks"])

    def test_extension_preserves_demand_and_uses_frontier_risk_breadth(self):
        result = complete_menu([1], [.01, .02, .03, .01], [.05, .1, .3, .4])
        self.assertTrue(result["complete"])
        self.assertEqual(result["indices"], [0, 1, 2])
        self.assertEqual(set(result["extension_indices"]), {0, 2})
        self.assertNotIn(3, result["reference_frontier"])
        self.assertTrue(result["three_distinct_reference_risks"])


class FitMenuTests(unittest.TestCase):
    def fixture(self):
        weights = np.array([[0, 0, 1], [.3, .3, .4], [.6, .3, .1], [1, 0, 0]])
        metrics = {15: {
            "log_growth": np.tile([.01, .02, .03, .04], (4, 1)),
            "mdd": np.tile([.02, .1, .2, .5], (4, 1)),
        }}
        profiles = [
            {"name": "Tight", "horizon": 15, "loss_tolerance": .05, "liquidity": "high"},
            {"name": "Loose", "horizon": 15, "loss_tolerance": .30, "liquidity": "low"},
            {"name": "Unfit", "horizon": 15, "loss_tolerance": .01, "liquidity": "high"},
        ]
        floors = {"high": .3, "medium": .1, "low": 0}
        return metrics, weights, profiles, floors

    def test_one_common_menu_with_feasible_assignment_and_review(self):
        metrics, weights, profiles, floors = self.fixture()
        result = fit_menu(metrics, weights, profiles, [0, 1], liquidity_floors=floors)
        self.assertEqual(result["n_training"], 2)
        self.assertEqual(result["selection"]["identified_indices"], [0, 2])
        self.assertEqual([item["candidate_index"] for item in result["menu"]], [0, 2, 3])
        self.assertEqual(result["assignments"], {"Tight": "Conservative", "Loose": "Balanced", "Unfit": "Review"})
        shared_ids = {item["candidate_index"] for item in result["menu"]}
        for profile in profiles:
            details = [row for row in result["client_product_details"] if row["client"] == profile["name"]]
            self.assertEqual({row["candidate_index"] for row in details}, shared_ids)
            chosen = [row for row in details if row["chosen"]]
            self.assertEqual(len(chosen), 0 if profile["name"] == "Unfit" else 1)
            self.assertTrue(all(row["sample_feasible"] for row in chosen))

    def test_unselected_scenarios_cannot_affect_any_fit_or_reference_extension(self):
        metrics, weights, profiles, floors = self.fixture()
        before = fit_menu(metrics, weights, profiles, [0, 1], liquidity_floors=floors)
        changed = copy.deepcopy(metrics)
        changed[15]["log_growth"][2:] = [[100, -100, 200, -200], [-200, 100, -100, 200]]
        changed[15]["mdd"][2:] = [[.99, .01, .99, 0], [0, .99, .01, .99]]
        after = fit_menu(changed, weights, profiles, [0, 1], liquidity_floors=floors)
        self.assertEqual(before, after)

    def test_repeated_indices_have_bootstrap_weight_and_count(self):
        metrics, weights, profiles, floors = self.fixture()
        metrics[15]["log_growth"][:2, 0] = [.03, 0]
        metrics[15]["mdd"][:2, 0] = [.02, .06]
        result = fit_menu(metrics, weights, profiles, [0, 0, 1], alpha=1, liquidity_floors=floors)
        detail = next(row for row in result["client_product_details"] if row["client"] == "Tight" and row["candidate_index"] == 0)
        self.assertEqual(detail["n"], 3)
        self.assertEqual(detail["breach_count"], 1)
        self.assertAlmostEqual(detail["breach_probability"], 1 / 3)
        self.assertAlmostEqual(detail["mean_log_growth"], .02)

    def test_alpha_and_liquidity_are_real_constraints_not_ranking_labels(self):
        metrics, weights, _, floors = self.fixture()
        profiles = [{"name": "Only", "horizon": 15, "loss_tolerance": .15, "liquidity": "high"}]
        # With alpha=0 no observed violation is permitted, and bills floor .3
        # excludes the high-equity candidates even when alpha is relaxed to 1.
        strict = fit_menu(metrics, weights, profiles, [0, 1], alpha=0, liquidity_floors=floors)
        relaxed = fit_menu(metrics, weights, profiles, [0, 1], alpha=1, liquidity_floors=floors)
        for result in (strict, relaxed):
            selected = next(row for row in result["client_product_details"] if row["chosen"])
            self.assertEqual(selected["candidate_index"], 1)
            self.assertTrue(selected["liquidity_pass"])
        no_floor = fit_menu(metrics, weights, profiles, [0, 1], alpha=1,
                            liquidity_floors={"high": 0, "medium": 0, "low": 0})
        self.assertEqual(next(row["candidate_index"] for row in no_floor["client_product_details"] if row["chosen"]), 3)

    def test_invalid_alpha_floors_and_training_indices_are_rejected(self):
        metrics, weights, profiles, floors = self.fixture()
        for alpha in (-.01, 1.01, np.nan):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                fit_menu(metrics, weights, profiles, [0], alpha=alpha, liquidity_floors=floors)
        for floor in (-.1, 1.1, np.nan):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                fit_menu(metrics, weights, profiles, [0], liquidity_floors={"high": floor, "medium": .1, "low": 0})
        for indices in ([], [0.0], [-1], [4]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                fit_menu(metrics, weights, profiles, indices, liquidity_floors=floors)


if __name__ == "__main__":
    unittest.main()
