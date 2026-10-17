"""Short synthetic risk-tree checks; no market/competition files are read."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]

from experimental_execution.risk_tree_reference import (  # noqa: E402
    PROBABILITY_ROUNDOFF_GUARD, PolicyNode, PolicyPoint, RiskState, action_for_history, evaluate_policy,
    finite_mixture_bound, pareto_frontier, select_deterministic,
    solve_risk_tree, step_state,
)


def fee_bisection(holdings, target, rate):
    wealth = float(np.sum(holdings))
    low, high = 0.0, wealth * rate / (1 + rate)
    for _ in range(100):
        fee = (low + high) / 2
        transfer = sum(max((wealth - fee) * w - x, 0) for x, w in zip(holdings, target))
        if fee > rate * transfer:
            high = fee
        else:
            low = fee
    return (low + high) / 2


def independent_h2_pairs(gross, inflation, probabilities, target, cost, loss):
    """Enumerate actions at root and each observed first-outcome history.

    No reference accounting, interpolation, tree generator or risk helper is
    used.  Both actions continue after breach for this eight-policy example.
    """
    pairs = []
    for decisions in product((0, 1), repeat=1 + len(gross)):
        growth, risk = 0.0, 0.0
        for first, second in product(range(len(gross)), repeat=2):
            probability = probabilities[first] * probabilities[second]
            x, cpi, peak, breached = np.array([0., 0., 100.]), 1., 100., False
            for outcome, restore in ((first, decisions[0]), (second, decisions[1 + first])):
                if restore:
                    fee = fee_bisection(x, target, cost)
                    x = (x.sum() - fee) * target
                x *= gross[outcome]
                cpi *= 1 + inflation[outcome]
                real = x.sum() / cpi
                peak = max(peak, real)
                breached = breached or 1 - real / peak - loss > 1e-12
            growth += probability * np.log((x.sum() / cpi) / 100)
            risk += probability * int(breached)
        pairs.append((growth, risk))
    return pairs


class StateAccountingTests(unittest.TestCase):
    def test_breach_is_absorbing_after_full_recovery_and_zero_horizon(self):
        state = RiskState((100., 0., 0.))
        bad, gain, fee = step_state(state, "hold", [.5, 1, 1], 0, [1, 0, 0], cost_rate=.0025, loss_tolerance=.2)
        self.assertTrue(bad.breached)
        self.assertAlmostEqual(gain, np.log(.5))
        self.assertEqual(fee, 0)
        recovered, _, _ = step_state(bad, "hold", [3, 1, 1], 0, [1, 0, 0], cost_rate=.0025, loss_tolerance=.2)
        self.assertEqual(recovered.real_wealth, 150)
        self.assertEqual(recovered.real_peak, 150)
        self.assertTrue(recovered.breached)
        result = solve_risk_tree([[1, 1, 1]], [0], [1], [1, 0, 0], horizon=0, loss_tolerance=.2,
                                 alpha=.99, on_breach="continue", initial_state=recovered)
        self.assertIsNone(result.selected)
        self.assertEqual(result.points[0].q, 1)
        self.assertEqual(result.points[0].g, 0)

    def test_already_observed_drawdown_latches_even_if_flag_was_false(self):
        initial = RiskState((0., 0., 100.), real_peak=200.)
        result = solve_risk_tree([[1, 1, 2]], [0], [1], [0, 0, 1], horizon=1,
                                 loss_tolerance=.25, alpha=.99, on_breach="hold_only", initial_state=initial)
        self.assertTrue(result.model.initial_state.breached)
        self.assertTrue(all(p.q == 1 for p in result.points))
        self.assertIsNone(result.selected)

    def test_inflation_alone_can_make_all_bills_breach(self):
        result = solve_risk_tree([[1, 1, 1]], [.2], [1], [0, 0, 1], horizon=1,
                                 loss_tolerance=.1, alpha=.1, on_breach="hold_only")
        self.assertIsNone(result.selected)
        self.assertEqual(result.points[0].q, 1)
        self.assertAlmostEqual(result.points[0].g, -np.log(1.2))
        self.assertFalse(result.randomized_bound.feasible)

    def test_fee_once_changes_actual_real_wealth_and_can_trigger_annual_breach(self):
        initial = RiskState((0., 0., 100.))
        after, growth, fee = step_state(initial, "restore", [1, 1, 1], .02, [1, 0, 0],
                                        cost_rate=.0025, loss_tolerance=.001)
        self.assertAlmostEqual(fee, 100 * .0025 / 1.0025, places=13)
        self.assertAlmostEqual(after.real_wealth, 100 / (1.0025 * 1.02), places=12)
        self.assertAlmostEqual(growth, -np.log1p(.0025) - np.log1p(.02), places=14)
        self.assertTrue(after.breached)
        again, _, second_fee = step_state(after, "restore", [1, 1, 1], 0, [1, 0, 0],
                                           cost_rate=.0025, loss_tolerance=.001)
        self.assertEqual(second_fee, 0)
        self.assertTrue(again.breached)

    def test_hold_only_is_an_explicit_nonresponse_branch_not_an_adviser_payoff(self):
        initial = RiskState((0., 0., 100.), breached=True)
        common = dict(gross=[[2, 1, 1]], inflation=[0], probabilities=[1], target=[1, 0, 0],
                      horizon=2, loss_tolerance=.2, alpha=1, initial_state=initial, cost_rate=0)
        pending = solve_risk_tree(**common, on_breach="hold_only")
        continuing = solve_risk_tree(**common, on_breach="continue")
        self.assertEqual(pending.selected.g, 0)
        self.assertAlmostEqual(continuing.selected.g, 2 * np.log(2))
        self.assertEqual(pending.selected.q, 1)
        self.assertEqual(continuing.selected.q, 1)
        self.assertEqual(action_for_history(pending.selected.policy, []), "hold")
        self.assertEqual(action_for_history(pending.selected.policy, [0]), "hold")
        self.assertIn("nonresponse", pending.metadata["on_breach_meaning"])


class EnumerationTests(unittest.TestCase):
    def test_one_period_two_asset_example_matches_direct_enumeration(self):
        # Restore half into equity: good wealth150, bad wealth75, q=.25.
        common = dict(gross=[[2, 1, 1], [.5, 1, 1]], inflation=[0, 0], probabilities=[.75, .25],
                      target=[.5, 0, .5], horizon=1, loss_tolerance=.2, cost_rate=0,
                      on_breach="continue", prune=False)
        result = solve_risk_tree(**common, alpha=.2)
        by_action = {point.policy.action: point for point in result.points}
        self.assertEqual(by_action["hold"].g, 0)
        self.assertEqual(by_action["hold"].q, 0)
        self.assertAlmostEqual(by_action["restore"].g, .75 * np.log(1.5) + .25 * np.log(.75), places=14)
        self.assertEqual(by_action["restore"].q, .25)
        self.assertEqual(result.selected.policy.action, "hold")
        allowed = solve_risk_tree(**common, alpha=.25)
        self.assertEqual(allowed.selected.policy.action, "restore")

    def test_two_period_event_probability_is_point19_not_reset_point1(self):
        result = solve_risk_tree([[1, 1, 1], [.5, .5, .5]], [0, 0], [.9, .1], [0, 0, 1],
                                 horizon=2, loss_tolerance=.2, alpha=.1, on_breach="hold_only")
        self.assertTrue(all(abs(p.q - .19) < 1e-14 for p in result.points))
        self.assertIsNone(result.selected)
        self.assertAlmostEqual(result.points[0].g, 2 * .1 * np.log(.5), places=14)
        self.assertTrue(result.metadata["alpha_used_only_at_root"])

    def test_history_specific_conditional_budget_is_not_locally_reset(self):
        result = solve_risk_tree([[2, 1, 1], [.5, 1, 1]], [0, 0], [.9, .1], [1, 0, 0],
                                 horizon=2, loss_tolerance=.25, alpha=.02, cost_rate=0, on_breach="continue")
        # Hold first.  After the rare observed bad market row, take the risky
        # target on the second step.  Financial states after either first row
        # are identical cash states; observable histories are different.
        self.assertAlmostEqual(result.selected.g, .1 * (.9 * np.log(2) + .1 * np.log(.5)), places=14)
        self.assertAlmostEqual(result.selected.q, .01, places=14)
        self.assertEqual(action_for_history(result.selected.policy, []), "hold")
        self.assertEqual(action_for_history(result.selected.policy, [0]), "hold")
        self.assertEqual(action_for_history(result.selected.policy, [1]), "restore")
        # The risky child's conditional .1 risk exceeds root alpha .02, but
        # its root-weighted .01 event is feasible.  Screening every child by
        # alpha would wrongly remove this valid deterministic history tree.
        self.assertGreater(.1, result.alpha)

    def test_full_h2_enumeration_matches_independent_path_accounting(self):
        gross = np.array([[1.4, .95, 1], [.65, 1.1, 1]])
        inflation, probability, target = np.array([.02, 0]), np.array([.7, .3]), np.array([.6, .2, .2])
        result = solve_risk_tree(gross, inflation, probability, target, horizon=2, loss_tolerance=.2,
                                 alpha=.2, cost_rate=.0025, on_breach="continue", prune=False)
        expected = independent_h2_pairs(gross, inflation, probability, target, .0025, .2)
        self.assertEqual(len(result.points), 8)
        actual = sorted((p.g, p.q) for p in result.points)
        expected.sort()
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
        for point in result.points:
            replay = evaluate_policy(result, point.policy)
            self.assertAlmostEqual(replay.g, point.g, places=14)
            self.assertAlmostEqual(replay.q, point.q, places=14)

    def test_recursive_dominance_preserves_root_options_in_full_h3_tree(self):
        common = dict(gross=[[1.3, .9, 1], [.75, 1.1, 1], [1.02, 1.01, 1]],
                      inflation=[.01, .02, 0], probabilities=[.5, .3, .2], target=[.6, .15, .25],
                      horizon=3, loss_tolerance=.2, alpha=.2, on_breach="continue")
        compact, exhaustive = solve_risk_tree(**common), solve_risk_tree(**common, prune=False)
        self.assertEqual(len(exhaustive.points), 8192)
        self.assertLess(len(compact.points), len(exhaustive.points))
        for alpha in (0, .1, .2, .35, .7, 1):
            left, right = select_deterministic(compact.points, alpha), select_deterministic(exhaustive.points, alpha)
            self.assertEqual(left is None, right is None)
            if left is not None:
                self.assertAlmostEqual(left.g, right.g, places=12)
        self.assertAlmostEqual(compact.randomized_bound.g, exhaustive.randomized_bound.g, places=12)

    def test_duplicate_observable_rows_cannot_be_hidden_random_coins(self):
        with self.assertRaises(ValueError):
            solve_risk_tree([[2, 1, 1], [2, 1, 1]], [0, 0], [.5, .5], [1, 0, 0],
                             horizon=2, loss_tolerance=.2, alpha=.1, on_breach="continue")
        # The same asset row with genuinely distinct observable inflation is
        # still a different joint outcome, with actual real-wealth effects.
        result = solve_risk_tree([[2, 1, 1], [2, 1, 1]], [0, .1], [.5, .5], [1, 0, 0],
                                 horizon=1, loss_tolerance=.2, alpha=.1, on_breach="continue")
        self.assertEqual(len(result.model.gross), 2)


class FrontierAndBoundTests(unittest.TestCase):
    def test_exact_dominance_retains_nearby_risk_growth_tradeoff(self):
        hold = PolicyNode("hold")
        restore = PolicyNode("restore")
        points = [PolicyPoint(0, 0), PolicyPoint(.1, .1, restore), PolicyPoint(.1, .1, hold),
                  PolicyPoint(.1 + 1e-14, .1 + 1e-14), PolicyPoint(.09, .11)]
        frontier = pareto_frontier(points)
        self.assertEqual(len(frontier), 3)
        self.assertEqual(frontier[1].policy.action, "hold")
        self.assertEqual(frontier[2].q, .1 + 1e-14)

    def test_root_mixture_exhibits_deterministic_relaxation_gap(self):
        points = (PolicyPoint(0, 0), PolicyPoint(.1, .2))
        self.assertEqual(select_deterministic(points, .1).g, 0)
        bound = finite_mixture_bound(points, .1)
        self.assertTrue(bound.feasible)
        self.assertAlmostEqual(bound.g, .05, places=12)
        self.assertAlmostEqual(bound.q, .1, places=12)
        self.assertAlmostEqual(bound.dual_lambda, .5, places=12)
        self.assertAlmostEqual(bound.dual_value, .05, places=12)
        self.assertAlmostEqual(bound.duality_gap, 0, places=12)
        self.assertAlmostEqual(bound.deterministic_relaxation_gap, .05, places=12)
        self.assertEqual([i for i, _ in bound.components], [0, 1])
        np.testing.assert_allclose([p for _, p in bound.components], [.5, .5], atol=1e-13, rtol=0)
        self.assertEqual(bound.requested_alpha, .1)
        self.assertEqual(bound.effective_alpha, .1 + PROBABILITY_ROUNDOFF_GUARD)

    def test_mixture_primal_and_dual_match_independent_linear_programs(self):
        rng = np.random.default_rng(441)
        for _ in range(15):
            points = tuple(PolicyPoint(float(g), float(q)) for g, q in zip(rng.normal(0, .2, 8), rng.uniform(0, 1, 8)))
            for alpha in (.05, .3, .8, 1):
                bound = finite_mixture_bound(points, alpha)
                lp = linprog(-np.array([p.g for p in points]), A_ub=[[p.q for p in points]], b_ub=[bound.effective_alpha],
                             A_eq=[np.ones(len(points))], b_eq=[1], bounds=(0, None), method="highs")
                self.assertEqual(bound.feasible, lp.success)
                if lp.success:
                    self.assertAlmostEqual(bound.g, -lp.fun, places=10)
                    self.assertAlmostEqual(bound.dual_value, -lp.fun, places=10)

    def test_infeasible_mixture_and_zero_risk_budget_remain_explicit(self):
        bound = finite_mixture_bound([PolicyPoint(.2, .3)], .1)
        self.assertFalse(bound.feasible)
        self.assertIsNone(bound.g)
        self.assertIsNone(bound.dual_value)
        self.assertEqual(bound.components, ())
        zero = finite_mixture_bound([PolicyPoint(0, 0), PolicyPoint(.1, .2)], 0)
        self.assertEqual(zero.g, 0)
        self.assertEqual(zero.q, 0)

    def test_roundoff_budget_is_explicit_and_zero_never_admits_positive_risk(self):
        points = [PolicyPoint(0, 0), PolicyPoint(.1, .1 * .1)]
        self.assertGreater(points[1].q, .01)
        selected = select_deterministic(points, .01)
        self.assertEqual(selected, points[1])
        bound = finite_mixture_bound(points, .01)
        self.assertEqual(bound.requested_alpha, .01)
        self.assertGreater(bound.effective_alpha, .01)
        self.assertGreater(bound.q_minus_requested_alpha, 0)
        tiny = [PolicyPoint(0, 0), PolicyPoint(.1, 1e-16)]
        self.assertEqual(select_deterministic(tiny, 0).q, 0)
        self.assertEqual(finite_mixture_bound(tiny, 0).g, 0)
        result = solve_risk_tree([[1, 1, 1]], [0], [1], [0, 0, 1], horizon=0,
                                 loss_tolerance=.2, alpha=.01, on_breach="hold_only")
        self.assertEqual(result.metadata["requested_alpha"], .01)
        self.assertEqual(result.metadata["effective_alpha"], .01 + PROBABILITY_ROUNDOFF_GUARD)
        self.assertEqual(result.metadata["selected_q_minus_requested_alpha"], -.01)
        self.assertIn("relative budget expansion", result.metadata["probability_guard_meaning"])

    def test_model_copies_inputs_and_ignores_zero_probability_branches(self):
        gross = np.array([[1.2, 1, 1], [.01, .01, .01], [.8, 1, 1]])
        inflation = np.array([.01, 10, .02]); probability = np.array([.8, 0, .2]); target = np.array([.5, .1, .4])
        result = solve_risk_tree(gross, inflation, probability, target, horizon=1, loss_tolerance=.2,
                                 alpha=.2, on_breach="continue")
        expected_gross = gross[[0, 2]].copy()
        gross[:] = 3; inflation[:] = 3; probability[:] = 1 / 3; target[:] = [1, 0, 0]
        np.testing.assert_array_equal(result.model.gross, expected_gross)
        self.assertEqual(result.model.outcome_ids, (0, 2))
        for array in (result.model.gross, result.model.inflation, result.model.probabilities, result.model.target):
            self.assertFalse(array.flags.writeable)
        with self.assertRaises(ValueError):
            action_for_history(result.points[0].policy, [1])

    def test_invalid_inputs_and_response_rules_are_rejected(self):
        base = dict(gross=[[1, 1, 1]], inflation=[0], probabilities=[1], target=[.5, .2, .3],
                    horizon=1, loss_tolerance=.2, alpha=.1, on_breach="hold_only")
        variants = [dict(horizon=-1), dict(horizon=4), dict(horizon=1.0), dict(horizon=True),
                    dict(gross=np.ones((4, 3)), inflation=[0] * 4, probabilities=[.25] * 4),
                    dict(gross=[[0, 1, 1]]), dict(inflation=[-1]), dict(probabilities=[.9]),
                    dict(probabilities=[-1]), dict(alpha=-.1), dict(alpha=1.1), dict(loss_tolerance=-.1),
                    dict(cost_rate=1), dict(target=[.5, .2, .2]), dict(on_breach="adviser_recovery"),
                    dict(initial_state=RiskState((0, 0, 0))), dict(initial_state=RiskState((0, 0, 100), cpi=0)),
                    dict(initial_state=RiskState((0, 0, 100), real_peak=50))]
        for change in variants:
            with self.subTest(change=change), self.assertRaises(ValueError):
                solve_risk_tree(**(base | change))
        missing = base.copy(); missing.pop("on_breach")
        with self.assertRaises(TypeError):
            solve_risk_tree(**missing)


if __name__ == "__main__":
    unittest.main()
