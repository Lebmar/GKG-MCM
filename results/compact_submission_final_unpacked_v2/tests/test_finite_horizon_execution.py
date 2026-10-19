"""Synthetic accounting/Bellman checks; no market files or fitted menus used."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from experimental_execution.finite_horizon import (  # noqa: E402
    ACTION_TOL,
    barycentric_coordinates,
    evaluate_action,
    exact_fee_fraction,
    interpolate_on_mesh,
    solve_finite_horizon,
    transition_terms,
    triangular_mesh,
)
from safeinvest.core import execute_target  # noqa: E402


def independent_fee(u, target, cost):
    """Scalar bisection of the money equation, independent of the executor."""
    low, high = 0.0, cost / (1.0 + cost)
    for _ in range(100):
        fee = (low + high) / 2
        transfer = sum(max((1 - fee) * w - x, 0) for x, w in zip(u, target))
        if fee > cost * transfer:
            high = fee
        else:
            low = fee
    return (low + high) / 2


def exact_small_tree(u, target, gross, remaining, cost):
    """Exhaust the two-action tree with no interpolation, for tiny horizons."""
    if remaining == 0:
        return 0.0
    action_values = []
    for after, fee in ((np.asarray(u), 0.0), (np.asarray(target), independent_fee(u, target, cost))):
        future = []
        for draw in gross:
            portfolio = float(after @ draw)
            drift = after * draw / portfolio
            future.append(np.log(portfolio) + exact_small_tree(drift, target, gross, remaining - 1, cost))
        action_values.append(np.log1p(-fee) + float(np.mean(future)))
    return max(action_values)


class TriangleTests(unittest.TestCase):
    def test_mesh_vertices_boundaries_and_no_extra_thirds(self):
        mesh = triangular_mesh()
        self.assertEqual(mesh.size, 231)
        self.assertEqual(triangular_mesh(.025).size, 861)
        np.testing.assert_allclose(mesh.states.sum(axis=1), 1, atol=2e-16, rtol=0)
        for vertex in np.eye(3):
            self.assertTrue(np.any(np.all(mesh.states == vertex, axis=1)))
        self.assertFalse(np.any(np.all(mesh.states == np.full(3, 1 / 3), axis=1)))
        self.assertFalse(mesh.states.flags.writeable)

    def test_nonnegative_barycentric_and_exact_linear_functions(self):
        mesh = triangular_mesh()
        rng = np.random.default_rng(103)
        points = np.vstack((np.eye(3), mesh.states, rng.dirichlet(np.ones(3), 250),
                            [[.025, .025, .95], [.49, .51, 0], [0, .37, .63], [1 / 3] * 3]))
        vertices, coefficients = barycentric_coordinates(points, mesh)
        self.assertTrue(np.all(coefficients >= 0))
        np.testing.assert_allclose(coefficients.sum(axis=1), 1, atol=2e-16, rtol=0)
        reconstructed = (mesh.states[vertices] * coefficients[..., None]).sum(axis=1)
        np.testing.assert_allclose(reconstructed, points, atol=2e-15, rtol=0)
        linear = 2 + mesh.states @ np.array([3.0, -2.0, .7])
        np.testing.assert_allclose(interpolate_on_mesh(linear, points, mesh),
                                   2 + points @ np.array([3.0, -2.0, .7]), atol=2e-14, rtol=0)
        np.testing.assert_allclose(interpolate_on_mesh(mesh.states, points.reshape(2, -1, 3), mesh),
                                   points.reshape(2, -1, 3), atol=2e-15, rtol=0)

    def test_transition_joint_mixture_is_stochastic_and_reproduces_affine_continuation(self):
        mesh = triangular_mesh(.1)
        gross = np.array([[[1.3, .8, 1.01], [.7, 1.2, 1.02]],
                          [[1.1, 1.3, .99], [1.4, .6, 1.0]]])
        states = np.vstack((mesh.states, [1 / 3] * 3))
        p, log_return = transition_terms(gross, states, mesh, chunk_size=7)
        self.assertTrue(np.all(p >= 0))
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=3e-16, rtol=0)
        draws = gross.reshape(-1, 3)
        money = states[:, None, :] * draws
        total = money.sum(axis=-1)
        drift = money / total[..., None]
        affine = lambda x: 4 + x @ np.array([2.0, -3.0, .5])
        np.testing.assert_allclose(p @ affine(mesh.states), affine(drift).mean(axis=1), atol=3e-14, rtol=0)
        np.testing.assert_allclose(log_return, np.log(total).mean(axis=1), atol=2e-15, rtol=0)
        # Joint observations are retained: replacing the log expectation by a
        # log of the arithmetic mean would violate Jensen in this example.
        self.assertGreater(float(np.max(np.log(total.mean(axis=1)) - log_return)), .01)

    def test_operator_unchanged_by_permuting_or_replicating_empirical_bank(self):
        mesh = triangular_mesh(.2)
        gross = np.array([[[1.4, .7, 1.01], [.8, 1.3, .99], [1.1, .9, 1.02]]])
        p, ell = transition_terms(gross, mesh.states, mesh)
        other_p, other_ell = transition_terms(np.tile(gross[:, ::-1], (3, 1, 1)), mesh.states, mesh,
                                              chunk_size=1)
        np.testing.assert_allclose(other_p, p, atol=3e-15, rtol=0)
        np.testing.assert_allclose(other_ell, ell, atol=2e-15, rtol=0)


class ExecutionTests(unittest.TestCase):
    def test_exact_fee_matches_independent_equation_and_scale_invariance(self):
        current = np.array([[0, 0, 1], [.23, .62, .15], [.5, .5, 0]])
        target = np.array([[1, 0, 0], [1 / 3] * 3, [.5, .5, 0]])
        fee = exact_fee_fraction(current, target)
        for index, (u, w) in enumerate(zip(current, target)):
            expected = independent_fee(u, w, .0025)
            self.assertAlmostEqual(fee[index], expected, places=15)
            self.assertAlmostEqual(fee[index], .0025 * np.maximum((1 - fee[index]) * w - u, 0).sum(), places=15)
            for wealth in (.01, 100, 1e8):
                post, cash_fee, moved = execute_target(wealth * u, w)
                self.assertAlmostEqual(cash_fee / wealth, expected, places=15)
                np.testing.assert_allclose(post.sum() + cash_fee, wealth, atol=wealth * 2e-15, rtol=0)
                self.assertAlmostEqual(cash_fee / wealth, .0025 * moved / wealth, places=15)
        self.assertAlmostEqual(fee[0], .0025 / 1.0025, places=15)
        self.assertEqual(fee[-1], 0.0)

    def test_horizon_one_matches_direct_enumeration_at_off_mesh_states(self):
        gross = np.array([[[1.3, .8, 1.01], [.7, 1.2, 1.02]],
                          [[1.1, 1.3, .99], [1.4, .6, 1.0]]])
        targets = np.array([[1 / 3] * 3, [.7, .1, .2], [0, 0, 1]])
        states = np.array([[.49, .51, 0], [.123, .234, .643], [.001, .998, .001]])
        solution = solve_finite_horizon(gross, targets, horizon=2)
        for k, target in enumerate(targets):
            decision = evaluate_action(solution, states, 1, k)
            holds = [np.mean([np.log(float(u @ draw)) for draw in gross.reshape(-1, 3)]) for u in states]
            trades = [np.log1p(-independent_fee(u, target, .0025))
                      + np.mean([np.log(float(target @ draw)) for draw in gross.reshape(-1, 3)]) for u in states]
            np.testing.assert_allclose(decision["hold_q"], holds, atol=2e-15, rtol=0)
            np.testing.assert_allclose(decision["trade_q"], trades, atol=2e-15, rtol=0)
            np.testing.assert_allclose(decision["one_period_gain"], np.array(trades) - holds, atol=2e-15, rtol=0)
            np.testing.assert_array_equal(decision["trade"], np.array(trades) - holds > ACTION_TOL)
        # The off-mesh equal-thirds target has its own exact current operator.
        p, ell = transition_terms(gross, targets, solution.mesh)
        np.testing.assert_allclose(solution.P_target, p, atol=2e-15, rtol=0)
        np.testing.assert_allclose(solution.L_target, ell, atol=2e-15, rtol=0)

    def test_two_period_gain_amortizes_initial_fee_despite_negative_myopic_gain(self):
        gross = np.array([[[1.0015, 1, 1]]])
        solution = solve_finite_horizon(gross, [1, 0, 0], horizon=30)
        one = evaluate_action(solution, [0, 0, 1], 1, 0)
        two = evaluate_action(solution, [0, 0, 1], 2, 0)
        self.assertFalse(one["trade"])
        self.assertLess(one["gain"], 0)
        self.assertTrue(two["trade"])
        self.assertGreater(two["gain"], 0)
        self.assertLess(two["one_period_gain"], 0)
        self.assertAlmostEqual(two["trade_q"], 2 * np.log(1.0015) - np.log1p(.0025), places=14)
        self.assertAlmostEqual(two["hold_q"], 0.0, places=14)
        self.assertAlmostEqual(two["gain"], exact_small_tree(np.array([0, 0, 1]), np.array([1, 0, 0]),
                                                              gross.reshape(-1, 3), 2, .0025), places=14)

    def test_three_period_example_and_client_horizon_are_distinct(self):
        solution = solve_finite_horizon([[[1.001, 1, 1]]], [1, 0, 0], horizon=30)
        decision = evaluate_action(solution, [0, 0, 1], np.array([0, 1, 2, 3]), 0)
        np.testing.assert_array_equal(decision["trade"], [False, False, False, True])
        self.assertAlmostEqual(decision["trade_q"][3], 3 * np.log(1.001) - np.log1p(.0025), places=14)

    def test_future_restoration_fee_is_included_and_matches_small_tree(self):
        draws = np.array([[1.32, .88, 1], [.88, 1.32, 1]])
        target = np.array([.5, .5, 0])
        solution = solve_finite_horizon(draws[None, :, :], target, horizon=2)
        where = np.flatnonzero(np.all(solution.mesh.states == target, axis=1))[0]
        future_fee = .00025 / 1.00125
        expected = 2 * np.log(1.1) + np.log1p(-future_fee)
        self.assertAlmostEqual(solution.V[2, where, 0], expected, places=14)
        self.assertLess(solution.V[2, where, 0], 2 * np.log(1.1))
        self.assertAlmostEqual(solution.V[2, where, 0], exact_small_tree(target, target, draws, 2, .0025), places=14)
        decision = evaluate_action(solution, [[.6, .4, 0], [.4, .6, 0]], 1, 0)
        np.testing.assert_array_equal(decision["trade"], [True, True])
        np.testing.assert_allclose(decision["fee_fraction"], future_fee, atol=2e-15, rtol=0)

    def test_direct_current_return_avoids_grid_q_interpolation_counterexample(self):
        gross = np.array([[[1.2, .8, 1], [.8, 1.2, 1]]])
        target, actual = [.5, .5, 0], [.49, .51, 0]
        for step in (.05, .025):
            solution = solve_finite_horizon(gross, target, horizon=1, step=step)
            decision = evaluate_action(solution, actual, 1, 0)
            self.assertFalse(decision["trade"])
            self.assertAlmostEqual(decision["hold_q"], .5 * np.log(.999984), places=14)
            self.assertLess(decision["gain"], 0)
            if step == .05:
                # Interpolating grid *Qhold*, rather than future V only,
                # would spuriously suggest restoring in this exact H1 case.
                wrong_hold = interpolate_on_mesh(solution.Qhold_grid[1, :, 0], actual, solution.mesh)
                self.assertGreater(decision["trade_q"] - wrong_hold, ACTION_TOL)

    def test_zero_cost_identity_and_terminal_hold_never_charge(self):
        gross = np.array([[[1.2, 1.1, 1], [.8, 1.1, 1]]])
        targets = np.array([[1 / 3] * 3, [0, 0, 1], [0, 1, 0]])
        solution = solve_finite_horizon(gross, targets, horizon=3)
        same = evaluate_action(solution, targets, 2, np.arange(3))
        np.testing.assert_array_equal(same["trade"], False)
        np.testing.assert_array_equal(same["gain"], 0)
        np.testing.assert_array_equal(same["fee_fraction"], 0)
        terminal = evaluate_action(solution, [1, 0, 0], 0, np.arange(3))
        for key in ("hold_q", "trade_q", "gain", "fee_fraction", "current_log_return", "one_period_gain"):
            np.testing.assert_array_equal(terminal[key], 0)
        np.testing.assert_array_equal(terminal["trade"], False)
        free = solve_finite_horizon(gross, [0, 1, 0], horizon=2, cost_rate=0)
        decision = evaluate_action(free, [0, 0, 1], 2, 0)
        self.assertTrue(decision["trade"])
        self.assertEqual(decision["fee_fraction"], 0)
        bills = evaluate_action(solution, [0, 0, 1], 3, 1)
        self.assertFalse(bills["trade"])
        self.assertEqual(bills["fee_fraction"], 0)

    def test_equal_asset_returns_give_no_fee_or_spurious_tie_trades(self):
        gross = np.array([[[1.1, 1.1, 1.1], [.9, .9, .9]]])
        solution = solve_finite_horizon(gross, [[1 / 3] * 3, [1, 0, 0]], horizon=3, cost_rate=0)
        for h in range(4):
            np.testing.assert_allclose(solution.V[h], h * .5 * np.log(.99), atol=2e-15, rtol=0)
        decision = evaluate_action(solution, [[.1, .2, .7], [0, 1, 0]], [1, 3], [0, 1])
        np.testing.assert_array_equal(decision["trade"], False)
        self.assertTrue(np.all(np.abs(decision["gain"]) < ACTION_TOL))

    def test_batch_targets_equal_separate_fixed_target_problems(self):
        gross = np.array([[[1.32, .88, 1], [.88, 1.32, 1]]])
        targets = np.array([[.5, .5, 0], [1 / 3] * 3, [0, 0, 1]])
        batch = solve_finite_horizon(gross, targets, horizon=3, step=.1)
        states = np.array([[.13, .17, .7], [.61, .24, .15], [.2, .3, .5]])
        combined = evaluate_action(batch, states, [1, 2, 3], np.arange(3))
        for k, target in enumerate(targets):
            single = solve_finite_horizon(gross, target, horizon=3, step=.1)
            np.testing.assert_allclose(batch.V[:, :, k], single.V[:, :, 0], atol=3e-15, rtol=0)
            separate = evaluate_action(single, states[k], k + 1, 0)
            for field in combined:
                np.testing.assert_allclose(combined[field][k], separate[field], atol=3e-15, rtol=0)

    def test_frozen_inputs_causal_signature_and_readonly_tables(self):
        gross = np.array([[[1.2, .9, 1], [.8, 1.1, 1]]])
        targets = np.array([[1 / 3] * 3])
        solution = solve_finite_horizon(gross, targets, horizon=2)
        before = evaluate_action(solution, [.2, .3, .5], 2, 0)
        gross[:] = 9
        targets[:] = [1, 0, 0]
        after = evaluate_action(solution, [.2, .3, .5], 2, 0)
        for field in before:
            np.testing.assert_array_equal(before[field], after[field])
        for value in (solution.targets, solution.gross_samples, solution.mesh.states, solution.V,
                      solution.Qhold_grid, solution.tradebase, solution.P_grid, solution.P_target):
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):
                value.flat[0] = 0
        self.assertEqual(set(inspect.signature(evaluate_action).parameters),
                         {"solution", "current_weights", "remaining", "target_id", "chunk_size"})
        with self.assertRaises(TypeError):
            evaluate_action(solution, [.2, .3, .5], 1, 0, future_gross=[2, 1, 1])
        self.assertFalse(solution.metadata["continuous_state_optimum_claimed"])
        self.assertFalse(solution.metadata["sample_independence_claimed"])
        self.assertIsNone(solution.metadata["continuous_state_error_bound"])

    def test_invalid_inputs_are_rejected(self):
        for step in (0, -1, .03, 1.1, np.nan, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                triangular_mesh(step)
        for gross in (np.ones((2, 3)), np.empty((0, 1, 3)), [[[1, 0, 1]]], [[[1, np.inf, 1]]]):
            with self.subTest(gross=str(gross)), self.assertRaises(ValueError):
                solve_finite_horizon(gross, [0, 0, 1], horizon=1)
        for target in ([.2, .3, .4], [-.1, .6, .5], [np.nan, 0, 1], [[1, 0]], []):
            with self.subTest(target=target), self.assertRaises(ValueError):
                solve_finite_horizon([[[1, 1, 1]]], target, horizon=1)
        for horizon in (-1, 2.0, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                solve_finite_horizon([[[1, 1, 1]]], [0, 0, 1], horizon=horizon)
        for cost in (-.1, 1, np.nan, False):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                solve_finite_horizon([[[1, 1, 1]]], [0, 0, 1], horizon=1, cost_rate=cost)
        solution = solve_finite_horizon([[[1, 1, 1]]], [0, 0, 1], horizon=1)
        for current, horizon, target_id in (([.2, .3, .4], 1, 0), ([0, 0, 1], 2, 0),
                                             ([0, 0, 1], -1, 0), ([0, 0, 1], 1, 1),
                                             ([0, 0, 1], 1.0, 0), ([0, 0, 1], True, 0)):
            with self.subTest(current=current, horizon=horizon, target_id=target_id), self.assertRaises(ValueError):
                evaluate_action(solution, current, horizon, target_id)
        with self.assertRaises(ValueError):
            evaluate_action(solution, [0, 0, 1], 1, 0, chunk_size=0)


if __name__ == "__main__":
    unittest.main()
